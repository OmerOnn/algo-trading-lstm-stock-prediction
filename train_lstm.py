"""
Train the LSTM future-return regressor.

Pipeline per horizon
--------------------
1. Build (or load) the supervised panel and the decomposed training target.
2. Split the timeline chronologically with a purge equal to the horizon.
3. Fit the market-return model and set its authority from purged folds.
4. Fit an ensemble of LSTM regressors on date-grouped batches, checkpointing on a
   criterion that requires both magnitude skill and cross-sectional skill.
5. Estimate predictive uncertainty with Monte Carlo dropout across the ensemble.
6. Hand everything to the shared evaluation layer, which selects calibration out
   of fold, builds calibrated intervals, freezes the decision rule on validation
   and then reports once on the development holdout.

Everything after step 5 is shared verbatim with the XGBoost trainer, so a
difference between the two reports is a difference between the two models rather
than between two evaluation implementations.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DateGroupedBatchSampler, StockSequenceDataset
from src.device import dataloader_device_kwargs, get_best_device, move_batch_to_device
from src.evaluation import build_forecast_table, evaluate_model, print_decile_table
from src.experiments import ExperimentCandidate, WalkForwardExperiment
from src.features import market_wide_feature_columns
from src.losses import CompositeRegressionLoss, LOSS_PRESETS
from src.market_model import (
    build_market_frame,
    evaluate_market_model,
    fit_market_return_model,
    fit_market_return_model_walk_forward,
)
from src.model import StockReturnPredictor
from src.plots import (
    plot_backtest_equity,
    plot_information_coefficient_series,
    plot_interval_calibration,
    plot_prediction_diagnostics,
    plot_training_history,
)
from src.regression import (
    SELECTION_SCORE_KEY,
    TOTAL_RETURN_COLUMN,
    add_selection_score,
    clipped_beta,
    full_metrics,
    summarise_folds,
    target_component_column,
)
from src.training_common import (
    ROOT,
    WALK_FORWARD_SUMMARY_KEYS,
    align_total_returns,
    artifact_path,
    copy_default_artifacts,
    get_horizons,
    json_safe,
    load_config,
    load_horizon_dataset,
    parse_horizon_list,
    print_fold_summary,
    print_metrics,
    scale_features,
    set_seed,
    strip_arrays,
)
from src.training_logger import training_log_context
from src.uncertainty import mc_dropout_predict_loader
from src.validation import chronological_train_validation_test_split, purged_walk_forward_splits


def save_out_of_fold_predictions(out_of_fold, report_dir, model_key: str, horizon: int):
    """
    Persist per-fold validation predictions so the two families can be blended.

    Blend weights have to be fitted on rows where *both* models produced a
    forecast, which means the row-level predictions must survive the training run
    rather than being summarised into fold metrics and discarded.
    """
    if not out_of_fold:
        return None
    frames = []
    for fold in out_of_fold:
        frames.append(
            pd.DataFrame(
                {
                    "fold": int(fold["fold"]),
                    "ticker": np.asarray(fold["tickers"], dtype=object),
                    "date": pd.to_datetime(pd.Series(list(fold["dates"]))).to_numpy(),
                    "true_return": np.asarray(fold["true_return"], dtype=float),
                    "predicted_return": np.asarray(fold["predicted_return"], dtype=float),
                    "reference_prediction": float(fold["reference_prediction"]),
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    path = report_dir / f"oof_predictions_{model_key}_h{horizon}.csv"
    frame.to_csv(path, index=False)
    print(f"Saved out-of-fold predictions to: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM future-return regression models.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Train only one prediction horizon. If omitted, trains all horizons in config.yaml.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated horizons to train, for example 21,63.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run purged walk-forward validation and report per-fold stability.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Override the number of seed-ensemble members configured in config.yaml.",
    )
    parser.add_argument(
        "--loss-experiment",
        action="store_true",
        help="Compare the loss presets (pure MSE, pure Huber, MSE+Huber, MSE+Huber+IC) on folds.",
    )
    parser.add_argument(
        "--architecture-experiment",
        action="store_true",
        help="Run the small documented architecture matrix on purged walk-forward folds.",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CompositeRegressionLoss,
    device: torch.device,
    max_grad_norm: float | None,
) -> dict:
    """
    One optimisation pass. Returns the mean total loss and each component.

    Date codes travel with every batch and are handed to the loss, so its
    cross-sectional term is computed within dates rather than across the whole
    batch. Component losses are returned separately because the composite total
    alone cannot show whether the magnitude terms or the ranking term is moving.
    """
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "huber": 0.0, "cross_sectional_ic": 0.0}
    row_count = 0

    for x, y_model_target, _, _, date_code in loader:
        x, y_model_target = move_batch_to_device((x, y_model_target), device)
        groups = date_code.to(device)
        optimizer.zero_grad()
        predicted = model(x)
        components = loss_fn.terms(predicted, y_model_target, groups)
        loss = loss_fn.combine(components)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_rows = int(x.size(0))
        row_count += batch_rows
        totals["loss"] += float(loss.item()) * batch_rows
        for key in ("mse", "huber", "cross_sectional_ic"):
            totals[key] += float(components[key].item()) * batch_rows

    row_count = max(1, row_count)
    return {key: value / row_count for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CompositeRegressionLoss,
    device: torch.device,
    dates,
    horizon: int,
    reference_prediction: float | None = None,
    selection_weight: float = 0.25,
) -> dict:
    """Deterministic evaluation in the modelled return space (excess return)."""
    model.eval()
    all_true: list[float] = []
    all_predicted: list[float] = []
    model_space_predicted: list[torch.Tensor] = []
    model_space_target: list[torch.Tensor] = []
    model_space_groups: list[torch.Tensor] = []

    for x, y_model_target, y_return, target_scale, date_code in loader:
        x_device, _ = move_batch_to_device((x, y_model_target), device)
        predicted_model_target = model(x_device).float().cpu()
        model_space_predicted.append(predicted_model_target)
        model_space_target.append(y_model_target.float().cpu())
        model_space_groups.append(date_code.cpu())
        all_true.extend(y_return.numpy().tolist())
        all_predicted.extend((predicted_model_target * target_scale).numpy().tolist())

    # The loss is computed once over the whole split rather than averaged over
    # batches. The cross-sectional term is a per-date average, and a batch only
    # ever holds a slice of each date, so a batch-wise average would not be the
    # same quantity as the loss the training objective defines.
    loss_value = float(
        loss_fn(
            torch.cat(model_space_predicted),
            torch.cat(model_space_target),
            torch.cat(model_space_groups),
        ).item()
    )

    true_array = np.asarray(all_true, dtype=float)
    predicted_array = np.asarray(all_predicted, dtype=float)
    metrics = full_metrics(
        dates,
        true_array,
        predicted_array,
        horizon=horizon,
        reference_prediction=reference_prediction,
    )
    metrics = add_selection_score(metrics, magnitude_weight=selection_weight)
    return {
        "loss": loss_value,
        **metrics,
        "true_return": true_array,
        "predicted_return": predicted_array,
    }


def build_loaders(
    scaled_full_df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    date_groups: dict[str, set],
    config: dict,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """
    Build one dataset and its loaders per split.

    The training loader can be date-grouped (``lstm_batching: date_grouped``), in
    which case each batch is made of whole date cross-sections so the ranking
    term of the loss has a real cross-section to work with. Evaluation loaders
    are always sequential and unshuffled.
    """
    loader_kwargs = dataloader_device_kwargs(
        device=device,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True)),
    )
    batch_size = int(config["batch_size"])
    batching = str(config.get("lstm_batching", "date_grouped")).lower().strip()
    if batching not in {"date_grouped", "shuffled"}:
        raise ValueError("lstm_batching must be 'date_grouped' or 'shuffled'")
    dates_per_batch = int(config.get("dates_per_batch", 3))

    result: dict = {}
    for name, dates in date_groups.items():
        dataset = StockSequenceDataset(scaled_full_df, feature_columns, window_size, target_dates=dates)
        if len(dataset) == 0:
            raise ValueError(
                f"The {name} split produced no sequences. Reduce window_size or widen the date range."
            )

        if name == "train" and batching == "date_grouped":
            sampler = DateGroupedBatchSampler(
                dataset,
                dates_per_batch=dates_per_batch,
                shuffle=True,
                seed=seed,
                max_rows_per_batch=int(config.get("max_rows_per_batch", 4096)),
            )
            train_loader = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(name == "train"),
                **loader_kwargs,
            )

        result[name] = {
            "dataset": dataset,
            "loader": train_loader,
            "eval_loader": DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs),
            "dates": [meta["date"] for meta in dataset.metadata],
            "batching": batching if name == "train" else "sequential",
        }
    return result


def build_model(config: dict, feature_count: int, device: torch.device) -> nn.Module:
    return StockReturnPredictor(
        input_size=feature_count,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        model_type=str(config.get("model_type", "lstm")),
        input_dropout=float(config.get("input_dropout", 0.10)),
        recurrent_dropout=float(config.get("recurrent_dropout", 0.0)),
        auxiliary_horizons=list(config.get("auxiliary_horizons", []) or []),
    ).to(device)


def fit_one_model(
    config: dict,
    feature_count: int,
    splits: dict,
    device: torch.device,
    horizon: int,
    seed: int,
    member_label: str,
    reference_prediction: float | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict], float]:
    """Fit a single ensemble member and return it restored to its best epoch."""
    set_seed(seed)
    model = build_model(config, feature_count, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    loss_fn = CompositeRegressionLoss.from_config(config.get("regression_loss"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config.get("lr_scheduler_factor", 0.5)),
        patience=int(config.get("lr_scheduler_patience", 3)),
        min_lr=float(config.get("minimum_learning_rate", 1e-6)),
    )

    early_stopping_metric = str(config.get("early_stopping_metric", SELECTION_SCORE_KEY))
    min_delta = float(config.get("early_stopping_min_delta", 1e-4))
    patience = int(config["early_stopping_patience"])
    selection_weight = float(config.get("selection_score_magnitude_weight", 0.25))

    best_score = -np.inf
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        train_terms = train_one_epoch(
            model,
            splits["train"]["loader"],
            optimizer,
            loss_fn,
            device,
            max_grad_norm=float(config["max_grad_norm"]),
        )
        validation = evaluate(
            model,
            splits["validation"]["eval_loader"],
            loss_fn,
            device,
            splits["validation"]["dates"],
            horizon,
            reference_prediction=reference_prediction,
            selection_weight=selection_weight,
        )
        if early_stopping_metric not in validation:
            raise KeyError(
                f"Unknown early_stopping_metric: {early_stopping_metric}. "
                f"Available keys include: {sorted(k for k in validation if isinstance(validation[k], float))[:12]}"
            )
        score = float(validation[early_stopping_metric])
        scheduler.step(score)

        history.append(
            {
                "horizon": horizon,
                "member": member_label,
                "epoch": epoch,
                "train_loss": train_terms["loss"],
                "train_mse": train_terms["mse"],
                "train_huber": train_terms["huber"],
                "train_cross_sectional_ic": train_terms["cross_sectional_ic"],
                "validation_loss": validation["loss"],
                "validation_mae": validation["mae"],
                "validation_mse": validation["mse"],
                "validation_rmse": validation["rmse"],
                "validation_r2": validation["r2"],
                "validation_direction_accuracy": validation["direction_accuracy"],
                "validation_return_correlation": validation["return_correlation"],
                "validation_rank_ic": validation["rank_information_coefficient"],
                "validation_cross_sectional_ic": validation["cross_sectional_ic"],
                "validation_cross_sectional_icir": validation["cross_sectional_icir"],
                "validation_mse_skill": validation.get("mse_skill_vs_historical_mean", 0.0),
                "validation_selection_score": validation[SELECTION_SCORE_KEY],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if verbose:
            print(
                f"  [{member_label}] epoch {epoch:03d} | "
                f"train {train_terms['loss']:.4f} "
                f"(mse {train_terms['mse']:.4f} ic {train_terms['cross_sectional_ic']:+.4f}) | "
                f"val MSE {validation['mse']:.6f} | "
                f"MSE skill {validation.get('mse_skill_vs_historical_mean', 0.0):+.4f} | "
                f"IC {validation['cross_sectional_ic']:+.4f} | "
                f"{early_stopping_metric} {score:+.4f}"
            )

        if score > best_score + min_delta:
            best_score = score
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(
                        f"  [{member_label}] early stop at epoch {epoch}: "
                        f"validation {early_stopping_metric} flat for {patience} epochs."
                    )
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history, float(best_score)


@torch.no_grad()
def ensemble_predict(
    models: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ensemble mean in return space, plus the realised returns."""
    predictions: list[np.ndarray] = []
    truths: list[float] = []

    for model in models:
        model.eval()

    for x, _, y_return, target_scale, *_ in loader:
        x_device = x.to(device, non_blocking=(device.type == "cuda"))
        scale = target_scale.float().cpu()
        batch = torch.stack([model(x_device).float().cpu() for model in models]).mean(dim=0)
        predictions.append((batch * scale).numpy())
        truths.extend(y_return.numpy().tolist())

    return np.concatenate(predictions), np.asarray(truths, dtype=float)


def ensemble_uncertainty(
    models: list[nn.Module],
    loader: DataLoader,
    passes: int,
    device: torch.device,
    label: str,
) -> dict:
    """
    Combine Monte Carlo dropout within members and disagreement between members.

    The law of total variance gives the exact combination:
    ``Var = E_m[Var_pass] + Var_m[E_pass]`` — the average within-member dropout
    variance plus the variance of the member means.
    """
    member_means: list[np.ndarray] = []
    member_variances: list[np.ndarray] = []
    truths = None
    scales = None

    for index, model in enumerate(models, start=1):
        result = mc_dropout_predict_loader(
            model,
            loader,
            passes=passes,
            device=device,
            progress_label=f"{label} (member {index}/{len(models)})",
        )
        member_means.append(result["mean"])
        member_variances.append(np.square(result["model_std"]))
        truths = result["true_return"]
        scales = result["target_scale"]

    means = np.stack(member_means)
    variances = np.stack(member_variances)
    combined_mean = means.mean(axis=0)
    within = variances.mean(axis=0)
    between = means.var(axis=0, ddof=1) if len(models) > 1 else np.zeros_like(combined_mean)

    return {
        "mean": combined_mean,
        "model_std": np.sqrt(within + between),
        "epistemic_within_member_std": np.sqrt(within),
        "epistemic_between_member_std": np.sqrt(between),
        "true_return": truths,
        "target_scale": scales,
    }

def prepare_split_loaders(
    config: dict,
    full_df: pd.DataFrame,
    feature_columns: list[str],
    date_groups: dict[str, set],
    device: torch.device,
    seed: int,
) -> tuple[dict, object]:
    """
    Fit the feature scaler on the training rows only and build every loader.

    The whole history is transformed with the train-only scaler so that a
    validation or test sequence can legitimately use its own past context across a
    split boundary. What must not happen — and does not — is the scaler seeing
    validation or test rows while it is being fitted.
    """
    train_dates = date_groups["train"]
    train_rows = full_df[pd.DatetimeIndex(full_df.index).isin(train_dates)]
    scaler = scale_features(train_rows, feature_columns, str(config.get("feature_scaler", "robust")))
    scaled = full_df.copy()
    scaled[feature_columns] = scaler.transform(scaled[feature_columns])
    loaders = build_loaders(
        scaled, feature_columns, int(config["window_size"]), date_groups, config, device, seed=seed
    )
    return loaders, scaler


def date_groups_from_split(split) -> dict[str, set]:
    return {
        "train": {pd.Timestamp(value) for value in split.train_dates},
        "validation": {pd.Timestamp(value) for value in split.validation_dates},
        "test": {pd.Timestamp(value) for value in split.test_dates},
    }


def run_walk_forward(
    config: dict,
    full_df: pd.DataFrame,
    feature_columns: list[str],
    target_config: dict,
    horizon: int,
    device: torch.device,
    ensemble_size: int = 1,
) -> dict:
    """
    Refit on successive out-of-sample windows and report per-fold stability.

    Each fold refits from scratch on data strictly before its own test window, and
    also returns its validation-window predictions, which is what the calibration
    and the LSTM/XGBoost blend are fitted on. Folds use a single ensemble member by
    default: the purpose here is measuring stability across periods, and paying for
    a full ensemble per fold would multiply the cost without changing the ranking.
    """
    wf_cfg = config.get("walk_forward", {}) or {}
    folds = int(wf_cfg.get("folds", 3))
    print(f"\nWalk-forward validation: {folds} purged folds")

    splits = purged_walk_forward_splits(
        full_df.index.unique(),
        folds=folds,
        initial_train_ratio=float(wf_cfg.get("initial_train_ratio", 0.50)),
        validation_ratio=float(wf_cfg.get("validation_ratio", 0.15)),
        purge_horizon=horizon if bool(config.get("purge_overlapping_labels", True)) else 0,
        expanding=bool(wf_cfg.get("expanding", True)),
    )

    component_column = target_component_column(target_config)
    fold_reports: list[dict] = []
    out_of_fold: list[dict] = []

    for split in splits:
        train_df, validation_df, test_df = split.frames(full_df)
        if train_df.empty or validation_df.empty or test_df.empty:
            continue
        bounds = split.describe()
        print(f"\n  {split.label}: test {bounds['test']['start']} .. {bounds['test']['end']}")

        seed = int(config.get("random_seed", 42)) + 101 * split.index
        loaders, _ = prepare_split_loaders(
            config, full_df, feature_columns, date_groups_from_split(split), device, seed
        )
        reference = float(train_df[component_column].astype(float).mean())

        members = [
            fit_one_model(
                config,
                len(feature_columns),
                loaders,
                device,
                horizon,
                seed=seed + 7 * member,
                member_label=f"{split.label}_m{member + 1}",
                reference_prediction=reference,
                verbose=False,
            )[0]
            for member in range(max(1, int(ensemble_size)))
        ]

        predicted, realised = ensemble_predict(members, loaders["test"]["eval_loader"], device)
        metrics = full_metrics(
            loaders["test"]["dates"], realised, predicted, horizon, reference_prediction=reference
        )
        report = bounds
        report["test_metrics"] = strip_arrays(metrics)
        report["component"] = component_column
        report["ensemble_size"] = len(members)
        fold_reports.append(report)

        validation_predicted, validation_realised = ensemble_predict(
            members, loaders["validation"]["eval_loader"], device
        )
        out_of_fold.append(
            {
                "fold": split.index,
                "dates": pd.to_datetime(pd.Series(loaders["validation"]["dates"])),
                "tickers": np.asarray(
                    [meta["ticker"] for meta in loaders["validation"]["dataset"].metadata]
                ),
                "true_return": validation_realised,
                "predicted_return": validation_predicted,
                "reference_prediction": reference,
            }
        )

        print(
            f"  {split.label} test IC {metrics['cross_sectional_ic']:+.4f} | "
            f"MSE {metrics['mse']:.6f} | "
            f"MSE skill {metrics.get('mse_skill_vs_historical_mean', 0.0):+.4f}"
        )

        del loaders, members
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarise_folds(
        [report["test_metrics"] for report in fold_reports], WALK_FORWARD_SUMMARY_KEYS
    )
    if summary:
        print("\nWalk-forward summary:")
        print_fold_summary(summary)

    return {
        "folds": fold_reports,
        "summary": summary,
        "out_of_fold": out_of_fold,
        "configuration": {
            "folds": folds,
            "expanding_origin": bool(wf_cfg.get("expanding", True)),
            "purge_trading_days": horizon,
            "ensemble_size_per_fold": int(ensemble_size),
            "note": "each fold is fitted from scratch on data strictly before its test window",
        },
    }


# ---------------------------------------------------------------------------
# Experiment matrices, selected on purged folds only
# ---------------------------------------------------------------------------

# A small, documented matrix rather than an unrestricted search. Each entry
# changes one thing relative to the configured baseline, so a win is attributable
# to that change. An unrestricted search over sequence length, width, depth and
# three dropout rates on three folds would mostly be selecting noise.
ARCHITECTURE_MATRIX = (
    ("baseline", {}),
    ("window_20", {"window_size": 20}),
    ("window_60", {"window_size": 60}),
    ("hidden_32", {"hidden_size": 32}),
    ("hidden_96", {"hidden_size": 96}),
    ("two_layers", {"num_layers": 2}),
    ("input_dropout_10", {"input_dropout": 0.10}),
    ("input_dropout_40", {"input_dropout": 0.40}),
    ("recurrent_dropout_20", {"recurrent_dropout": 0.20}),
    ("weight_decay_5e3", {"weight_decay": 0.005}),
    ("shuffled_batches", {"lstm_batching": "shuffled"}),
)


def run_experiment_matrix(
    config: dict,
    full_df: pd.DataFrame,
    feature_columns: list[str],
    target_config: dict,
    horizon: int,
    device: torch.device,
    variants: list[tuple[str, dict]],
    experiment_name: str,
    epoch_cap: int | None = None,
) -> dict:
    """
    Score each variant on identical purged folds and rank by consistency.

    ``epoch_cap`` bounds the budget per fit. Every variant gets exactly the same
    cap, so the comparison stays like-for-like; without it a matrix of this size
    would not finish in a usable time on this panel.
    """
    experiment = WalkForwardExperiment(
        name=experiment_name,
        full_df=full_df,
        horizon=horizon,
        folds=int((config.get("walk_forward", {}) or {}).get("folds", 3)),
        initial_train_ratio=0.50,
        validation_ratio=0.15,
        base_seed=int(config.get("random_seed", 42)),
        output_dir=ROOT / "reports" / "experiments",
    )
    component_column = target_component_column(target_config)
    candidates = [
        ExperimentCandidate(name=name, overrides=dict(overrides), description=str(overrides))
        for name, overrides in variants
    ]

    def evaluate_candidate(candidate, split, panel, seed) -> dict:
        variant_config = {**config, **candidate.overrides}
        if epoch_cap is not None:
            variant_config["epochs"] = int(epoch_cap)
        train_df, _, _ = split.frames(panel)
        reference = float(train_df[component_column].astype(float).mean())

        loaders, _ = prepare_split_loaders(
            variant_config,
            panel,
            feature_columns,
            date_groups_from_split(split),
            device,
            seed,
        )
        model, _, _ = fit_one_model(
            variant_config,
            len(feature_columns),
            loaders,
            device,
            horizon,
            seed=seed,
            member_label=candidate.name,
            reference_prediction=reference,
            verbose=False,
        )
        predicted, realised = ensemble_predict([model], loaders["test"]["eval_loader"], device)
        metrics = strip_arrays(
            full_metrics(
                loaders["test"]["dates"],
                realised,
                predicted,
                horizon,
                reference_prediction=reference,
            )
        )
        del loaders, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics

    experiment.run(candidates, evaluate_candidate)
    paths = experiment.save(
        candidates,
        WALK_FORWARD_SUMMARY_KEYS,
        configuration={
            "epoch_cap": epoch_cap,
            "window_size": int(config["window_size"]),
            "ensemble_size_per_fit": 1,
            "note": "every variant receives an identical epoch budget and identical folds",
        },
    )
    ranking = experiment.rank()
    print(f"\n{experiment_name} ranking (mean - std across folds):")
    for position, row in enumerate(ranking, start=1):
        if row.get("failed"):
            print(f"  {position:>2}. {row['candidate']:<24} FAILED")
            continue
        print(
            f"  {position:>2}. {row['candidate']:<24} score {row['selection_score']:+.4f} "
            f"(mean {row['mean_score']:+.4f} std {row['score_std']:.4f}) "
            f"IC {row['mean_objective']:+.4f}"
        )
    print(f"  saved: {paths['markdown'].name}")
    return {"ranking": ranking, "paths": {key: str(value) for key, value in paths.items()}}


def train_for_horizon(
    config: dict,
    horizon: int,
    device_info,
    walk_forward: bool = False,
    ensemble_override: int | None = None,
    loss_experiment: bool = False,
    architecture_experiment: bool = False,
) -> None:
    print("\n" + "=" * 78)
    print(f"Training LSTM regressor for {horizon} trading days ahead")
    print("=" * 78)

    report_dir = ROOT / "reports"
    plots_dir = ROOT / config["plots_output_dir"] / f"h{horizon}"
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_path(config["model_output_path"], horizon)
    scaler_path = artifact_path(config["scaler_output_path"], horizon)
    metadata_path = artifact_path(config["metadata_output_path"], horizon)
    metrics_path = artifact_path(config["metrics_output_path"], horizon)
    backtest_path = artifact_path(config["backtest_output_path"], horizon)

    full_df, feature_columns, target_config = load_horizon_dataset(config, horizon)
    component_column = target_component_column(target_config)
    device = device_info.device
    print(f"Using device: {device} ({device_info.accelerator} - {device_info.device_name})")
    print(f"Panel: {len(full_df):,} rows | {len(feature_columns)} features")
    print(f"Training target: {component_column} scaled by trailing volatility")

    train_df, validation_df, test_df = chronological_train_validation_test_split(
        full_df,
        train_ratio=float(config["train_ratio"]),
        validation_ratio=float(config["validation_ratio"]),
        purge_horizon=horizon if bool(config.get("purge_overlapping_labels", True)) else 0,
    )

    # Everything strictly before the test period. Every selection decision is
    # confined to this slice, so the development holdout cannot influence a choice.
    history = full_df[full_df.index <= validation_df.index.max()]

    # ------------------------------------------------------------------
    # Stage 1: the market-return model, authority set by fold agreement.
    # ------------------------------------------------------------------
    market_cfg = config.get("market_model", {}) or {}
    market_features = market_wide_feature_columns(full_df)
    history_splits = purged_walk_forward_splits(
        history.index.unique(),
        folds=int((config.get("walk_forward", {}) or {}).get("folds", 3)),
        initial_train_ratio=0.50,
        validation_ratio=0.15,
        purge_horizon=horizon,
        expanding=True,
    )
    shrinkage, shrinkage_report = fit_market_return_model_walk_forward(
        history,
        history_splits,
        market_features,
        alpha=float(market_cfg.get("ridge_alpha", 10.0)),
        maximum_shrinkage=float(market_cfg.get("maximum_shrinkage", 0.60)),
    )
    market_model = fit_market_return_model(
        train_df,
        validation_df,
        market_features,
        alpha=float(market_cfg.get("ridge_alpha", 10.0)),
        maximum_shrinkage=1.0,
    )
    market_model.shrinkage = float(shrinkage)
    market_model.maximum_shrinkage = float(market_cfg.get("maximum_shrinkage", 0.60))
    print(
        f"\nMarket model: drift {market_model.drift * 100:.2f}% over {horizon} sessions | "
        f"walk-forward shrinkage {market_model.shrinkage:.3f}"
    )
    print(f"  {shrinkage_report['reason']}")
    market_evaluation = evaluate_market_model(market_model, test_df)

    # ------------------------------------------------------------------
    # Optional experiment matrices, on history only.
    # ------------------------------------------------------------------
    experiments: dict = {}
    epoch_cap = int(config.get("experiment_epoch_cap", 8))
    if loss_experiment:
        experiments["loss"] = run_experiment_matrix(
            config,
            history,
            feature_columns,
            target_config,
            horizon,
            device,
            [(name, {"regression_loss": {**config.get("regression_loss", {}), **preset}})
             for name, preset in LOSS_PRESETS.items()],
            "lstm_loss",
            epoch_cap=epoch_cap,
        )
    if architecture_experiment:
        experiments["architecture"] = run_experiment_matrix(
            config,
            history,
            feature_columns,
            target_config,
            horizon,
            device,
            list(ARCHITECTURE_MATRIX),
            "lstm_architecture",
            epoch_cap=epoch_cap,
        )

    # ------------------------------------------------------------------
    # Final fit on the shipped split.
    # ------------------------------------------------------------------
    date_groups = {
        "train": {pd.Timestamp(value) for value in train_df.index.unique()},
        "validation": {pd.Timestamp(value) for value in validation_df.index.unique()},
        "test": {pd.Timestamp(value) for value in test_df.index.unique()},
    }
    base_seed = int(config.get("random_seed", 42))
    scaler = scale_features(
        train_df, feature_columns, str(config.get("feature_scaler", "robust")), scaler_path=scaler_path
    )
    scaled_full_df = full_df.copy()
    scaled_full_df[feature_columns] = scaler.transform(scaled_full_df[feature_columns])
    splits = build_loaders(
        scaled_full_df,
        feature_columns,
        int(config["window_size"]),
        date_groups,
        config,
        device,
        seed=base_seed,
    )
    print(
        f"Sequences: train {len(splits['train']['dataset']):,} | "
        f"validation {len(splits['validation']['dataset']):,} | "
        f"test {len(splits['test']['dataset']):,}"
    )
    print(f"Training batches: {splits['train']['batching']} ({len(splits['train']['loader'])} per epoch)")

    reference_component = float(train_df[component_column].astype(float).mean())
    ensemble_size = int(ensemble_override or config.get("ensemble_size", 3))
    models: list[nn.Module] = []
    history_rows: list[dict] = []
    member_scores: list[float] = []

    for member in range(ensemble_size):
        print(f"\nEnsemble member {member + 1}/{ensemble_size}")
        model, member_history, best_score = fit_one_model(
            config,
            len(feature_columns),
            splits,
            device,
            horizon,
            seed=base_seed + 101 * member,
            member_label=f"m{member + 1}",
            reference_prediction=reference_component,
        )
        models.append(model)
        history_rows.extend(member_history)
        member_scores.append(best_score)

    pd.DataFrame(history_rows).to_csv(report_dir / f"training_history_h{horizon}.csv", index=False)
    plot_training_history([row for row in history_rows if row["member"] == "m1"], plots_dir)

    # ------------------------------------------------------------------
    # Walk-forward, which also supplies the out-of-fold calibration inputs.
    # ------------------------------------------------------------------
    walk_forward_report = None
    out_of_fold = None
    if walk_forward:
        walk_forward_report = run_walk_forward(
            config, history, feature_columns, target_config, horizon, device
        )
        out_of_fold = walk_forward_report.pop("out_of_fold", None)
        save_out_of_fold_predictions(out_of_fold, report_dir, "lstm", horizon)

    # ------------------------------------------------------------------
    # Uncertainty: MC dropout across the ensemble.
    # ------------------------------------------------------------------
    mc_passes = int((config.get("uncertainty", {}) or {}).get("mc_dropout_passes", 10))
    validation_uncertainty = ensemble_uncertainty(
        models, splits["validation"]["eval_loader"], mc_passes, device, "validation"
    )
    test_uncertainty = ensemble_uncertainty(
        models, splits["test"]["eval_loader"], mc_passes, device, "test"
    )

    # One market forecast per date, then broadcast onto the sequence rows by date.
    # A positional broadcast would be wrong here: the sequence dataset emits rows
    # grouped by ticker, so row order does not follow the panel's date order.
    market_frame = build_market_frame(full_df, market_features)
    market_by_date = pd.Series(market_model.predict(market_frame), index=market_frame.index)

    sector_by_ticker = {
        str(key).upper(): str(value) for key, value in (config.get("ticker_sectors") or {}).items()
    }

    def table_for(name: str, uncertainty: dict):
        metadata = splits[name]["dataset"].metadata
        dates = pd.DatetimeIndex([meta["date"] for meta in metadata])
        tickers = [str(meta["ticker"]) for meta in metadata]
        market_forecast = pd.Series(dates.map(market_by_date)).fillna(market_model.drift)
        return build_forecast_table(
            name=name,
            tickers=tickers,
            dates=dates,
            true_total_return=align_total_returns(full_df, metadata, TOTAL_RETURN_COLUMN),
            true_component_return=uncertainty["true_return"],
            raw_prediction=uncertainty["mean"],
            model_std=uncertainty["model_std"],
            target_scale=uncertainty["target_scale"],
            beta=(
                align_total_returns(full_df, metadata, "target_beta")
                if "target_beta" in full_df.columns
                else None
            ),
            sector=[sector_by_ticker.get(ticker.upper(), "Unclassified") for ticker in tickers],
            volatility_scale=uncertainty["target_scale"],
            market_return_forecast=market_forecast.to_numpy(),
        )

    validation_table = table_for("validation", validation_uncertainty)
    test_table = table_for("test", test_uncertainty)

    result = evaluate_model(
        config=config,
        horizon=horizon,
        model_name="LSTMEnsembleRegressor",
        validation=validation_table,
        test=test_table,
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
        market_model=market_model,
        target_config=target_config,
        component_column=component_column,
        walk_forward=walk_forward_report,
        calibration_folds=out_of_fold,
        extra={
            "train_size": int(len(splits["train"]["dataset"])),
            "ensemble_size": ensemble_size,
            "member_validation_scores": member_scores,
            "mc_dropout_passes_per_member": mc_passes,
            "total_stochastic_samples": mc_passes * ensemble_size,
            "regression_loss": CompositeRegressionLoss.from_config(
                config.get("regression_loss")
            ).describe(),
            "early_stopping_metric": str(config.get("early_stopping_metric", SELECTION_SCORE_KEY)),
            "batching": splits["train"]["batching"],
            "dates_per_batch": int(config.get("dates_per_batch", 3)),
            "architecture": {
                "model_type": str(config.get("model_type", "lstm")),
                "window_size": int(config["window_size"]),
                "hidden_size": int(config["hidden_size"]),
                "num_layers": int(config["num_layers"]),
                "dropout": float(config["dropout"]),
                "input_dropout": float(config.get("input_dropout", 0.10)),
                "recurrent_dropout": float(config.get("recurrent_dropout", 0.0)),
                "auxiliary_horizons": list(config.get("auxiliary_horizons", []) or []),
            },
            "epistemic_within_member_std": float(
                np.mean(test_uncertainty["epistemic_within_member_std"])
            ),
            "epistemic_between_member_std": float(
                np.mean(test_uncertainty["epistemic_between_member_std"])
            ),
            "market_model_shrinkage_selection": shrinkage_report,
            "market_model_test_evaluation": market_evaluation,
            "experiments": experiments,
        },
    )

    print("\nTest performance on the modelled component:")
    print_metrics(result.payload["component_metrics"]["test"])
    print("\nTest performance on the user-facing total return:")
    print_metrics(result.payload["total_return_metrics"]["test"])
    print_decile_table(
        result.payload["decile_calibration_total_return"], "Decile calibration (total return)"
    )

    # ------------------------------------------------------------------
    # Persist.
    # ------------------------------------------------------------------
    calibrated_test = result.calibration.apply(test_table.column("raw_prediction"), test_table.dates)
    market_leg = test_table.column("beta", 1.0) * test_table.column(
        "market_return_forecast", market_model.drift
    )

    predictions_path = report_dir / f"test_predictions_h{horizon}.csv"
    result.signal_frame.to_csv(predictions_path, index=False)
    result.backtest_frame.to_csv(backtest_path, index=False)
    if not result.portfolio_frame.empty:
        result.portfolio_frame.to_csv(report_dir / f"portfolio_h{horizon}.csv", index=False)
    pd.DataFrame(result.payload["decile_calibration_total_return"]).to_csv(
        report_dir / f"decile_calibration_h{horizon}.csv", index=False
    )

    plot_backtest_equity(result.backtest_frame, plots_dir)
    plot_prediction_diagnostics(
        test_table.column("true_component_return"), calibrated_test, plots_dir,
        "(modelled component, test)",
    )
    plot_interval_calibration(
        test_table.column("true_component_return"),
        calibrated_test,
        result.signal_frame["lower_bound"].to_numpy() - market_leg,
        result.signal_frame["upper_bound"].to_numpy() - market_leg,
        plots_dir,
        float((config.get("uncertainty", {}) or {}).get("confidence_level", 0.80)),
    )
    plot_information_coefficient_series(
        test_table.dates, test_table.column("true_component_return"), calibrated_test, plots_dir
    )

    torch.save(
        {
            "ensemble_state_dicts": [model.state_dict() for model in models],
            "input_size": len(feature_columns),
        },
        model_path,
    )
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(result.payload), file, indent=2)

    metadata = {
        "artifact_schema_version": 4,
        "tickers": config["tickers"],
        "benchmark_ticker": config["benchmark_ticker"],
        "feature_columns": feature_columns,
        "window_size": int(config["window_size"]),
        "prediction_horizon": horizon,
        "available_prediction_horizons": get_horizons(config),
        "model_type": str(config.get("model_type", "lstm")),
        "hidden_size": int(config["hidden_size"]),
        "num_layers": int(config["num_layers"]),
        "dropout": float(config["dropout"]),
        "input_dropout": float(config.get("input_dropout", 0.10)),
        "recurrent_dropout": float(config.get("recurrent_dropout", 0.0)),
        "auxiliary_horizons": list(config.get("auxiliary_horizons", []) or []),
        "ensemble_size": ensemble_size,
        "feature_scaler": str(config.get("feature_scaler", "robust")),
        "target_configuration": target_config,
        "modelled_component": component_column,
        "market_model": market_model.to_dict(),
        "return_calibration": result.calibration.to_dict(),
        "interval_calibration": result.interval_calibration.to_dict(),
        "mc_dropout_passes": mc_passes,
        "decision_rule": result.decision_config.to_dict(),
        "final_test_metrics": result.payload["total_return_metrics"]["test"],
        "final_test_component_metrics": result.payload["component_metrics"]["test"],
        "test_interval_metrics": result.payload["uncertainty"]["test_interval_metrics"],
        "backtest_metrics": result.payload["backtest_metrics"],
        "acceptance_gates": result.payload["acceptance_gates"],
        "device_used": str(device),
        "device_accelerator": device_info.accelerator,
        "device_name": device_info.device_name,
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(metadata), file, indent=2)

    copy_default_artifacts(
        config,
        horizon,
        [
            (model_path, "model_output_path"),
            (scaler_path, "scaler_output_path"),
            (metadata_path, "metadata_output_path"),
            (metrics_path, "metrics_output_path"),
            (backtest_path, "backtest_output_path"),
        ],
    )

    print(f"\nSaved ensemble ({ensemble_size} members) to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved plots to: {plots_dir}")


def train_models_for_horizons(
    config: dict,
    horizons: list[int],
    walk_forward: bool = False,
    ensemble_override: int | None = None,
    loss_experiment: bool = False,
    architecture_experiment: bool = False,
) -> None:
    set_seed(int(config.get("random_seed", 42)))
    logs_dir = ROOT / "logs"

    with training_log_context(logs_dir, horizons) as log_path:
        device_info = get_best_device(config.get("device", "auto"))
        print("Training configuration")
        print("----------------------")
        print(f"Model: {config.get('model_type', 'lstm').upper()} ensemble regressor")
        print(f"Ensemble size: {int(ensemble_override or config.get('ensemble_size', 3))}")
        print(f"Horizons: {horizons}")
        print(f"Maximum epochs per member: {config['epochs']}")
        print(f"Early stopping: {config.get('early_stopping_metric', 'cross_sectional_ic')} "
              f"with patience {config['early_stopping_patience']}")
        print(f"Walk-forward validation: {'enabled' if walk_forward else 'disabled'}")
        print(f"Logs will be saved to: {log_path}")

        for horizon in horizons:
            train_for_horizon(
                config,
                int(horizon),
                device_info,
                walk_forward,
                ensemble_override,
                loss_experiment=loss_experiment,
                architecture_experiment=architecture_experiment,
            )

        print("\nTraining finished.")
        print("Important: this project is for academic research and simulation only, not financial advice.")


def main() -> None:
    config = load_config()
    set_seed(int(config.get("random_seed", 42)))
    args = parse_args()
    selected_horizons = parse_horizon_list(args.horizons)
    train_models_for_horizons(
        config,
        get_horizons(config, args.horizon, selected_horizons),
        walk_forward=bool(args.walk_forward),
        ensemble_override=args.ensemble_size,
        loss_experiment=bool(args.loss_experiment),
        architecture_experiment=bool(args.architecture_experiment),
    )


if __name__ == "__main__":
    main()
