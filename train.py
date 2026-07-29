"""
Train the LSTM future-return regressor.

Pipeline per horizon
--------------------
1. Build (or load) the supervised panel and the market-excess training target.
2. Split the timeline chronologically with a purge equal to the horizon.
3. Fit an ensemble of LSTM regressors, checkpointing on validation
   cross-sectional information coefficient.
4. Estimate predictive uncertainty with Monte Carlo dropout across the ensemble
   and calibrate prediction intervals on the validation set.
5. Freeze the decision rule on validation, then evaluate once on the test set
   against baselines, regime blocks and a cost-aware backtest.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame, tune_decision_config
from src.dataset import StockSequenceDataset
from src.decision import SIGNAL_LABELS, DecisionConfig
from src.device import dataloader_device_kwargs, get_best_device, move_batch_to_device
from src.model import StockReturnPredictor
from src.pipeline import build_or_load_dataset_for_tickers
from src.plots import (
    plot_backtest_equity,
    plot_information_coefficient_series,
    plot_interval_calibration,
    plot_prediction_diagnostics,
    plot_training_history,
)
from src.regression import (
    EXCESS_RETURN_COLUMN,
    TOTAL_RETURN_COLUMN,
    add_model_target,
    apply_return_calibration,
    chronological_block_metrics,
    compose_total_return,
    estimate_market_drift,
    evaluate_baselines,
    fit_return_calibration,
    full_metrics,
    regression_metrics as calculate_regression_metrics,
    resolve_target_config,
    summarise_folds,
    target_component_column,
)
from src.training_logger import training_log_context
from src.uncertainty import (
    fit_interval_calibration,
    interval_metrics,
    mc_dropout_predict_loader,
)
from src.validation import chronological_train_validation_test_split, purged_walk_forward_splits


ROOT = Path(__file__).resolve().parent

WALK_FORWARD_SUMMARY_KEYS = (
    "mae",
    "rmse",
    "direction_accuracy",
    "return_correlation",
    "cross_sectional_ic",
    "cross_sectional_icir",
    "cross_sectional_long_short_spread_annualised",
)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
    return parser.parse_args()


def get_horizons(
    config: dict,
    selected_horizon: int | None = None,
    selected_horizons: list[int] | None = None,
) -> list[int]:
    if selected_horizons is not None:
        return [int(h) for h in selected_horizons]
    if selected_horizon is not None:
        return [int(selected_horizon)]
    if "prediction_horizons" in config and config["prediction_horizons"]:
        return [int(h) for h in config["prediction_horizons"]]
    return [int(config.get("prediction_horizon", 21))]


def parse_horizon_list(raw_horizons: str | None) -> list[int] | None:
    if raw_horizons is None:
        return None
    items = [item.strip() for item in str(raw_horizons).split(",")]
    horizons = [int(item) for item in items if item]
    if not horizons:
        raise ValueError("At least one horizon must be provided in --horizons.")
    ordered_unique: list[int] = []
    for horizon in horizons:
        if horizon not in ordered_unique:
            ordered_unique.append(horizon)
    return ordered_unique


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def artifact_path(base_path: str | Path, horizon: int) -> Path:
    path = Path(base_path)
    return ROOT / path.with_name(f"{path.stem}_h{horizon}{path.suffix}")


def copy_default_artifacts(
    config: dict,
    horizon: int,
    model_path: Path,
    scaler_path: Path,
    metadata_path: Path,
    metrics_path: Path,
    backtest_path: Path,
) -> None:
    default_horizon = int(config.get("default_prediction_horizon", config.get("prediction_horizon", 21)))
    if horizon != default_horizon:
        return

    copies = [
        (model_path, ROOT / config["model_output_path"]),
        (scaler_path, ROOT / config["scaler_output_path"]),
        (metadata_path, ROOT / config["metadata_output_path"]),
        (metrics_path, ROOT / config["metrics_output_path"]),
        (backtest_path, ROOT / config["backtest_output_path"]),
    ]
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, target)


def scale_features(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    scaler_kind: str = "robust",
    scaler_path: Path | None = None,
):
    """Fit the feature scaler on training rows only and persist it."""
    scaler_kind = scaler_kind.lower().strip()
    if scaler_kind == "robust":
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
    elif scaler_kind == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("feature_scaler must be 'robust' or 'standard'")

    scaler.fit(train_df[feature_columns])
    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return scaler


def regression_metrics(true_return: np.ndarray, predicted_return: np.ndarray) -> dict:
    return calculate_regression_metrics(true_return, predicted_return)


class RobustRegressionLoss(nn.Module):
    """
    Huber loss with a batch-level correlation term.

    Huber caps the influence of extreme return events, which otherwise dominate
    the gradient on a fat-tailed panel. The correlation term is what stops the
    model from collapsing onto the constant-mean solution: predicting a constant
    is optimal for squared error on a near-unpredictable target, but it scores
    zero correlation, so the combined objective penalises it.
    """

    def __init__(self, huber_beta: float = 0.5, correlation_weight: float = 0.10) -> None:
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=float(huber_beta))
        self.correlation_weight = float(correlation_weight)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        robust_loss = self.huber(predicted, target)
        if self.correlation_weight <= 0 or len(predicted) < 3:
            return robust_loss
        predicted_centered = predicted - predicted.mean()
        target_centered = target - target.mean()
        denominator = torch.sqrt(
            torch.sum(predicted_centered.square()) * torch.sum(target_centered.square())
        ).clamp_min(1e-8)
        correlation = torch.sum(predicted_centered * target_centered) / denominator
        return robust_loss + self.correlation_weight * (1.0 - correlation)


def derived_signal_metrics(signal_df: pd.DataFrame) -> dict:
    true_signal = signal_df["true_signal"].astype(str)
    predicted_signal = signal_df["predicted_signal"].astype(str)
    labels = list(SIGNAL_LABELS)
    return {
        "accuracy": float(accuracy_score(true_signal, predicted_signal)),
        "classification_report": classification_report(
            true_signal,
            predicted_signal,
            labels=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true_signal, predicted_signal, labels=labels).tolist(),
        "labels": labels,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_grad_norm: float | None,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y_model_target, _, _ in loader:
        x, y_model_target = move_batch_to_device((x, y_model_target), device)
        optimizer.zero_grad()
        predicted = model(x)
        loss = loss_fn(predicted, y_model_target)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    dates,
    horizon: int,
) -> dict:
    """Deterministic evaluation in the modelled return space (excess return)."""
    model.eval()
    all_true: list[float] = []
    all_predicted: list[float] = []
    total_loss = 0.0

    for x, y_model_target, y_return, target_scale in loader:
        x_device, y_device = move_batch_to_device((x, y_model_target), device)
        predicted_model_target = model(x_device)
        total_loss += loss_fn(predicted_model_target, y_device).item() * x.size(0)
        all_true.extend(y_return.numpy().tolist())
        all_predicted.extend((predicted_model_target.cpu() * target_scale).numpy().tolist())

    true_array = np.asarray(all_true, dtype=float)
    predicted_array = np.asarray(all_predicted, dtype=float)
    return {
        "loss": float(total_loss / len(loader.dataset)),
        **full_metrics(dates, true_array, predicted_array, horizon=horizon),
        "true_return": true_array,
        "predicted_return": predicted_array,
    }


def strip_arrays(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray) and key not in {"true_return", "predicted_return"}
    }


def build_loaders(
    scaled_full_df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    date_groups: dict[str, set],
    config: dict,
    device: torch.device,
) -> dict:
    loader_kwargs = dataloader_device_kwargs(
        device=device,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True)),
    )
    batch_size = int(config["batch_size"])
    result: dict = {}
    for name, dates in date_groups.items():
        dataset = StockSequenceDataset(scaled_full_df, feature_columns, window_size, target_dates=dates)
        if len(dataset) == 0:
            raise ValueError(
                f"The {name} split produced no sequences. Reduce window_size or widen the date range."
            )
        result[name] = {
            "dataset": dataset,
            "loader": DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(name == "train"),
                **loader_kwargs,
            ),
            "eval_loader": DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs),
            "dates": [meta["date"] for meta in dataset.metadata],
        }
    return result


def fit_one_model(
    config: dict,
    feature_count: int,
    splits: dict,
    device: torch.device,
    horizon: int,
    seed: int,
    member_label: str,
) -> tuple[nn.Module, list[dict], float]:
    """Fit a single ensemble member and return it restored to its best epoch."""
    set_seed(seed)
    model = StockReturnPredictor(
        input_size=feature_count,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        model_type=str(config.get("model_type", "lstm")),
        input_dropout=float(config.get("input_dropout", 0.10)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    loss_cfg = config.get("regression_loss", {})
    loss_fn = RobustRegressionLoss(
        huber_beta=float(loss_cfg.get("huber_beta", 0.5)),
        correlation_weight=float(loss_cfg.get("correlation_weight", 0.10)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config.get("lr_scheduler_factor", 0.5)),
        patience=int(config.get("lr_scheduler_patience", 3)),
        min_lr=float(config.get("minimum_learning_rate", 1e-6)),
    )

    early_stopping_metric = str(config.get("early_stopping_metric", "cross_sectional_ic"))
    min_delta = float(config.get("early_stopping_min_delta", 1e-4))
    patience = int(config["early_stopping_patience"])

    best_score = -np.inf
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(
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
        )
        if early_stopping_metric not in validation:
            raise KeyError(f"Unknown early_stopping_metric: {early_stopping_metric}")
        score = float(validation[early_stopping_metric])
        scheduler.step(score)

        history.append(
            {
                "horizon": horizon,
                "member": member_label,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_mae": validation["mae"],
                "validation_rmse": validation["rmse"],
                "validation_direction_accuracy": validation["direction_accuracy"],
                "validation_return_correlation": validation["return_correlation"],
                "validation_rank_ic": validation["rank_information_coefficient"],
                "validation_cross_sectional_ic": validation["cross_sectional_ic"],
                "validation_cross_sectional_icir": validation["cross_sectional_icir"],
                "validation_predictive_score": validation["predictive_score"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"  [{member_label}] epoch {epoch:03d} | "
            f"train {train_loss:.4f} | "
            f"val MAE {validation['mae']:.4f} | "
            f"dir {validation['direction_accuracy']:.4f} | "
            f"IC {validation['cross_sectional_ic']:+.4f} | "
            f"ICIR {validation['cross_sectional_icir']:+.2f} | "
            f"{early_stopping_metric} {score:+.4f}"
        )

        if score > best_score + min_delta:
            best_score = score
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
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

    for x, _, y_return, target_scale in loader:
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


def load_horizon_dataset(config: dict, horizon: int) -> tuple[pd.DataFrame, list[str], dict]:
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_path = cache_dir / f"full_dataset_h{horizon}.parquet"

    full_df, feature_columns = build_or_load_dataset_for_tickers(
        tickers=config["tickers"],
        benchmark_ticker=config["benchmark_ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        prediction_horizon=horizon,
        buy_threshold=float(config["buy_threshold"]),
        sell_threshold=float(config["sell_threshold"]),
        macro_tickers=config.get("macro_tickers"),
        cache_path=dataset_cache_path,
        use_cache=bool(config.get("use_dataset_cache", True)),
        force_rebuild=bool(config.get("force_rebuild_dataset_cache", False)),
        use_earnings_features=bool(config.get("use_earnings_features", False)),
        regime_normalization_window=int(config.get("regime_normalization_window", 252)),
        exclude_market_wide_features=bool(config.get("exclude_market_wide_features", True)),
    )
    target_config = resolve_target_config(config.get("regression_target"))
    full_df = add_model_target(full_df, horizon, target_config)
    return full_df, feature_columns, target_config


def run_walk_forward(
    config: dict,
    full_df: pd.DataFrame,
    feature_columns: list[str],
    target_config: dict,
    horizon: int,
    device: torch.device,
) -> dict:
    """Repeat the fit on successive out-of-sample windows and report stability."""
    wf_cfg = config.get("walk_forward", {})
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

    for split in splits:
        train_df, validation_df, test_df = split.frames(full_df)
        if train_df.empty or validation_df.empty or test_df.empty:
            continue
        print(f"\n  {split.label}: {split.describe()['test']['start']} .. {split.describe()['test']['end']}")

        scaler = scale_features(train_df, feature_columns, str(config.get("feature_scaler", "robust")))
        scaled = full_df.copy()
        scaled[feature_columns] = scaler.transform(scaled[feature_columns])

        loaders = build_loaders(
            scaled,
            feature_columns,
            int(config["window_size"]),
            {
                "train": {pd.Timestamp(v) for v in split.train_dates},
                "validation": {pd.Timestamp(v) for v in split.validation_dates},
                "test": {pd.Timestamp(v) for v in split.test_dates},
            },
            config,
            device,
        )

        model, _, _ = fit_one_model(
            config,
            len(feature_columns),
            loaders,
            device,
            horizon,
            seed=int(config.get("random_seed", 42)) + split.index,
            member_label=split.label,
        )
        predicted, realised = ensemble_predict([model], loaders["test"]["eval_loader"], device)
        metrics = full_metrics(loaders["test"]["dates"], realised, predicted, horizon=horizon)

        report = split.describe()
        report["test_metrics"] = strip_arrays(metrics)
        report["component"] = component_column
        fold_reports.append(report)
        print(
            f"  {split.label} test IC {metrics['cross_sectional_ic']:+.4f} | "
            f"dir {metrics['direction_accuracy']:.4f} | MAE {metrics['mae']:.4f}"
        )

        del loaders, scaled
        torch.cuda.empty_cache() if device.type == "cuda" else None

    return {
        "folds": fold_reports,
        "summary": summarise_folds(
            [report["test_metrics"] for report in fold_reports], WALK_FORWARD_SUMMARY_KEYS
        ),
        "configuration": {
            "folds": folds,
            "expanding_origin": bool(wf_cfg.get("expanding", True)),
            "purge_trading_days": horizon,
            "note": "each fold is fitted from scratch on data strictly before its test window",
        },
    }


def train_for_horizon(
    config: dict,
    horizon: int,
    device_info,
    walk_forward: bool = False,
    ensemble_override: int | None = None,
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
    market_drift = estimate_market_drift(train_df, target_config)
    print(
        f"Market drift over {horizon} sessions, estimated on train only: "
        f"{market_drift['market_drift'] * 100:.2f}%"
    )

    scaler = scale_features(
        train_df,
        feature_columns,
        str(config.get("feature_scaler", "robust")),
        scaler_path=scaler_path,
    )
    # Scale the whole history with the train-only scaler so validation and test
    # sequences can legitimately use past context across split boundaries.
    scaled_full_df = full_df.copy()
    scaled_full_df[feature_columns] = scaler.transform(scaled_full_df[feature_columns])

    splits = build_loaders(
        scaled_full_df,
        feature_columns,
        int(config["window_size"]),
        {
            "train": {pd.Timestamp(value) for value in train_df.index.unique()},
            "validation": {pd.Timestamp(value) for value in validation_df.index.unique()},
            "test": {pd.Timestamp(value) for value in test_df.index.unique()},
        },
        config,
        device,
    )
    print(
        f"Sequences: train {len(splits['train']['dataset']):,} | "
        f"validation {len(splits['validation']['dataset']):,} | "
        f"test {len(splits['test']['dataset']):,}"
    )

    ensemble_size = int(ensemble_override or config.get("ensemble_size", 3))
    base_seed = int(config.get("random_seed", 42))
    models: list[nn.Module] = []
    history: list[dict] = []
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
        )
        models.append(model)
        history.extend(member_history)
        member_scores.append(best_score)

    pd.DataFrame(history).to_csv(report_dir / f"training_history_h{horizon}.csv", index=False)
    plot_training_history(
        [row for row in history if row["member"] == "m1"],
        plots_dir,
    )

    # ------------------------------------------------------------------
    # Uncertainty: MC dropout across the ensemble, calibrated on validation.
    # ------------------------------------------------------------------
    mc_passes = int(config.get("uncertainty", {}).get("mc_dropout_passes", 10))
    confidence_level = float(config.get("uncertainty", {}).get("confidence_level", 0.80))

    validation_uncertainty = ensemble_uncertainty(
        models, splits["validation"]["eval_loader"], mc_passes, device, "validation"
    )
    test_uncertainty = ensemble_uncertainty(
        models, splits["test"]["eval_loader"], mc_passes, device, "test"
    )

    validation_component = validation_uncertainty["mean"]
    test_component = test_uncertainty["mean"]
    validation_truth = validation_uncertainty["true_return"]
    test_truth = test_uncertainty["true_return"]

    calibration = fit_return_calibration(validation_truth, validation_component)
    validation_component = apply_return_calibration(validation_component, calibration)
    test_component = apply_return_calibration(test_component, calibration)

    interval_calibration = fit_interval_calibration(
        validation_truth,
        validation_component,
        validation_uncertainty["model_std"],
        validation_uncertainty["target_scale"],
        confidence_level=confidence_level,
        minimum_sigma=float(config.get("uncertainty", {}).get("minimum_sigma", 0.005)),
    )
    print(
        f"\nInterval calibration: level {confidence_level:.0%} | "
        f"multiplier {interval_calibration.conformal_multiplier:.3f} | "
        f"validation coverage {interval_calibration.validation_coverage:.3f} | "
        f"mean width {interval_calibration.validation_mean_width * 100:.2f}%"
    )

    validation_lower, validation_upper, validation_sigma = interval_calibration.interval(
        validation_component, validation_uncertainty["model_std"], validation_uncertainty["target_scale"]
    )
    test_lower, test_upper, test_sigma = interval_calibration.interval(
        test_component, test_uncertainty["model_std"], test_uncertainty["target_scale"]
    )

    # ------------------------------------------------------------------
    # Metrics in both spaces: what the model learns, and what the user sees.
    # ------------------------------------------------------------------
    drift = float(market_drift["market_drift"])
    validation_total = compose_total_return(validation_component, drift)
    test_total = compose_total_return(test_component, drift)
    validation_total_actual = _align_total_returns(full_df, splits["validation"]["dataset"].metadata)
    test_total_actual = _align_total_returns(full_df, splits["test"]["dataset"].metadata)

    component_metrics = {
        "validation": strip_arrays(
            full_metrics(splits["validation"]["dates"], validation_truth, validation_component, horizon)
        ),
        "test": strip_arrays(
            full_metrics(splits["test"]["dates"], test_truth, test_component, horizon)
        ),
    }
    total_metrics = {
        "validation": strip_arrays(
            full_metrics(splits["validation"]["dates"], validation_total_actual, validation_total, horizon)
        ),
        "test": strip_arrays(
            full_metrics(splits["test"]["dates"], test_total_actual, test_total, horizon)
        ),
    }

    print("\nTest performance on the modelled component (market-excess return):")
    _print_metrics(component_metrics["test"])
    print("\nTest performance on the user-facing total return:")
    _print_metrics(total_metrics["test"])

    uncertainty_report = {
        "method": "Monte Carlo dropout across a seed ensemble, normalised split-conformal interval",
        "mc_dropout_passes_per_member": mc_passes,
        "ensemble_size": ensemble_size,
        "total_stochastic_samples": mc_passes * ensemble_size,
        "calibration": interval_calibration.to_dict(),
        "validation_interval_metrics": interval_metrics(
            validation_truth, validation_lower, validation_upper, confidence_level
        ),
        "test_interval_metrics": interval_metrics(
            test_truth, test_lower, test_upper, confidence_level
        ),
        "mean_epistemic_std": float(np.mean(test_uncertainty["model_std"])),
        "mean_total_sigma": float(np.mean(test_sigma)),
        "epistemic_share_of_variance": float(
            np.mean(np.square(test_uncertainty["model_std"])) / max(float(np.mean(np.square(test_sigma))), 1e-12)
        ),
    }
    print(
        f"Test interval coverage {uncertainty_report['test_interval_metrics']['coverage_picp']:.3f} "
        f"(nominal {confidence_level:.2f}) | mean width "
        f"{uncertainty_report['test_interval_metrics']['mean_interval_width_mpiw'] * 100:.2f}%"
    )

    # ------------------------------------------------------------------
    # Decision rule and backtest.
    # ------------------------------------------------------------------
    backtest_cfg = BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        allow_short=bool(config["allow_short"]),
        signal_threshold_multiplier=float(config.get("signal_threshold_multiplier", 1.0)),
        min_signal_edge=float(config.get("min_signal_edge", 0.0)),
    )
    decision_defaults = config.get("decision", {})
    base_decision_cfg = DecisionConfig(
        rule=str(decision_defaults.get("rule", "risk_adjusted")),
        allow_short=bool(config["allow_short"]),
        position_sizing=str(decision_defaults.get("position_sizing", "binary")),
        min_direction_probability=float(decision_defaults.get("min_direction_probability", 0.0)),
    )

    decision_cfg, decision_tuning = tune_decision_config(
        metadata=splits["validation"]["dataset"].metadata,
        true_return=validation_total_actual,
        predicted_return=validation_total,
        sigma=validation_sigma,
        cfg=backtest_cfg,
        horizon=horizon,
        base_decision_cfg=base_decision_cfg,
        min_active_trades=int(config.get("threshold_min_active_trades", 20)),
    )
    print(
        f"\nValidation-selected decision rule: {decision_cfg.rule} | "
        f"threshold {decision_cfg.threshold * 100:.2f}% | min z {decision_cfg.min_z_score:.2f}"
    )

    signal_df = build_signal_frame(
        metadata=splits["test"]["dataset"].metadata,
        true_return=test_total_actual,
        predicted_return=test_total,
        cfg=backtest_cfg,
        threshold=decision_cfg.threshold,
        sigma=test_sigma,
        decision_cfg=decision_cfg,
    )
    signal_df["lower_bound"] = test_lower + drift
    signal_df["upper_bound"] = test_upper + drift
    signal_metrics = derived_signal_metrics(signal_df)
    predictions_path = report_dir / f"test_predictions_h{horizon}.csv"
    signal_df.to_csv(predictions_path, index=False)

    backtest_df, backtest_metrics = backtest_signals(signal_df, backtest_cfg, horizon=horizon)
    backtest_df.to_csv(backtest_path, index=False)

    # Ablation: identical predictions, plain point-threshold rule with no
    # uncertainty input. Isolates the contribution of the risk-adjusted rule.
    point_frame = build_signal_frame(
        metadata=splits["test"]["dataset"].metadata,
        true_return=test_total_actual,
        predicted_return=test_total,
        cfg=backtest_cfg,
        threshold=decision_cfg.threshold,
        sigma=None,
    )
    _, point_backtest_metrics = backtest_signals(point_frame, backtest_cfg, horizon=horizon)

    print("\nDerived signal classification report:")
    print(signal_metrics["classification_report"])

    # ------------------------------------------------------------------
    # Plots.
    # ------------------------------------------------------------------
    plot_backtest_equity(backtest_df, plots_dir)
    plot_prediction_diagnostics(test_truth, test_component, plots_dir, "(market-excess, test)")
    plot_interval_calibration(test_truth, test_component, test_lower, test_upper, plots_dir, confidence_level)
    plot_information_coefficient_series(splits["test"]["dates"], test_truth, test_component, plots_dir)

    walk_forward_report = None
    if walk_forward:
        walk_forward_report = run_walk_forward(
            config, full_df, feature_columns, target_config, horizon, device
        )

    # ------------------------------------------------------------------
    # Persist artifacts.
    # ------------------------------------------------------------------
    torch.save(
        {
            "ensemble_state_dicts": [model.state_dict() for model in models],
            "input_size": len(feature_columns),
        },
        model_path,
    )

    metrics_payload = {
        "model_name": "LSTMEnsembleRegressor",
        "prediction_horizon": horizon,
        "train_size": len(splits["train"]["dataset"]),
        "validation_size": len(splits["validation"]["dataset"]),
        "test_size": len(splits["test"]["dataset"]),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "ensemble_size": ensemble_size,
        "member_validation_scores": member_scores,
        "target_configuration": target_config,
        "modelled_component": component_column,
        "market_drift": market_drift,
        "component_metrics": component_metrics,
        "total_return_metrics": total_metrics,
        # Reported at the top level too, so existing tooling keeps working.
        "validation_metrics": total_metrics["validation"],
        "test_metrics": total_metrics["test"],
        "regression_baselines": evaluate_baselines(train_df, test_df, horizon, TOTAL_RETURN_COLUMN),
        "regression_baselines_excess": evaluate_baselines(
            train_df, test_df, horizon, EXCESS_RETURN_COLUMN
        ),
        "uncertainty": uncertainty_report,
        "test_regime_blocks": chronological_block_metrics(
            splits["test"]["dataset"].metadata,
            test_truth,
            test_component,
            blocks=int(config.get("evaluation_regime_blocks", 4)),
            horizon=horizon,
        ),
        "decision_rule": decision_cfg.to_dict(),
        "decision_rule_tuning": decision_tuning,
        "signal_metrics": signal_metrics,
        "backtest_metrics": backtest_metrics,
        "backtest_metrics_point_rule_ablation": point_backtest_metrics,
        "return_calibration": calibration,
        "walk_forward": walk_forward_report,
        "split_method": "purged_chronological_holdout",
        "purge_trading_days": horizon if bool(config.get("purge_overlapping_labels", True)) else 0,
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2, default=float)

    metadata = {
        "artifact_schema_version": 3,
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
        "ensemble_size": ensemble_size,
        "feature_scaler": str(config.get("feature_scaler", "robust")),
        "target_configuration": target_config,
        "modelled_component": component_column,
        "market_drift": market_drift,
        "return_calibration": calibration,
        "interval_calibration": interval_calibration.to_dict(),
        "mc_dropout_passes": mc_passes,
        "decision_rule": decision_cfg.to_dict(),
        "final_test_metrics": total_metrics["test"],
        "final_test_component_metrics": component_metrics["test"],
        "test_interval_metrics": uncertainty_report["test_interval_metrics"],
        "signal_metrics": {
            key: value for key, value in signal_metrics.items() if key != "classification_report"
        },
        "backtest_metrics": backtest_metrics,
        "device_used": str(device),
        "device_accelerator": device_info.accelerator,
        "device_name": device_info.device_name,
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=float)

    copy_default_artifacts(config, horizon, model_path, scaler_path, metadata_path, metrics_path, backtest_path)

    print(f"\nSaved ensemble ({ensemble_size} members) to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved backtest results to: {backtest_path}")
    print(f"Saved plots to: {plots_dir}")


def _align_total_returns(full_df: pd.DataFrame, metadata: list[dict]) -> np.ndarray:
    """Look up the realised total return for each (ticker, date) sequence target."""
    lookup = full_df.reset_index().rename(columns={"index": "date"})
    date_column = lookup.columns[0]
    lookup["_key"] = (
        lookup["Ticker"].astype(str) + "|" + pd.to_datetime(lookup[date_column]).dt.date.astype(str)
    )
    mapping = dict(zip(lookup["_key"], lookup[TOTAL_RETURN_COLUMN].astype(float)))
    keys = [f"{meta['ticker']}|{meta['date']}" for meta in metadata]
    return np.asarray([mapping.get(key, np.nan) for key in keys], dtype=float)


def _print_metrics(metrics: dict) -> None:
    print(
        f"  MAE {metrics['mae']:.4f} | RMSE {metrics['rmse']:.4f} | "
        f"direction {metrics['direction_accuracy']:.4f} | "
        f"corr {metrics['return_correlation']:+.4f}"
    )
    print(
        f"  Cross-sectional IC {metrics['cross_sectional_mean_ic']:+.4f} | "
        f"ICIR {metrics['cross_sectional_icir']:+.2f} | "
        f"t-stat {metrics['cross_sectional_ic_t_statistic']:+.2f} | "
        f"IC>0 on {metrics['cross_sectional_ic_positive_rate']:.1%} of dates"
    )
    print(
        f"  Top-minus-bottom quintile spread per period: "
        f"{metrics['cross_sectional_long_short_spread_per_period'] * 100:+.2f}% "
        f"({metrics['cross_sectional_long_short_spread_annualised'] * 100:+.2f}% annualised)"
    )


def train_models_for_horizons(
    config: dict,
    horizons: list[int],
    walk_forward: bool = False,
    ensemble_override: int | None = None,
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
            train_for_horizon(config, int(horizon), device_info, walk_forward, ensemble_override)

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
    )


if __name__ == "__main__":
    main()
