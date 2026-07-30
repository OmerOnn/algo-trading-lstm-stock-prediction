"""
Train the XGBoost future-return regressor.

It solves the same problem as the LSTM on the same rows, splits, target,
calibration, uncertainty treatment and decision rule, so the two are directly
comparable. Uncertainty comes from a moving-block bootstrap ensemble: each member
is fitted on a resample of contiguous date blocks, so the spread of member
predictions reflects how much the fit depends on which slice of history it
happened to see.

Model selection
---------------
Boosting rounds are chosen by evaluating a fitted booster at a ladder of
iteration counts on purged walk-forward folds, scored by per-date cross-sectional
IC plus magnitude skill. There is no minimum-round floor: the previous version
reported ``best_iteration = 0`` and then silently used 50 trees anyway, so the
number recorded in the metadata described nothing that had actually happened.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.boosting import (
    DEFAULT_ROUND_LADDER,
    ensemble_feature_importance,
    evaluate_rounds,
    permutation_importance,
    recommend_feature_blocklist,
    select_rounds,
)
from src.evaluation import build_forecast_table, evaluate_model, print_decile_table
from src.features import market_wide_feature_columns
from src.market_model import (
    evaluate_market_model,
    fit_market_return_model,
    fit_market_return_model_walk_forward,
)
from src.plots import (
    plot_backtest_equity,
    plot_information_coefficient_series,
    plot_interval_calibration,
    plot_prediction_diagnostics,
)
from src.regression import (
    TOTAL_RETURN_COLUMN,
    clipped_beta,
    full_metrics,
    summarise_folds,
    target_component_column,
)
from src.training_common import (
    ROOT,
    WALK_FORWARD_SUMMARY_KEYS,
    get_horizons,
    json_safe,
    load_config,
    load_horizon_dataset,
    parse_horizon_list,
    print_fold_summary,
    print_metrics,
    set_seed,
    strip_arrays,
)
from src.training_logger import training_log_context
from src.uncertainty import BootstrapEnsemble
from src.validation import chronological_train_validation_test_split, purged_walk_forward_splits


try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = Any


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
    parser = argparse.ArgumentParser(description="Train XGBoost future-return regression models.")
    parser.add_argument("--horizon", type=int, default=None, help="Train one prediction horizon.")
    parser.add_argument(
        "--horizons", type=str, default=None, help="Comma-separated horizons, for example 21,63."
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run purged walk-forward validation and use it to select rounds and calibration.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Also run the small documented hyperparameter grid on walk-forward folds.",
    )
    parser.add_argument(
        "--permutation-importance",
        action="store_true",
        help="Compute out-of-fold permutation importance and recommend a feature blocklist.",
    )
    return parser.parse_args()


def resolve_xgboost_backend(config: dict) -> tuple[str, str]:
    requested = str(config.get("xgboost", {}).get("device", "auto")).lower().strip()
    aliases = {"gpu": "cuda", "metal": "mps", "mac": "mps", "apple": "mps"}
    requested = aliases.get(requested, requested)

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda", "CUDA GPU acceleration is active for XGBoost."
        raise RuntimeError("XGBoost CUDA was requested, but no CUDA-capable NVIDIA GPU is available.")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda", "CUDA GPU acceleration is active for XGBoost."
        return "cpu", "No CUDA-capable NVIDIA GPU detected, so XGBoost is running on CPU."
    if requested == "mps":
        return "cpu", "Apple Metal GPU is not supported by XGBoost, so it is running on CPU."
    if requested == "cpu":
        return "cpu", "XGBoost is configured to run on CPU."
    raise ValueError("Invalid XGBoost device. Use one of: auto, cuda, gpu, cpu, mps, metal, mac, apple")


def make_regressor(
    xgb_cfg: dict,
    device: str,
    seed: int,
    n_estimators: int | None = None,
    objective: str | None = None,
) -> "XGBRegressor":
    """
    Build a booster with no early stopping.

    Early stopping is deliberately absent. Rounds are selected afterwards by
    scoring one fitted booster at many iteration counts, which is both cheaper
    (one fit instead of one per candidate) and more correct: XGBoost's early
    stopping has to optimise a pooled metric, and the metric that matters here is
    an average of per-date correlations, which its callback interface cannot
    express.
    """
    resolved_objective = str(objective or xgb_cfg.get("objective", "reg:squarederror"))
    kwargs = dict(
        objective=resolved_objective,
        n_estimators=int(n_estimators or xgb_cfg.get("n_estimators", 800)),
        max_depth=int(xgb_cfg.get("max_depth", 5)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.02)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.6)),
        random_state=int(seed),
        n_jobs=-1,
        tree_method="hist",
        device=device,
        reg_alpha=float(xgb_cfg.get("reg_alpha", 0.10)),
        reg_lambda=float(xgb_cfg.get("reg_lambda", 3.0)),
        min_child_weight=float(xgb_cfg.get("min_child_weight", 20.0)),
    )
    if resolved_objective == "reg:pseudohubererror":
        kwargs["huber_slope"] = float(xgb_cfg.get("huber_slope", 1.0))
    return XGBRegressor(**kwargs)


def hyperparameter_grid(xgb_cfg: dict) -> list[dict]:
    """
    A small, documented grid rather than an unrestricted search.

    Each entry changes one thing relative to the configured baseline, so a win is
    attributable to that change. An unrestricted search over this many correlated
    hyperparameters on three folds would mostly be selecting noise.
    """
    base = dict(xgb_cfg)
    variants: list[dict] = [{"name": "baseline", "overrides": {}}]
    for name, overrides in (
        ("shallow_depth3", {"max_depth": 3}),
        ("deep_depth7", {"max_depth": 7}),
        ("high_min_child_weight", {"min_child_weight": 60.0}),
        ("slow_learning", {"learning_rate": 0.01}),
        ("low_colsample", {"colsample_bytree": 0.35}),
        ("low_subsample", {"subsample": 0.6}),
        ("strong_l2", {"reg_lambda": 12.0}),
        ("strong_l1", {"reg_alpha": 1.0}),
        ("pseudo_huber", {"objective": "reg:pseudohubererror"}),
    ):
        variants.append({"name": name, "overrides": overrides})
    for variant in variants:
        variant["config"] = {**base, **variant["overrides"]}
    return variants


def predict_component(
    ensemble: BootstrapEnsemble,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Ensemble mean and member disagreement, decoded into return space."""
    mean_scaled, std_scaled = ensemble.predict(df[feature_columns])
    scales = df["target_scale"].astype(float).to_numpy()
    return {
        "mean": np.asarray(mean_scaled, dtype=float) * scales,
        "model_std": np.asarray(std_scaled, dtype=float) * scales,
        "target_scale": scales,
    }


def forecast_table_from_frame(
    name: str,
    df: pd.DataFrame,
    prediction: dict,
    component_column: str,
    market_forecast: np.ndarray,
    target_config: dict,
):
    """Build the shared evaluation table from a plain panel slice."""
    return build_forecast_table(
        name=name,
        tickers=df["Ticker"].astype(str).to_numpy(),
        dates=df.index,
        true_total_return=df[TOTAL_RETURN_COLUMN].astype(float).to_numpy(),
        true_component_return=df[component_column].astype(float).to_numpy(),
        raw_prediction=prediction["mean"],
        model_std=prediction["model_std"],
        target_scale=prediction["target_scale"],
        beta=clipped_beta(df, target_config),
        sector=df["sector"].astype(str).to_numpy() if "sector" in df.columns else None,
        volatility_scale=prediction["target_scale"],
        market_return_forecast=market_forecast,
    )


def run_walk_forward(
    config: dict,
    full_df: pd.DataFrame,
    feature_columns: list[str],
    target_config: dict,
    horizon: int,
    xgb_device: str,
    xgb_cfg: dict,
    round_ladder=DEFAULT_ROUND_LADDER,
    collect_permutation: bool = False,
) -> dict:
    """
    Purged walk-forward validation that also produces every selection input.

    One pass yields the fold metrics, the round-count curves, the out-of-fold
    predictions that calibration and blending are fitted on, and optionally
    permutation importance — so nothing downstream needs to touch the test period.
    """
    wf_cfg = config.get("walk_forward", {}) or {}
    folds = int(wf_cfg.get("folds", 3))
    component_column = target_component_column(target_config)
    print(f"\nWalk-forward validation: {folds} purged folds")

    splits = purged_walk_forward_splits(
        full_df.index.unique(),
        folds=folds,
        initial_train_ratio=float(wf_cfg.get("initial_train_ratio", 0.50)),
        validation_ratio=float(wf_cfg.get("validation_ratio", 0.15)),
        purge_horizon=horizon if bool(config.get("purge_overlapping_labels", True)) else 0,
        expanding=bool(wf_cfg.get("expanding", True)),
    )

    fold_reports: list[dict] = []
    round_curves: list[list[dict]] = []
    out_of_fold: list[dict] = []
    permutation_frames: list[pd.DataFrame] = []

    for split in splits:
        train_df, validation_df, test_df = split.frames(full_df)
        if train_df.empty or validation_df.empty or test_df.empty:
            continue

        seed = int(config.get("random_seed", 42)) + split.index
        maximum_rounds = int(xgb_cfg.get("n_estimators", 800))
        regressor = make_regressor(xgb_cfg, xgb_device, seed=seed, n_estimators=maximum_rounds)
        regressor.fit(
            train_df[feature_columns], train_df["model_target"].astype(float), verbose=False
        )

        train_component_mean = float(train_df[component_column].astype(float).mean())

        # The round curve is measured on this fold's *validation* window, so the
        # round count is never chosen using any fold's test window.
        curve = evaluate_rounds(
            regressor,
            validation_df[feature_columns],
            validation_df[component_column].astype(float).to_numpy(),
            validation_df.index,
            horizon,
            target_scale=validation_df["target_scale"].astype(float).to_numpy(),
            reference_prediction=train_component_mean,
            ladder=round_ladder,
        )
        round_curves.append(curve)

        fold_rounds = select_rounds([curve]).rounds
        predicted = (
            regressor.predict(test_df[feature_columns], iteration_range=(0, fold_rounds))
            * test_df["target_scale"].to_numpy()
        )
        metrics = full_metrics(
            test_df.index,
            test_df[component_column].astype(float).to_numpy(),
            predicted,
            horizon,
            reference_prediction=train_component_mean,
        )

        report = split.describe()
        report["test_metrics"] = strip_arrays(metrics)
        report["component"] = component_column
        report["selected_rounds"] = int(fold_rounds)
        report["maximum_rounds_fitted"] = maximum_rounds
        fold_reports.append(report)

        validation_predicted = (
            regressor.predict(validation_df[feature_columns], iteration_range=(0, fold_rounds))
            * validation_df["target_scale"].to_numpy()
        )
        out_of_fold.append(
            {
                "fold": split.index,
                "dates": validation_df.index,
                "tickers": validation_df["Ticker"].astype(str).to_numpy(),
                "true_return": validation_df[component_column].astype(float).to_numpy(),
                "predicted_return": np.asarray(validation_predicted, dtype=float),
                "reference_prediction": train_component_mean,
            }
        )

        print(
            f"  {split.label} rounds {fold_rounds:>4} | test IC "
            f"{metrics['cross_sectional_ic']:+.4f} | MSE {metrics['mse']:.6f} | "
            f"MSE skill {metrics.get('mse_skill_vs_historical_mean', 0.0):+.4f}"
        )

        if collect_permutation:
            frame = permutation_importance(
                lambda data, model=regressor, rounds=fold_rounds: model.predict(
                    data, iteration_range=(0, rounds)
                ),
                validation_df[feature_columns],
                validation_df[component_column].astype(float).to_numpy(),
                validation_df.index,
                horizon,
                feature_columns=feature_columns,
                repeats=int(config.get("permutation_repeats", 2)),
                seed=seed,
                target_scale=validation_df["target_scale"].astype(float).to_numpy(),
            )
            permutation_frames.append(frame)

    selection = select_rounds(round_curves)
    print(f"\nBoosting rounds selected out of fold: {selection.rounds}")
    print(f"  {selection.reason}")

    summary = summarise_folds(
        [report["test_metrics"] for report in fold_reports], WALK_FORWARD_SUMMARY_KEYS
    )
    if summary:
        print("\nWalk-forward summary:")
        print_fold_summary(summary)

    return {
        "folds": fold_reports,
        "summary": summary,
        "round_selection": selection.to_dict(),
        "out_of_fold": out_of_fold,
        "permutation_importance": [frame.to_dict("records") for frame in permutation_frames],
        "permutation_recommendation": (
            recommend_feature_blocklist(permutation_frames) if permutation_frames else None
        ),
        "configuration": {
            "folds": folds,
            "expanding_origin": bool(wf_cfg.get("expanding", True)),
            "purge_trading_days": horizon,
            "note": "each fold is fitted from scratch on data strictly before its test window",
        },
    }


def run_grid_search(
    config: dict,
    full_df: pd.DataFrame,
    feature_columns: list[str],
    target_config: dict,
    horizon: int,
    xgb_device: str,
) -> dict:
    """Score the documented grid on purged folds, ranked by out-of-fold consistency."""
    wf_cfg = config.get("walk_forward", {}) or {}
    component_column = target_component_column(target_config)
    splits = purged_walk_forward_splits(
        full_df.index.unique(),
        folds=int(wf_cfg.get("folds", 3)),
        initial_train_ratio=float(wf_cfg.get("initial_train_ratio", 0.50)),
        validation_ratio=float(wf_cfg.get("validation_ratio", 0.15)),
        purge_horizon=horizon if bool(config.get("purge_overlapping_labels", True)) else 0,
        expanding=bool(wf_cfg.get("expanding", True)),
    )

    variants = hyperparameter_grid(config.get("xgboost", {}) or {})
    print(f"\nXGBoost grid: {len(variants)} variants x {len(splits)} folds")
    results: list[dict] = []

    for variant in variants:
        scores: list[float] = []
        ics: list[float] = []
        rounds_chosen: list[int] = []
        for split in splits:
            train_df, validation_df, _ = split.frames(full_df)
            if train_df.empty or validation_df.empty:
                continue
            regressor = make_regressor(
                variant["config"],
                xgb_device,
                seed=int(config.get("random_seed", 42)) + split.index,
                n_estimators=int(variant["config"].get("n_estimators", 800)),
                objective=variant["config"].get("objective"),
            )
            regressor.fit(
                train_df[feature_columns], train_df["model_target"].astype(float), verbose=False
            )
            train_component_mean = float(train_df[component_column].astype(float).mean())
            curve = evaluate_rounds(
                regressor,
                validation_df[feature_columns],
                validation_df[component_column].astype(float).to_numpy(),
                validation_df.index,
                horizon,
                target_scale=validation_df["target_scale"].astype(float).to_numpy(),
                reference_prediction=train_component_mean,
            )
            if not curve:
                continue
            selection = select_rounds([curve])
            row = next(item for item in curve if item["rounds"] == selection.rounds)
            scores.append(float(row["cross_sectional_ic"] + 0.25 * np.clip(row["mse_skill"], -1, 1)))
            ics.append(float(row["cross_sectional_ic"]))
            rounds_chosen.append(int(selection.rounds))

        if not scores:
            continue
        values = np.asarray(scores, dtype=float)
        dispersion = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        results.append(
            {
                "name": variant["name"],
                "overrides": variant["overrides"],
                "mean_score": float(values.mean()),
                "score_std": dispersion,
                "selection_score": float(values.mean() - dispersion),
                "mean_cross_sectional_ic": float(np.mean(ics)),
                "per_fold_ic": ics,
                "rounds_per_fold": rounds_chosen,
            }
        )
        print(
            f"  {variant['name']:<24} score {results[-1]['selection_score']:+.4f} "
            f"(mean {results[-1]['mean_score']:+.4f} std {dispersion:.4f}) "
            f"IC {results[-1]['mean_cross_sectional_ic']:+.4f} rounds {rounds_chosen}"
        )

    results.sort(key=lambda row: row["selection_score"], reverse=True)
    winner = results[0] if results else None
    if winner:
        print(f"\nGrid winner: {winner['name']} {winner['overrides'] or '(configured baseline)'}")
    return {
        "selection_rule": "mean(out-of-fold score) - std(out-of-fold score)",
        "variants": results,
        "selected": winner,
        "note": (
            "each variant changes one hyperparameter relative to the baseline, so a "
            "win is attributable; an unrestricted search on three folds would "
            "mostly select noise"
        ),
    }


def train_for_horizon(
    config: dict,
    horizon: int,
    walk_forward: bool = False,
    grid_search: bool = False,
    permutation: bool = False,
) -> None:
    if XGBRegressor is Any:
        raise ModuleNotFoundError(
            "xgboost is not installed in this environment. Install dependencies before training."
        )

    print("\n" + "=" * 78)
    print(f"Training XGBoost regressor for {horizon} trading days ahead")
    print("=" * 78)
    xgb_device, xgb_backend_message = resolve_xgboost_backend(config)
    print(f"Execution backend: {xgb_device.upper()} - {xgb_backend_message}")

    report_dir = ROOT / "reports"
    model_dir = ROOT / "models"
    plots_dir = ROOT / config["plots_output_dir"] / f"xgboost_h{horizon}"
    for directory in (report_dir, model_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    full_df, feature_columns, target_config = load_horizon_dataset(config, horizon)
    component_column = target_component_column(target_config)
    print(f"Panel: {len(full_df):,} rows | {len(feature_columns)} features")
    print(f"Training target: {component_column} scaled by trailing volatility")

    train_df, validation_df, test_df = chronological_train_validation_test_split(
        full_df,
        train_ratio=float(config["train_ratio"]),
        validation_ratio=float(config["validation_ratio"]),
        purge_horizon=horizon if bool(config.get("purge_overlapping_labels", True)) else 0,
    )

    xgb_cfg = dict(config.get("xgboost", {}) or {})
    seed = int(xgb_cfg.get("random_state", config.get("random_seed", 42)))

    # Everything strictly before the test period. Every selection decision is
    # confined to this slice, so the development holdout cannot influence a choice.
    history = full_df[full_df.index <= validation_df.index.max()]

    # ------------------------------------------------------------------
    # Stage 1: the market-return model, with its authority set by folds.
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
    # Selection: optional grid, then walk-forward round selection.
    # ------------------------------------------------------------------
    grid_report = None
    if grid_search:
        grid_report = run_grid_search(
            config, history, feature_columns, target_config, horizon, xgb_device
        )
        if grid_report.get("selected") and grid_report["selected"]["overrides"]:
            xgb_cfg = {**xgb_cfg, **grid_report["selected"]["overrides"]}
            print(f"Applying grid winner overrides: {grid_report['selected']['overrides']}")

    walk_forward_report = None
    out_of_fold = None
    if walk_forward:
        walk_forward_report = run_walk_forward(
            config,
            history,
            feature_columns,
            target_config,
            horizon,
            xgb_device,
            xgb_cfg,
            collect_permutation=permutation,
        )
        tuned_rounds = int(walk_forward_report["round_selection"]["rounds"])
        out_of_fold = walk_forward_report.pop("out_of_fold", None)
        save_out_of_fold_predictions(out_of_fold, report_dir, "xgboost", horizon)
    else:
        # Without folds the rounds come from a single validation curve. Still not
        # the test set, but weaker evidence, and the payload records that.
        maximum_rounds = int(xgb_cfg.get("n_estimators", 800))
        probe = make_regressor(xgb_cfg, xgb_device, seed=seed, n_estimators=maximum_rounds)
        probe.fit(train_df[feature_columns], train_df["model_target"].astype(float), verbose=False)
        curve = evaluate_rounds(
            probe,
            validation_df[feature_columns],
            validation_df[component_column].astype(float).to_numpy(),
            validation_df.index,
            horizon,
            target_scale=validation_df["target_scale"].astype(float).to_numpy(),
            reference_prediction=float(train_df[component_column].astype(float).mean()),
        )
        selection = select_rounds([curve])
        tuned_rounds = int(selection.rounds)
        print(f"\nBoosting rounds selected on validation: {tuned_rounds}")
        print(f"  {selection.reason}")

    # ------------------------------------------------------------------
    # Stage 2: bag over moving-block resamples.
    # ------------------------------------------------------------------
    uncertainty_cfg = config.get("uncertainty", {}) or {}
    bootstrap_models = int(uncertainty_cfg.get("bootstrap_models", 15))
    block_size = int(uncertainty_cfg.get("bootstrap_block_size", max(21, horizon)))
    print(
        f"\nFitting {bootstrap_models} block-bootstrap members "
        f"({tuned_rounds} rounds each, block = {block_size} sessions)..."
    )
    ensemble = BootstrapEnsemble.fit(
        model_factory=lambda member_seed: make_regressor(
            xgb_cfg, xgb_device, seed=member_seed, n_estimators=tuned_rounds
        ),
        x_train=train_df[feature_columns],
        y_train=train_df["model_target"].astype(float).to_numpy(),
        dates=train_df.index,
        n_models=bootstrap_models,
        block_size=block_size,
        seed=seed,
    )

    validation_table = forecast_table_from_frame(
        "validation",
        validation_df,
        predict_component(ensemble, validation_df, feature_columns),
        component_column,
        market_model.predict_for_panel(validation_df),
        target_config,
    )
    test_table = forecast_table_from_frame(
        "test",
        test_df,
        predict_component(ensemble, test_df, feature_columns),
        component_column,
        market_model.predict_for_panel(test_df),
        target_config,
    )

    # Importance from the ensemble that is actually used for inference.
    importance = ensemble_feature_importance(ensemble.models, feature_columns)
    importance.to_csv(report_dir / f"feature_importance_xgboost_h{horizon}.csv", index=False)

    result = evaluate_model(
        config=config,
        horizon=horizon,
        model_name="XGBoostBootstrapRegressor",
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
            "train_size": int(len(train_df)),
            "boosting_rounds": int(tuned_rounds),
            "boosting_round_selection": (
                walk_forward_report["round_selection"] if walk_forward_report else None
            ),
            "minimum_round_floor_applied": False,
            "bootstrap_models": len(ensemble),
            "bootstrap_block_size": block_size,
            "xgboost_parameters": xgb_cfg,
            "hyperparameter_grid": grid_report,
            "market_model_shrinkage_selection": shrinkage_report,
            "market_model_test_evaluation": market_evaluation,
            "feature_importance_source": (
                f"mean gain across the {len(ensemble)} bootstrap members actually used "
                "for inference, not a separate one-off reference model"
            ),
            "top_features": importance.head(20).to_dict("records"),
        },
    )

    print("\nTest performance on the modelled component:")
    print_metrics(result.payload["component_metrics"]["test"])
    print("\nTest performance on the user-facing total return:")
    print_metrics(result.payload["total_return_metrics"]["test"])
    print_decile_table(
        result.payload["decile_calibration_total_return"], "Decile calibration (total return)"
    )

    print("\nTop 12 features by mean gain across the bootstrap ensemble:")
    print(
        importance.head(12)[
            ["feature", "gain_importance", "gain_importance_std", "members_using_feature"]
        ].to_string(index=False)
    )

    if walk_forward_report and walk_forward_report.get("permutation_recommendation"):
        recommendation = walk_forward_report["permutation_recommendation"]
        print(
            f"\nPermutation importance: {len(recommendation['recommended_blocklist'])} features "
            f"harmful in >= {recommendation['minimum_folds_harmful']} folds"
        )
        if recommendation["recommended_blocklist"]:
            print(f"  recommended blocklist: {recommendation['recommended_blocklist']}")
            print("  (not applied automatically; add to feature_blocklist to act on it)")

    # ------------------------------------------------------------------
    # Persist.
    # ------------------------------------------------------------------
    calibrated_test = result.calibration.apply(
        test_table.column("raw_prediction"), test_table.dates
    )
    market_leg = test_table.column("beta", 1.0) * test_table.column(
        "market_return_forecast", market_model.drift
    )

    predictions_path = report_dir / f"test_predictions_xgboost_h{horizon}.csv"
    result.signal_frame.to_csv(predictions_path, index=False)
    result.backtest_frame.to_csv(report_dir / f"backtest_results_xgboost_h{horizon}.csv", index=False)
    if not result.portfolio_frame.empty:
        result.portfolio_frame.to_csv(report_dir / f"portfolio_xgboost_h{horizon}.csv", index=False)
    pd.DataFrame(result.payload["decile_calibration_total_return"]).to_csv(
        report_dir / f"decile_calibration_xgboost_h{horizon}.csv", index=False
    )

    plot_backtest_equity(result.backtest_frame, plots_dir)
    plot_prediction_diagnostics(
        test_table.column("true_component_return"),
        calibrated_test,
        plots_dir,
        "(modelled component, test)",
    )
    plot_interval_calibration(
        test_table.column("true_component_return"),
        calibrated_test,
        result.signal_frame["lower_bound"].to_numpy() - market_leg,
        result.signal_frame["upper_bound"].to_numpy() - market_leg,
        plots_dir,
        float(uncertainty_cfg.get("confidence_level", 0.80)),
    )
    plot_information_coefficient_series(
        test_table.dates, test_table.column("true_component_return"), calibrated_test, plots_dir
    )

    metrics_path = report_dir / f"metrics_xgboost_h{horizon}.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(result.payload), file, indent=2)

    regressor_path = model_dir / f"xgboost_regressor_h{horizon}.joblib"
    joblib.dump(ensemble, regressor_path)

    xgboost_metadata_path = model_dir / f"xgboost_metadata_h{horizon}.json"
    with open(xgboost_metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(
                {
                    "artifact_schema_version": 4,
                    "model_name": "XGBoostBootstrapRegressor",
                    "prediction_horizon": horizon,
                    "feature_columns": feature_columns,
                    "target_configuration": target_config,
                    "modelled_component": component_column,
                    "market_model": market_model.to_dict(),
                    "return_calibration": result.calibration.to_dict(),
                    "interval_calibration": result.interval_calibration.to_dict(),
                    "decision_rule": result.decision_config.to_dict(),
                    "bootstrap_models": len(ensemble),
                    "boosting_rounds": int(tuned_rounds),
                    "final_test_metrics": result.payload["total_return_metrics"]["test"],
                    "final_test_component_metrics": result.payload["component_metrics"]["test"],
                    "test_interval_metrics": result.payload["uncertainty"]["test_interval_metrics"],
                    "acceptance_gates": result.payload["acceptance_gates"],
                }
            ),
            file,
            indent=2,
        )

    print(f"\nSaved XGBoost bootstrap ensemble to: {regressor_path}")
    print(f"Saved XGBoost metadata to: {xgboost_metadata_path}")
    print(f"Saved XGBoost metrics to: {metrics_path}")
    print(f"Saved XGBoost predictions to: {predictions_path}")
    print(f"Saved plots to: {plots_dir}")


def train_models_for_horizons(
    config: dict,
    horizons: list[int],
    walk_forward: bool = False,
    grid_search: bool = False,
    permutation: bool = False,
) -> None:
    set_seed(int(config.get("random_seed", 42)))
    for horizon in horizons:
        train_for_horizon(
            config,
            int(horizon),
            walk_forward=walk_forward,
            grid_search=grid_search,
            permutation=permutation,
        )


def main() -> None:
    config = load_config()
    set_seed(int(config.get("random_seed", 42)))
    args = parse_args()
    selected_horizons = parse_horizon_list(args.horizons)
    horizons = get_horizons(config, args.horizon, selected_horizons)

    with training_log_context(ROOT / "logs", horizons, prefix="xgboost") as log_path:
        print(f"Logs will be saved to: {log_path}")
        train_models_for_horizons(
            config,
            horizons,
            walk_forward=bool(args.walk_forward),
            grid_search=bool(args.grid_search),
            permutation=bool(args.permutation_importance),
        )
        print("\nXGBoost training finished.")
        print("Important: this project is for academic research and simulation only, not financial advice.")


if __name__ == "__main__":
    main()
