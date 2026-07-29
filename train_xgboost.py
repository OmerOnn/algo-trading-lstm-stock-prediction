"""
Train the XGBoost future-return regressor.

It solves the same problem as the LSTM on the same rows, splits, target and
decision rule, so the two are directly comparable. Uncertainty comes from a
moving-block bootstrap ensemble: each member is fitted on a resample of
contiguous date blocks, so the spread of member predictions reflects how much
the fit depends on which slice of history it happened to see.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame, tune_decision_config
from src.decision import DecisionConfig
from src.plots import (
    plot_backtest_equity,
    plot_information_coefficient_series,
    plot_interval_calibration,
    plot_prediction_diagnostics,
)
from src.regression import (
    EXCESS_RETURN_COLUMN,
    TOTAL_RETURN_COLUMN,
    apply_return_calibration,
    chronological_block_metrics,
    compose_total_return,
    estimate_market_drift,
    evaluate_baselines,
    fit_return_calibration,
    full_metrics,
    summarise_folds,
    target_component_column,
)
from src.training_logger import training_log_context
from src.uncertainty import BootstrapEnsemble, fit_interval_calibration, interval_metrics
from src.validation import chronological_train_validation_test_split, purged_walk_forward_splits
from train import (
    WALK_FORWARD_SUMMARY_KEYS,
    derived_signal_metrics,
    get_horizons,
    load_horizon_dataset,
    parse_horizon_list,
    strip_arrays,
    _print_metrics,
)


ROOT = Path(__file__).resolve().parent

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = Any


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost future-return regression models.")
    parser.add_argument("--horizon", type=int, default=None, help="Train one prediction horizon.")
    parser.add_argument(
        "--horizons", type=str, default=None, help="Comma-separated horizons, for example 21,63."
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run purged walk-forward validation and report per-fold stability.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def negative_rank_correlation(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Early-stopping objective: minimise the negative Spearman correlation.

    Because the target is already market-excess, the pooled rank correlation is a
    good proxy for the cross-sectional information coefficient, and unlike RMSE
    it cannot be gamed by shrinking every prediction towards the mean.
    """
    from scipy.stats import spearmanr

    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_predicted, dtype=float)
    if len(true_values) < 3 or np.std(predicted_values) == 0:
        return 0.0
    value = float(spearmanr(true_values, predicted_values).statistic)
    return -value if np.isfinite(value) else 0.0


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
    early_stopping_rounds: int | None = None,
) -> "XGBRegressor":
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(n_estimators or xgb_cfg.get("n_estimators", 800)),
        max_depth=int(xgb_cfg.get("max_depth", 4)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        random_state=int(seed),
        n_jobs=-1,
        tree_method="hist",
        device=device,
        eval_metric=negative_rank_correlation if early_stopping_rounds else None,
        early_stopping_rounds=early_stopping_rounds,
        reg_alpha=float(xgb_cfg.get("reg_alpha", 0.05)),
        reg_lambda=float(xgb_cfg.get("reg_lambda", 1.5)),
        min_child_weight=float(xgb_cfg.get("min_child_weight", 5.0)),
    )


def predict_component(
    ensemble: BootstrapEnsemble,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Ensemble mean and disagreement, decoded into return space."""
    mean_scaled, std_scaled = ensemble.predict(df[feature_columns])
    scales = df["target_scale"].astype(float).to_numpy()
    return {
        "mean": np.asarray(mean_scaled, dtype=float) * scales,
        "model_std": np.asarray(std_scaled, dtype=float) * scales,
        "target_scale": scales,
    }


def run_walk_forward(
    config: dict,
    full_df: pd.DataFrame,
    feature_columns: list[str],
    target_config: dict,
    horizon: int,
    xgb_device: str,
) -> dict:
    wf_cfg = config.get("walk_forward", {})
    folds = int(wf_cfg.get("folds", 3))
    xgb_cfg = config.get("xgboost", {})
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
    for split in splits:
        train_df, validation_df, test_df = split.frames(full_df)
        if train_df.empty or validation_df.empty or test_df.empty:
            continue
        regressor = make_regressor(
            xgb_cfg,
            xgb_device,
            seed=int(config.get("random_seed", 42)) + split.index,
            early_stopping_rounds=int(xgb_cfg.get("early_stopping_rounds", 50)),
        )
        regressor.fit(
            train_df[feature_columns],
            train_df["model_target"].astype(float),
            eval_set=[(validation_df[feature_columns], validation_df["model_target"].astype(float))],
            verbose=False,
        )
        predicted = regressor.predict(test_df[feature_columns]) * test_df["target_scale"].to_numpy()
        metrics = full_metrics(
            test_df.index, test_df[component_column].astype(float).to_numpy(), predicted, horizon
        )
        report = split.describe()
        report["test_metrics"] = strip_arrays(metrics)
        report["component"] = component_column
        report["best_iteration"] = int(getattr(regressor, "best_iteration", -1))
        fold_reports.append(report)
        print(
            f"  {split.label} test IC {metrics['cross_sectional_ic']:+.4f} | "
            f"dir {metrics['direction_accuracy']:.4f} | MAE {metrics['mae']:.4f}"
        )

    return {
        "folds": fold_reports,
        "summary": summarise_folds(
            [report["test_metrics"] for report in fold_reports], WALK_FORWARD_SUMMARY_KEYS
        ),
        "configuration": {
            "folds": folds,
            "expanding_origin": bool(wf_cfg.get("expanding", True)),
            "purge_trading_days": horizon,
        },
    }


def train_for_horizon(config: dict, horizon: int, walk_forward: bool = False) -> None:
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
    market_drift = estimate_market_drift(train_df, target_config)
    drift = float(market_drift["market_drift"])

    xgb_cfg = config.get("xgboost", {})
    seed = int(xgb_cfg.get("random_state", config.get("random_seed", 42)))

    # Stage 1: one early-stopped fit to discover a good number of boosting rounds.
    print("\nStage 1: fitting the reference model with early stopping...")
    reference = make_regressor(
        xgb_cfg,
        xgb_device,
        seed=seed,
        early_stopping_rounds=int(xgb_cfg.get("early_stopping_rounds", 50)),
    )
    reference.fit(
        train_df[feature_columns],
        train_df["model_target"].astype(float),
        eval_set=[(validation_df[feature_columns], validation_df["model_target"].astype(float))],
        verbose=False,
    )
    best_iteration = int(getattr(reference, "best_iteration", 0) or 0)
    tuned_rounds = max(50, best_iteration + 1)
    print(f"  best iteration on validation: {best_iteration} (bootstrap members use {tuned_rounds} rounds)")

    # Stage 2: bag over moving-block resamples for variance reduction + uncertainty.
    bootstrap_models = int(config.get("uncertainty", {}).get("bootstrap_models", 15))
    block_size = int(config.get("uncertainty", {}).get("bootstrap_block_size", max(21, horizon)))
    print(f"\nStage 2: fitting {bootstrap_models} block-bootstrap members (block = {block_size} sessions)...")
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

    validation_prediction = predict_component(ensemble, validation_df, feature_columns)
    test_prediction = predict_component(ensemble, test_df, feature_columns)

    validation_truth = validation_df[component_column].astype(float).to_numpy()
    test_truth = test_df[component_column].astype(float).to_numpy()

    calibration = fit_return_calibration(validation_truth, validation_prediction["mean"])
    validation_component = apply_return_calibration(validation_prediction["mean"], calibration)
    test_component = apply_return_calibration(test_prediction["mean"], calibration)

    confidence_level = float(config.get("uncertainty", {}).get("confidence_level", 0.80))
    interval_calibration = fit_interval_calibration(
        validation_truth,
        validation_component,
        validation_prediction["model_std"],
        validation_prediction["target_scale"],
        confidence_level=confidence_level,
        minimum_sigma=float(config.get("uncertainty", {}).get("minimum_sigma", 0.005)),
    )
    validation_lower, validation_upper, validation_sigma = interval_calibration.interval(
        validation_component, validation_prediction["model_std"], validation_prediction["target_scale"]
    )
    test_lower, test_upper, test_sigma = interval_calibration.interval(
        test_component, test_prediction["model_std"], test_prediction["target_scale"]
    )
    print(
        f"\nInterval calibration: level {confidence_level:.0%} | "
        f"multiplier {interval_calibration.conformal_multiplier:.3f} | "
        f"validation coverage {interval_calibration.validation_coverage:.3f}"
    )

    validation_total = compose_total_return(validation_component, drift)
    test_total = compose_total_return(test_component, drift)
    validation_total_actual = validation_df[TOTAL_RETURN_COLUMN].astype(float).to_numpy()
    test_total_actual = test_df[TOTAL_RETURN_COLUMN].astype(float).to_numpy()

    component_metrics = {
        "validation": strip_arrays(
            full_metrics(validation_df.index, validation_truth, validation_component, horizon)
        ),
        "test": strip_arrays(full_metrics(test_df.index, test_truth, test_component, horizon)),
    }
    total_metrics = {
        "validation": strip_arrays(
            full_metrics(validation_df.index, validation_total_actual, validation_total, horizon)
        ),
        "test": strip_arrays(full_metrics(test_df.index, test_total_actual, test_total, horizon)),
    }

    print("\nTest performance on the modelled component (market-excess return):")
    _print_metrics(component_metrics["test"])
    print("\nTest performance on the user-facing total return:")
    _print_metrics(total_metrics["test"])

    uncertainty_report = {
        "method": "moving-block bootstrap ensemble, normalised split-conformal interval",
        "bootstrap_models": len(ensemble),
        "bootstrap_block_size": block_size,
        "boosting_rounds_per_member": tuned_rounds,
        "calibration": interval_calibration.to_dict(),
        "validation_interval_metrics": interval_metrics(
            validation_truth, validation_lower, validation_upper, confidence_level
        ),
        "test_interval_metrics": interval_metrics(test_truth, test_lower, test_upper, confidence_level),
        "mean_epistemic_std": float(np.mean(test_prediction["model_std"])),
        "mean_total_sigma": float(np.mean(test_sigma)),
        "epistemic_share_of_variance": float(
            np.mean(np.square(test_prediction["model_std"]))
            / max(float(np.mean(np.square(test_sigma))), 1e-12)
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

    validation_metadata = [
        {"ticker": row, "date": str(pd.to_datetime(index).date())}
        for index, row in zip(validation_df.index, validation_df["Ticker"])
    ]
    test_metadata = [
        {"ticker": row, "date": str(pd.to_datetime(index).date())}
        for index, row in zip(test_df.index, test_df["Ticker"])
    ]

    decision_cfg, decision_tuning = tune_decision_config(
        metadata=validation_metadata,
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
        metadata=test_metadata,
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

    predictions_path = report_dir / f"test_predictions_xgboost_h{horizon}.csv"
    signal_df.to_csv(predictions_path, index=False)

    backtest_df, backtest_metrics = backtest_signals(signal_df, backtest_cfg, horizon=horizon)
    backtest_path = report_dir / f"backtest_results_xgboost_h{horizon}.csv"
    backtest_df.to_csv(backtest_path, index=False)

    point_frame = build_signal_frame(
        metadata=test_metadata,
        true_return=test_total_actual,
        predicted_return=test_total,
        cfg=backtest_cfg,
        threshold=decision_cfg.threshold,
        sigma=None,
    )
    _, point_backtest_metrics = backtest_signals(point_frame, backtest_cfg, horizon=horizon)

    print("\nDerived signal classification report:")
    print(signal_metrics["classification_report"])

    plot_backtest_equity(backtest_df, plots_dir)
    plot_prediction_diagnostics(test_truth, test_component, plots_dir, "(market-excess, test)")
    plot_interval_calibration(test_truth, test_component, test_lower, test_upper, plots_dir, confidence_level)
    plot_information_coefficient_series(test_df.index, test_truth, test_component, plots_dir)

    walk_forward_report = None
    if walk_forward:
        walk_forward_report = run_walk_forward(
            config, full_df, feature_columns, target_config, horizon, xgb_device
        )

    # ------------------------------------------------------------------
    # Persist artifacts.
    # ------------------------------------------------------------------
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_importance": reference.feature_importances_.astype(float),
        }
    ).sort_values("gain_importance", ascending=False)
    importance.to_csv(report_dir / f"feature_importance_xgboost_h{horizon}.csv", index=False)
    print("\nTop 10 features by gain:")
    print(importance.head(10).to_string(index=False))

    regressor_path = model_dir / f"xgboost_regressor_h{horizon}.joblib"
    joblib.dump(ensemble, regressor_path)

    all_metrics = {
        "model_name": "XGBoostBootstrapRegressor",
        "prediction_horizon": horizon,
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target_configuration": target_config,
        "modelled_component": component_column,
        "market_drift": market_drift,
        "component_metrics": component_metrics,
        "total_return_metrics": total_metrics,
        "validation_metrics": total_metrics["validation"],
        "test_metrics": total_metrics["test"],
        "regression_baselines": evaluate_baselines(train_df, test_df, horizon, TOTAL_RETURN_COLUMN),
        "regression_baselines_excess": evaluate_baselines(
            train_df, test_df, horizon, EXCESS_RETURN_COLUMN
        ),
        "uncertainty": uncertainty_report,
        "test_regime_blocks": chronological_block_metrics(
            test_metadata,
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
        "best_iteration": best_iteration,
        "split_method": "purged_chronological_holdout",
        "purge_trading_days": horizon if bool(config.get("purge_overlapping_labels", True)) else 0,
    }

    metrics_path = report_dir / f"metrics_xgboost_h{horizon}.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2, default=float)

    xgboost_metadata_path = model_dir / f"xgboost_metadata_h{horizon}.json"
    with open(xgboost_metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "artifact_schema_version": 3,
                "model_name": "XGBoostBootstrapRegressor",
                "prediction_horizon": horizon,
                "feature_columns": feature_columns,
                "target_configuration": target_config,
                "modelled_component": component_column,
                "market_drift": market_drift,
                "return_calibration": calibration,
                "interval_calibration": interval_calibration.to_dict(),
                "decision_rule": decision_cfg.to_dict(),
                "bootstrap_models": len(ensemble),
                "best_iteration": best_iteration,
                "final_test_metrics": total_metrics["test"],
                "test_interval_metrics": uncertainty_report["test_interval_metrics"],
            },
            file,
            indent=2,
            default=float,
        )

    print(f"\nSaved XGBoost bootstrap ensemble to: {regressor_path}")
    print(f"Saved XGBoost metadata to: {xgboost_metadata_path}")
    print(f"Saved XGBoost metrics to: {metrics_path}")
    print(f"Saved XGBoost predictions to: {predictions_path}")
    print(f"Saved XGBoost backtest to: {backtest_path}")
    print(f"Saved plots to: {plots_dir}")


def train_models_for_horizons(
    config: dict, horizons: list[int], walk_forward: bool = False
) -> None:
    set_seed(int(config.get("random_seed", 42)))
    for horizon in horizons:
        train_for_horizon(config, int(horizon), walk_forward=walk_forward)


def main() -> None:
    config = load_config()
    set_seed(int(config.get("random_seed", 42)))
    args = parse_args()
    selected_horizons = parse_horizon_list(args.horizons)
    horizons = get_horizons(config, args.horizon, selected_horizons)

    with training_log_context(ROOT / "logs", horizons, prefix="xgboost") as log_path:
        print(f"Logs will be saved to: {log_path}")
        train_models_for_horizons(config, horizons, walk_forward=bool(args.walk_forward))
        print("\nXGBoost training finished.")
        print("Important: this project is for academic research and simulation only, not financial advice.")


if __name__ == "__main__":
    main()
