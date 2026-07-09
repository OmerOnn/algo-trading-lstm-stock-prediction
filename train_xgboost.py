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

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame, cost_aware_signal_threshold
from src.pipeline import build_or_load_dataset_for_tickers
from src.plots import plot_backtest_equity
from train import (
    chronological_train_validation_test_split,
    derived_signal_metrics,
    get_horizons,
    parse_horizon_list,
    regression_metrics,
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
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Train one prediction horizon. If omitted, trains all configured horizons.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated horizons to train, for example 21,252.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def evaluate_xgboost(
    regressor: XGBRegressor,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    x_values = df[feature_columns]
    true_return = df["future_return"].astype(float).values
    predicted_return = regressor.predict(x_values)
    metrics = regression_metrics(true_return, predicted_return)
    return {
        **metrics,
        "true_return": true_return,
        "predicted_return": predicted_return,
    }


def strip_arrays(metrics: dict) -> dict:
    skipped = {"true_return", "predicted_return"}
    return {key: value for key, value in metrics.items() if key not in skipped}


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


def train_for_horizon(config: dict, horizon: int) -> None:
    if XGBRegressor is Any:
        raise ModuleNotFoundError(
            "xgboost is not installed in this environment. Install dependencies before running XGBoost training."
        )

    print("\n" + "=" * 72)
    print(f"Training XGBoost regressor for {horizon} trading days ahead")
    print("=" * 72)
    xgb_device, xgb_backend_message = resolve_xgboost_backend(config)
    print(f"Execution backend: {xgb_device.upper()}")
    print(xgb_backend_message)

    processed_dir = ROOT / "data" / "processed"
    cache_dir = ROOT / "data" / "cache"
    report_dir = ROOT / "reports"
    model_dir = ROOT / "models"
    plots_dir = ROOT / config["plots_output_dir"] / f"xgboost_h{horizon}"

    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache_path = cache_dir / f"full_dataset_h{horizon}.csv"
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
    )

    full_df.to_csv(processed_dir / f"full_dataset_xgboost_h{horizon}.csv")
    train_df, validation_df, test_df = chronological_train_validation_test_split(
        full_df,
        train_ratio=float(config["train_ratio"]),
        validation_ratio=float(config["validation_ratio"]),
    )

    x_train = train_df[feature_columns]
    y_train = train_df["future_return"].astype(float)
    x_validation = validation_df[feature_columns]
    y_validation = validation_df["future_return"].astype(float)

    xgb_cfg = config.get("xgboost", {})
    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(xgb_cfg.get("n_estimators", 400)),
        max_depth=int(xgb_cfg.get("max_depth", 4)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        random_state=int(xgb_cfg.get("random_state", config.get("random_seed", 42))),
        n_jobs=-1,
        tree_method="hist",
        device=xgb_device,
    )

    regressor.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )

    validation_metrics = evaluate_xgboost(regressor, validation_df, feature_columns)
    test_metrics = evaluate_xgboost(regressor, test_df, feature_columns)

    print("\nFinal XGBoost test regression metrics:")
    print(
        f"MAE: {test_metrics['mae']:.4f} | "
        f"RMSE: {test_metrics['rmse']:.4f} | "
        f"Direction Acc: {test_metrics['direction_accuracy']:.4f} | "
        f"Corr: {test_metrics['return_correlation']:.4f}"
    )

    metadata = [
        {"ticker": row["Ticker"], "date": str(pd.to_datetime(idx).date())}
        for idx, row in test_df.iterrows()
    ]

    backtest_cfg = BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        allow_short=bool(config["allow_short"]),
        signal_threshold_multiplier=float(config.get("signal_threshold_multiplier", 1.0)),
        min_signal_edge=float(config.get("min_signal_edge", 0.0)),
    )
    signal_df = build_signal_frame(
        metadata=metadata,
        true_return=test_metrics["true_return"],
        predicted_return=test_metrics["predicted_return"],
        cfg=backtest_cfg,
    )
    signal_metrics = derived_signal_metrics(signal_df)

    predictions_path = report_dir / f"test_predictions_xgboost_h{horizon}.csv"
    signal_df.to_csv(predictions_path, index=False)

    backtest_df, backtest_metrics = backtest_signals(signal_df, backtest_cfg)
    backtest_path = report_dir / f"backtest_results_xgboost_h{horizon}.csv"
    backtest_df.to_csv(backtest_path, index=False)
    plot_backtest_equity(backtest_df, plots_dir)

    print("\nDerived signal classification report:")
    print(signal_metrics["classification_report"])

    regressor_path = model_dir / f"xgboost_regressor_h{horizon}.joblib"
    joblib.dump(regressor, regressor_path)

    all_metrics = {
        "model_name": "XGBoostRegressor",
        "prediction_horizon": horizon,
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "validation_metrics": strip_arrays(validation_metrics),
        "test_metrics": strip_arrays(test_metrics),
        "signal_metrics": signal_metrics,
        "backtest_metrics": backtest_metrics,
        "signal_threshold": cost_aware_signal_threshold(backtest_cfg),
    }

    metrics_path = report_dir / f"metrics_xgboost_h{horizon}.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    print(f"\nSaved XGBoost regressor to: {regressor_path}")
    print(f"Saved XGBoost metrics to: {metrics_path}")
    print(f"Saved XGBoost predictions to: {predictions_path}")
    print(f"Saved XGBoost backtest to: {backtest_path}")


def main() -> None:
    config = load_config()
    set_seed(int(config.get("random_seed", 42)))
    args = parse_args()
    selected_horizons = parse_horizon_list(args.horizons)
    train_models_for_horizons(config, get_horizons(config, args.horizon, selected_horizons))
    print("\nXGBoost training finished.")
    print("Important: this project is for academic research and simulation only, not financial advice.")


def train_models_for_horizons(config: dict, horizons: list[int]) -> None:
    set_seed(int(config.get("random_seed", 42)))
    for horizon in horizons:
        train_for_horizon(config, int(horizon))


if __name__ == "__main__":
    main()
