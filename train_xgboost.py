from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame
from src.baselines import add_sma_crossover_baseline, baseline_accuracy
from src.pipeline import build_dataset_for_tickers
from src.features import ID_TO_CLASS

ROOT = Path(__file__).resolve().parent


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost stock signal models.")
    parser.add_argument("--horizon", type=int, default=None, help="Train only one prediction horizon")
    return parser.parse_args()


def get_horizons(config: dict, selected_horizon: int | None = None) -> list[int]:
    if selected_horizon is not None:
        return [int(selected_horizon)]
    if "prediction_horizons" in config and config["prediction_horizons"]:
        return [int(h) for h in config["prediction_horizons"]]
    return [int(config.get("prediction_horizon", 10))]


def artifact_path(base_path: str | Path, horizon: int) -> Path:
    path = Path(base_path)
    return ROOT / path.with_name(f"{path.stem}_h{horizon}{path.suffix}")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def train_for_horizon(config: dict, horizon: int) -> None:
    print("\n" + "=" * 72)
    print(f"Training XGBoost model for {horizon} trading days ahead")
    print("=" * 72)

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = artifact_path("models/xgboost_classifier.joblib", horizon)
    regressor_path = artifact_path("models/xgboost_regressor.joblib", horizon)
    metrics_path = artifact_path("reports/metrics_xgboost.json", horizon)
    backtest_path = artifact_path("reports/backtest_results_xgboost.csv", horizon)
    predictions_path = artifact_path("reports/test_predictions_xgboost.csv", horizon)

    full_df, feature_columns = build_dataset_for_tickers(
        tickers=config["tickers"],
        benchmark_ticker=config["benchmark_ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        prediction_horizon=horizon,
        buy_threshold=float(config["buy_threshold"]),
        sell_threshold=float(config["sell_threshold"]),
        macro_tickers=config.get("macro_tickers"),
    )
    full_df = add_sma_crossover_baseline(full_df)

    unique_dates = sorted(full_df.index.unique())
    if len(unique_dates) < 10:
        raise ValueError("Not enough rows for a chronological train/validation/test split")

    train_idx = int(len(unique_dates) * float(config["train_ratio"]))
    validation_idx = int(len(unique_dates) * (float(config["train_ratio"]) + float(config["validation_ratio"])))
    train_end_date = unique_dates[train_idx]
    validation_end_date = unique_dates[validation_idx]
    train_df = full_df[full_df.index < train_end_date].copy()
    validation_df = full_df[(full_df.index >= train_end_date) & (full_df.index < validation_end_date)].copy()
    test_df = full_df[full_df.index >= validation_end_date].copy()

    feature_columns = [col for col in feature_columns if col in full_df.columns]
    X_train = train_df[feature_columns].fillna(0.0)
    X_val = validation_df[feature_columns].fillna(0.0)
    X_test = test_df[feature_columns].fillna(0.0)
    y_class_train = train_df["signal_label"].astype(int)
    y_class_val = validation_df["signal_label"].astype(int)
    y_class_test = test_df["signal_label"].astype(int)
    y_return_train = train_df["future_return"].astype(float)
    y_return_val = validation_df["future_return"].astype(float)
    y_return_test = test_df["future_return"].astype(float)

    xgb_cfg = config.get("xgboost", {})
    classifier = XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 400)),
        max_depth=int(xgb_cfg.get("max_depth", 4)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        random_state=int(xgb_cfg.get("random_state", 42)),
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    regressor = XGBRegressor(
        n_estimators=int(xgb_cfg.get("n_estimators", 400)),
        max_depth=int(xgb_cfg.get("max_depth", 4)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        random_state=int(xgb_cfg.get("random_state", 42)),
        objective="reg:squarederror",
        n_jobs=-1,
    )

    classifier.fit(X_train, y_class_train)
    regressor.fit(X_train, y_return_train)

    pred_class = classifier.predict(X_test)
    pred_return = regressor.predict(X_test)

    test_accuracy = accuracy_score(y_class_test, pred_class)
    test_return_mae = mean_absolute_error(y_return_test, pred_return)

    probabilities = classifier.predict_proba(X_test)
    signal_df = build_signal_frame(
        metadata=[{"ticker": str(ticker), "date": idx} for ticker, idx in zip(test_df["Ticker"].tolist(), test_df.index.tolist())],
        true_class=y_class_test.to_numpy(),
        predicted_class=pred_class.astype(int),
        true_return=y_return_test.to_numpy(),
        predicted_return=pred_return.astype(float),
        probabilities=probabilities,
    )
    signal_df.to_csv(predictions_path, index=False)

    backtest_cfg = BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        min_signal_confidence=float(config["min_signal_confidence"]),
        allow_short=bool(config["allow_short"]),
    )
    backtest_df, backtest_metrics = backtest_signals(signal_df, backtest_cfg)
    backtest_df.to_csv(backtest_path, index=False)

    baselines = baseline_accuracy(test_df)
    metrics = {
        "model_name": "XGBoost",
        "prediction_horizon": horizon,
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "test_metrics": {
            "accuracy": float(test_accuracy),
            "return_mae": float(test_return_mae),
        },
        "baseline_metrics": baselines,
        "backtest_metrics": backtest_metrics,
        "class_mapping": ID_TO_CLASS,
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    print(f"Saved classifier to: {classifier_path}")
    print(f"Saved regressor to: {regressor_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved backtest results to: {backtest_path}")


def main() -> None:
    config = load_config()
    args = parse_args()
    set_seed(config.get("random_seed", 42))
    horizons = get_horizons(config, args.horizon)
    for horizon in horizons:
        train_for_horizon(config, int(horizon))


if __name__ == "__main__":
    main()
