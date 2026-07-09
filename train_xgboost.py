from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame
from src.baselines import add_sma_crossover_baseline, baseline_accuracy
from src.features import ID_TO_CLASS
from src.pipeline import build_or_load_dataset_for_tickers
from src.plots import plot_backtest_equity
from train import chronological_train_validation_test_split


ROOT = Path(__file__).resolve().parent


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost stock signal model.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Train one prediction horizon. If omitted, trains all configured horizons.",
    )
    return parser.parse_args()


def get_horizons(config: dict, selected_horizon: int | None = None) -> list[int]:
    if selected_horizon is not None:
        return [int(selected_horizon)]

    if "prediction_horizons" in config and config["prediction_horizons"]:
        return [int(h) for h in config["prediction_horizons"]]

    return [int(config.get("prediction_horizon", 10))]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def evaluate_xgboost(
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    x_test = test_df[feature_columns]
    y_class = test_df["signal_label"].astype(int).values
    y_return = test_df["future_return"].astype(float).values

    predicted_class = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)
    predicted_return = regressor.predict(x_test)

    accuracy = accuracy_score(y_class, predicted_class)
    mae = mean_absolute_error(y_return, predicted_return)

    report = classification_report(
        y_class,
        predicted_class,
        labels=[0, 1, 2],
        target_names=[ID_TO_CLASS[i] for i in range(3)],
        zero_division=0,
    )

    matrix = confusion_matrix(y_class, predicted_class, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": float(accuracy),
        "return_mae": float(mae),
        "classification_report": report,
        "confusion_matrix": matrix,
        "true_class": y_class,
        "predicted_class": predicted_class,
        "true_return": y_return,
        "predicted_return": predicted_return,
        "probabilities": probabilities,
    }


def strip_arrays(metrics: dict) -> dict:
    skipped = {
        "true_class",
        "predicted_class",
        "true_return",
        "predicted_return",
        "probabilities",
    }
    return {key: value for key, value in metrics.items() if key not in skipped}


def train_for_horizon(config: dict, horizon: int) -> None:
    print("\n" + "=" * 72)
    print(f"Training XGBoost model for {horizon} trading days ahead")
    print("=" * 72)

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

    full_df = add_sma_crossover_baseline(full_df)
    full_df.to_csv(processed_dir / f"full_dataset_xgboost_h{horizon}.csv")

    train_df, validation_df, test_df = chronological_train_validation_test_split(
        full_df,
        train_ratio=float(config["train_ratio"]),
        validation_ratio=float(config["validation_ratio"]),
    )

    x_train = train_df[feature_columns]
    y_train_class = train_df["signal_label"].astype(int)
    y_train_return = train_df["future_return"].astype(float)

    xgb_cfg = config.get("xgboost", {})

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=int(xgb_cfg.get("n_estimators", 400)),
        max_depth=int(xgb_cfg.get("max_depth", 4)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        random_state=int(xgb_cfg.get("random_state", config.get("random_seed", 42))),
        n_jobs=-1,
    )

    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(xgb_cfg.get("n_estimators", 400)),
        max_depth=int(xgb_cfg.get("max_depth", 4)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        random_state=int(xgb_cfg.get("random_state", config.get("random_seed", 42))),
        n_jobs=-1,
    )

    classifier.fit(x_train, y_train_class)
    regressor.fit(x_train, y_train_return)

    test_metrics = evaluate_xgboost(
        classifier=classifier,
        regressor=regressor,
        test_df=test_df,
        feature_columns=feature_columns,
    )

    print("\nFinal XGBoost test classification report:")
    print(test_metrics["classification_report"])

    metadata = [
        {
            "ticker": row["Ticker"],
            "date": str(pd.to_datetime(idx).date()),
        }
        for idx, row in test_df.iterrows()
    ]

    signal_df = build_signal_frame(
        metadata=metadata,
        true_class=test_metrics["true_class"],
        predicted_class=test_metrics["predicted_class"],
        true_return=test_metrics["true_return"],
        predicted_return=test_metrics["predicted_return"],
        probabilities=test_metrics["probabilities"],
    )

    predictions_path = report_dir / f"test_predictions_xgboost_h{horizon}.csv"
    signal_df.to_csv(predictions_path, index=False)

    backtest_cfg = BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        min_signal_confidence=float(config["min_signal_confidence"]),
        allow_short=bool(config["allow_short"]),
    )

    backtest_df, backtest_metrics = backtest_signals(signal_df, backtest_cfg)
    backtest_path = report_dir / f"backtest_results_xgboost_h{horizon}.csv"
    backtest_df.to_csv(backtest_path, index=False)

    plot_backtest_equity(backtest_df, plots_dir)

    classifier_path = model_dir / f"xgboost_classifier_h{horizon}.joblib"
    regressor_path = model_dir / f"xgboost_regressor_h{horizon}.joblib"
    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    baselines = baseline_accuracy(test_df)

    all_metrics = {
        "model_name": "XGBoost",
        "prediction_horizon": horizon,
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "test_metrics": strip_arrays(test_metrics),
        "baseline_metrics": baselines,
        "backtest_metrics": backtest_metrics,
        "class_mapping": ID_TO_CLASS,
    }

    metrics_path = report_dir / f"metrics_xgboost_h{horizon}.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    print(f"\nSaved XGBoost classifier to: {classifier_path}")
    print(f"Saved XGBoost regressor to: {regressor_path}")
    print(f"Saved XGBoost metrics to: {metrics_path}")
    print(f"Saved XGBoost predictions to: {predictions_path}")
    print(f"Saved XGBoost backtest to: {backtest_path}")


def main() -> None:
    config = load_config()
    args = parse_args()
    set_seed(int(config.get("random_seed", 42)))

    for horizon in get_horizons(config, args.horizon):
        train_for_horizon(config, int(horizon))

    print("\nXGBoost training finished.")
    print("Important: this project is for academic research and simulation only, not financial advice.")


if __name__ == "__main__":
    main()
