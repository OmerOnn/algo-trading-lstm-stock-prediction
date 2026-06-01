from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()
    metrics_path = ROOT / config["metrics_output_path"]
    backtest_path = ROOT / config["backtest_output_path"]
    predictions_path = ROOT / "reports" / "test_predictions.csv"

    if not metrics_path.exists():
        raise FileNotFoundError("Run python train.py first. metrics.json was not found.")

    metrics = load_json(metrics_path)
    print("\nModel Evaluation Summary")
    print("========================")
    print(f"Test accuracy: {metrics['test_metrics']['accuracy']:.4f}")
    print(f"Test return MAE: {metrics['test_metrics']['return_mae']:.4f}")
    print(f"Majority baseline accuracy: {metrics['baseline_metrics']['majority_class_accuracy']:.4f}")
    print(f"SMA 20/50 baseline accuracy: {metrics['baseline_metrics']['sma_20_50_crossover_accuracy']:.4f}")

    print("\nBacktest Summary")
    print("================")
    bt = metrics["backtest_metrics"]
    print(f"Final equity: ${bt['final_equity']:.2f}")
    print(f"Total return: {bt['total_return'] * 100:.2f}%")
    print(f"Buy-and-hold comparison: {bt['buy_and_hold_total_return'] * 100:.2f}%")
    print(f"Max drawdown: {bt['max_drawdown'] * 100:.2f}%")
    print(f"Sharpe ratio: {bt['sharpe_ratio']:.4f}")
    print(f"Active trades: {bt['number_of_active_trades']}")
    print(f"Trade activation rate: {bt['trade_activation_rate'] * 100:.2f}%")

    if predictions_path.exists():
        preds = pd.read_csv(predictions_path)
        print("\nSignal Distribution")
        print("===================")
        print(preds["predicted_signal"].value_counts().to_string())

    if backtest_path.exists():
        print(f"\nDetailed backtest file: {backtest_path}")
    print("Important: this is an academic simulation only, not financial advice.")


if __name__ == "__main__":
    main()
