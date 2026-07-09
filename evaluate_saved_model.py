from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model artifact.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "xgboost"], help="Model family to inspect")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon")
    return parser.parse_args()


def main() -> None:
    config = load_config()
    args = parse_args()
    horizon = int(args.horizon or config.get("default_prediction_horizon", config.get("prediction_horizon", 10)))
    prefix = "" if args.model == "lstm" else "_xgboost"
    metrics_path = ROOT / "reports" / f"metrics{prefix}_h{horizon}.json"
    backtest_path = ROOT / "reports" / f"backtest_results{prefix}_h{horizon}.csv"
    predictions_path = ROOT / "reports" / f"test_predictions{prefix}_h{horizon}.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Run training first. Metrics file not found: {metrics_path}")

    metrics = load_json(metrics_path)
    print(f"\n{args.model.upper()} Evaluation Summary")
    print("=" * 32)
    print(f"Test accuracy: {metrics['test_metrics']['accuracy']:.4f}")
    print(f"Test return MAE: {metrics['test_metrics']['return_mae']:.4f}")
    print(f"Majority baseline accuracy: {metrics['baseline_metrics']['majority_class_accuracy']:.4f}")
    print(f"SMA 20/50 baseline accuracy: {metrics['baseline_metrics']['sma_20_50_crossover_accuracy']:.4f}")

    print("\nBacktest Summary")
    print("=" * 16)
    bt = metrics["backtest_metrics"]
    print(f"Final equity: ${bt['final_equity']:.2f}")
    print(f"Total return: {bt['total_return'] * 100:.2f}%")
    print(f"Buy-and-hold comparison: {bt['buy_and_hold_total_return'] * 100:.2f}%")
    print(f"Excess return vs buy-and-hold: {bt['excess_return_vs_buy_hold'] * 100:.2f}%")
    print(f"Max drawdown: {bt['max_drawdown'] * 100:.2f}%")
    print(f"Sharpe ratio: {bt['sharpe_ratio']:.4f}")
    print(f"Win rate: {bt['win_rate'] * 100:.2f}%")
    print(f"Active trades: {bt['number_of_active_trades']}")
    print(f"Trade activation rate: {bt['trade_activation_rate'] * 100:.2f}%")

    if predictions_path.exists():
        preds = pd.read_csv(predictions_path)
        print("\nSignal Distribution")
        print("=" * 20)
        print(preds["predicted_signal"].value_counts().to_string())

    if backtest_path.exists():
        print(f"\nDetailed backtest file: {backtest_path}")
    print("Important: this is an academic simulation only, not financial advice.")


if __name__ == "__main__":
    main()
