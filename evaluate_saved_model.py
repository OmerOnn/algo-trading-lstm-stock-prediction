from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print saved model evaluation summary.")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "xgboost"])
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    horizon = int(args.horizon)

    if args.model == "lstm":
        metrics_path = ROOT / "reports" / f"metrics_h{horizon}.json"
        backtest_path = ROOT / "reports" / f"backtest_results_h{horizon}.csv"
        predictions_path = ROOT / "reports" / f"test_predictions_h{horizon}.csv"
    else:
        metrics_path = ROOT / "reports" / f"metrics_xgboost_h{horizon}.json"
        backtest_path = ROOT / "reports" / f"backtest_results_xgboost_h{horizon}.csv"
        predictions_path = ROOT / "reports" / f"test_predictions_xgboost_h{horizon}.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file was not found: {metrics_path}")

    metrics = load_json(metrics_path)

    print("\nModel Evaluation Summary")
    print("========================")
    print(f"Model: {metrics.get('model_name', args.model)}")
    print(f"Horizon: {metrics['prediction_horizon']} trading days")
    print(f"Test MAE: {metrics['test_metrics']['mae']:.4f}")
    print(f"Test RMSE: {metrics['test_metrics']['rmse']:.4f}")
    print(f"Direction accuracy: {metrics['test_metrics']['direction_accuracy']:.4f}")
    print(f"Return correlation: {metrics['test_metrics']['return_correlation']:.4f}")

    print("\nBacktest Summary")
    print("================")
    bt = metrics["backtest_metrics"]
    print(f"Final equity: ${bt['final_equity']:.2f}")
    print(f"Total return: {bt['total_return'] * 100:.2f}%")
    print(f"Buy-and-hold comparison: {bt['buy_and_hold_total_return'] * 100:.2f}%")
    print(f"Excess return vs buy-and-hold: {bt.get('excess_return_vs_buy_hold', 0.0) * 100:.2f}%")
    print(f"Max drawdown: {bt['max_drawdown'] * 100:.2f}%")
    print(f"Sharpe ratio: {bt['sharpe_ratio']:.4f}")
    print(f"Win rate: {bt.get('win_rate', 0.0) * 100:.2f}%")
    print(f"Active trades: {bt['number_of_active_trades']}")
    print(f"Trade threshold: {metrics.get('signal_threshold', 0.0) * 100:.2f}%")

    if predictions_path.exists():
        preds = pd.read_csv(predictions_path)
        if "predicted_signal" in preds.columns:
            print("\nSignal Distribution")
            print("===================")
            print(preds["predicted_signal"].value_counts().to_string())

    if backtest_path.exists():
        print(f"\nDetailed backtest file: {backtest_path}")

    print("Important: this is an academic simulation only, not financial advice.")


if __name__ == "__main__":
    main()
