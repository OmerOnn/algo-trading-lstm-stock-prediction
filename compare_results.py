from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LSTM and XGBoost results.")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    horizon = args.horizon or 10
    lstm_metrics = load_json(ROOT / "reports" / f"metrics_h{horizon}.json")
    xgb_metrics = load_json(ROOT / "reports" / f"metrics_xgboost_h{horizon}.json")

    rows = []
    for name, metrics in [("LSTM", lstm_metrics), ("XGBoost", xgb_metrics)]:
        bt = metrics["backtest_metrics"]
        rows.append(
            {
                "model": name,
                "accuracy": metrics["test_metrics"]["accuracy"],
                "return_mae": metrics["test_metrics"]["return_mae"],
                "total_return": bt["total_return"],
                "buy_and_hold_total_return": bt["buy_and_hold_total_return"],
                "excess_return_vs_buy_hold": bt["excess_return_vs_buy_hold"],
                "max_drawdown": bt["max_drawdown"],
                "sharpe_ratio": bt["sharpe_ratio"],
                "sortino_ratio": bt["sortino_ratio"],
                "win_rate": bt["win_rate"],
                "average_trade_return": bt["average_trade_return"],
                "number_of_active_trades": bt["number_of_active_trades"],
                "trade_activation_rate": bt["trade_activation_rate"],
                "feature_count": metrics.get("feature_count", 0),
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(ROOT / "reports" / f"model_comparison_h{horizon}.csv", index=False)

    md_lines = ["# Model Comparison", "", f"## Horizon {horizon}", "", comparison_df.to_markdown(index=False)]
    (ROOT / "reports" / f"model_comparison_h{horizon}.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved comparison CSV to: {ROOT / 'reports' / f'model_comparison_h{horizon}.csv'}")
    print(f"Saved comparison markdown to: {ROOT / 'reports' / f'model_comparison_h{horizon}.md'}")


if __name__ == "__main__":
    main()
