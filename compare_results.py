from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


ROOT = Path(__file__).resolve().parent

METRIC_SPECS = [
    ("accuracy", ("test_metrics", "accuracy"), True),
    ("return_mae", ("test_metrics", "return_mae"), False),
    ("total_return", ("backtest_metrics", "total_return"), True),
    ("buy_and_hold_total_return", ("backtest_metrics", "buy_and_hold_total_return"), True),
    ("excess_return_vs_buy_hold", ("backtest_metrics", "excess_return_vs_buy_hold"), True),
    ("max_drawdown", ("backtest_metrics", "max_drawdown"), False),
    ("sharpe_ratio", ("backtest_metrics", "sharpe_ratio"), True),
    ("sortino_ratio", ("backtest_metrics", "sortino_ratio"), True),
    ("win_rate", ("backtest_metrics", "win_rate"), True),
    ("average_trade_return", ("backtest_metrics", "average_trade_return"), True),
    ("number_of_active_trades", ("backtest_metrics", "number_of_active_trades"), True),
    ("trade_activation_rate", ("backtest_metrics", "trade_activation_rate"), True),
    ("feature_count", ("feature_count",), True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LSTM and XGBoost results.")
    parser.add_argument("--horizon", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def nested_lookup(metrics: dict, path: tuple[str, ...]):
    value = metrics
    for key in path:
        value = value[key]
    return value


def build_comparison_rows(lstm_metrics: dict, xgb_metrics: dict) -> pd.DataFrame:
    rows = []

    for metric_name, path, higher_is_better in METRIC_SPECS:
        lstm_value = nested_lookup(lstm_metrics, path)
        xgb_value = nested_lookup(xgb_metrics, path)

        if isinstance(lstm_value, (int, float)) and isinstance(xgb_value, (int, float)):
            delta = float(xgb_value) - float(lstm_value)
            if metric_name == "feature_count":
                winner = "same" if xgb_value == lstm_value else "xgboost" if xgb_value > lstm_value else "lstm"
            elif higher_is_better:
                winner = "xgboost" if xgb_value > lstm_value else "lstm" if xgb_value < lstm_value else "tie"
            else:
                winner = "xgboost" if xgb_value < lstm_value else "lstm" if xgb_value > lstm_value else "tie"
        else:
            delta = None
            winner = "n/a"

        rows.append(
            {
                "metric": metric_name,
                "lstm": round(float(lstm_value), 6) if isinstance(lstm_value, float) else lstm_value,
                "xgboost": round(float(xgb_value), 6) if isinstance(xgb_value, float) else xgb_value,
                "delta_xgboost_minus_lstm": round(delta, 6) if delta is not None else None,
                "higher_is_better": higher_is_better,
                "winner": winner,
            }
        )

    return pd.DataFrame(rows)


def write_markdown_report(df: pd.DataFrame, horizon: int, output_path: Path) -> None:
    if tabulate is not None:
        table = tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".6f")
    else:
        headers = list(df.columns)
        rows = ["| " + " | ".join(headers) + " |"]
        rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        table = "\n".join(rows)

    markdown = [
        f"# Model Comparison - Horizon {horizon}",
        "",
        "This report compares the saved LSTM and XGBoost artifacts using the same chronological split and backtest settings.",
        "",
        table,
        "",
    ]
    output_path.write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    args = parse_args()
    horizon = int(args.horizon)

    lstm_path = ROOT / "reports" / f"metrics_h{horizon}.json"
    xgb_path = ROOT / "reports" / f"metrics_xgboost_h{horizon}.json"

    if not lstm_path.exists():
        raise FileNotFoundError(f"Missing LSTM metrics: {lstm_path}. Run: python train.py --horizon {horizon}")

    if not xgb_path.exists():
        raise FileNotFoundError(f"Missing XGBoost metrics: {xgb_path}. Run: python train_xgboost.py --horizon {horizon}")

    lstm_metrics = load_json(lstm_path)
    xgb_metrics = load_json(xgb_path)
    comparison = build_comparison_rows(lstm_metrics, xgb_metrics)

    output_csv = ROOT / "reports" / f"model_comparison_h{horizon}.csv"
    output_md = ROOT / "reports" / f"model_comparison_h{horizon}.md"

    comparison.to_csv(output_csv, index=False)

    write_markdown_report(comparison, horizon, output_md)

    print("\nModel comparison")
    print("================")
    print(comparison.to_string(index=False))

    print(f"\nSaved comparison CSV to: {output_csv}")
    print(f"Saved comparison Markdown to: {output_md}")


if __name__ == "__main__":
    main()
