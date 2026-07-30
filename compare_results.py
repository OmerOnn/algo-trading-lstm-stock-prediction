"""
Compare the LSTM and XGBoost regressors for one horizon.

Both models are evaluated on the same rows, the same purged split, the same
target and the same decision rule, so every difference in the table is a
property of the model rather than of the experimental setup. Baselines and
interval-quality metrics are included because a model that loses to a
historical-mean forecast, or whose 80% interval covers 55% of outcomes, has not
demonstrated anything regardless of its headline accuracy.
"""

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

# (display name, path in the metrics JSON, higher_is_better)
METRIC_SPECS = [
    ("cross_sectional_ic", ("component_metrics", "test", "cross_sectional_mean_ic"), True),
    ("cross_sectional_icir", ("component_metrics", "test", "cross_sectional_icir"), True),
    ("ic_t_statistic", ("component_metrics", "test", "cross_sectional_ic_t_statistic"), True),
    ("ic_positive_date_rate", ("component_metrics", "test", "cross_sectional_ic_positive_rate"), True),
    (
        "long_short_spread_annualised",
        ("component_metrics", "test", "cross_sectional_long_short_spread_annualised"),
        True,
    ),
    ("direction_accuracy", ("test_metrics", "direction_accuracy"), True),
    ("return_correlation", ("test_metrics", "return_correlation"), True),
    ("mae", ("test_metrics", "mae"), False),
    ("mse", ("test_metrics", "mse"), False),
    ("rmse", ("test_metrics", "rmse"), False),
    ("r2", ("test_metrics", "r2"), True),
    ("mse_skill_vs_historical_mean", ("test_metrics", "mse_skill_vs_historical_mean"), True),
    ("mae_skill_vs_historical_mean", ("test_metrics", "mae_skill_vs_historical_mean"), True),
    ("rmse_skill_vs_zero", ("test_metrics", "rmse_skill_vs_zero"), True),
    ("interval_coverage_picp", ("uncertainty", "test_interval_metrics", "coverage_picp"), True),
    ("interval_coverage_error", ("uncertainty", "test_interval_metrics", "coverage_error"), True),
    ("interval_width_mpiw", ("uncertainty", "test_interval_metrics", "mean_interval_width_mpiw"), False),
    ("winkler_score", ("uncertainty", "test_interval_metrics", "winkler_score"), False),
    ("backtest_total_return", ("backtest_metrics", "total_return"), True),
    ("buy_and_hold_total_return", ("backtest_metrics", "buy_and_hold_total_return"), True),
    ("excess_return_vs_buy_hold", ("backtest_metrics", "excess_return_vs_buy_hold"), True),
    ("backtest_sharpe_ratio", ("backtest_metrics", "sharpe_ratio"), True),
    ("information_ratio_vs_universe", ("backtest_metrics", "information_ratio_vs_universe"), True),
    ("max_drawdown", ("backtest_metrics", "max_drawdown"), True),
    ("win_rate", ("backtest_metrics", "win_rate"), True),
    ("number_of_active_trades", ("backtest_metrics", "number_of_active_trades"), True),
    (
        "portfolio_sharpe_mean_over_offsets",
        ("portfolio_backtest", "top_k_long_only", "distribution", "sharpe_ratio", "mean"),
        True,
    ),
    (
        "portfolio_sharpe_worst_offset",
        ("portfolio_backtest", "top_k_long_only", "distribution", "sharpe_ratio", "worst"),
        True,
    ),
    (
        "portfolio_information_ratio_mean",
        ("portfolio_backtest", "top_k_long_only", "distribution",
         "information_ratio_vs_universe", "mean"),
        True,
    ),
    (
        "portfolio_excess_vs_universe_mean",
        ("portfolio_backtest", "top_k_long_only", "distribution",
         "excess_return_vs_universe", "mean"),
        True,
    ),
    (
        "portfolio_annualised_turnover",
        ("portfolio_backtest", "top_k_long_only", "distribution", "annualized_turnover", "mean"),
        False,
    ),
    ("acceptance_gates_passed", ("acceptance_gates", "passed"), True),
    ("feature_count", ("feature_count",), True),
]

BASELINE_METRIC_KEYS = [
    ("cross_sectional_mean_ic", "cross_sectional_ic"),
    ("direction_accuracy", "direction_accuracy"),
    ("mae", "mae"),
    ("rmse", "rmse"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LSTM and XGBoost results.")
    parser.add_argument("--horizon", type=int, default=21)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def nested_lookup(metrics: dict, path: tuple[str, ...]):
    value = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def build_comparison_rows(lstm_metrics: dict, xgb_metrics: dict) -> pd.DataFrame:
    rows = []
    for metric_name, path, higher_is_better in METRIC_SPECS:
        lstm_value = nested_lookup(lstm_metrics, path)
        xgb_value = nested_lookup(xgb_metrics, path)

        if isinstance(lstm_value, (int, float)) and isinstance(xgb_value, (int, float)):
            delta = float(xgb_value) - float(lstm_value)
            if metric_name in {"feature_count", "interval_coverage_picp"}:
                # Coverage is a target, not a maximum: closer to nominal wins.
                winner = "same" if xgb_value == lstm_value else "n/a"
            elif higher_is_better:
                winner = "xgboost" if xgb_value > lstm_value else "lstm" if xgb_value < lstm_value else "tie"
            else:
                winner = "xgboost" if xgb_value < lstm_value else "lstm" if xgb_value > lstm_value else "tie"
        else:
            delta, winner = None, "n/a"

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


def build_baseline_rows(lstm_metrics: dict, xgb_metrics: dict) -> pd.DataFrame:
    """Both models plus every reference forecast, scored on identical test rows."""
    rows = []

    def add(name: str, metrics: dict | None) -> None:
        if not metrics:
            return
        row = {"model": name}
        for source_key, display_key in BASELINE_METRIC_KEYS:
            value = metrics.get(source_key)
            row[display_key] = round(float(value), 6) if isinstance(value, (int, float)) else None
        rows.append(row)

    add("LSTM ensemble", nested_lookup(lstm_metrics, ("component_metrics", "test")))
    add("XGBoost bootstrap", nested_lookup(xgb_metrics, ("component_metrics", "test")))

    baseline_block = (
        lstm_metrics.get("regression_baselines_component")
        or lstm_metrics.get("regression_baselines_excess")
        or {}
    )
    for name, values in baseline_block.items():
        add(f"baseline: {name.replace('_', ' ')}", values)

    return pd.DataFrame(rows)


def build_walk_forward_rows(metrics: dict, model_name: str) -> pd.DataFrame:
    walk_forward = metrics.get("walk_forward")
    if not walk_forward or not walk_forward.get("folds"):
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "fold": fold["label"],
                "test_start": fold["test"]["start"],
                "test_end": fold["test"]["end"],
                "cross_sectional_ic": round(
                    float(fold["test_metrics"].get("cross_sectional_mean_ic", 0.0)), 6
                ),
                "direction_accuracy": round(
                    float(fold["test_metrics"].get("direction_accuracy", 0.0)), 6
                ),
                "mae": round(float(fold["test_metrics"].get("mae", 0.0)), 6),
            }
            for fold in walk_forward["folds"]
        ]
    )


def to_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    if tabulate is not None:
        return tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".6f")
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_markdown_report(
    comparison: pd.DataFrame,
    baselines: pd.DataFrame,
    walk_forward: pd.DataFrame,
    horizon: int,
    lstm_metrics: dict,
    output_path: Path,
) -> None:
    confidence_level = (
        nested_lookup(lstm_metrics, ("uncertainty", "calibration", "confidence_level")) or 0.80
    )
    markdown = [
        f"# Model Comparison - Horizon {horizon} trading days",
        "",
        "Both models use the same panel, the same purged chronological split, the same "
        "market-excess target, the same calibration and uncertainty treatment, and the "
        "same validation-frozen decision rule. Every difference below is therefore a "
        "property of the model rather than of the experimental setup.",
        "",
        "> **Which backtest rows to trust.** The `backtest_*`, "
        "`buy_and_hold_total_return`, `excess_return_vs_buy_hold` and "
        "`information_ratio_vs_universe` rows come from the *signal* backtest, which "
        "holds cash whenever no name clears the hurdle. It is only partly invested, so "
        "comparing its total return to 100%-invested buy-and-hold is not a like-for-like "
        "comparison and its large negative excess is an artefact of exposure, not of "
        "selection skill. The `portfolio_*` rows are the fair test: a fully invested "
        "top-k book measured against the equal-weight universe, averaged over every "
        "rebalance offset.",
        "",
        "## Head-to-head metrics",
        "",
        to_table(comparison),
        "",
        "## Against baselines",
        "",
        "Scored on identical test rows, in market-excess return space, which is the "
        "quantity the models are trained to predict.",
        "",
        to_table(baselines),
        "",
        f"## Interval quality (nominal level {float(confidence_level):.0%})",
        "",
        "`interval_coverage_picp` should sit near the nominal level. A much higher value "
        "means the interval is uninformatively wide; a much lower value means it is "
        "overconfident. `winkler_score` scores width and coverage jointly; lower is better.",
        "",
    ]
    if not walk_forward.empty:
        markdown += [
            "## Purged walk-forward folds",
            "",
            "Each fold is refitted from scratch on data strictly before its own test window.",
            "",
            to_table(walk_forward),
            "",
        ]
    markdown += ["---", "", "_Academic research and simulation only. Not financial advice._", ""]
    output_path.write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    args = parse_args()
    horizon = int(args.horizon)

    lstm_path = ROOT / "reports" / f"metrics_h{horizon}.json"
    xgb_path = ROOT / "reports" / f"metrics_xgboost_h{horizon}.json"

    if not lstm_path.exists():
        raise FileNotFoundError(f"Missing LSTM metrics: {lstm_path}. Run: python3 train_lstm.py --horizon {horizon}")
    if not xgb_path.exists():
        raise FileNotFoundError(
            f"Missing XGBoost metrics: {xgb_path}. Run: python3 train_xgboost.py --horizon {horizon}"
        )

    lstm_metrics = load_json(lstm_path)
    xgb_metrics = load_json(xgb_path)

    comparison = build_comparison_rows(lstm_metrics, xgb_metrics)
    baselines = build_baseline_rows(lstm_metrics, xgb_metrics)
    walk_forward = pd.concat(
        [
            build_walk_forward_rows(lstm_metrics, "lstm"),
            build_walk_forward_rows(xgb_metrics, "xgboost"),
        ],
        ignore_index=True,
    )

    output_csv = ROOT / "reports" / f"model_comparison_h{horizon}.csv"
    output_md = ROOT / "reports" / f"model_comparison_h{horizon}.md"
    comparison.to_csv(output_csv, index=False)
    baselines.to_csv(ROOT / "reports" / f"baseline_comparison_h{horizon}.csv", index=False)
    write_markdown_report(comparison, baselines, walk_forward, horizon, lstm_metrics, output_md)

    print("\nModel comparison")
    print("================")
    print(comparison.to_string(index=False))
    print("\nAgainst baselines (market-excess return space)")
    print("==============================================")
    print(baselines.to_string(index=False))
    if not walk_forward.empty:
        print("\nWalk-forward folds")
        print("==================")
        print(walk_forward.to_string(index=False))

    print(f"\nSaved comparison CSV to: {output_csv}")
    print(f"Saved comparison Markdown to: {output_md}")


if __name__ == "__main__":
    main()
