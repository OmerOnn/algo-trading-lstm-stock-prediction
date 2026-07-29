"""
Print the saved evaluation report for a trained model.

Reads the metrics JSON written at the end of training and renders it as a
readable summary: point-forecast accuracy, cross-sectional skill, interval
quality, baseline comparison, regime stability and the cost-aware backtest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print saved model evaluation summary.")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "xgboost"])
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def section(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def print_point_metrics(metrics: dict, label: str) -> None:
    section(label)
    print(f"MAE                     : {metrics.get('mae', 0.0) * 100:.2f}%")
    print(f"Median absolute error   : {metrics.get('median_absolute_error', 0.0) * 100:.2f}%")
    print(f"RMSE                    : {metrics.get('rmse', 0.0) * 100:.2f}%")
    print(f"R^2                     : {metrics.get('r2', 0.0):+.4f}")
    print(f"Directional accuracy    : {metrics.get('direction_accuracy', 0.0):.4f}")
    print(f"Return correlation      : {metrics.get('return_correlation', 0.0):+.4f}")
    print(f"RMSE skill vs zero      : {metrics.get('rmse_skill_vs_zero', 0.0):+.4f}")
    print(f"Cross-sectional IC      : {metrics.get('cross_sectional_mean_ic', 0.0):+.4f}")
    print(f"Cross-sectional ICIR    : {metrics.get('cross_sectional_icir', 0.0):+.2f}")
    print(f"IC t-statistic          : {metrics.get('cross_sectional_ic_t_statistic', 0.0):+.2f}")
    print(f"IC p-value              : {metrics.get('cross_sectional_ic_p_value', 1.0):.4f}")
    print(f"Dates with positive IC  : {metrics.get('cross_sectional_ic_positive_rate', 0.0):.1%}")
    print(
        "Quintile spread         : "
        f"{metrics.get('cross_sectional_long_short_spread_per_period', 0.0) * 100:+.2f}% per period "
        f"({metrics.get('cross_sectional_long_short_spread_annualised', 0.0) * 100:+.2f}% annualised)"
    )


def main() -> None:
    args = parse_args()
    horizon = int(args.horizon)
    suffix = "" if args.model == "lstm" else "_xgboost"

    metrics_path = ROOT / "reports" / f"metrics{suffix}_h{horizon}.json"
    backtest_path = ROOT / "reports" / f"backtest_results{suffix}_h{horizon}.csv"
    predictions_path = ROOT / "reports" / f"test_predictions{suffix}_h{horizon}.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file was not found: {metrics_path}. "
            f"Run: python3 train{'_xgboost' if args.model == 'xgboost' else ''}.py --horizon {horizon}"
        )

    metrics = load_json(metrics_path)

    section("Model")
    print(f"Name                    : {metrics.get('model_name', args.model)}")
    print(f"Horizon                 : {metrics['prediction_horizon']} trading days")
    print(f"Modelled component      : {metrics.get('modelled_component', 'future_return')}")
    print(f"Features                : {metrics.get('feature_count', 0)}")
    print(
        f"Rows                    : train {metrics.get('train_size', 0):,} | "
        f"validation {metrics.get('validation_size', 0):,} | test {metrics.get('test_size', 0):,}"
    )
    drift = metrics.get("market_drift", {})
    if drift:
        print(f"Market drift (train)    : {drift.get('market_drift', 0.0) * 100:+.2f}% over the horizon")
    print(
        f"Split                   : {metrics.get('split_method', 'n/a')}, "
        f"purge {metrics.get('purge_trading_days', 0)} sessions"
    )

    component = metrics.get("component_metrics", {}).get("test")
    if component:
        print_point_metrics(component, "Test metrics - modelled component (market-excess return)")
    print_point_metrics(metrics.get("test_metrics", {}), "Test metrics - total return (what the user sees)")

    baselines = metrics.get("regression_baselines_excess") or metrics.get("regression_baselines")
    if baselines:
        section("Baseline comparison (same test rows)")
        rows = []
        if component:
            rows.append(
                {
                    "model": "this model",
                    "cross_sectional_ic": round(component.get("cross_sectional_mean_ic", 0.0), 5),
                    "direction_accuracy": round(component.get("direction_accuracy", 0.0), 4),
                    "mae": round(component.get("mae", 0.0), 5),
                    "rmse": round(component.get("rmse", 0.0), 5),
                }
            )
        for name, values in baselines.items():
            rows.append(
                {
                    "model": name,
                    "cross_sectional_ic": round(values.get("cross_sectional_mean_ic", 0.0), 5),
                    "direction_accuracy": round(values.get("direction_accuracy", 0.0), 4),
                    "mae": round(values.get("mae", 0.0), 5),
                    "rmse": round(values.get("rmse", 0.0), 5),
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))

    uncertainty = metrics.get("uncertainty")
    if uncertainty:
        section("Uncertainty and interval quality")
        print(f"Method                  : {uncertainty.get('method', 'n/a')}")
        test_intervals = uncertainty.get("test_interval_metrics", {})
        print(f"Nominal level           : {test_intervals.get('nominal_confidence_level', 0.0):.0%}")
        print(
            f"Empirical coverage      : {test_intervals.get('coverage_picp', 0.0):.1%} "
            f"({test_intervals.get('coverage_error', 0.0):+.1%} vs nominal)"
        )
        print(f"Mean interval width     : {test_intervals.get('mean_interval_width_mpiw', 0.0) * 100:.2f}%")
        print(
            f"Normalised width        : {test_intervals.get('normalized_interval_width', 0.0):.2f} x return std"
        )
        print(f"Winkler score           : {test_intervals.get('winkler_score', 0.0):.4f}")
        print(f"Mean total sigma        : {uncertainty.get('mean_total_sigma', 0.0) * 100:.2f}%")
        print(
            "Epistemic share of var. : "
            f"{uncertainty.get('epistemic_share_of_variance', 0.0):.1%} "
            "(the rest is irreducible return noise)"
        )

    blocks = metrics.get("test_regime_blocks")
    if blocks:
        section("Stability across consecutive test regimes")
        print(
            pd.DataFrame(
                [
                    {
                        "block": block["block"],
                        "start": block["start_date"],
                        "end": block["end_date"],
                        "rows": block["sample_size"],
                        "cross_sectional_ic": round(
                            block["metrics"].get("cross_sectional_mean_ic", 0.0), 5
                        ),
                        "direction_accuracy": round(block["metrics"].get("direction_accuracy", 0.0), 4),
                    }
                    for block in blocks
                ]
            ).to_string(index=False)
        )

    walk_forward = metrics.get("walk_forward")
    if walk_forward and walk_forward.get("summary"):
        section("Purged walk-forward summary (mean +/- std across folds)")
        for key, stats in walk_forward["summary"].items():
            print(
                f"{key:<42}: {stats['mean']:+.4f} +/- {stats['std']:.4f} "
                f"(min {stats['min']:+.4f}, max {stats['max']:+.4f}, {stats['folds']} folds)"
            )

    backtest = metrics.get("backtest_metrics", {})
    if backtest:
        section("Cost-aware backtest (non-overlapping horizon dates)")
        print(f"Decision rule           : {metrics.get('decision_rule', {})}")
        print(f"Final equity            : ${backtest.get('final_equity', 0.0):,.2f}")
        print(f"Total return            : {backtest.get('total_return', 0.0) * 100:+.2f}%")
        print(f"Equal-weight buy & hold : {backtest.get('buy_and_hold_total_return', 0.0) * 100:+.2f}%")
        print(f"Excess vs buy & hold    : {backtest.get('excess_return_vs_buy_hold', 0.0) * 100:+.2f}%")
        print(f"Sharpe ratio            : {backtest.get('sharpe_ratio', 0.0):.3f}")
        print(f"Information ratio       : {backtest.get('information_ratio_vs_universe', 0.0):.3f}")
        print(f"Max drawdown            : {backtest.get('max_drawdown', 0.0) * 100:.2f}%")
        print(f"Win rate                : {backtest.get('win_rate', 0.0) * 100:.2f}%")
        print(f"Active trades           : {backtest.get('number_of_active_trades', 0):,}")

        ablation = metrics.get("backtest_metrics_point_rule_ablation")
        if ablation:
            print("\nAblation, same forecasts with the plain point-threshold rule (no uncertainty input):")
            print(f"  Sharpe ratio          : {ablation.get('sharpe_ratio', 0.0):.3f}")
            print(f"  Total return          : {ablation.get('total_return', 0.0) * 100:+.2f}%")
            print(f"  Active trades         : {ablation.get('number_of_active_trades', 0):,}")

    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        if "predicted_signal" in predictions.columns:
            section("Signal distribution")
            print(predictions["predicted_signal"].value_counts().to_string())

    if backtest_path.exists():
        print(f"\nDetailed backtest file  : {backtest_path}")
    print("\nImportant: this is an academic simulation only, not financial advice.")


if __name__ == "__main__":
    main()
