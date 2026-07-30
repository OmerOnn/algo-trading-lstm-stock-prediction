"""
Acceptance gates for a candidate model.

These are the pre-registered criteria a model has to clear before it can be
called an improvement. They exist so the verdict is decided by a checklist agreed
in advance rather than by whichever metric happened to look good afterwards.

The gates are *reported*, not enforced. Nothing here silently tunes a model until
it passes; a failed gate is printed as a failure and carried into the final
report. Optimising against these thresholds would defeat the point of having
them, so a gate that fails is stated plainly and the reason recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class Gate:
    """One acceptance criterion and its outcome."""

    name: str
    description: str
    passed: bool
    observed: Any
    threshold: Any
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "passed": bool(self.passed),
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


DEFAULT_THRESHOLDS = {
    "walk_forward_mean_ic": 0.03,
    "ic_t_statistic": 2.0,
    "minimum_coverage_error": 0.05,
    "maximum_normalized_interval_width": 3.0,
    "regime_collapse_ic": 0.0,
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def evaluate_acceptance_gates(
    walk_forward: dict | None,
    test_metrics: dict | None,
    baselines: dict | None,
    interval_metrics: dict | None,
    portfolio: dict | None,
    regime_blocks: list[dict] | None = None,
    calibration_stability: dict | None = None,
    thresholds: dict | None = None,
) -> dict:
    """
    Score every gate and return the full checklist plus a summary verdict.

    Each gate is evaluated defensively: a missing input makes the gate fail with
    an explanation rather than raise, because "we did not measure this" and "this
    passed" must never look the same in the report.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    gates: list[Gate] = []

    summary = (walk_forward or {}).get("summary", {}) or {}
    folds = (walk_forward or {}).get("folds", []) or []
    ic_summary = summary.get("cross_sectional_ic", {}) or {}

    # 1. Walk-forward mean cross-sectional IC above the threshold.
    mean_ic = _finite(ic_summary.get("mean"))
    gates.append(
        Gate(
            "walk_forward_mean_ic",
            f"Mean out-of-fold cross-sectional IC above {limits['walk_forward_mean_ic']:.3f}",
            bool(ic_summary) and mean_ic > limits["walk_forward_mean_ic"],
            mean_ic,
            limits["walk_forward_mean_ic"],
            "no walk-forward summary available" if not ic_summary else f"{len(folds)} folds",
        )
    )

    # 2. Positive IC in every fold.
    fold_ics = [
        _finite((fold.get("test_metrics") or {}).get("cross_sectional_ic"))
        for fold in folds
    ]
    gates.append(
        Gate(
            "positive_ic_every_fold",
            "Cross-sectional IC positive in every purged fold",
            bool(fold_ics) and all(value > 0 for value in fold_ics),
            [round(value, 4) for value in fold_ics],
            "> 0 in all folds",
            f"{sum(1 for value in fold_ics if value > 0)}/{len(fold_ics)} folds positive"
            if fold_ics
            else "no folds evaluated",
        )
    )

    # 3. Aggregate IC t-statistic above 2.
    t_statistic = _finite((test_metrics or {}).get("cross_sectional_ic_t_statistic"))
    gates.append(
        Gate(
            "ic_t_statistic",
            f"Aggregate IC t-statistic above {limits['ic_t_statistic']:.1f}",
            t_statistic > limits["ic_t_statistic"],
            t_statistic,
            limits["ic_t_statistic"],
            "overlap-discounted effective sample size",
        )
    )

    # 4. MSE, RMSE and MAE better than the historical-mean forecast.
    historical = (baselines or {}).get("historical_mean", {}) or {}
    model_errors = {key: _finite((test_metrics or {}).get(key), np.inf) for key in ("mse", "rmse", "mae")}
    baseline_errors = {key: _finite(historical.get(key), 0.0) for key in ("mse", "rmse", "mae")}
    beaten = {
        key: bool(baseline_errors[key] > 0 and model_errors[key] < baseline_errors[key])
        for key in model_errors
    }
    gates.append(
        Gate(
            "magnitude_beats_historical_mean",
            "MSE, RMSE and MAE all better than predicting the training-window mean",
            bool(historical) and all(beaten.values()),
            {key: round(model_errors[key], 6) for key in model_errors},
            {key: round(baseline_errors[key], 6) for key in baseline_errors},
            ", ".join(f"{key}: {'better' if value else 'not better'}" for key, value in beaten.items())
            if historical
            else "no historical-mean baseline available",
        )
    )

    # 5. Positive out-of-fold information ratio after costs.
    distribution = (portfolio or {}).get("distribution", {}) or {}
    information_ratio = (distribution.get("information_ratio_vs_universe") or {})
    mean_ir = _finite(information_ratio.get("mean"))
    gates.append(
        Gate(
            "positive_information_ratio_after_costs",
            "Positive information ratio versus the equal-weight universe, net of costs",
            bool(information_ratio) and mean_ir > 0,
            mean_ir,
            "> 0",
            f"averaged over {(portfolio or {}).get('offsets_evaluated', 0)} rebalance offsets"
            if information_ratio
            else "no portfolio backtest available",
        )
    )

    # 6. Positive top-minus-bottom spread after costs in every fold.
    fold_spreads = [
        _finite(
            (fold.get("test_metrics") or {}).get("cross_sectional_long_short_spread_annualised")
        )
        for fold in folds
    ]
    gates.append(
        Gate(
            "positive_spread_every_fold",
            "Positive top-minus-bottom quintile spread in every fold",
            bool(fold_spreads) and all(value > 0 for value in fold_spreads),
            [round(value, 4) for value in fold_spreads],
            "> 0 in all folds",
            f"{sum(1 for value in fold_spreads if value > 0)}/{len(fold_spreads)} folds positive"
            if fold_spreads
            else "no folds evaluated",
        )
    )

    # 7. Stable calibration and decision parameters across folds.
    stability = calibration_stability or {}
    gates.append(
        Gate(
            "stable_calibration_across_folds",
            "The same calibration family and decision rule is selected across folds",
            bool(stability.get("stable", False)),
            stability.get("selected"),
            "one consistent selection",
            str(stability.get("detail", "calibration stability was not measured")),
        )
    )

    # 8. No material regime collapse.
    block_ics = [
        _finite((block.get("metrics") or {}).get("cross_sectional_ic"))
        for block in (regime_blocks or [])
    ]
    worst_block = min(block_ics) if block_ics else 0.0
    gates.append(
        Gate(
            "no_material_regime_collapse",
            "Cross-sectional IC does not go negative in any out-of-sample regime block",
            bool(block_ics) and worst_block >= limits["regime_collapse_ic"],
            round(worst_block, 4),
            limits["regime_collapse_ic"],
            f"{len(block_ics)} consecutive regime blocks"
            if block_ics
            else "no regime blocks evaluated",
        )
    )

    # 9. Interval coverage near nominal without excessive width.
    coverage_error = abs(_finite((interval_metrics or {}).get("coverage_error"), 1.0))
    normalized_width = _finite((interval_metrics or {}).get("normalized_interval_width"), np.inf)
    gates.append(
        Gate(
            "interval_coverage_and_width",
            (
                f"Coverage within {limits['minimum_coverage_error']:.2f} of nominal and "
                f"normalised width below {limits['maximum_normalized_interval_width']:.1f}"
            ),
            bool(interval_metrics)
            and coverage_error <= limits["minimum_coverage_error"]
            and normalized_width <= limits["maximum_normalized_interval_width"],
            {"coverage_error": round(coverage_error, 4), "normalized_width": round(normalized_width, 3)},
            {
                "coverage_error": limits["minimum_coverage_error"],
                "normalized_width": limits["maximum_normalized_interval_width"],
            },
            "width is measured in units of the realised return's standard deviation"
            if interval_metrics
            else "no interval metrics available",
        )
    )

    passed = sum(1 for gate in gates if gate.passed)
    return {
        "gates": [gate.to_dict() for gate in gates],
        "passed": passed,
        "total": len(gates),
        "all_passed": passed == len(gates),
        "failed_gates": [gate.name for gate in gates if not gate.passed],
        "note": (
            "Gates are reported, never optimised against. A failing gate is a "
            "finding about the model, not a target to tune towards."
        ),
    }


def format_acceptance_table(report: dict) -> str:
    """Render the checklist as plain text for the console and the report."""
    lines = [
        f"Acceptance gates: {report['passed']}/{report['total']} passed",
        "",
        f"{'':2} {'gate':<44} {'observed':<28} {'threshold':<20}",
        "-" * 96,
    ]
    for gate in report["gates"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        observed = str(gate["observed"])
        threshold = str(gate["threshold"])
        lines.append(f"{mark:<4} {gate['name']:<44} {observed[:27]:<28} {threshold[:19]:<20}")
        if gate["detail"]:
            lines.append(f"     {gate['detail']}")
    return "\n".join(lines)


def calibration_stability(selections: list[dict]) -> dict:
    """
    Did the folds agree on the calibration family and the decision rule?

    Agreement matters more than which option won. A pipeline that picks affine
    calibration in one fold, isotonic in the next and identity in the third has
    not discovered a calibration; it has discovered that the choice is noise, and
    the stable default should be preferred.
    """
    if not selections:
        return {"stable": False, "detail": "no fold selections recorded"}

    calibrations = [str(item.get("calibration", "unknown")) for item in selections]
    rules = [str(item.get("decision_rule", "unknown")) for item in selections]
    unique_calibrations = sorted(set(calibrations))
    unique_rules = sorted(set(rules))
    stable = len(unique_calibrations) == 1 and len(unique_rules) == 1

    return {
        "stable": bool(stable),
        "selected": {"calibration": unique_calibrations, "decision_rule": unique_rules},
        "folds": len(selections),
        "detail": (
            f"calibration {unique_calibrations} and rule {unique_rules} across {len(selections)} folds"
        ),
    }
