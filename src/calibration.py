"""
Turning a ranking into a percentage return.

The models here reliably produce *ordering* skill and unreliably produce
*magnitude* skill. That gap is the whole problem this module addresses, because
the product promises an expected percentage return, not a rank.

Rules the calibration must obey
-------------------------------
1. **It must not destroy the ranking.** A least-squares fit on a low-signal panel
    will happily return a near-zero or negative slope, because flattening every
    prediction towards a constant genuinely improves squared error. That
    "improvement" deletes the only thing the model had. Any candidate whose
    Spearman correlation with the outcome is materially worse than the raw
    prediction's is rejected outright.
2. **The residual leg may not carry a common intercept.** The residual is a
    *relative* view: the claim is "this stock beats its peers", not "everything
    goes up 2%". A large intercept there is a market call smuggled into the alpha
    leg, and it would double-count the market component that the hierarchical
    composition already supplies.
3. **The residual leg is cross-sectionally centred.** Subtracting each date's
    mean prediction makes the alpha sum to roughly zero across the universe,
    which is what a relative view means, and it removes any date-level drift the
    model picked up.
4. **Shrinkage towards zero, not towards the mean.** For an alpha forecast, zero
   is the honest default — "no view" — and the shrinkage factor is fitted, so a
   model with weak magnitude skill has its magnitudes pulled in rather than
   presented at face value.
5. **The method is chosen out of fold.** Affine, ridge and isotonic are compared
   on purged walk-forward predictions, never on the test set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.regression import regression_metrics


CALIBRATION_METHODS = ("identity", "affine", "ridge", "isotonic")

# A candidate is rejected if it costs more than this much rank correlation.
RANK_TOLERANCE = 1e-4


@dataclass
class ReturnCalibration:
    """A fitted magnitude correction that can be replayed at inference time."""

    method: str = "identity"
    slope: float = 1.0
    intercept: float = 0.0
    shrinkage: float = 1.0
    allow_intercept: bool = True
    cross_sectional_centering: bool = False
    # Isotonic support points, stored so inference needs no sklearn model object.
    knots_x: list[float] = field(default_factory=list)
    knots_y: list[float] = field(default_factory=list)
    validation_rank_ic: float = 0.0
    validation_mse: float = 0.0
    validation_mse_skill: float = 0.0
    # Rank correlation between the calibrated output and the raw model output.
    # Must be ~1: a calibration is a monotone rescaling, never a re-ordering.
    rank_preservation: float = 1.0
    # Mean raw prediction at fit time. Stands in for cross-sectional centring at
    # inference, where there is no cross-section to centre against.
    centering_offset: float = 0.0
    reason: str = "identity calibration retained"

    def apply(
        self,
        predictions: np.ndarray,
        dates: Sequence | None = None,
    ) -> np.ndarray:
        """
        Map raw model output into calibrated return units.

        ``dates`` is required for cross-sectional centring. At inference there is
        usually no cross-section to centre against — the user asks about one or
        three tickers, not the whole universe — and centring a single value
        against itself would return exactly zero, silently destroying every
        single-stock forecast. When dates are absent the stored
        ``centering_offset`` (the mean raw prediction observed at fit time) is
        subtracted instead, which is the constant that centring applies on
        average and leaves a one-stock forecast intact.
        """
        values = np.asarray(predictions, dtype=float)

        if self.cross_sectional_centering:
            if dates is not None and len(np.unique(np.asarray(dates))) < len(values):
                values = cross_sectional_center(values, dates)
            else:
                values = values - float(self.centering_offset)

        if self.method == "isotonic" and self.knots_x:
            values = np.interp(
                values,
                np.asarray(self.knots_x, dtype=float),
                np.asarray(self.knots_y, dtype=float),
            )
        elif self.method in {"affine", "ridge"}:
            intercept = float(self.intercept) if self.allow_intercept else 0.0
            values = intercept + float(self.slope) * values

        return float(self.shrinkage) * values

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["enabled"] = self.method != "identity" or self.shrinkage != 1.0
        return payload

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ReturnCalibration":
        if not payload:
            return cls()
        known = {key: payload[key] for key in asdict(cls()).keys() if key in payload}
        return cls(**known)


def cross_sectional_center(predictions: np.ndarray, dates: Sequence) -> np.ndarray:
    """
    Remove each date's mean prediction, leaving a purely relative view.

    This is what makes the residual leg an *alpha* forecast. Any level the model
    emits on a given date is a market call, and the market call belongs to the
    market model; leaving it in the residual would count it twice.
    """
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(list(dates))).to_numpy(),
            "value": np.asarray(predictions, dtype=float),
        }
    )
    centred = frame["value"] - frame.groupby("date")["value"].transform("mean")
    return np.asarray(centred, dtype=float)


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    from scipy import stats

    value = float(stats.spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def _fit_affine(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    allow_intercept: bool,
    ridge_alpha: float = 0.0,
) -> tuple[float, float]:
    """Least-squares (optionally ridge-penalised) slope and intercept."""
    if allow_intercept:
        design = np.column_stack([predicted_values, np.ones(len(predicted_values))])
    else:
        design = predicted_values.reshape(-1, 1)

    gram = design.T @ design
    if ridge_alpha > 0:
        penalty = np.eye(gram.shape[0]) * float(ridge_alpha)
        if allow_intercept:
            # Never penalise the intercept: shrinking it towards zero is a
            # statement about the mean return, not about model complexity.
            penalty[-1, -1] = 0.0
        gram = gram + penalty

    try:
        solution = np.linalg.solve(gram, design.T @ true_values)
    except np.linalg.LinAlgError:
        return 1.0, 0.0

    if allow_intercept:
        return float(solution[0]), float(solution[1])
    return float(solution[0]), 0.0


def fit_calibration_candidates(
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    dates: Sequence | None = None,
    allow_intercept: bool = True,
    cross_sectional_centering: bool = False,
    minimum_samples: int = 500,
    ridge_alpha: float | None = None,
    shrinkage_candidates: Sequence[float] = (1.0, 0.75, 0.50),
) -> dict[str, ReturnCalibration]:
    """
    Fit every calibration family on one set of out-of-fold predictions.

    Returns a dictionary keyed by method name. Candidates that would flatten or
    invert the ranking are still returned, but carry the rejection reason and an
    identity transform, so the selection log shows what was tried and why it lost.
    """
    true_values = np.asarray(true_return, dtype=float)
    raw = np.asarray(predicted_return, dtype=float)

    identity = ReturnCalibration(
        cross_sectional_centering=bool(cross_sectional_centering),
        allow_intercept=bool(allow_intercept),
    )
    offset = float(np.mean(raw[np.isfinite(raw)])) if np.any(np.isfinite(raw)) else 0.0
    identity.centering_offset = offset
    finite = np.isfinite(true_values) & np.isfinite(raw)
    true_values, raw = true_values[finite], raw[finite]
    if dates is not None:
        dates = np.asarray(pd.to_datetime(pd.Series(list(dates))))[finite]

    baseline_rank = _rank_correlation(true_values, raw)
    reference = float(np.mean(true_values)) if len(true_values) else 0.0

    def score(calibration: ReturnCalibration) -> ReturnCalibration:
        calibrated = calibration.apply(raw, dates)
        metrics = regression_metrics(true_values, calibrated, reference_prediction=reference)
        calibration.validation_rank_ic = _rank_correlation(true_values, calibrated)
        calibration.validation_mse = float(metrics["mse"])
        calibration.validation_mse_skill = float(metrics["mse_skill_vs_historical_mean"])
        return calibration

    candidates: dict[str, ReturnCalibration] = {"identity": score(identity)}

    if len(true_values) < int(minimum_samples) or np.std(raw) < 1e-12:
        candidates["identity"].reason = "not enough out-of-fold signal to calibrate"
        return candidates

    centred = cross_sectional_center(raw, dates) if (cross_sectional_centering and dates is not None) else raw
    intercept_limit = max(0.005, float(np.quantile(np.abs(true_values), 0.50)))
    default_ridge = float(len(centred)) * 0.01 if ridge_alpha is None else float(ridge_alpha)

    for method, alpha in (("affine", 0.0), ("ridge", default_ridge)):
        slope, intercept = _fit_affine(true_values, centred, allow_intercept, alpha)
        for shrinkage in shrinkage_candidates:
            candidate = ReturnCalibration(
                method=method,
                slope=float(np.clip(slope, 0.0, 5.0)),
                intercept=float(np.clip(intercept, -intercept_limit, intercept_limit)),
                shrinkage=float(shrinkage),
                allow_intercept=bool(allow_intercept),
                cross_sectional_centering=bool(cross_sectional_centering),
                centering_offset=offset,
            )
            if not np.isfinite(slope) or slope <= 0.0:
                candidate = ReturnCalibration(
                    allow_intercept=bool(allow_intercept),
                    cross_sectional_centering=bool(cross_sectional_centering),
                    centering_offset=offset,
                    reason=f"rejected non-positive {method} slope {float(slope):.4f}",
                )
                candidates[f"{method}_s{shrinkage:g}"] = score(candidate)
                continue
            candidate.reason = f"{method} magnitude fit, shrinkage {shrinkage:g}"
            candidates[f"{method}_s{shrinkage:g}"] = score(candidate)

    # Isotonic: a monotone step function, so it can correct a non-linear
    # magnitude bias while preserving the ordering by construction.
    try:
        from sklearn.isotonic import IsotonicRegression
    except ModuleNotFoundError:
        return candidates

    order = np.argsort(centred)
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    isotonic.fit(centred[order], true_values[order])
    # Subsample the step function to a manageable number of knots so the fitted
    # calibration can be stored in JSON metadata and replayed without sklearn.
    knot_x = np.quantile(centred, np.linspace(0.0, 1.0, 64))
    knot_x = np.unique(knot_x)
    if len(knot_x) >= 4:
        for shrinkage in shrinkage_candidates:
            candidate = ReturnCalibration(
                method="isotonic",
                shrinkage=float(shrinkage),
                allow_intercept=bool(allow_intercept),
                cross_sectional_centering=bool(cross_sectional_centering),
                centering_offset=offset,
                knots_x=[float(value) for value in knot_x],
                knots_y=[float(value) for value in isotonic.predict(knot_x)],
                reason=f"monotone (isotonic) magnitude fit, shrinkage {shrinkage:g}",
            )
            candidates[f"isotonic_s{shrinkage:g}"] = score(candidate)

    # Rank-preservation guard.
    #
    # The test is monotonicity *with respect to the model's own output*, not
    # agreement with the outcome. Comparing to the outcome is the wrong test and
    # silently accepts the worst case: when the raw prediction happens to be
    # negatively correlated with the truth in the fitting window, flattening
    # everything to a constant *improves* agreement, so a collapse would be waved
    # through as an upgrade. Requiring rank correlation ~1 against the raw
    # prediction makes a calibration what it is supposed to be — a monotone
    # rescaling of the model's ordering — and rejects anything that erases or
    # inverts that ordering regardless of how it scores against the outcome.
    for calibration in candidates.values():
        if calibration.method == "identity":
            continue
        calibrated = calibration.apply(raw, dates)
        if np.std(calibrated) <= 1e-12:
            calibration.reason = (
                f"{calibration.method} rejected: collapsed the prediction to a constant"
            )
        else:
            preserved = _rank_correlation(calibrated, centred)
            calibration.rank_preservation = float(preserved)
            if preserved >= 1.0 - 1e-6:
                continue
            calibration.reason = (
                f"{calibration.method} rejected: not monotone in the model output "
                f"(rank correlation with the raw forecast {preserved:+.4f})"
            )
        calibration.method = "identity"
        calibration.slope, calibration.intercept, calibration.shrinkage = 1.0, 0.0, 1.0
        calibration.knots_x, calibration.knots_y = [], []

    return candidates


def select_calibration(
    fold_candidates: list[dict[str, ReturnCalibration]],
    objective: str = "mse_skill",
) -> tuple[str, dict]:
    """
    Pick one calibration name using the mean score across purged folds.

    Selection is by mean out-of-fold score *minus* one standard deviation across
    folds, the same consistency requirement applied to the market model: a
    calibration that helps in one period and hurts in another is not a
    calibration, it is an artefact of that period.
    """
    if not fold_candidates:
        return "identity", {"reason": "no folds available", "candidates": {}}

    shared = set(fold_candidates[0])
    for candidates in fold_candidates[1:]:
        shared &= set(candidates)
    if not shared:
        return "identity", {"reason": "folds produced no common candidates", "candidates": {}}

    summary: dict[str, dict] = {}
    for name in sorted(shared):
        if objective == "mse_skill":
            scores = [float(fold[name].validation_mse_skill) for fold in fold_candidates]
        else:
            scores = [float(fold[name].validation_rank_ic) for fold in fold_candidates]
        values = np.asarray(scores, dtype=float)
        dispersion = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[name] = {
            "mean": float(values.mean()),
            "std": dispersion,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "selection_score": float(values.mean() - dispersion),
            "per_fold": scores,
        }

    best = max(summary, key=lambda name: summary[name]["selection_score"])
    # Only displace the identity transform on a strictly better consistent score.
    if summary.get("identity", {}).get("selection_score", -np.inf) >= summary[best]["selection_score"]:
        best = "identity"

    return best, {
        "objective": objective,
        "rule": "mean(out-of-fold score) - std(out-of-fold score)",
        "selected": best,
        "folds": len(fold_candidates),
        "candidates": summary,
    }


def decile_calibration_report(
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    deciles: int = 10,
) -> list[dict]:
    """
    Predicted versus realised return, bucketed by predicted decile.

    This is the single most revealing diagnostic for a return model. A model with
    real magnitude skill produces a monotone increasing realised column whose
    values sit near the predicted column. A model with ranking skill but no
    magnitude skill produces a monotone realised column with a much *flatter*
    slope than predicted — visible immediately here and invisible in a single
    aggregate MAE. The standard error makes it clear which of those differences
    survive sampling noise.
    """
    true_values = np.asarray(true_return, dtype=float)
    predicted_values = np.asarray(predicted_return, dtype=float)
    finite = np.isfinite(true_values) & np.isfinite(predicted_values)
    true_values, predicted_values = true_values[finite], predicted_values[finite]
    if len(true_values) < int(deciles):
        return []

    ranks = pd.qcut(
        pd.Series(predicted_values).rank(method="first"),
        int(deciles),
        labels=False,
        duplicates="drop",
    )

    rows: list[dict] = []
    for bucket in sorted(pd.unique(ranks.dropna())):
        mask = (ranks == bucket).to_numpy()
        realised = true_values[mask]
        forecast = predicted_values[mask]
        count = int(len(realised))
        if count == 0:
            continue
        standard_error = float(np.std(realised, ddof=1) / np.sqrt(count)) if count > 1 else 0.0
        rows.append(
            {
                "decile": int(bucket) + 1,
                "count": count,
                "mean_predicted_return": float(np.mean(forecast)),
                "mean_realized_return": float(np.mean(realised)),
                "median_realized_return": float(np.median(realised)),
                "realized_standard_error": standard_error,
                "directional_hit_rate": float(
                    np.mean(np.sign(forecast) == np.sign(realised))
                ),
                "realized_minus_predicted": float(np.mean(realised) - np.mean(forecast)),
            }
        )
    return rows


def calibration_monotonicity(rows: list[dict]) -> dict:
    """
    Summarise a decile table: is realised return actually increasing in forecast?

    ``realized_slope_vs_predicted`` is the regression of realised decile means on
    predicted decile means. A value near 1 means the magnitudes are right; a
    small positive value means the model ranks well but overstates how much the
    top names will actually beat the bottom ones, which is the normal outcome for
    an equity return model and is exactly what the shrinkage above corrects.
    """
    if len(rows) < 3:
        return {"deciles": len(rows)}

    predicted = np.asarray([row["mean_predicted_return"] for row in rows], dtype=float)
    realised = np.asarray([row["mean_realized_return"] for row in rows], dtype=float)
    slope = 0.0
    if np.std(predicted) > 0:
        slope = float(np.polyfit(predicted, realised, 1)[0])

    return {
        "deciles": len(rows),
        "realized_slope_vs_predicted": slope,
        "rank_correlation": _rank_correlation(predicted, realised),
        "top_minus_bottom_realized": float(realised[-1] - realised[0]),
        "top_minus_bottom_predicted": float(predicted[-1] - predicted[0]),
        "monotone_increasing": bool(np.all(np.diff(realised) >= 0)),
        "increasing_step_share": float(np.mean(np.diff(realised) > 0)),
    }
