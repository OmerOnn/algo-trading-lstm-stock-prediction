"""
Boosting-round selection and feature importance for the gradient-boosted model.

The bug this module replaces
----------------------------
The previous trainer early-stopped a reference booster on *pooled* Spearman
correlation, read back ``best_iteration``, and then did:

```python
tuned_rounds = max(50, best_iteration + 1)
```

Two things were wrong with that. First, the run reported ``best_iteration = 0``,
meaning the validation objective never improved after the very first tree — and
the code then silently overrode that with 50 trees. So the number of trees the
ensemble actually used had no relationship to the number the selection procedure
chose, and the metadata recorded a value that was not used. Second, pooled
Spearman is the wrong early-stopping objective: it mixes "did the market rise?"
with "which stock beat which?", so it can rise while cross-sectional skill is
flat, and its per-iteration evaluation is over a stacked panel rather than per
date.

What it does instead
--------------------
Boosters are fitted once to a generous number of rounds, then *evaluated at a
ladder of iteration counts* using the metric the project actually cares about:
per-date cross-sectional IC, together with MSE and MAE skill. Prediction at a
given round uses XGBoost's ``iteration_range``, so one fit yields every candidate
without refitting. The chosen count comes from walk-forward folds, and there is
no floor: if two trees is what the evidence supports, two trees is what gets
used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.regression import cross_sectional_metrics, regression_metrics


# Geometric ladder: boosting improvements are multiplicative in round count, so
# equal ratios rather than equal steps is the right resolution.
DEFAULT_ROUND_LADDER = (1, 5, 10, 20, 40, 80, 160, 320, 640)


@dataclass
class RoundSelection:
    """The chosen number of boosting rounds and the evidence behind it."""

    rounds: int = 100
    objective: str = "cross_sectional_ic"
    selection_rule: str = "mean(out-of-fold score) - std(out-of-fold score)"
    ladder: list[int] = field(default_factory=list)
    per_round: list[dict] = field(default_factory=list)
    folds: list[dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "rounds": int(self.rounds),
            "objective": self.objective,
            "selection_rule": self.selection_rule,
            "ladder": [int(value) for value in self.ladder],
            "per_round": self.per_round,
            "folds": self.folds,
            "reason": self.reason,
            "note": (
                "no minimum-round floor is applied; the ensemble uses exactly the "
                "number of rounds the out-of-fold evidence selected"
            ),
        }


def round_ladder(maximum_rounds: int, ladder=DEFAULT_ROUND_LADDER) -> list[int]:
    """The ladder truncated to what the fitted booster can actually serve."""
    values = sorted({int(value) for value in ladder if 1 <= int(value) <= int(maximum_rounds)})
    if int(maximum_rounds) not in values:
        values.append(int(maximum_rounds))
    return values


def evaluate_rounds(
    booster,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    dates,
    horizon: int,
    target_scale: np.ndarray | None = None,
    reference_prediction: float | None = None,
    ladder=DEFAULT_ROUND_LADDER,
) -> list[dict]:
    """
    Score one fitted booster at every rung of the ladder.

    ``iteration_range=(0, n)`` asks the fitted model to predict using only its
    first ``n`` trees, so the whole curve costs one fit rather than one fit per
    candidate. Scores are computed in return space when ``target_scale`` is
    supplied, because that is the space the reported metrics live in.
    """
    total = int(getattr(booster, "n_estimators", 0) or 0)
    if total <= 0:
        return []

    truth = np.asarray(y_validation, dtype=float)
    scales = None if target_scale is None else np.asarray(target_scale, dtype=float)

    rows: list[dict] = []
    for rounds in round_ladder(total, ladder):
        raw = booster.predict(x_validation, iteration_range=(0, int(rounds)))
        predicted = np.asarray(raw, dtype=float)
        if scales is not None:
            predicted = predicted * scales

        point = regression_metrics(truth, predicted, reference_prediction=reference_prediction)
        ranking = cross_sectional_metrics(dates, truth, predicted, horizon=horizon)
        rows.append(
            {
                "rounds": int(rounds),
                "cross_sectional_ic": float(ranking["mean_ic"]),
                "icir": float(ranking["icir"]),
                "ic_positive_rate": float(ranking["ic_positive_rate"]),
                "mse": float(point["mse"]),
                "mae": float(point["mae"]),
                "rmse": float(point["rmse"]),
                "mse_skill": float(point.get("mse_skill_vs_historical_mean", 0.0)),
                "mae_skill": float(point.get("mae_skill_vs_historical_mean", 0.0)),
                "direction_accuracy": float(point["direction_accuracy"]),
                "prediction_std": float(point["prediction_std"]),
            }
        )
    return rows


def round_score(row: dict, magnitude_weight: float = 0.25) -> float:
    """
    The scalar used to choose a round count.

    Deliberately the same shape as the LSTM's checkpoint criterion, so both model
    families are selected on the same definition of "better": cross-sectional
    ranking skill, plus a weighted magnitude-skill term that stops a
    ranking-only model from winning while its reported percentages are wrong.
    """
    ic = float(row.get("cross_sectional_ic", 0.0))
    magnitude = float(row.get("mse_skill", 0.0))
    if not np.isfinite(magnitude):
        magnitude = 0.0
    return float(ic + float(magnitude_weight) * float(np.clip(magnitude, -1.0, 1.0)))


def select_rounds(
    fold_curves: list[list[dict]],
    magnitude_weight: float = 0.25,
    stability_penalty: bool = True,
) -> RoundSelection:
    """
    Choose the round count that scores best consistently across folds.

    Only rungs evaluated in *every* fold are eligible, so folds are compared on
    identical candidates. The score is the mean across folds minus their standard
    deviation when ``stability_penalty`` is on: a round count that is excellent in
    one period and poor in another is worse than a slightly lower count that holds
    up everywhere, because the latter is the one that will generalise.
    """
    usable = [curve for curve in fold_curves if curve]
    if not usable:
        return RoundSelection(rounds=100, reason="no fold curves available; using a default of 100")

    shared = set(row["rounds"] for row in usable[0])
    for curve in usable[1:]:
        shared &= set(row["rounds"] for row in curve)
    if not shared:
        return RoundSelection(rounds=100, reason="folds share no common round counts")

    per_round: list[dict] = []
    for rounds in sorted(shared):
        scores, ics, skills = [], [], []
        for curve in usable:
            row = next(item for item in curve if item["rounds"] == rounds)
            scores.append(round_score(row, magnitude_weight))
            ics.append(float(row["cross_sectional_ic"]))
            skills.append(float(row.get("mse_skill", 0.0)))
        values = np.asarray(scores, dtype=float)
        dispersion = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        per_round.append(
            {
                "rounds": int(rounds),
                "mean_score": float(values.mean()),
                "score_std": dispersion,
                "selection_score": float(values.mean() - (dispersion if stability_penalty else 0.0)),
                "mean_cross_sectional_ic": float(np.mean(ics)),
                "minimum_cross_sectional_ic": float(np.min(ics)),
                "mean_mse_skill": float(np.mean(skills)),
                "per_fold_score": [float(value) for value in scores],
            }
        )

    best = max(per_round, key=lambda row: row["selection_score"])
    return RoundSelection(
        rounds=int(best["rounds"]),
        ladder=sorted(shared),
        per_round=per_round,
        folds=[{"rungs": len(curve)} for curve in usable],
        reason=(
            f"{best['rounds']} rounds scored {best['selection_score']:+.4f} "
            f"(mean {best['mean_score']:+.4f} - std {best['score_std']:.4f}) across "
            f"{len(usable)} purged folds; mean IC {best['mean_cross_sectional_ic']:+.4f}, "
            f"worst fold IC {best['minimum_cross_sectional_ic']:+.4f}"
        ),
    )


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def ensemble_feature_importance(
    models: list,
    feature_columns: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Average gain importance across the *actual* bootstrap ensemble.

    The previous report came from a single reference booster that was never used
    for inference. Since each bootstrap member sees a different resample and a
    different column subsample, their importances differ, and the ensemble's
    behaviour is the average — not any one member's. The dispersion across members
    is reported too, because a feature with high mean importance and huge spread is
    being relied on by a minority of members and is not a stable signal.
    """
    if not models:
        return pd.DataFrame(columns=["feature", "gain_importance"])

    matrix = []
    for model in models:
        values = getattr(model, "feature_importances_", None)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if len(values) == len(feature_columns):
            matrix.append(values)

    if not matrix:
        return pd.DataFrame(columns=["feature", "gain_importance"])

    stacked = np.vstack(matrix)
    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_importance": stacked.mean(axis=0),
            "gain_importance_std": stacked.std(axis=0, ddof=1) if len(matrix) > 1 else 0.0,
            "members_using_feature": (stacked > 0).sum(axis=0),
            "member_count": len(matrix),
            "importance_type": importance_type,
        }
    )
    frame["importance_stability"] = np.where(
        frame["gain_importance"] > 0,
        1.0 - frame["gain_importance_std"] / frame["gain_importance"].replace(0, np.nan),
        0.0,
    )
    return frame.sort_values("gain_importance", ascending=False).reset_index(drop=True)


def permutation_importance(
    predict_fn,
    x: pd.DataFrame,
    y_true: np.ndarray,
    dates,
    horizon: int,
    feature_columns: list[str] | None = None,
    repeats: int = 3,
    seed: int = 42,
    target_scale: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Out-of-fold permutation importance measured in cross-sectional IC.

    Gain importance answers "how much did the trees use this feature?", which is
    a statement about the fitting process and can be large for a feature that
    contributes nothing out of sample. Permutation importance answers the question
    that matters: "how much out-of-sample skill is lost if this feature is
    replaced by noise?" A feature with negative permutation importance is actively
    harmful — the model does better without it — and is a candidate for the
    blocklist.

    Values are shuffled *within each date*, so the permuted column keeps its
    cross-sectional distribution for that day and only the assignment to stocks is
    destroyed. Shuffling globally would also destroy the time-series distribution
    and confound "this feature is informative" with "this feature has a trend".
    """
    columns = feature_columns or list(x.columns)
    rng = np.random.default_rng(int(seed))
    truth = np.asarray(y_true, dtype=float)
    scales = None if target_scale is None else np.asarray(target_scale, dtype=float)
    date_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates))))

    def scored(frame: pd.DataFrame) -> float:
        predicted = np.asarray(predict_fn(frame), dtype=float)
        if scales is not None:
            predicted = predicted * scales
        return float(cross_sectional_metrics(date_index, truth, predicted, horizon=horizon)["mean_ic"])

    baseline = scored(x)
    date_codes = pd.factorize(date_index, sort=True)[0]
    groups = pd.Series(np.arange(len(date_codes))).groupby(date_codes).apply(np.asarray)

    rows: list[dict] = []
    for column in columns:
        if column not in x.columns:
            continue
        drops: list[float] = []
        original = x[column].to_numpy(copy=True)
        for _ in range(max(1, int(repeats))):
            shuffled = original.copy()
            for positions in groups:
                if len(positions) > 1:
                    shuffled[positions] = original[rng.permutation(positions)]
            permuted = x.copy()
            permuted[column] = shuffled
            drops.append(baseline - scored(permuted))
        values = np.asarray(drops, dtype=float)
        rows.append(
            {
                "feature": column,
                "ic_drop_mean": float(values.mean()),
                "ic_drop_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "repeats": int(len(values)),
                "baseline_ic": baseline,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Harmful: removing the feature would have *improved* out-of-sample IC by
    # more than the noise in the estimate.
    frame["harmful"] = frame["ic_drop_mean"] < -frame["ic_drop_std"]
    frame["negligible"] = frame["ic_drop_mean"].abs() <= frame["ic_drop_std"]
    return frame.sort_values("ic_drop_mean", ascending=False).reset_index(drop=True)


def recommend_feature_blocklist(
    permutation_frames: list[pd.DataFrame],
    minimum_folds_harmful: int = 2,
) -> dict:
    """
    Features that were harmful in several folds, as a blocklist recommendation.

    Requiring a feature to be harmful in more than one fold is the whole point.
    On ~100 features and one evaluation window, several will look harmful by
    chance; demanding repetition across purged folds is what separates a genuinely
    damaging feature from sampling noise. The output is a *recommendation* written
    to the report — it is not applied automatically, because dropping features on
    the strength of a noisy estimate is its own form of overfitting.
    """
    if not permutation_frames:
        return {"recommended_blocklist": [], "reason": "no permutation runs available"}

    counts: dict[str, int] = {}
    drops: dict[str, list[float]] = {}
    for frame in permutation_frames:
        if frame.empty or "harmful" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            feature = str(row["feature"])
            drops.setdefault(feature, []).append(float(row["ic_drop_mean"]))
            if bool(row["harmful"]):
                counts[feature] = counts.get(feature, 0) + 1

    recommended = sorted(
        feature for feature, count in counts.items() if count >= int(minimum_folds_harmful)
    )
    return {
        "recommended_blocklist": recommended,
        "folds_evaluated": len(permutation_frames),
        "minimum_folds_harmful": int(minimum_folds_harmful),
        "harmful_fold_counts": dict(sorted(counts.items(), key=lambda item: -item[1])),
        "mean_ic_drop": {
            feature: float(np.mean(values)) for feature, values in sorted(drops.items())
        },
        "reason": (
            "a feature must be harmful in at least "
            f"{minimum_folds_harmful} purged folds to be recommended; on a wide "
            "feature set a single fold will always flag some features by chance"
        ),
        "applied": False,
        "how_to_apply": "add the recommended names to feature_blocklist in configs/config.yaml",
    }
