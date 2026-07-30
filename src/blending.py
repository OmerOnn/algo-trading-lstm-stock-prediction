"""
Combining the LSTM and the XGBoost forecast.

Two models trained on the same target with completely different inductive biases
make different mistakes. Averaging them cancels part of the independent error,
which on a signal-to-noise ratio this low is usually a larger gain than any
improvement available from tuning either model further.

Constraints on the weights
--------------------------
```text
blend = w_lstm * lstm + w_xgboost * xgboost,   w >= 0,   sum(w) = 1
```

* **Non-negative.** A negative weight means "predict the opposite of what this
  model says", which on out-of-fold data is almost always noise-fitting: it
  reverses a model that happened to be wrong in the fitting window and will not
  stay wrong.
* **Summing to one.** This keeps the blend in return units. Weights that sum to
  more than one silently inflate every magnitude, which would corrupt the
  percentage return the product reports and the interval built around it.
* **Fitted out of fold.** Weights come from purged walk-forward predictions
  only. Fitting them on the test set would be selecting a model on the test set.
* **Not assumed equal.** 50/50 is evaluated as one candidate among many, and it
  wins only if it earns it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.regression import cross_sectional_metrics, regression_metrics


@dataclass
class BlendWeights:
    """Fitted, constrained combination weights for the two model families."""

    weights: dict[str, float] = field(default_factory=dict)
    objective: str = "0.5 * mse_skill + 0.5 * normalised cross-sectional IC"
    selection: str = "purged walk-forward out-of-fold predictions"
    out_of_fold_score: float = 0.0
    out_of_fold_score_std: float = 0.0
    per_model_scores: dict[str, float] = field(default_factory=dict)
    improves_on_best_single_model: bool = False
    retained: bool = False
    reason: str = "not fitted"

    def apply(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Weighted combination. Missing models are dropped and weights renormalised."""
        available = {
            name: np.asarray(values, dtype=float)
            for name, values in predictions.items()
            if name in self.weights and values is not None
        }
        if not available:
            raise ValueError("No model predictions available to blend.")

        total = sum(self.weights[name] for name in available)
        if total <= 0:
            # Fall back to an equal-weight average rather than dividing by zero.
            return np.mean(np.stack(list(available.values())), axis=0)

        stacked = np.zeros_like(next(iter(available.values())), dtype=float)
        for name, values in available.items():
            stacked = stacked + (self.weights[name] / total) * values
        return stacked

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "BlendWeights":
        if not payload:
            return cls()
        known = {key: payload[key] for key in asdict(cls()).keys() if key in payload}
        return cls(**known)


def weight_grid(model_names: Sequence[str], step: float = 0.05) -> list[dict[str, float]]:
    """
    Every non-negative weight vector on the simplex, at the given resolution.

    An exhaustive grid rather than an optimiser: with two models and a 0.05 step
    this is 21 candidates, so the global optimum under the constraints is found
    exactly and there is no question of an optimiser stopping in a local minimum
    or violating the simplex constraint.
    """
    names = list(model_names)
    if len(names) == 1:
        return [{names[0]: 1.0}]
    if len(names) != 2:
        raise ValueError("weight_grid currently supports one or two models.")

    steps = int(round(1.0 / float(step)))
    grid = []
    for index in range(steps + 1):
        first = index / steps
        grid.append({names[0]: float(first), names[1]: float(1.0 - first)})
    return grid


def blend_score(
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    dates: Sequence,
    horizon: int,
    reference_prediction: float,
    ic_weight: float = 0.5,
    ic_scale: float = 0.05,
) -> float:
    """
    The documented blend objective: half magnitude skill, half ranking skill.

    ```text
    score = (1 - ic_weight) * mse_skill_vs_historical_mean
          + ic_weight * (cross_sectional_ic / ic_scale)
    ```

    Both legs are needed for the same reason the checkpoint criterion needs both:
    optimising MSE skill alone would favour whichever model shrinks hardest
    towards the mean, and optimising IC alone would ignore whether the reported
    percentage is anywhere near right. ``ic_scale`` puts an IC of 0.05 — a
    realistic good result on monthly equity cross-sections — on the same numeric
    footing as an MSE skill of 1.0, so neither term silently dominates.
    """
    magnitude = regression_metrics(
        true_return, predicted_return, reference_prediction=reference_prediction
    )["mse_skill_vs_historical_mean"]
    ranking = cross_sectional_metrics(dates, true_return, predicted_return, horizon=horizon)[
        "mean_ic"
    ]
    return float(
        (1.0 - float(ic_weight)) * float(magnitude)
        + float(ic_weight) * (float(ranking) / float(ic_scale))
    )


def fit_blend_weights(
    folds: list[dict],
    horizon: int,
    step: float = 0.05,
    ic_weight: float = 0.5,
    minimum_improvement: float = 0.01,
) -> BlendWeights:
    """
    Fit blend weights on aligned out-of-fold predictions.

    ``folds`` is a list of dictionaries, one per walk-forward fold, each holding
    ``true_return``, ``dates``, ``reference_prediction`` and a ``predictions``
    mapping of model name to array.

    The blend is retained only if it beats the better standalone model by
    ``minimum_improvement`` on the mean-minus-one-standard-deviation score. A
    blend that wins by a hair on the mean while being less stable across folds is
    not an improvement; it is a more complicated way to get the same answer.
    """
    usable = [
        fold
        for fold in folds
        if fold.get("predictions") and len(fold.get("true_return", [])) > 0
    ]
    if not usable:
        return BlendWeights(reason="no aligned out-of-fold predictions available")

    model_names = sorted(set.intersection(*[set(fold["predictions"]) for fold in usable]))
    if not model_names:
        return BlendWeights(reason="folds share no common models")
    if len(model_names) == 1:
        only = model_names[0]
        return BlendWeights(
            weights={only: 1.0},
            retained=False,
            reason=f"only {only} produced out-of-fold predictions; nothing to blend",
        )

    def score_weights(weights: dict[str, float]) -> list[float]:
        scores = []
        for fold in usable:
            combined = np.zeros(len(fold["true_return"]), dtype=float)
            for name, weight in weights.items():
                combined = combined + weight * np.asarray(fold["predictions"][name], dtype=float)
            scores.append(
                blend_score(
                    fold["true_return"],
                    combined,
                    fold["dates"],
                    horizon,
                    fold["reference_prediction"],
                    ic_weight=ic_weight,
                )
            )
        return scores

    evaluated: list[dict] = []
    for weights in weight_grid(model_names, step=step):
        scores = np.asarray(score_weights(weights), dtype=float)
        dispersion = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        evaluated.append(
            {
                "weights": dict(weights),
                "mean": float(scores.mean()),
                "std": dispersion,
                "selection_score": float(scores.mean() - dispersion),
                "per_fold": [float(value) for value in scores],
            }
        )

    best = max(evaluated, key=lambda row: row["selection_score"])

    single_model_scores: dict[str, float] = {}
    for name in model_names:
        pure = {model: (1.0 if model == name else 0.0) for model in model_names}
        match = next(row for row in evaluated if row["weights"] == pure)
        single_model_scores[name] = float(match["selection_score"])

    best_single_name = max(single_model_scores, key=lambda name: single_model_scores[name])
    best_single_score = single_model_scores[best_single_name]
    improvement = best["selection_score"] - best_single_score
    retained = improvement > float(minimum_improvement)

    if retained:
        weights = dict(best["weights"])
        reason = (
            f"blend improved the out-of-fold score by {improvement:+.4f} over the best "
            f"single model ({best_single_name}), above the {minimum_improvement:.3f} bar"
        )
    else:
        weights = {model: (1.0 if model == best_single_name else 0.0) for model in model_names}
        reason = (
            f"blend improved by only {improvement:+.4f} over {best_single_name}, below the "
            f"{minimum_improvement:.3f} bar; keeping the single model"
        )

    result = BlendWeights(
        weights=weights,
        out_of_fold_score=float(best["mean"] if retained else best_single_score),
        out_of_fold_score_std=float(best["std"]),
        per_model_scores=single_model_scores,
        improves_on_best_single_model=bool(retained),
        retained=bool(retained),
        reason=reason,
    )
    result.candidate_grid = evaluated  # type: ignore[attr-defined]
    return result


def align_model_predictions(
    frames: dict[str, pd.DataFrame],
    prediction_column: str = "predicted_return",
    truth_column: str = "true_return",
) -> pd.DataFrame:
    """
    Inner-join per-model prediction frames on ``(ticker, date)``.

    An inner join, not a concatenation: blending is only meaningful on rows where
    every model produced a forecast. The two families do not cover identical rows
    — the sequence model needs a full look-back window and so drops the first
    rows of every ticker — and lining them up positionally would compare one
    model's forecast for AAPL against another's for XOM.
    """
    aligned: pd.DataFrame | None = None
    for name, frame in frames.items():
        subset = frame[["ticker", "date", prediction_column, truth_column]].copy()
        subset["date"] = pd.to_datetime(subset["date"])
        subset = subset.rename(columns={prediction_column: f"prediction_{name}"})
        if aligned is None:
            aligned = subset.rename(columns={truth_column: "true_return"})
        else:
            aligned = aligned.merge(
                subset.drop(columns=[truth_column]),
                on=["ticker", "date"],
                how="inner",
            )
    if aligned is None:
        return pd.DataFrame()
    return aligned.sort_values(["date", "ticker"]).reset_index(drop=True)
