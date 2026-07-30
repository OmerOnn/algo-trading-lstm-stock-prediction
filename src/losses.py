"""
Regression objectives for the sequence model.

The composite objective
-----------------------
```text
total = mse_weight * MSE
      + huber_weight * Huber
      + cross_sectional_ic_weight * (1 - mean per-date correlation)
```

Each term does a different job and none of them is sufficient alone:

* **MSE** is the loss function of a squared-error regression. It is what makes
  the *magnitude* of the forecast meaningful, so the output can honestly be
  reported as an expected percentage return rather than a ranking score.
* **Huber** caps the gradient contribution of extreme return events. On a
  fat-tailed panel a handful of crash days otherwise dominate every batch.
* **Cross-sectional IC** is the only term that rewards getting the *ordering*
  right. It matters because a constant prediction is a strong MSE solution on a
  near-unforecastable target — it achieves the variance of the target and cannot
  be beaten by any unskilled model — while carrying exactly zero
  stock-selection information. Without a ranking term, minimising squared error
  drives the model towards that useless optimum.

Why per-date and not per-batch
------------------------------
The reported headline metric is the mean *per-date* rank correlation. A
correlation computed over a batch of randomly mixed rows measures something
else: it is inflated by the market factor common to all names on a date, so a
model that only tracks "the market was up that day" scores well on it while
adding no cross-sectional skill. Grouping the correlation by date makes the
training objective measure the same thing as the evaluation metric.

Pearson correlation is used as a differentiable surrogate for the Spearman rank
IC that is reported. Ranks are not differentiable, and on a de-meaned
volatility-scaled target the two track each other closely.
"""

from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_LOSS_CONFIG = {
    "mse_weight": 0.40,
    "huber_weight": 0.40,
    "cross_sectional_ic_weight": 0.20,
    "huber_beta": 0.5,
    # A date with only a couple of names gives a meaningless correlation.
    "minimum_names_per_date": 5,
}


def resolve_loss_config(config: dict | None) -> dict:
    resolved = dict(DEFAULT_LOSS_CONFIG)
    for key, value in (config or {}).items():
        if key in resolved:
            resolved[key] = value
    weights = (
        float(resolved["mse_weight"]),
        float(resolved["huber_weight"]),
        float(resolved["cross_sectional_ic_weight"]),
    )
    if all(weight <= 0 for weight in weights):
        raise ValueError(
            "regression_loss must give a positive weight to at least one of "
            "mse_weight, huber_weight, cross_sectional_ic_weight."
        )
    if float(resolved["huber_beta"]) <= 0:
        raise ValueError("regression_loss.huber_beta must be positive.")
    return resolved


def grouped_correlation(
    predicted: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
    minimum_names_per_date: int = 5,
) -> torch.Tensor:
    """
    Mean Pearson correlation computed *within* each group, one group per date.

    Every qualifying group contributes equally to the mean regardless of how many
    names it holds. That is deliberate: weighting by group size would let the
    handful of dates on which the whole universe is present dominate the
    gradient, and the evaluation metric averages dates, not rows.

    Returns a zero-dimensional tensor. When no group has enough names the result
    is exactly zero, so a caller adding ``1 - correlation`` sees a constant and
    contributes no gradient rather than a NaN.
    """
    predicted = predicted.reshape(-1)
    target = target.reshape(-1)
    groups = groups.reshape(-1).long()

    if predicted.numel() == 0:
        return predicted.new_zeros(())

    group_count = int(groups.max().item()) + 1 if groups.numel() else 0
    if group_count == 0:
        return predicted.new_zeros(())

    ones = torch.ones_like(predicted)
    counts = torch.zeros(group_count, device=predicted.device, dtype=predicted.dtype)
    counts = counts.index_add(0, groups, ones)
    safe_counts = counts.clamp_min(1.0)

    sum_predicted = torch.zeros_like(counts).index_add(0, groups, predicted)
    sum_target = torch.zeros_like(counts).index_add(0, groups, target)
    centered_predicted = predicted - (sum_predicted / safe_counts)[groups]
    centered_target = target - (sum_target / safe_counts)[groups]

    covariance = torch.zeros_like(counts).index_add(
        0, groups, centered_predicted * centered_target
    )
    variance_predicted = torch.zeros_like(counts).index_add(
        0, groups, centered_predicted.square()
    )
    variance_target = torch.zeros_like(counts).index_add(0, groups, centered_target.square())

    denominator = torch.sqrt(variance_predicted * variance_target).clamp_min(1e-8)
    correlation = covariance / denominator

    valid = (counts >= float(minimum_names_per_date)) & (variance_predicted > 1e-12)
    if not bool(valid.any()):
        return predicted.new_zeros(())
    return correlation[valid].mean()


class CompositeRegressionLoss(nn.Module):
    """
    Weighted sum of MSE, Huber and a per-date cross-sectional correlation loss.

    ``groups`` carries an integer date code per row. When it is omitted the
    correlation term degrades to a single whole-batch correlation, which is what
    a randomly shuffled loader can support; passing date codes is what makes the
    term match the reported metric.
    """

    def __init__(
        self,
        mse_weight: float = 0.40,
        huber_weight: float = 0.40,
        cross_sectional_ic_weight: float = 0.20,
        huber_beta: float = 0.5,
        minimum_names_per_date: int = 5,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.huber_weight = float(huber_weight)
        self.cross_sectional_ic_weight = float(cross_sectional_ic_weight)
        self.minimum_names_per_date = int(minimum_names_per_date)
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss(beta=float(huber_beta))

    @classmethod
    def from_config(cls, config: dict | None) -> "CompositeRegressionLoss":
        resolved = resolve_loss_config(config)
        return cls(
            mse_weight=float(resolved["mse_weight"]),
            huber_weight=float(resolved["huber_weight"]),
            cross_sectional_ic_weight=float(resolved["cross_sectional_ic_weight"]),
            huber_beta=float(resolved["huber_beta"]),
            minimum_names_per_date=int(resolved["minimum_names_per_date"]),
        )

    def terms(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        groups: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Individual loss components, before weighting. Useful for logging."""
        predicted = predicted.reshape(-1)
        target = target.reshape(-1)
        components = {
            "mse": self.mse(predicted, target),
            "huber": self.huber(predicted, target),
        }

        if self.cross_sectional_ic_weight <= 0 or predicted.numel() < 3:
            components["cross_sectional_ic"] = predicted.new_zeros(())
        elif groups is None:
            # No date information available: fall back to one batch-wide
            # correlation. Reported separately so the difference is visible.
            centered_predicted = predicted - predicted.mean()
            centered_target = target - target.mean()
            denominator = torch.sqrt(
                centered_predicted.square().sum() * centered_target.square().sum()
            ).clamp_min(1e-8)
            components["cross_sectional_ic"] = (
                centered_predicted * centered_target
            ).sum() / denominator
        else:
            components["cross_sectional_ic"] = grouped_correlation(
                predicted, target, groups, self.minimum_names_per_date
            )
        return components

    def combine(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Weighted total from pre-computed components.

        Exposed separately from ``forward`` so a training loop can log the
        individual terms without paying for a second forward pass over them.
        """
        loss = self.mse_weight * components["mse"] + self.huber_weight * components["huber"]
        if self.cross_sectional_ic_weight > 0:
            # 1 - correlation, so a perfect ordering costs nothing and an
            # inverted ordering costs twice as much as no signal at all.
            loss = loss + self.cross_sectional_ic_weight * (
                1.0 - components["cross_sectional_ic"]
            )
        return loss

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.combine(self.terms(predicted, target, groups))

    def describe(self) -> dict:
        return {
            "mse_weight": self.mse_weight,
            "huber_weight": self.huber_weight,
            "cross_sectional_ic_weight": self.cross_sectional_ic_weight,
            "huber_beta": float(self.huber.beta),
            "minimum_names_per_date": self.minimum_names_per_date,
            "correlation_surrogate": "per-date Pearson (differentiable proxy for Spearman IC)",
        }


# Named presets used by the loss-selection experiment. Comparing these on
# identical purged folds is what decides which objective ships.
LOSS_PRESETS: dict[str, dict] = {
    "pure_mse": {
        "mse_weight": 1.0,
        "huber_weight": 0.0,
        "cross_sectional_ic_weight": 0.0,
        "huber_beta": 0.5,
    },
    "pure_huber": {
        "mse_weight": 0.0,
        "huber_weight": 1.0,
        "cross_sectional_ic_weight": 0.0,
        "huber_beta": 0.5,
    },
    "mse_huber": {
        "mse_weight": 0.5,
        "huber_weight": 0.5,
        "cross_sectional_ic_weight": 0.0,
        "huber_beta": 0.5,
    },
    "mse_huber_ic": {
        "mse_weight": 0.40,
        "huber_weight": 0.40,
        "cross_sectional_ic_weight": 0.20,
        "huber_beta": 0.5,
    },
}
