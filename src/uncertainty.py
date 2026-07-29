"""
Uncertainty estimation for the return regressors.

Two sources of uncertainty are separated and then recombined:

* **Epistemic** (model) uncertainty — how much the forecast would change if the
  model had been fitted slightly differently. Estimated with Monte Carlo dropout
  for the LSTM and with a block-bootstrap ensemble for XGBoost.
* **Aleatoric** (irreducible) uncertainty — the noise in the return itself.
  Dominant for equity returns. Estimated as a multiple of the stock's trailing
  volatility scale, so it widens for volatile names and volatile regimes.

The combined standard deviation is then turned into an interval with
**normalised split-conformal calibration**: the interval multiplier is the
empirical quantile of the standardised absolute validation errors. Under
exchangeability this gives the requested marginal coverage regardless of whether
the underlying error distribution is Gaussian, which a plain ``mean ± 1.96 sigma``
interval does not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEFAULT_CONFIDENCE_LEVEL = 0.80


# ---------------------------------------------------------------------------
# Monte Carlo dropout (LSTM)
# ---------------------------------------------------------------------------


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """
    Put the model in evaluation mode but keep every dropout layer stochastic.

    Batch/layer normalisation must stay in eval mode, otherwise the running
    statistics change between passes and the spread no longer reflects model
    uncertainty alone.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.AlphaDropout)):
            module.train()
    return model


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    passes: int = 30,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) of ``passes`` stochastic forward passes for one batch."""
    was_training = model.training
    enable_mc_dropout(model)
    if device is not None:
        x = x.to(device, non_blocking=(device.type == "cuda"))

    samples = torch.stack([model(x).float().cpu() for _ in range(max(2, int(passes)))])
    model.train(was_training)
    model.eval()
    return samples.mean(dim=0).numpy(), samples.std(dim=0, unbiased=True).numpy()


@torch.no_grad()
def mc_dropout_predict_loader(
    model: nn.Module,
    loader: Iterable,
    passes: int = 30,
    device: torch.device | None = None,
    progress_label: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Run Monte Carlo dropout over a dataloader that yields
    ``(x, model_target, true_return, target_scale)``.

    Returns the decoded return-space mean and standard deviation, i.e. both are
    already multiplied by the per-row target scale.
    """
    was_training = model.training
    enable_mc_dropout(model)

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    scales: list[np.ndarray] = []

    passes = max(2, int(passes))
    if progress_label:
        print(f"Monte Carlo dropout ({passes} passes) over {progress_label}...")

    for x, _, y_return, target_scale in loader:
        if device is not None:
            x = x.to(device, non_blocking=(device.type == "cuda"))
        samples = torch.stack([model(x).float().cpu() for _ in range(passes)])
        scale = target_scale.float().cpu()
        means.append((samples.mean(dim=0) * scale).numpy())
        stds.append((samples.std(dim=0, unbiased=True) * scale).numpy())
        truths.append(y_return.float().cpu().numpy())
        scales.append(scale.numpy())

    model.train(was_training)
    model.eval()

    return {
        "mean": np.concatenate(means) if means else np.empty(0),
        "model_std": np.concatenate(stds) if stds else np.empty(0),
        "true_return": np.concatenate(truths) if truths else np.empty(0),
        "target_scale": np.concatenate(scales) if scales else np.empty(0),
    }


# ---------------------------------------------------------------------------
# Block bootstrap ensemble (XGBoost)
# ---------------------------------------------------------------------------


def block_bootstrap_row_indices(
    dates: Sequence,
    rng: np.random.Generator,
    block_size: int = 21,
) -> np.ndarray:
    """
    Draw a bootstrap resample of rows by sampling contiguous blocks of dates.

    Sampling individual rows would ignore the strong serial dependence of a
    price panel and produce far too narrow an interval. Resampling blocks of
    consecutive trading dates (moving-block bootstrap) preserves it, and keeping
    a whole date together preserves the cross-section.
    """
    # Work in integer date codes rather than timestamps. Mixing pandas Timestamp
    # and numpy datetime64 keys is a trap: they compare equal but hash
    # differently, so a dict keyed on one and queried with the other silently
    # returns nothing and the resample comes back empty.
    codes, uniques = pd.factorize(pd.to_datetime(pd.Series(list(dates))), sort=True)
    date_count = len(uniques)
    block_size = max(1, int(block_size))
    if date_count == 0:
        return np.arange(len(codes), dtype=int)

    block_starts = range(0, date_count, block_size)
    blocks = [np.arange(start, min(start + block_size, date_count)) for start in block_starts]
    chosen = rng.integers(0, len(blocks), size=len(blocks))
    selected_codes = np.concatenate([blocks[index] for index in chosen])

    # Group row positions by date code once, then gather the selected codes.
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, selected_codes, side="left")
    ends = np.searchsorted(sorted_codes, selected_codes, side="right")
    segments = [order[start:end] for start, end in zip(starts, ends) if end > start]
    if not segments:
        return np.arange(len(codes), dtype=int)
    return np.concatenate(segments).astype(int)


class BootstrapEnsemble:
    """
    Bagged ensemble whose spread estimates model uncertainty.

    Bagging is not only an uncertainty device here: averaging de-correlated
    trees fitted on different resamples also reduces variance, which on a
    low-signal panel is usually a genuine accuracy gain over a single fit.
    """

    def __init__(self, models: list) -> None:
        if not models:
            raise ValueError("BootstrapEnsemble requires at least one fitted model.")
        self.models = models

    @classmethod
    def fit(
        cls,
        model_factory: Callable[[int], object],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        dates: Sequence,
        n_models: int = 20,
        block_size: int = 21,
        seed: int = 42,
        fit_kwargs: dict | None = None,
        progress: bool = True,
    ) -> "BootstrapEnsemble":
        rng = np.random.default_rng(int(seed))
        y_values = np.asarray(y_train, dtype=float)
        models = []
        for index in range(max(1, int(n_models))):
            rows = block_bootstrap_row_indices(dates, rng, block_size=block_size)
            model = model_factory(int(seed) + index)
            model.fit(x_train.iloc[rows], y_values[rows], **(fit_kwargs or {}))
            models.append(model)
            if progress:
                print(f"  bootstrap model {index + 1}/{int(n_models)} fitted")
        return cls(models)

    def predict(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.stack([model.predict(x) for model in self.models]).astype(float)
        std = predictions.std(axis=0, ddof=1) if len(self.models) > 1 else np.zeros(len(x))
        return predictions.mean(axis=0), std

    def __len__(self) -> int:
        return len(self.models)


# ---------------------------------------------------------------------------
# Interval calibration
# ---------------------------------------------------------------------------


@dataclass
class IntervalCalibration:
    """Everything needed to rebuild a calibrated interval at inference time."""

    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    # Aleatoric noise expressed as a multiple of the row's volatility scale.
    residual_scale_ratio: float = 1.0
    # Floor so a stock with a degenerate volatility estimate still gets a band.
    minimum_sigma: float = 0.005
    # Conformal multiplier applied to the combined sigma.
    conformal_multiplier: float = 1.0
    model_std_weight: float = 1.0
    validation_coverage: float = 0.0
    validation_mean_width: float = 0.0
    typical_sigma: float = 0.0
    method: str = "normalised split conformal (MC dropout / bootstrap + volatility-scaled residual)"

    def sigma(self, model_std: np.ndarray, target_scale: np.ndarray) -> np.ndarray:
        """Combine epistemic and aleatoric standard deviations."""
        epistemic = self.model_std_weight * np.asarray(model_std, dtype=float)
        aleatoric = self.residual_scale_ratio * np.asarray(target_scale, dtype=float)
        combined = np.sqrt(np.square(epistemic) + np.square(aleatoric))
        return np.maximum(combined, self.minimum_sigma)

    def interval(
        self,
        prediction: np.ndarray,
        model_std: np.ndarray,
        target_scale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, upper, sigma) for the calibrated confidence level."""
        sigma = self.sigma(model_std, target_scale)
        half_width = self.conformal_multiplier * sigma
        prediction = np.asarray(prediction, dtype=float)
        return prediction - half_width, prediction + half_width, sigma

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "IntervalCalibration":
        if not payload:
            return cls()
        known = {key: payload[key] for key in asdict(cls()).keys() if key in payload}
        return cls(**known)


def fit_interval_calibration(
    true_return: np.ndarray,
    prediction: np.ndarray,
    model_std: np.ndarray,
    target_scale: np.ndarray,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    minimum_sigma: float = 0.005,
) -> IntervalCalibration:
    """
    Calibrate prediction intervals on validation data only.

    Step 1 estimates the aleatoric component as the standard deviation of the
    volatility-standardised residual. Step 2 finds the conformal multiplier that
    makes empirical validation coverage equal the requested level.
    """
    true_values = np.asarray(true_return, dtype=float)
    predictions = np.asarray(prediction, dtype=float)
    model_std_values = np.asarray(model_std, dtype=float)
    scales = np.maximum(np.asarray(target_scale, dtype=float), 1e-8)

    residual = true_values - predictions
    standardised_residual = residual / scales
    residual_scale_ratio = float(np.std(standardised_residual, ddof=1))
    if not np.isfinite(residual_scale_ratio) or residual_scale_ratio <= 0:
        residual_scale_ratio = 1.0

    calibration = IntervalCalibration(
        confidence_level=float(confidence_level),
        residual_scale_ratio=residual_scale_ratio,
        minimum_sigma=float(minimum_sigma),
        conformal_multiplier=1.0,
    )

    sigma = calibration.sigma(model_std_values, scales)
    nonconformity = np.abs(residual) / sigma
    nonconformity = nonconformity[np.isfinite(nonconformity)]

    if len(nonconformity) > 0:
        # Finite-sample split-conformal quantile level.
        n = len(nonconformity)
        level = min(1.0, np.ceil((n + 1) * float(confidence_level)) / n)
        calibration.conformal_multiplier = float(np.quantile(nonconformity, level))
    else:
        calibration.conformal_multiplier = 1.28  # normal 80% two-sided fallback

    lower, upper, sigma = calibration.interval(predictions, model_std_values, scales)
    calibration.validation_coverage = float(np.mean((true_values >= lower) & (true_values <= upper)))
    calibration.validation_mean_width = float(np.mean(upper - lower))
    calibration.typical_sigma = float(np.median(sigma))
    return calibration


def interval_metrics(
    true_return: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict:
    """
    Standard prediction-interval quality metrics.

    * **PICP** — prediction interval coverage probability; should sit near the
      nominal level. Far above it means the interval is uninformatively wide.
    * **MPIW** — mean prediction interval width; narrower is better *given*
      correct coverage.
    * **Winkler score** — the proper scoring rule that trades the two off: width
      plus a penalty proportional to how far an observation falls outside.
      Lower is better.
    """
    true_values = np.asarray(true_return, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)

    inside = (true_values >= lower_values) & (true_values <= upper_values)
    width = upper_values - lower_values
    alpha = 1.0 - float(confidence_level)

    penalty = np.zeros_like(width)
    below = true_values < lower_values
    above = true_values > upper_values
    if alpha > 0:
        penalty[below] = (2.0 / alpha) * (lower_values[below] - true_values[below])
        penalty[above] = (2.0 / alpha) * (true_values[above] - upper_values[above])
    winkler = width + penalty

    observed_std = float(np.std(true_values, ddof=1)) if len(true_values) > 1 else 0.0
    coverage = float(np.mean(inside))

    return {
        "nominal_confidence_level": float(confidence_level),
        "coverage_picp": coverage,
        "coverage_error": coverage - float(confidence_level),
        "mean_interval_width_mpiw": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "normalized_interval_width": float(np.mean(width) / observed_std) if observed_std > 0 else 0.0,
        "winkler_score": float(np.mean(winkler)),
        "below_interval_rate": float(np.mean(below)),
        "above_interval_rate": float(np.mean(above)),
        "sample_size": int(len(true_values)),
    }


# ---------------------------------------------------------------------------
# Human-facing confidence
# ---------------------------------------------------------------------------

CONFIDENCE_TIERS = (
    (1.00, "High", "The expected move is large relative to the model's uncertainty."),
    (0.50, "Moderate", "The expected move is visible but comparable to the uncertainty band."),
    (0.00, "Low", "The uncertainty band is wider than the expected move; treat as inconclusive."),
)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF without pulling in scipy on the inference path."""
    from math import erf, sqrt

    values = np.asarray(x, dtype=float)
    return np.asarray([0.5 * (1.0 + erf(float(v) / sqrt(2.0))) for v in np.atleast_1d(values)])


def describe_confidence(prediction: float, sigma: float) -> dict:
    """
    Translate a prediction and its sigma into a label the UI can show.

    ``direction_probability`` is the probability that the realised return has the
    same sign as the forecast, under a normal approximation of the calibrated
    predictive distribution. It is the most directly interpretable number a user
    can be given, and it degrades gracefully towards 0.5 when the model is unsure.
    """
    sigma = float(max(sigma, 1e-9))
    signal_to_noise = abs(float(prediction)) / sigma
    direction_probability = float(normal_cdf(np.asarray([signal_to_noise]))[0])

    for cutoff, label, explanation in CONFIDENCE_TIERS:
        if signal_to_noise >= cutoff:
            break

    return {
        "confidence_label": label,
        "confidence_explanation": explanation,
        "signal_to_noise": float(signal_to_noise),
        "direction_probability": direction_probability,
        "sigma": sigma,
    }
