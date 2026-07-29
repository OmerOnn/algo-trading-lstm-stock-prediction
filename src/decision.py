"""
Deriving BUY / HOLD / SELL from a regression forecast and its uncertainty.

Why keep a discrete signal at all
---------------------------------
The model is a regressor and the reported output is a percentage return. A
discrete signal is kept because it is what makes the forecast testable: it turns
a number into a decision that a backtest can charge transaction costs against.
Without it there is no way to answer "is this edge economically real?".

Why the rule is risk-adjusted rather than a fixed percentage
------------------------------------------------------------
A fixed "+3% means BUY" rule is arbitrary and, worse, it is not comparable
across stocks: +3% expected on a low-volatility utility is a much stronger claim
than +3% on a high-beta semiconductor name. The default rule therefore scores
each forecast by its edge *per unit of forecast uncertainty*

```text
z = (predicted_return - threshold) / sigma
```

and acts only when ``z`` clears a floor tuned on validation data. ``threshold``
is never below the round-trip trading cost, so a signal must be profitable after
costs before it can be emitted. This is the same quantity a practitioner would
call a forecast Sharpe ratio, and it makes the uncertainty estimate a first-class
input to the decision rather than a decoration on the output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"
SIGNAL_LABELS = (SELL, HOLD, BUY)

DECISION_RULES = {"risk_adjusted", "point"}


@dataclass
class DecisionConfig:
    """Parameters of the signal rule. Tuned on validation, frozen before test."""

    rule: str = "risk_adjusted"
    # Minimum expected return, in return units. Floored at the round-trip cost.
    threshold: float = 0.003
    # Minimum edge-per-risk required to act (risk_adjusted rule only).
    min_z_score: float = 0.15
    allow_short: bool = False
    # Optional extra screen on the probability the direction is right.
    min_direction_probability: float = 0.0
    # "binary" trades a full unit; "confidence" scales exposure by conviction.
    position_sizing: str = "binary"
    max_position: float = 1.0

    def __post_init__(self) -> None:
        rule = str(self.rule).lower().strip()
        if rule not in DECISION_RULES:
            raise ValueError(f"decision rule must be one of {sorted(DECISION_RULES)}")
        self.rule = rule
        if str(self.position_sizing).lower().strip() not in {"binary", "confidence"}:
            raise ValueError("position_sizing must be 'binary' or 'confidence'")
        self.position_sizing = str(self.position_sizing).lower().strip()

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "threshold": float(self.threshold),
            "min_z_score": float(self.min_z_score),
            "allow_short": bool(self.allow_short),
            "min_direction_probability": float(self.min_direction_probability),
            "position_sizing": self.position_sizing,
            "max_position": float(self.max_position),
        }


def edge_z_score(
    predicted_return: np.ndarray,
    sigma: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Signed edge beyond the cost hurdle, expressed in units of forecast sigma."""
    predictions = np.asarray(predicted_return, dtype=float)
    sigma_values = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    threshold = abs(float(threshold))
    # Shrink towards zero by the hurdle on whichever side the forecast points.
    adjusted = np.where(
        predictions >= 0,
        np.maximum(predictions - threshold, 0.0),
        np.minimum(predictions + threshold, 0.0),
    )
    return adjusted / sigma_values


def direction_probability(predicted_return: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """P(realised return has the forecast's sign) under a normal approximation."""
    from math import erf, sqrt

    predictions = np.asarray(predicted_return, dtype=float)
    sigma_values = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    ratio = np.abs(predictions) / sigma_values
    return np.asarray([0.5 * (1.0 + erf(float(value) / sqrt(2.0))) for value in ratio])


def decide_batch(
    predicted_return: np.ndarray,
    sigma: np.ndarray,
    cfg: DecisionConfig,
) -> np.ndarray:
    """Vectorised BUY / HOLD / SELL assignment."""
    predictions = np.asarray(predicted_return, dtype=float)
    sigma_values = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    threshold = abs(float(cfg.threshold))

    # Both sides must require a *strictly signed* edge. Without the sign test a
    # zero edge satisfies `>= 0` and `<= -0` simultaneously, and whichever side
    # is assigned last silently wins -- which produced SELL on a positive
    # forecast whenever the tuned floor happened to be zero.
    if cfg.rule == "point":
        long_side = (predictions > 0.0) & (predictions >= threshold)
        short_side = (predictions < 0.0) & (predictions <= -threshold)
    else:
        z = edge_z_score(predictions, sigma_values, threshold)
        floor = max(float(cfg.min_z_score), 0.0)
        long_side = (z > 0.0) & (z >= floor)
        short_side = (z < 0.0) & (z <= -floor)

    if cfg.min_direction_probability > 0:
        confident = direction_probability(predictions, sigma_values) >= float(
            cfg.min_direction_probability
        )
        long_side &= confident
        short_side &= confident

    signals = np.full(len(predictions), HOLD, dtype=object)
    signals[long_side] = BUY
    signals[short_side] = SELL
    return signals


def decide(predicted_return: float, sigma: float, cfg: DecisionConfig) -> str:
    """Single-observation convenience wrapper used by the prediction path."""
    return str(decide_batch(np.asarray([predicted_return]), np.asarray([sigma]), cfg)[0])


def position_sizes(
    signals: np.ndarray,
    predicted_return: np.ndarray,
    sigma: np.ndarray,
    cfg: DecisionConfig,
) -> np.ndarray:
    """
    Map signals to portfolio exposure in [-max_position, max_position].

    Under "confidence" sizing the exposure grows with the risk-adjusted edge and
    saturates at ``max_position``, so a marginal signal risks less capital than a
    strong one. Under "binary" sizing every active signal takes a full unit,
    which keeps the backtest directly comparable to the classic threshold rule.
    """
    signal_array = np.asarray(signals, dtype=object)
    direction = np.where(signal_array == BUY, 1.0, np.where(signal_array == SELL, -1.0, 0.0))
    if not cfg.allow_short:
        direction = np.where(direction < 0, 0.0, direction)

    if cfg.position_sizing == "binary":
        return direction * float(cfg.max_position)

    z = np.abs(edge_z_score(predicted_return, sigma, cfg.threshold))
    floor = max(float(cfg.min_z_score), 1e-6)
    conviction = np.clip(z / (2.0 * floor), 0.0, 1.0)
    return direction * float(cfg.max_position) * conviction


def realised_signal(true_return: np.ndarray, threshold: float) -> np.ndarray:
    """
    Label what actually happened, for the confusion matrix only.

    The realised label uses the plain return threshold: uncertainty is a property
    of the forecast, not of the outcome.
    """
    values = np.asarray(true_return, dtype=float)
    threshold = abs(float(threshold))
    labels = np.full(len(values), HOLD, dtype=object)
    labels[values >= threshold] = BUY
    labels[values <= -threshold] = SELL
    return labels
