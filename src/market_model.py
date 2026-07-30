"""
Two-stage hierarchical return decomposition.

The forecast a user sees is a stock's total future return. That total is the sum
of things with completely different forecastability, and mixing them into one
regression target is what made the first version of this project score close to
zero:

```text
expected_stock_return = beta * expected_market_return   <- one number per date
                      + expected_sector_return          <- one number per sector-date
                      + expected_stock_residual         <- what a stock model can learn
```

The market leg dominates the variance of a pooled panel and is essentially
unforecastable from a single stock's technical indicators. The residual leg is
small but genuinely carries cross-sectional information. Training one model on
the sum spends nearly all of its capacity on the unforecastable part.

What replaced the constant drift
--------------------------------
The previous version added a *constant* historical benchmark drift to the
predicted excess return. That is a defensible baseline but it throws away the
fact that the market's expected return is not constant — it is higher after a
drawdown and lower when volatility is elevated. This module fits an explicit
market-return regressor on one observation per date, using only market-state
variables known at that date.

Crucially it is **shrunk towards the historical drift** unless validation
demonstrates skill. Predicting the index return over a month is close to
impossible, and an unshrunk market model would inject a large, confident, wrong
number straight into every stock's forecast. The shrinkage weight is fitted on
validation data by out-of-sample MSE skill, so a market model with no skill
collapses to exactly the old constant-drift behaviour, and one with real skill is
allowed to contribute in proportion to how much it demonstrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.features import market_wide_feature_columns
from src.regression import BENCHMARK_RETURN_COLUMN, regression_metrics


# ---------------------------------------------------------------------------
# Stage 1: the market-return model
# ---------------------------------------------------------------------------


def build_market_frame(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collapse the panel to one row per date for the market model.

    Market-state variables are identical for every ticker on a date, so taking
    the first row per date is exact rather than an approximation. Doing this is
    also what makes the market model honest: it sees ~5,000 observations, not
    ~500,000 copies of the same 5,000, so its validation statistics are not
    inflated a hundredfold by duplicated rows.
    """
    columns = feature_columns or market_wide_feature_columns(df)
    columns = [column for column in columns if column in df.columns]
    wanted = columns + [BENCHMARK_RETURN_COLUMN]
    wanted = [column for column in wanted if column in df.columns]

    frame = df[wanted].groupby(level=0).first().sort_index()
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass
class MarketReturnModel:
    """
    Ridge regression on market-state features, shrunk towards historical drift.

    ```text
    prediction = drift + shrinkage * (ridge_prediction - drift)
    ```

    ``shrinkage`` is 0 when the model showed no validation skill (reducing this
    to the historical-drift baseline exactly) and at most ``maximum_shrinkage``
    when it did. It is never 1: a month-ahead index forecast that good does not
    exist, and leaving headroom means a lucky validation window cannot hand the
    market model full authority over every stock's forecast.
    """

    drift: float = 0.0
    shrinkage: float = 0.0
    maximum_shrinkage: float = 0.60
    alpha: float = 10.0
    feature_columns: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    feature_mean: list[float] = field(default_factory=list)
    feature_std: list[float] = field(default_factory=list)
    train_dates: int = 0
    validation_dates: int = 0
    validation_mse_skill: float = 0.0
    validation_correlation: float = 0.0
    drift_validation_mse: float = 0.0
    model_validation_mse: float = 0.0
    reason: str = "not fitted"

    def raw_predict(self, frame: pd.DataFrame) -> np.ndarray:
        """The ridge prediction before shrinkage."""
        mean = np.asarray(self.feature_mean, dtype=float)
        std = np.asarray(self.feature_std, dtype=float)
        # Every piece has to be present and consistently sized. A metadata payload
        # that names features but carries no standardisation moments (an older
        # artifact, or a model that was never fitted) must degrade to the drift
        # rather than raise from inside a matmul.
        if (
            not self.feature_columns
            or len(mean) != len(self.feature_columns)
            or len(std) != len(self.feature_columns)
        ):
            return np.full(len(frame), self.drift, dtype=float)
        values = frame.reindex(columns=self.feature_columns).astype(float).to_numpy()

        # A missing or infinite input becomes the training mean, i.e. it
        # contributes exactly zero to the standardised prediction. Left
        # unguarded, one non-finite macro reading propagates through the matmul
        # and turns the whole date's market forecast into a NaN, which would
        # then be broadcast onto every stock on that date.
        values = np.where(np.isfinite(values), values, mean)
        standardised = (values - mean) / np.where(std > 0, std, 1.0)
        standardised = np.clip(np.nan_to_num(standardised, nan=0.0), -10.0, 10.0)

        weights = np.asarray(
            [self.coefficients.get(column, 0.0) for column in self.feature_columns], dtype=float
        )
        return self.intercept + standardised @ weights

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """The shrunk market-return forecast, in return units."""
        raw = self.raw_predict(frame)
        raw = np.where(np.isfinite(raw), raw, self.drift)
        return self.drift + float(self.shrinkage) * (raw - self.drift)

    def predict_for_panel(self, df: pd.DataFrame) -> np.ndarray:
        """Broadcast the per-date market forecast back onto every panel row."""
        per_date = build_market_frame(df, self.feature_columns or None)
        if per_date.empty:
            return np.full(len(df), self.drift, dtype=float)
        predictions = pd.Series(self.predict(per_date), index=per_date.index)
        mapped = pd.DatetimeIndex(df.index).map(predictions)
        return np.asarray(pd.Series(mapped).fillna(self.drift), dtype=float)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["method"] = (
            "ridge on date-level market-state features, shrunk towards the "
            "training-window historical drift by validated MSE skill"
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict | None) -> "MarketReturnModel":
        if not payload:
            return cls()
        known = {key: payload[key] for key in asdict(cls()).keys() if key in payload}
        return cls(**known)


def fit_market_return_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    alpha: float = 10.0,
    maximum_shrinkage: float = 0.60,
    minimum_train_dates: int = 250,
) -> MarketReturnModel:
    """
    Fit stage 1 on training dates and set its authority from validation dates.

    The shrinkage weight is chosen analytically rather than by a grid search. For
    a prediction of the form ``drift + w * (model - drift)`` the validation MSE is
    a quadratic in ``w``, so its minimiser is available in closed form:

    ```text
    w* = cov(residual_from_drift, model_deviation) / var(model_deviation)
    ```

    which is then clipped into ``[0, maximum_shrinkage]``. A model that is
    anti-correlated with the truth gets exactly 0 and the system falls back to
    the constant drift.
    """
    features = feature_columns or market_wide_feature_columns(train_df)
    train_frame = build_market_frame(train_df, features)
    validation_frame = build_market_frame(validation_df, features)

    usable = [column for column in features if column in train_frame.columns]
    drift = (
        float(train_frame[BENCHMARK_RETURN_COLUMN].mean())
        if BENCHMARK_RETURN_COLUMN in train_frame.columns and not train_frame.empty
        else 0.0
    )

    model = MarketReturnModel(
        drift=drift,
        shrinkage=0.0,
        maximum_shrinkage=float(maximum_shrinkage),
        alpha=float(alpha),
        train_dates=int(len(train_frame)),
        validation_dates=int(len(validation_frame)),
        reason="historical drift only",
    )

    if len(train_frame) < int(minimum_train_dates) or not usable or validation_frame.empty:
        model.reason = "not enough date-level history to fit a market model"
        return model

    x_train = train_frame[usable].astype(float).to_numpy()
    y_train = train_frame[BENCHMARK_RETURN_COLUMN].astype(float).to_numpy()

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    ridge = Ridge(alpha=float(alpha))
    ridge.fit((x_train - mean) / std, y_train)

    model.feature_columns = list(usable)
    model.feature_mean = [float(value) for value in mean]
    model.feature_std = [float(value) for value in std]
    model.coefficients = {
        column: float(weight) for column, weight in zip(usable, ridge.coef_.ravel())
    }
    model.intercept = float(ridge.intercept_)

    y_validation = validation_frame[BENCHMARK_RETURN_COLUMN].astype(float).to_numpy()
    raw = model.raw_predict(validation_frame[usable])
    raw = np.where(np.isfinite(raw), raw, drift)

    deviation = raw - drift
    residual = y_validation - drift
    variance = float(np.dot(deviation, deviation))
    if variance <= 1e-18:
        model.reason = "market model produced no variation beyond the drift"
        return model

    optimal = float(np.dot(residual, deviation) / variance)
    model.shrinkage = float(np.clip(optimal, 0.0, float(maximum_shrinkage)))

    drift_mse = float(np.mean(np.square(residual)))
    model_mse = float(np.mean(np.square(y_validation - model.predict(validation_frame[usable]))))
    model.drift_validation_mse = drift_mse
    model.model_validation_mse = model_mse
    model.validation_mse_skill = float(1.0 - model_mse / drift_mse) if drift_mse > 0 else 0.0
    model.validation_correlation = (
        float(np.corrcoef(y_validation, raw)[0, 1])
        if np.std(raw) > 0 and np.std(y_validation) > 0
        else 0.0
    )
    model.reason = (
        f"unconstrained optimum {optimal:+.3f}, applied {model.shrinkage:.3f} "
        f"(cap {maximum_shrinkage:.2f}); validation MSE skill "
        f"{model.validation_mse_skill:+.4f}"
    )
    return model


def fit_market_return_model_walk_forward(
    full_df: pd.DataFrame,
    splits,
    feature_columns: list[str] | None = None,
    alpha: float = 10.0,
    maximum_shrinkage: float = 0.60,
) -> tuple[float, dict]:
    """
    Choose the market model's shrinkage from *consistency across folds*.

    A single validation window is not enough to grant the market model authority.
    Measured on this panel, the one-window fit wanted a shrinkage of 0.74 on the
    strength of +0.14 validation MSE skill, and that same setting then scored
    −0.11 MSE skill on the test period: the fit was a property of the window, not
    of the market.

    The rule used instead requires the folds to agree:

    ```text
    shrinkage = clip(mean(fold optima) - std(fold optima), 0, cap)
    ```

    One standard deviation of disagreement is subtracted from the mean, so folds
    that point in different directions cancel each other out and the shrinkage
    collapses to zero — which reduces the composition to the constant historical
    drift exactly. Skill has to be *repeatable* to be used, not merely present
    once.
    """
    fold_optima: list[float] = []
    fold_reports: list[dict] = []

    for split in splits:
        train_df, validation_df, _ = split.frames(full_df)
        if train_df.empty or validation_df.empty:
            continue
        # Fitted with the cap wide open so the fold's unconstrained preference is
        # visible; the aggregate below is what actually gets clipped.
        fold_model = fit_market_return_model(
            train_df,
            validation_df,
            feature_columns,
            alpha=alpha,
            maximum_shrinkage=1.0,
        )
        fold_optima.append(float(fold_model.shrinkage))
        fold_reports.append(
            {
                "fold": split.index,
                "label": split.label,
                "optimal_shrinkage": float(fold_model.shrinkage),
                "validation_mse_skill": float(fold_model.validation_mse_skill),
                "validation_correlation": float(fold_model.validation_correlation),
                "train_dates": int(fold_model.train_dates),
            }
        )

    if not fold_optima:
        return 0.0, {
            "selected_shrinkage": 0.0,
            "reason": "no usable folds; falling back to constant historical drift",
            "folds": [],
        }

    values = np.asarray(fold_optima, dtype=float)
    dispersion = float(values.std(ddof=1)) if len(values) > 1 else float(values.mean())
    selected = float(np.clip(values.mean() - dispersion, 0.0, float(maximum_shrinkage)))

    return selected, {
        "selected_shrinkage": selected,
        "rule": "clip(mean(fold optima) - std(fold optima), 0, cap)",
        "fold_mean": float(values.mean()),
        "fold_std": dispersion,
        "fold_minimum": float(values.min()),
        "fold_maximum": float(values.max()),
        "maximum_shrinkage": float(maximum_shrinkage),
        "folds": fold_reports,
        "reason": (
            "market-return skill must repeat across purged folds before it is "
            "allowed to move a stock forecast; disagreement collapses the weight "
            "to the constant drift"
        ),
    }


def evaluate_market_model(
    model: MarketReturnModel,
    evaluation_df: pd.DataFrame,
) -> dict:
    """Score the market model on held-out dates against the constant drift."""
    frame = build_market_frame(evaluation_df, model.feature_columns or None)
    if frame.empty or BENCHMARK_RETURN_COLUMN not in frame.columns:
        return {"evaluated_dates": 0}

    truth = frame[BENCHMARK_RETURN_COLUMN].astype(float).to_numpy()
    predicted = model.predict(frame)
    drift_reference = np.full_like(truth, model.drift)

    return {
        "evaluated_dates": int(len(truth)),
        "shrinkage": float(model.shrinkage),
        "model": regression_metrics(truth, predicted, reference_prediction=drift_reference),
        "historical_drift_baseline": regression_metrics(truth, drift_reference),
        "unshrunk_model": regression_metrics(
            truth, model.raw_predict(frame), reference_prediction=drift_reference
        ),
    }


# ---------------------------------------------------------------------------
# Stage 2: composing the total return from its parts
# ---------------------------------------------------------------------------

BETA_COLUMN = "market_beta_60d"
SECTOR_BETA_COLUMN = "sector_beta_60d"


def rolling_beta(df: pd.DataFrame, column: str = BETA_COLUMN, default: float = 1.0) -> np.ndarray:
    """
    The stock's beta as known at prediction time, cleaned for use as a multiplier.

    Betas are clipped to [0, 3]. A rolling 60-day regression on noisy daily
    returns occasionally produces a negative or enormous beta for a single stock,
    and because beta multiplies the market forecast, one bad estimate would
    otherwise flip or explode that stock's entire expected return.
    """
    if column not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    values = df[column].astype(float).to_numpy()
    values = np.where(np.isfinite(values), values, float(default))
    return np.clip(values, 0.0, 3.0)


def compose_hierarchical_return(
    market_component: np.ndarray,
    residual_component: np.ndarray,
    beta: np.ndarray | None = None,
    sector_component: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Rebuild the user-facing total return and return every component separately.

    Components are reported alongside the total, never folded away, because the
    honest reading of the forecast depends on which leg it came from: "+3%
    because the market is expected to rise" and "+3% because this stock is
    expected to beat its peers" are different claims with different reliability.
    """
    market = np.asarray(market_component, dtype=float)
    residual = np.asarray(residual_component, dtype=float)
    beta_values = np.ones_like(market) if beta is None else np.asarray(beta, dtype=float)
    market_leg = beta_values * market
    sector_leg = (
        np.zeros_like(market) if sector_component is None else np.asarray(sector_component, dtype=float)
    )
    return {
        "total": market_leg + sector_leg + residual,
        "market_component": market_leg,
        "sector_component": sector_leg,
        "residual_component": residual,
        "beta": beta_values,
        "market_return_forecast": market,
    }
