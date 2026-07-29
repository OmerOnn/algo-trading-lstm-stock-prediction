"""
Regression target construction and evaluation metrics.

Design summary
--------------
The public prediction is always a future percentage return over ``horizon``
trading sessions. Internally the model is trained on a decomposition:

```text
future_return = benchmark_future_return + future_excess_return
```

The benchmark (market) leg carries most of the variance of a pooled multi-stock
panel and is essentially unforecastable from technical features, so learning it
only injects noise. The model therefore learns the *excess* leg, and the market
leg is supplied by a train-only drift estimate. The user still sees a single
total expected return, and the two components are reported separately so the
decomposition is transparent.

For optimisation the excess leg is additionally divided by a past-only
volatility scale, which makes one loss scale valid across stocks and horizons.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score


TOTAL_RETURN_COLUMN = "future_return"
EXCESS_RETURN_COLUMN = "future_excess_return"
BENCHMARK_RETURN_COLUMN = "benchmark_future_return"

TARGET_MODES = {"raw_return", "volatility_scaled", "market_excess"}

DEFAULT_TARGET_CONFIG = {
    # "market_excess" trains on benchmark-relative return scaled by volatility.
    "mode": "market_excess",
    "volatility_column": "idiosyncratic_volatility_20d",
    "fallback_volatility_column": "volatility_20d",
    "daily_volatility_floor": 0.005,
    "target_clip": 4.0,
}


def resolve_target_config(config: dict | None) -> dict:
    resolved = dict(DEFAULT_TARGET_CONFIG)
    if config:
        resolved.update(config)
    mode = str(resolved["mode"]).lower().strip()
    if mode not in TARGET_MODES:
        raise ValueError(f"regression_target.mode must be one of {sorted(TARGET_MODES)}")
    resolved["mode"] = mode
    return resolved


def target_component_column(target_config: dict | None) -> str:
    """Name of the return column the model actually learns."""
    cfg = resolve_target_config(target_config)
    return EXCESS_RETURN_COLUMN if cfg["mode"] == "market_excess" else TOTAL_RETURN_COLUMN


def target_scale(
    df: pd.DataFrame,
    horizon: int,
    target_config: dict | None,
) -> np.ndarray:
    """Return a past-only scale that makes the training target more stationary."""
    cfg = resolve_target_config(target_config)
    if cfg["mode"] == "raw_return":
        return np.ones(len(df), dtype=float)

    column = str(cfg["volatility_column"])
    if column not in df.columns:
        fallback = str(cfg.get("fallback_volatility_column", "volatility_20d"))
        if fallback not in df.columns:
            raise KeyError(
                f"Target normalisation requires feature column '{column}' or '{fallback}'"
            )
        column = fallback

    daily_floor = float(cfg["daily_volatility_floor"])
    daily_volatility = df[column].astype(float).to_numpy()
    daily_volatility = np.where(np.isfinite(daily_volatility), daily_volatility, daily_floor)
    daily_volatility = np.maximum(daily_volatility, daily_floor)
    return daily_volatility * np.sqrt(max(1, int(horizon)))


def add_model_target(
    df: pd.DataFrame,
    horizon: int,
    target_config: dict | None,
) -> pd.DataFrame:
    """
    Add ``model_target``, ``target_scale`` and the excess-return column.

    ``future_return`` is never overwritten: it stays the ground-truth total
    return used for reporting and backtesting.
    """
    out = df.copy()
    cfg = resolve_target_config(target_config)

    if BENCHMARK_RETURN_COLUMN in out.columns:
        out[EXCESS_RETURN_COLUMN] = (
            out[TOTAL_RETURN_COLUMN].astype(float) - out[BENCHMARK_RETURN_COLUMN].astype(float)
        )
    elif cfg["mode"] == "market_excess":
        raise KeyError(
            f"regression_target.mode='market_excess' requires the '{BENCHMARK_RETURN_COLUMN}' "
            "column. Rebuild the dataset cache so benchmark forward returns are included."
        )

    component_column = target_component_column(cfg)
    out["target_scale"] = target_scale(out, horizon, cfg)
    normalised = out[component_column].astype(float) / out["target_scale"]
    clip = float(cfg.get("target_clip", 0.0))
    if clip > 0:
        normalised = normalised.clip(-clip, clip)
    out["model_target"] = normalised.astype(float)
    return out


def decode_model_output(model_output: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Convert raw network output back into the modelled return component."""
    return np.asarray(model_output, dtype=float) * np.asarray(scales, dtype=float)


def estimate_market_drift(train_df: pd.DataFrame, target_config: dict | None) -> dict:
    """
    Estimate the market leg of the decomposition from training data only.

    Returned as a dictionary so the value, its dispersion and its provenance are
    all stored with the model artifacts.
    """
    cfg = resolve_target_config(target_config)
    if cfg["mode"] != "market_excess" or BENCHMARK_RETURN_COLUMN not in train_df.columns:
        return {"mode": cfg["mode"], "market_drift": 0.0, "market_drift_std": 0.0, "sample_size": 0}

    # One observation per date: the benchmark return is identical across tickers.
    per_date = train_df.groupby(train_df.index)[BENCHMARK_RETURN_COLUMN].first().astype(float)
    per_date = per_date[np.isfinite(per_date)]
    if per_date.empty:
        return {"mode": cfg["mode"], "market_drift": 0.0, "market_drift_std": 0.0, "sample_size": 0}

    return {
        "mode": cfg["mode"],
        "market_drift": float(per_date.mean()),
        "market_drift_median": float(per_date.median()),
        "market_drift_std": float(per_date.std(ddof=0)),
        "sample_size": int(len(per_date)),
        "source": "mean benchmark forward return over the training window",
    }


def compose_total_return(component_return: np.ndarray, market_drift: float) -> np.ndarray:
    """Rebuild the user-facing total return from the modelled component."""
    return np.asarray(component_return, dtype=float) + float(market_drift)


def apply_return_calibration(predicted_return: np.ndarray, calibration: dict | None) -> np.ndarray:
    calibration = calibration or {}
    if not bool(calibration.get("enabled", False)):
        return np.asarray(predicted_return, dtype=float)
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    return intercept + slope * np.asarray(predicted_return, dtype=float)


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def _safe_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    value = float(stats.spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def regression_metrics(true_return: np.ndarray, predicted_return: np.ndarray) -> dict:
    """Point-forecast quality metrics for a return regression."""
    true_values = np.asarray(true_return, dtype=float)
    predicted_values = np.asarray(predicted_return, dtype=float)
    errors = predicted_values - true_values
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    median_ae = float(np.median(np.abs(errors)))
    direction_accuracy = float(np.mean(np.sign(predicted_values) == np.sign(true_values)))
    zero_mae = float(np.mean(np.abs(true_values)))
    zero_rmse = float(np.sqrt(np.mean(np.square(true_values))))
    normalised_mae = mae / zero_mae if zero_mae > 0 else 1.0
    normalised_rmse = rmse / zero_rmse if zero_rmse > 0 else 1.0
    pearson = _safe_correlation(true_values, predicted_values)
    rank_ic = _safe_rank_correlation(true_values, predicted_values)
    r2 = float(r2_score(true_values, predicted_values)) if len(true_values) > 1 else 0.0

    rmse_skill = float(np.clip(1.0 - normalised_rmse, -1.0, 1.0))
    direction_skill = 2.0 * direction_accuracy - 1.0
    predictive_score = (
        0.35 * rank_ic + 0.25 * pearson + 0.20 * direction_skill + 0.20 * rmse_skill
    )

    return {
        "mae": mae,
        "median_absolute_error": median_ae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
        "return_correlation": pearson,
        "rank_information_coefficient": rank_ic,
        "prediction_std": float(np.std(predicted_values)),
        "prediction_mean": float(np.mean(predicted_values)),
        "normalized_mae_vs_zero": float(normalised_mae),
        "normalized_rmse_vs_zero": float(normalised_rmse),
        "rmse_skill_vs_zero": rmse_skill,
        "predictive_score": float(predictive_score),
    }


# ---------------------------------------------------------------------------
# Cross-sectional evaluation
#
# A pooled correlation over a stacked panel mixes two very different questions:
# "did the market go up?" and "which stock beat which?". Only the second is what
# a stock-selection model can actually learn, and it is measured per date.
# ---------------------------------------------------------------------------


def cross_sectional_metrics(
    dates: Sequence[Any],
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    horizon: int = 1,
    minimum_names_per_date: int = 5,
    quantile: float = 0.2,
) -> dict:
    """
    Per-date information coefficient statistics and a long/short quantile spread.

    ``mean_ic`` is the average daily Spearman rank correlation between the
    prediction and the realised return. ``icir`` is its information ratio,
    annualised with the number of independent (non-overlapping) periods per
    year, and ``ic_t_statistic`` tests whether the mean IC differs from zero.
    """
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(list(dates))),
            "true_return": np.asarray(true_return, dtype=float),
            "predicted_return": np.asarray(predicted_return, dtype=float),
        }
    )

    daily_ic: list[float] = []
    daily_spread: list[float] = []
    daily_top: list[float] = []
    daily_bottom: list[float] = []

    for _, group in frame.groupby("date"):
        if len(group) < int(minimum_names_per_date):
            continue
        predictions = group["predicted_return"].to_numpy()
        realised = group["true_return"].to_numpy()
        if np.std(predictions) == 0:
            continue
        daily_ic.append(_safe_rank_correlation(realised, predictions))

        count = max(1, int(round(len(group) * float(quantile))))
        order = np.argsort(predictions)
        bottom = float(np.mean(realised[order[:count]]))
        top = float(np.mean(realised[order[-count:]]))
        daily_top.append(top)
        daily_bottom.append(bottom)
        daily_spread.append(top - bottom)

    if not daily_ic:
        return {
            "mean_ic": 0.0,
            "median_ic": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "ic_t_statistic": 0.0,
            "ic_p_value": 1.0,
            "ic_positive_rate": 0.0,
            "evaluated_dates": 0,
            "long_short_spread_per_period": 0.0,
            "long_short_spread_annualised": 0.0,
            "top_quantile_return": 0.0,
            "bottom_quantile_return": 0.0,
        }

    ic_values = np.asarray(daily_ic, dtype=float)
    spread_values = np.asarray(daily_spread, dtype=float)
    ic_mean = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0

    # Overlapping daily observations of an h-day label are not independent, so
    # annualisation uses 252/h effective periods rather than 252.
    periods_per_year = 252.0 / max(1, int(horizon))
    icir = float(ic_mean / ic_std * np.sqrt(periods_per_year)) if ic_std > 0 else 0.0

    if len(ic_values) > 1 and ic_std > 0:
        # Effective sample size discounts the overlap of adjacent forward windows.
        effective_n = max(2.0, len(ic_values) / max(1, int(horizon)))
        t_statistic = float(ic_mean / (ic_std / np.sqrt(effective_n)))
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=effective_n - 1))
    else:
        t_statistic, p_value = 0.0, 1.0

    return {
        "mean_ic": ic_mean,
        "median_ic": float(np.median(ic_values)),
        "ic_std": ic_std,
        "icir": icir,
        "ic_t_statistic": t_statistic,
        "ic_p_value": p_value,
        "ic_positive_rate": float(np.mean(ic_values > 0)),
        "evaluated_dates": int(len(ic_values)),
        "long_short_spread_per_period": float(np.mean(spread_values)),
        "long_short_spread_annualised": float(np.mean(spread_values) * periods_per_year),
        "top_quantile_return": float(np.mean(daily_top)),
        "bottom_quantile_return": float(np.mean(daily_bottom)),
    }


def full_metrics(
    dates: Sequence[Any],
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    horizon: int = 1,
) -> dict:
    """Point-forecast metrics plus the cross-sectional block, in one dictionary."""
    metrics = regression_metrics(true_return, predicted_return)
    cross_sectional = cross_sectional_metrics(dates, true_return, predicted_return, horizon=horizon)
    metrics.update({f"cross_sectional_{key}": value for key, value in cross_sectional.items()})
    # Convenience aliases usable directly as an early-stopping metric name.
    metrics["cross_sectional_ic"] = cross_sectional["mean_ic"]
    metrics["cross_sectional_icir"] = cross_sectional["icir"]
    return metrics


def fit_return_calibration(true_return: np.ndarray, predicted_return: np.ndarray) -> dict:
    """
    Fit a conservative affine calibration of prediction magnitude on validation data.

    Guard rails matter here: an unconstrained least-squares fit on a low-signal
    panel can return a near-zero or negative slope, which collapses the model to
    a constant while *improving* MAE. Such a fit is rejected outright.
    """
    true_values = np.asarray(true_return, dtype=float)
    predicted_values = np.asarray(predicted_return, dtype=float)
    identity = {
        "slope": 1.0,
        "intercept": 0.0,
        "enabled": False,
        "reason": "identity calibration retained",
    }
    if len(true_values) < 100 or np.std(predicted_values) < 1e-12:
        return {**identity, "reason": "not enough validation signal to calibrate"}

    design = np.column_stack([predicted_values, np.ones(len(predicted_values))])
    slope, intercept = np.linalg.lstsq(design, true_values, rcond=None)[0]

    # A calibration is only a magnitude correction. It must never flip or erase
    # the ranking the model produced.
    if not np.isfinite(slope) or slope < 0.25:
        return {
            **identity,
            "reason": f"rejected degenerate slope {float(slope):.4f} (would flatten predictions)",
        }

    slope = float(np.clip(slope, 0.25, 3.0))
    intercept_limit = max(0.005, float(np.quantile(np.abs(true_values), 0.50)))
    intercept = float(np.clip(intercept, -intercept_limit, intercept_limit))

    candidate = intercept + slope * predicted_values
    before = regression_metrics(true_values, predicted_values)
    after = regression_metrics(true_values, candidate)
    correlation_kept = after["return_correlation"] >= before["return_correlation"] - 1e-9
    if after["mae"] <= before["mae"] and correlation_kept:
        return {
            "slope": slope,
            "intercept": intercept,
            "enabled": True,
            "validation_mae_before": before["mae"],
            "validation_mae_after": after["mae"],
            "reason": "affine magnitude calibration improved validation MAE",
        }
    return {**identity, "reason": "calibration did not improve validation MAE"}


def baseline_predictions(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    horizon: int,
    target_column: str = TOTAL_RETURN_COLUMN,
) -> dict[str, np.ndarray]:
    """Transparent reference forecasts evaluated on exactly the same rows."""
    row_count = len(evaluation_df)
    train_mean = float(train_df[target_column].mean())
    ticker_means = train_df.groupby("Ticker")[target_column].mean()
    ticker_mean_prediction = (
        evaluation_df["Ticker"].map(ticker_means).fillna(train_mean).astype(float).to_numpy()
    )

    for candidate in (f"return_{horizon}d", "return_60d", "return_20d", "return_5d"):
        if candidate in evaluation_df.columns:
            momentum = evaluation_df[candidate].astype(float).to_numpy()
            break
    else:
        momentum = np.zeros(row_count, dtype=float)

    baselines = {
        "zero_return": np.zeros(row_count, dtype=float),
        "historical_mean": np.full(row_count, train_mean, dtype=float),
        "ticker_historical_mean": ticker_mean_prediction,
        "momentum": momentum,
        "reversal": -momentum,
    }

    if target_column == EXCESS_RETURN_COLUMN and "excess_return_20d" in evaluation_df.columns:
        baselines["excess_momentum"] = (
            evaluation_df["excess_return_20d"].astype(float).to_numpy()
        )
    if target_column == TOTAL_RETURN_COLUMN and BENCHMARK_RETURN_COLUMN in train_df.columns:
        drift = estimate_market_drift(train_df, {"mode": "market_excess"})["market_drift"]
        baselines["market_drift"] = np.full(row_count, drift, dtype=float)

    return baselines


def evaluate_baselines(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    horizon: int,
    target_column: str = TOTAL_RETURN_COLUMN,
) -> dict[str, dict]:
    true_values = evaluation_df[target_column].astype(float).to_numpy()
    dates = evaluation_df.index
    return {
        name: full_metrics(dates, true_values, values, horizon=horizon)
        for name, values in baseline_predictions(
            train_df, evaluation_df, horizon, target_column
        ).items()
    }


def chronological_block_metrics(
    metadata: list[dict[str, Any]],
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    blocks: int = 4,
    horizon: int = 1,
) -> list[dict]:
    """Expose consistency across consecutive out-of-sample market regimes."""
    frame = pd.DataFrame(metadata)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["true_return"] = np.asarray(true_return, dtype=float)
    frame["predicted_return"] = np.asarray(predicted_return, dtype=float)
    unique_dates = np.asarray(sorted(frame["date"].unique()))
    results = []
    for index, dates in enumerate(np.array_split(unique_dates, max(1, int(blocks))), start=1):
        if len(dates) == 0:
            continue
        subset = frame[frame["date"].isin(dates)]
        results.append(
            {
                "block": index,
                "start_date": str(pd.Timestamp(dates[0]).date()),
                "end_date": str(pd.Timestamp(dates[-1]).date()),
                "sample_size": int(len(subset)),
                "metrics": full_metrics(
                    subset["date"],
                    subset["true_return"].to_numpy(),
                    subset["predicted_return"].to_numpy(),
                    horizon=horizon,
                ),
            }
        )
    return results


def summarise_folds(fold_metrics: list[dict], keys: Sequence[str]) -> dict:
    """Aggregate walk-forward fold metrics into mean / std / min / max."""
    summary: dict[str, dict] = {}
    for key in keys:
        values = [
            float(fold[key])
            for fold in fold_metrics
            if key in fold and fold[key] is not None and np.isfinite(float(fold[key]))
        ]
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        summary[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
            "folds": int(len(array)),
        }
    return summary
