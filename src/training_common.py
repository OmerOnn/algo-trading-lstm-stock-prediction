"""
Orchestration helpers shared by ``train_lstm.py`` and ``train_xgboost.py``.

Everything here is model-family agnostic: configuration, horizon parsing, dataset
and target preparation, artifact paths, feature scaling, metric serialisation and
console reporting. The two trainers own only the parts that are genuinely
specific to their model family.

This module exists so neither trainer has to import the other. A trainer
importing another trainer means loading its argument parser, its torch setup and
its side effects just to reach a helper, and it makes the dependency direction
between the two families ambiguous.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.backtest import BacktestConfig
from src.decision import SIGNAL_LABELS, DecisionConfig
from src.pipeline import build_or_load_dataset_for_tickers
from src.regression import (
    add_model_target,
    resolve_target_config,
)


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = "configs/config.yaml"

# Metrics aggregated across walk-forward folds. Kept in one place so the LSTM,
# XGBoost and blended reports are summarised over an identical key set and are
# therefore directly comparable.
WALK_FORWARD_SUMMARY_KEYS = (
    "mae",
    "mse",
    "rmse",
    "r2",
    "direction_accuracy",
    "return_correlation",
    "cross_sectional_ic",
    "cross_sectional_icir",
    "cross_sectional_ic_t_statistic",
    "cross_sectional_long_short_spread_annualised",
    "mse_skill_vs_historical_mean",
    "mae_skill_vs_historical_mean",
)


# ---------------------------------------------------------------------------
# Configuration and horizons
# ---------------------------------------------------------------------------


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_horizons(
    config: dict,
    selected_horizon: int | None = None,
    selected_horizons: list[int] | None = None,
) -> list[int]:
    if selected_horizons is not None:
        return [int(h) for h in selected_horizons]
    if selected_horizon is not None:
        return [int(selected_horizon)]
    if "prediction_horizons" in config and config["prediction_horizons"]:
        return [int(h) for h in config["prediction_horizons"]]
    return [int(config.get("prediction_horizon", 21))]


def parse_horizon_list(raw_horizons: str | None) -> list[int] | None:
    if raw_horizons is None:
        return None
    items = [item.strip() for item in str(raw_horizons).split(",")]
    horizons = [int(item) for item in items if item]
    if not horizons:
        raise ValueError("At least one horizon must be provided in --horizons.")
    ordered_unique: list[int] = []
    for horizon in horizons:
        if horizon not in ordered_unique:
            ordered_unique.append(horizon)
    return ordered_unique


def set_seed(seed: int) -> None:
    """Seed every generator that can affect a fit, including torch when present."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset and target preparation
# ---------------------------------------------------------------------------


def load_horizon_dataset(
    config: dict,
    horizon: int,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Build or load the supervised panel and attach the training target.

    Returns the panel, the feature columns to train on, and the resolved target
    configuration. Both model families consume exactly this, which is what makes
    their metrics comparable.
    """
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_path = cache_dir / f"full_dataset_h{horizon}.parquet"

    full_df, feature_columns = build_or_load_dataset_for_tickers(
        tickers=config["tickers"],
        benchmark_ticker=config["benchmark_ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        prediction_horizon=horizon,
        buy_threshold=float(config["buy_threshold"]),
        sell_threshold=float(config["sell_threshold"]),
        macro_tickers=config.get("macro_tickers"),
        ticker_sectors=config.get("ticker_sectors"),
        cache_path=dataset_cache_path,
        use_cache=bool(config.get("use_dataset_cache", True)),
        force_rebuild=bool(config.get("force_rebuild_dataset_cache", False)),
        use_earnings_features=bool(config.get("use_earnings_features", False)),
        regime_normalization_window=int(config.get("regime_normalization_window", 252)),
        exclude_market_wide_features=bool(config.get("exclude_market_wide_features", True)),
        feature_blocklist=config.get("feature_blocklist"),
        minimum_sector_members=int(config.get("minimum_sector_members", 3)),
    )
    target_config = resolve_target_config(config.get("regression_target"))
    full_df = add_model_target(full_df, horizon, target_config)
    return full_df, feature_columns, target_config


def scale_features(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    scaler_kind: str = "robust",
    scaler_path: Path | None = None,
):
    """Fit the feature scaler on training rows only and optionally persist it."""
    scaler_kind = str(scaler_kind).lower().strip()
    if scaler_kind == "robust":
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
    elif scaler_kind == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("feature_scaler must be 'robust' or 'standard'")

    scaler.fit(train_df[feature_columns])
    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return scaler


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------


def artifact_path(base_path: str | Path, horizon: int) -> Path:
    path = Path(base_path)
    return ROOT / path.with_name(f"{path.stem}_h{horizon}{path.suffix}")


def copy_default_artifacts(config: dict, horizon: int, copies: Sequence[tuple[Path, str]]) -> None:
    """
    Mirror horizon-specific artifacts to the unsuffixed default paths.

    Only the configured default horizon is mirrored, so the unsuffixed files
    always describe one coherent model rather than whichever horizon ran last.
    """
    default_horizon = int(
        config.get("default_prediction_horizon", config.get("prediction_horizon", 21))
    )
    if horizon != default_horizon:
        return
    for source, config_key in copies:
        if config_key not in config:
            continue
        target = ROOT / config[config_key]
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).exists():
            shutil.copy2(source, target)


# ---------------------------------------------------------------------------
# Configuration objects shared by both trainers
# ---------------------------------------------------------------------------


def build_backtest_config(config: dict) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        allow_short=bool(config["allow_short"]),
        signal_threshold_multiplier=float(config.get("signal_threshold_multiplier", 1.0)),
        min_signal_edge=float(config.get("min_signal_edge", 0.0)),
    )


def build_base_decision_config(config: dict) -> DecisionConfig:
    decision_defaults = config.get("decision", {}) or {}
    return DecisionConfig(
        rule=str(decision_defaults.get("rule", "risk_adjusted")),
        allow_short=bool(config["allow_short"]),
        position_sizing=str(decision_defaults.get("position_sizing", "binary")),
        min_direction_probability=float(decision_defaults.get("min_direction_probability", 0.0)),
    )


# ---------------------------------------------------------------------------
# Metric serialisation and reporting
# ---------------------------------------------------------------------------


def strip_arrays(metrics: dict) -> dict:
    """Drop array payloads so a metrics dictionary is JSON-serialisable."""
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray) and key not in {"true_return", "predicted_return"}
    }


def metadata_records(df: pd.DataFrame) -> list[dict]:
    """``(ticker, date)`` records in the row order of a dated panel."""
    return [
        {"ticker": str(ticker), "date": str(pd.to_datetime(index).date())}
        for index, ticker in zip(df.index, df["Ticker"])
    ]


def derived_signal_metrics(signal_df: pd.DataFrame) -> dict:
    """Classification view of the *derived* signal, never a supervised target."""
    true_signal = signal_df["true_signal"].astype(str)
    predicted_signal = signal_df["predicted_signal"].astype(str)
    labels = list(SIGNAL_LABELS)
    return {
        "accuracy": float(accuracy_score(true_signal, predicted_signal)),
        "classification_report": classification_report(
            true_signal,
            predicted_signal,
            labels=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true_signal, predicted_signal, labels=labels).tolist(),
        "labels": labels,
    }


def print_metrics(metrics: dict) -> None:
    """Console summary covering magnitude skill and cross-sectional skill."""
    print(
        f"  MAE {metrics['mae']:.4f} | MSE {metrics['mse']:.6f} | RMSE {metrics['rmse']:.4f} | "
        f"R2 {metrics['r2']:+.4f}"
    )
    print(
        f"  direction {metrics['direction_accuracy']:.4f} | "
        f"corr {metrics['return_correlation']:+.4f} | "
        f"rank IC {metrics['rank_information_coefficient']:+.4f}"
    )
    print(
        f"  Cross-sectional IC {metrics['cross_sectional_mean_ic']:+.4f} | "
        f"ICIR {metrics['cross_sectional_icir']:+.2f} | "
        f"t-stat {metrics['cross_sectional_ic_t_statistic']:+.2f} | "
        f"IC>0 on {metrics['cross_sectional_ic_positive_rate']:.1%} of dates"
    )
    print(
        f"  Top-minus-bottom quintile spread per period: "
        f"{metrics['cross_sectional_long_short_spread_per_period'] * 100:+.2f}% "
        f"({metrics['cross_sectional_long_short_spread_annualised'] * 100:+.2f}% annualised)"
    )
    if "mse_skill_vs_historical_mean" in metrics:
        print(
            f"  Skill vs historical mean: MSE {metrics['mse_skill_vs_historical_mean']:+.4f} | "
            f"RMSE {metrics['rmse_skill_vs_historical_mean']:+.4f} | "
            f"MAE {metrics['mae_skill_vs_historical_mean']:+.4f}"
        )


def print_fold_summary(summary: dict[str, dict], keys: Sequence[str] | None = None) -> None:
    """Print mean / std / min / max for each aggregated walk-forward metric."""
    keys = keys or list(summary.keys())
    header = f"  {'metric':<48} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in keys:
        stats = summary.get(key)
        if not stats:
            continue
        print(
            f"  {key:<48} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} {stats['max']:>10.4f}"
        )


def align_total_returns(
    full_df: pd.DataFrame,
    metadata: list[dict],
    column: str = "future_return",
) -> np.ndarray:
    """
    Look up a per-row panel column for each ``(ticker, date)`` sequence target.

    Sequence datasets emit rows in their own order, so panel columns cannot be
    read positionally; they have to be joined back on the identifying pair.
    """
    lookup = full_df.reset_index()
    date_column = lookup.columns[0]
    keys = (
        lookup["Ticker"].astype(str)
        + "|"
        + pd.to_datetime(lookup[date_column]).dt.date.astype(str)
    )
    mapping = dict(zip(keys, lookup[column].astype(float)))
    wanted = [f"{meta['ticker']}|{meta['date']}" for meta in metadata]
    return np.asarray([mapping.get(key, np.nan) for key in wanted], dtype=float)


def json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays so ``json.dump`` cannot fail."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    return value
