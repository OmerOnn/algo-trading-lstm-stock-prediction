from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_download import download_earnings_data, download_macro_data, download_price_data
from src.features import (
    add_benchmark_features,
    add_earnings_features,
    add_labels,
    add_macro_features,
    add_regime_normalized_features,
    add_technical_indicators,
    get_feature_columns,
    is_market_wide_feature,
)

# Version 4 adds benchmark forward returns (for the market-excess target) and
# per-ticker trailing z-scored features. Older caches are rebuilt automatically.
DATASET_SCHEMA_VERSION = 4


def save_dataset_cache(
    df: pd.DataFrame,
    feature_columns: list[str],
    cache_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save a processed supervised dataset and its feature-column contract."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        # Parquet keeps dtypes and is roughly an order of magnitude smaller and
        # faster than CSV for a panel of this size.
        df.to_parquet(path)
    else:
        df.to_csv(path)

    metadata_path = path.with_suffix(".features.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        payload = {"feature_columns": feature_columns}
        if metadata:
            payload.update(metadata)
        json.dump(payload, file, indent=2)


def load_dataset_cache(cache_path: str | Path) -> tuple[pd.DataFrame, list[str], dict]:
    """Load a cached processed dataset built by save_dataset_cache."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset cache does not exist: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = None

    metadata_path = path.with_suffix(".features.json")
    metadata: dict = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
            feature_columns = list(metadata["feature_columns"])
    else:
        feature_columns = get_feature_columns(df)

    return df, feature_columns, metadata


def build_or_load_dataset_for_tickers(
    tickers: list[str],
    benchmark_ticker: str,
    start_date: str,
    end_date: str | None,
    prediction_horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    macro_tickers: dict[str, str] | None = None,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    use_earnings_features: bool = False,
    regime_normalization_window: int = 252,
    exclude_market_wide_features: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the processed dataset from cache, or build and cache it.

    ``exclude_market_wide_features`` is applied to the returned feature list
    rather than to the stored panel, so the same cache serves both settings.
    """
    if cache_path and use_cache and not force_rebuild and Path(cache_path).exists():
        print(f"Loading processed dataset cache: {cache_path}")
        cached_df, cached_features, cached_metadata = load_dataset_cache(cache_path)
        expected_contract = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "prediction_horizon": int(prediction_horizon),
            "tickers": list(tickers),
            "benchmark_ticker": str(benchmark_ticker),
            "start_date": str(start_date),
            "end_date": end_date,
            "macro_tickers": macro_tickers or {},
            "use_earnings_features": bool(use_earnings_features),
            "regime_normalization_window": int(regime_normalization_window),
        }
        if all(cached_metadata.get(key) == value for key, value in expected_contract.items()):
            return cached_df, _select_features(cached_features, exclude_market_wide_features)
        print("Cached dataset contract does not match the current data/feature configuration. Rebuilding cache.")

    full_df, feature_columns = build_dataset_for_tickers(
        tickers=tickers,
        benchmark_ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
        prediction_horizon=prediction_horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        macro_tickers=macro_tickers,
        use_earnings_features=use_earnings_features,
        regime_normalization_window=regime_normalization_window,
    )

    if cache_path and use_cache:
        print(f"Saving processed dataset cache: {cache_path}")
        save_dataset_cache(
            full_df,
            feature_columns,
            cache_path,
            metadata={
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "prediction_horizon": int(prediction_horizon),
                "buy_threshold": float(buy_threshold),
                "sell_threshold": float(sell_threshold),
                "tickers": list(tickers),
                "benchmark_ticker": str(benchmark_ticker),
                "start_date": str(start_date),
                "end_date": end_date,
                "macro_tickers": macro_tickers or {},
                "use_earnings_features": bool(use_earnings_features),
                "regime_normalization_window": int(regime_normalization_window),
            },
        )

    return full_df, _select_features(feature_columns, exclude_market_wide_features)


def _select_features(feature_columns: list[str], exclude_market_wide: bool) -> list[str]:
    if not exclude_market_wide:
        return list(feature_columns)
    return [column for column in feature_columns if not is_market_wide_feature(column)]


def build_dataset_for_tickers(
    tickers: list[str],
    benchmark_ticker: str,
    start_date: str,
    end_date: str | None,
    prediction_horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    macro_tickers: dict[str, str] | None = None,
    use_earnings_features: bool = False,
    regime_normalization_window: int = 252,
) -> tuple[pd.DataFrame, list[str]]:
    """Download data, build features, and create labels for multiple tickers."""
    try:
        benchmark_df = download_price_data(benchmark_ticker, start_date, end_date)
    except Exception as exc:
        print(f"Warning: benchmark data download failed for {benchmark_ticker}: {exc}")
        benchmark_df = pd.DataFrame()

    macro_df = pd.DataFrame()
    if macro_tickers:
        print("Downloading macro / alternative market data...")
        macro_df = download_macro_data(macro_tickers, start_date, end_date)

    all_frames = []
    for ticker in tickers:
        print(f"Building dataset for {ticker}...")
        try:
            price_df = download_price_data(ticker, start_date, end_date)
        except Exception as exc:
            print(f"Warning: price data download failed for {ticker}: {exc}")
            continue
        df = add_technical_indicators(price_df)
        df = add_benchmark_features(df, benchmark_df)
        df = add_macro_features(df, macro_df)
        if use_earnings_features:
            # Disabled by default: the source is a present-day snapshot rather
            # than a point-in-time historical feed and can create hindsight bias.
            earnings_df = download_earnings_data(ticker)
            df = add_earnings_features(df, earnings_df)
        # Trailing z-scores are computed per ticker, before the panel is pooled.
        df = add_regime_normalized_features(df, window=int(regime_normalization_window))
        df = add_labels(df, prediction_horizon, buy_threshold, sell_threshold)
        df = df.replace([np.inf, -np.inf], np.nan)
        all_frames.append(df)

    if not all_frames:
        raise ValueError("No ticker data could be downloaded. Check connectivity and the ticker list.")

    full_df = pd.concat(all_frames).sort_index()
    feature_columns = get_feature_columns(full_df)

    label_columns = [
        column
        for column in ("future_return", "benchmark_future_return")
        if column in full_df.columns
    ]

    # Keep only rows that are usable for training.
    full_df = full_df.dropna(subset=feature_columns + label_columns).copy()
    full_df[feature_columns] = full_df[feature_columns].astype(float)
    for column in label_columns:
        full_df[column] = full_df[column].astype(float)

    return full_df, feature_columns
