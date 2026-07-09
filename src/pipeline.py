from __future__ import annotations

import pandas as pd

from src.data_download import download_earnings_data, download_macro_data, download_price_data
from src.features import (
    add_benchmark_features,
    add_earnings_features,
    add_labels,
    add_macro_features,
    add_technical_indicators,
    get_feature_columns,
)


def build_dataset_for_tickers(
    tickers: list[str],
    benchmark_ticker: str,
    start_date: str,
    end_date: str | None,
    prediction_horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    macro_tickers: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download data, build features, and create labels for multiple tickers."""
    try:
        benchmark_df = download_price_data(benchmark_ticker, start_date, end_date)
    except Exception as exc:
        print(f"Warning: benchmark data download failed for {benchmark_ticker}: {exc}")
        benchmark_df = pd.DataFrame()
    macro_df = download_macro_data(macro_tickers, start_date, end_date)

    all_frames = []
    for ticker in tickers:
        print(f"Building dataset for {ticker}...")
        try:
            price_df = download_price_data(ticker, start_date, end_date)
        except Exception as exc:
            print(f"Warning: price data download failed for {ticker}: {exc}")
            continue
        earnings_df = download_earnings_data(ticker)

        df = add_technical_indicators(price_df)
        df = add_benchmark_features(df, benchmark_df)
        df = add_macro_features(df, macro_df)
        df = add_earnings_features(df, earnings_df)
        df = add_labels(df, prediction_horizon, buy_threshold, sell_threshold)
        df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        all_frames.append(df)

    full_df = pd.concat(all_frames).sort_index()
    feature_columns = get_feature_columns(full_df)

    # Keep only rows that are usable for training.
    full_df = full_df.dropna(subset=feature_columns + ["future_return", "signal_label"]).copy()
    full_df[feature_columns] = full_df[feature_columns].astype(float)
    full_df["signal_label"] = full_df["signal_label"].astype(int)
    full_df["future_return"] = full_df["future_return"].astype(float)

    return full_df, feature_columns
