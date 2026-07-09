from __future__ import annotations

import os
import shutil
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional

import certifi
import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
_PRICE_CACHE: dict[tuple[str, str, str | None], pd.DataFrame] = {}
_MACRO_CACHE: dict[tuple[tuple[tuple[str, str], ...], str, str | None], pd.DataFrame] = {}
_EARNINGS_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def _configure_download_environment() -> None:
    """Set UTF-8 / CA bundle environment variables for Windows paths and SSL downloads."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")

    cert_path = certifi.where()
    temp_cert_dir = Path("C:/temp/certifi")
    temp_cert_dir.mkdir(parents=True, exist_ok=True)
    ascii_cert_path = temp_cert_dir / "cacert.pem"
    if not ascii_cert_path.exists():
        shutil.copy2(cert_path, ascii_cert_path)

    os.environ["SSL_CERT_FILE"] = str(ascii_cert_path)
    os.environ["REQUESTS_CA_BUNDLE"] = str(ascii_cert_path)
    os.environ["CURL_CA_BUNDLE"] = str(ascii_cert_path)


_configure_download_environment()

import yfinance as yf


def download_price_data(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Download daily OHLCV data for one ticker from Yahoo Finance through yfinance."""
    cache_key = (str(ticker), str(start), str(end) if end is not None else None)
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key].copy(deep=True)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
            timeout=30,
        )
    except Exception as exc:
        raise ValueError(f"No price data returned for ticker: {ticker}: {exc}") from exc

    if df.empty:
        raise ValueError(f"No price data returned for ticker: {ticker}")

    # yfinance may return multi-index columns in some cases.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {missing}")

    df = df[REQUIRED_PRICE_COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df["Ticker"] = ticker
    _PRICE_CACHE[cache_key] = df.copy(deep=True)
    return df.copy(deep=True)


def download_macro_data(
    macro_tickers: dict[str, str] | list[tuple[str, str]] | None,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download macro/alternative market indicators and create simple return features."""
    if not macro_tickers:
        return pd.DataFrame()

    if isinstance(macro_tickers, dict):
        items = list(macro_tickers.items())
    else:
        items = list(macro_tickers)

    cache_key = (
        tuple((str(name), str(ticker)) for name, ticker in items),
        str(start),
        str(end) if end is not None else None,
    )
    if cache_key in _MACRO_CACHE:
        return _MACRO_CACHE[cache_key].copy(deep=True)

    frames: list[pd.DataFrame] = []
    for name, ticker in items:
        try:
            df = download_price_data(ticker, start, end)
        except Exception as exc:
            print(f"Warning: macro data download failed for {name} ({ticker}): {exc}")
            continue

        feature_frame = pd.DataFrame(index=df.index)
        close = df["Adj Close"].astype(float)
        returns_1d = close.pct_change()
        feature_frame[f"macro_{name}_close"] = close
        feature_frame[f"macro_{name}_return_1d"] = returns_1d
        feature_frame[f"macro_{name}_return_5d"] = close.pct_change(5)
        feature_frame[f"macro_{name}_volatility_20d"] = returns_1d.rolling(20).std()
        frames.append(feature_frame)

    if not frames:
        return pd.DataFrame()

    macro_df = pd.concat(frames, axis=1).sort_index()
    macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
    macro_df = macro_df.ffill()
    _MACRO_CACHE[cache_key] = macro_df.copy(deep=True)
    return macro_df.copy(deep=True)


def download_earnings_data(ticker: str, limit: int = 100) -> pd.DataFrame:
    """
    Download earnings dates and earnings surprise fields when available.

    Note: yfinance earnings data availability can vary by ticker and by time.
    The rest of the project handles missing earnings data safely.
    """
    cache_key = (str(ticker), int(limit))
    if cache_key in _EARNINGS_CACHE:
        return _EARNINGS_CACHE[cache_key].copy(deep=True)

    try:
        stock = yf.Ticker(ticker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            earnings = stock.get_earnings_dates(limit=limit)
    except Exception:
        earnings = None

    if earnings is None or len(earnings) == 0:
        empty = pd.DataFrame(columns=["Earnings Date", "EPS Estimate", "Reported EPS", "Surprise(%)"])
        _EARNINGS_CACHE[cache_key] = empty.copy(deep=True)
        return empty.copy(deep=True)

    earnings = earnings.reset_index()

    # yfinance usually names the date column either "Earnings Date" or keeps it as the index name.
    if "Earnings Date" not in earnings.columns:
        earnings.rename(columns={earnings.columns[0]: "Earnings Date"}, inplace=True)

    earnings["Earnings Date"] = pd.to_datetime(earnings["Earnings Date"], errors="coerce").dt.tz_localize(None)
    earnings.dropna(subset=["Earnings Date"], inplace=True)
    earnings.sort_values("Earnings Date", inplace=True)
    earnings["Ticker"] = ticker
    _EARNINGS_CACHE[cache_key] = earnings.copy(deep=True)
    return earnings.copy(deep=True)


def preload_training_data(
    tickers: list[str],
    benchmark_ticker: str,
    start: str,
    end: Optional[str] = None,
    macro_tickers: dict[str, str] | list[tuple[str, str]] | None = None,
    earnings_limit: int = 100,
) -> None:
    """Preload shared raw datasets once so repeated model/horizon training reuses them in memory."""
    print("Preloading shared market data into memory cache...")
    all_price_tickers = [benchmark_ticker, *tickers]
    for ticker in all_price_tickers:
        try:
            download_price_data(ticker, start, end)
        except Exception as exc:
            print(f"Warning: preload skipped price data for {ticker}: {exc}")
    if macro_tickers:
        try:
            download_macro_data(macro_tickers, start, end)
        except Exception as exc:
            print(f"Warning: preload skipped macro data: {exc}")
    for ticker in tickers:
        try:
            download_earnings_data(ticker, limit=earnings_limit)
        except Exception as exc:
            print(f"Warning: preload skipped earnings data for {ticker}: {exc}")
    print("Shared market data cache is ready.")
