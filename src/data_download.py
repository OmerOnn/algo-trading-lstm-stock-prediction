from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import yfinance as yf


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_price_data(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Download daily OHLCV data for one ticker from Yahoo Finance through yfinance."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

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
    return df


def download_earnings_data(ticker: str, limit: int = 100) -> pd.DataFrame:
    """
    Download earnings dates and earnings surprise fields when available.

    Note: yfinance earnings data availability can vary by ticker and by time.
    The rest of the project handles missing earnings data safely.
    """
    try:
        stock = yf.Ticker(ticker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            earnings = stock.get_earnings_dates(limit=limit)
    except Exception:
        earnings = None

    if earnings is None or len(earnings) == 0:
        return pd.DataFrame(columns=["Earnings Date", "EPS Estimate", "Reported EPS", "Surprise(%)"])

    earnings = earnings.reset_index()

    # yfinance usually names the date column either "Earnings Date" or keeps it as the index name.
    if "Earnings Date" not in earnings.columns:
        earnings.rename(columns={earnings.columns[0]: "Earnings Date"}, inplace=True)

    earnings["Earnings Date"] = pd.to_datetime(earnings["Earnings Date"], errors="coerce").dt.tz_localize(None)
    earnings.dropna(subset=["Earnings Date"], inplace=True)
    earnings.sort_values("Earnings Date", inplace=True)
    earnings["Ticker"] = ticker
    return earnings
