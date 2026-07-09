from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Optional

import certifi
import pandas as pd


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


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
    return df


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

    return pd.concat(frames, axis=1).sort_index()


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
