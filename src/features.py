from __future__ import annotations

import numpy as np
import pandas as pd


CLASS_TO_ID = {"SELL": 0, "HOLD": 1, "BUY": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators manually to avoid dependency issues."""
    out = df.copy()

    close = out["Adj Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)

    out["return_1d"] = close.pct_change()
    out["log_return_1d"] = np.log(close / close.shift(1))
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)
    out["return_20d"] = close.pct_change(20)
    out["return_60d"] = close.pct_change(60)

    for window in [20, 50, 200]:
        out[f"sma_{window}"] = close.rolling(window).mean()
        out[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
        out[f"price_to_sma_{window}"] = close / out[f"sma_{window}"] - 1.0

    out["sma_20_above_50"] = (out["sma_20"] > out["sma_50"]).astype(int)
    out["sma_50_above_200"] = (out["sma_50"] > out["sma_200"]).astype(int)
    out["sma_20_50_cross"] = out["sma_20_above_50"].diff().fillna(0)
    out["sma_50_200_cross"] = out["sma_50_above_200"].diff().fillna(0)

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # ATR 14
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean()
    out["atr_ratio_14"] = out["atr_14"] / close

    # Bollinger Bands 20
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    out["bb_upper_20"] = rolling_mean + 2 * rolling_std
    out["bb_lower_20"] = rolling_mean - 2 * rolling_std
    out["bb_position_20"] = (close - out["bb_lower_20"]) / (out["bb_upper_20"] - out["bb_lower_20"])
    out["bb_width_20"] = (out["bb_upper_20"] - out["bb_lower_20"]) / rolling_mean

    # Volume features
    out["volume_sma_20"] = volume.rolling(20).mean()
    out["volume_ratio_20"] = volume / out["volume_sma_20"]
    out["volume_change_5d"] = volume.pct_change(5)
    out["volume_change_20d"] = volume.pct_change(20)

    # Realized volatility
    out["volatility_10d"] = out["return_1d"].rolling(10).std()
    out["volatility_20d"] = out["return_1d"].rolling(20).std()
    out["volatility_60d"] = out["return_1d"].rolling(60).std()

    return out


def add_benchmark_features(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Add market-wide benchmark return features, usually SPY."""
    out = stock_df.copy()
    if benchmark_df is None or benchmark_df.empty:
        return out

    bench = benchmark_df.copy()
    if "Adj Close" not in bench.columns:
        return out

    bench_close = bench["Adj Close"].astype(float)
    bench_features = pd.DataFrame(index=bench.index)
    bench_features["benchmark_return_1d"] = bench_close.pct_change()
    bench_features["benchmark_return_5d"] = bench_close.pct_change(5)
    bench_features["benchmark_volatility_20d"] = bench_features["benchmark_return_1d"].rolling(20).std()
    out = out.join(bench_features, how="left")
    return out


def add_macro_features(stock_df: pd.DataFrame, macro_df: pd.DataFrame | None) -> pd.DataFrame:
    """Join macro/alternative features by date and forward-fill missing values."""
    out = stock_df.copy()
    if macro_df is None or macro_df.empty:
        return out

    macro = macro_df.copy().sort_index()
    out = out.join(macro, how="left")
    for column in macro.columns:
        out[column] = out[column].ffill()
    return out


def add_earnings_features(price_df: pd.DataFrame, earnings_df: pd.DataFrame) -> pd.DataFrame:
    """Add earnings proximity and earnings surprise features to each trading day."""
    out = price_df.copy()
    out["is_earnings_day"] = 0
    out["is_near_earnings"] = 0
    out["days_since_earnings"] = 999.0
    out["days_to_earnings"] = 999.0
    out["eps_estimate"] = 0.0
    out["reported_eps"] = 0.0
    out["eps_surprise_pct"] = 0.0

    if earnings_df is None or earnings_df.empty:
        return out

    earnings = earnings_df.copy()
    earnings["Earnings Date"] = pd.to_datetime(earnings["Earnings Date"], errors="coerce").dt.normalize()
    earnings.dropna(subset=["Earnings Date"], inplace=True)

    if earnings.empty:
        return out

    trading_dates = pd.Series(out.index.normalize(), index=out.index)
    earnings_dates = earnings["Earnings Date"].sort_values().drop_duplicates().to_list()

    for idx, current_date in trading_dates.items():
        deltas = np.array([(ed - current_date).days for ed in earnings_dates])
        if len(deltas) == 0:
            continue

        past_deltas = deltas[deltas <= 0]
        future_deltas = deltas[deltas >= 0]

        if len(past_deltas) > 0:
            out.at[idx, "days_since_earnings"] = float(abs(past_deltas.max()))
        if len(future_deltas) > 0:
            out.at[idx, "days_to_earnings"] = float(future_deltas.min())

        if np.any(deltas == 0):
            out.at[idx, "is_earnings_day"] = 1

        if np.any(np.abs(deltas) <= 2):
            out.at[idx, "is_near_earnings"] = 1

    # Attach the most recently known earnings values after each earnings date.
    known = earnings.set_index("Earnings Date").sort_index()
    known = known.rename(
        columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "reported_eps",
            "Surprise(%)": "eps_surprise_pct",
        }
    )
    known = known[[col for col in ["eps_estimate", "reported_eps", "eps_surprise_pct"] if col in known.columns]]

    if not known.empty:
        merged = pd.merge_asof(
            out.reset_index().rename(columns={"index": "Date"}).sort_values("Date"),
            known.reset_index().rename(columns={"Earnings Date": "Date"}).sort_values("Date"),
            on="Date",
            direction="backward",
        ).set_index("Date")

        for col in ["eps_estimate", "reported_eps", "eps_surprise_pct"]:
            if f"{col}_y" in merged.columns:
                out[col] = merged[f"{col}_y"].fillna(0.0).values
            elif col in merged.columns:
                out[col] = merged[col].fillna(0.0).values

    return out


def add_labels(
    df: pd.DataFrame,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """Create future return regression target and BUY/HOLD/SELL classification target."""
    out = df.copy()
    close = out["Adj Close"].astype(float)
    out["future_return"] = close.shift(-horizon) / close - 1.0

    conditions = [
        out["future_return"] <= sell_threshold,
        out["future_return"] >= buy_threshold,
    ]
    choices = [CLASS_TO_ID["SELL"], CLASS_TO_ID["BUY"]]
    out["signal_label"] = np.select(conditions, choices, default=CLASS_TO_ID["HOLD"])
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "Ticker",
        "signal_label",
        "future_return",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]
