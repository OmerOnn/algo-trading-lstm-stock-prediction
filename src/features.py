from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators manually to avoid dependency issues."""
    out = df.copy()

    close = out["Adj Close"].astype(float)
    # Put OHLC values on the same split/dividend-adjusted scale as the target.
    adjustment_factor = close / out["Close"].astype(float).replace(0, np.nan)
    high = out["High"].astype(float) * adjustment_factor
    low = out["Low"].astype(float) * adjustment_factor
    volume = out["Volume"].astype(float)

    out["return_1d"] = close.pct_change()
    out["log_return_1d"] = np.log(close / close.shift(1))
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)
    out["return_20d"] = close.pct_change(20)
    out["return_60d"] = close.pct_change(60)
    out["return_120d"] = close.pct_change(120)

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
    out["macd_ratio"] = out["macd"] / close
    out["macd_signal_ratio"] = out["macd_signal"] / close
    out["macd_hist_ratio"] = out["macd_hist"] / close

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
    out["downside_volatility_20d"] = (
        out["return_1d"].clip(upper=0).rolling(20).std()
    )
    out["return_to_volatility_20d"] = out["return_20d"] / (
        out["volatility_20d"] * np.sqrt(20)
    )
    out["return_to_volatility_60d"] = out["return_60d"] / (
        out["volatility_60d"] * np.sqrt(60)
    )
    volume_mean = volume.rolling(60).mean()
    volume_std = volume.rolling(60).std()
    out["volume_zscore_60d"] = (volume - volume_mean) / volume_std

    return out


def add_benchmark_features(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Add market-wide benchmark return features, usually SPY."""
    out = stock_df.copy()
    if benchmark_df is None or benchmark_df.empty:
        return out

    bench = benchmark_df.copy()
    bench_close = bench["Adj Close"].astype(float)
    bench_features = pd.DataFrame(index=bench.index)
    # Kept as a helper column (excluded from the feature set) so forward
    # benchmark returns can be built alongside the stock label.
    bench_features["benchmark_close"] = bench_close
    bench_features["benchmark_return_1d"] = bench_close.pct_change()
    bench_features["benchmark_return_5d"] = bench_close.pct_change(5)
    bench_features["benchmark_return_20d"] = bench_close.pct_change(20)
    bench_features["benchmark_return_60d"] = bench_close.pct_change(60)
    bench_features["benchmark_volatility_20d"] = bench_features["benchmark_return_1d"].rolling(20).std()
    out = out.join(bench_features, how="left")
    out["excess_return_1d"] = out["return_1d"] - out["benchmark_return_1d"]
    out["excess_return_5d"] = out["return_5d"] - out["benchmark_return_5d"]
    out["excess_return_20d"] = out["return_20d"] - out["benchmark_return_20d"]
    rolling_covariance = out["return_1d"].rolling(60).cov(out["benchmark_return_1d"])
    rolling_variance = out["benchmark_return_1d"].rolling(60).var()
    out["market_beta_60d"] = rolling_covariance / rolling_variance
    out["idiosyncratic_return_1d"] = (
        out["return_1d"] - out["market_beta_60d"] * out["benchmark_return_1d"]
    )
    out["idiosyncratic_volatility_20d"] = out["idiosyncratic_return_1d"].rolling(20).std()
    return out


def add_macro_features(stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add alternative / macro market features to the stock dataframe.

    These features are not stock-specific, but they describe the broader market
    environment, for example volatility, interest rates, and dollar strength.
    """
    out = stock_df.copy()

    if macro_df is None or macro_df.empty:
        return out

    out = out.join(macro_df, how="left")
    macro_columns = list(macro_df.columns)
    out[macro_columns] = out[macro_columns].ffill()
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


# Features whose informative content is their position relative to the stock's
# own recent history rather than their absolute level. A trailing z-score makes
# them comparable across stocks and adaptive to volatility regimes, which a
# single global scaler fitted once on the training period cannot do.
REGIME_NORMALIZED_FEATURES = (
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_200",
    "macd_hist_ratio",
    "atr_ratio_14",
    "bb_width_20",
    "volatility_20d",
    "volatility_60d",
    "volume_ratio_20",
    "excess_return_5d",
    "excess_return_20d",
    "idiosyncratic_volatility_20d",
)


def add_regime_normalized_features(
    df: pd.DataFrame,
    window: int = 252,
    columns: tuple[str, ...] = REGIME_NORMALIZED_FEATURES,
    clip: float = 5.0,
) -> pd.DataFrame:
    """
    Add trailing z-scores of selected features, computed per ticker.

    Only past observations enter each z-score, so the transformation is free of
    look-ahead. It is applied per ticker before the panel is concatenated.
    """
    out = df.copy()
    window = max(20, int(window))
    minimum_periods = max(20, window // 4)

    for column in columns:
        if column not in out.columns:
            continue
        series = out[column].astype(float)
        rolling = series.rolling(window, min_periods=minimum_periods)
        mean = rolling.mean()
        std = rolling.std()
        z_score = (series - mean) / std.replace(0.0, np.nan)
        out[f"{column}_z{window}"] = z_score.clip(-clip, clip)

    return out


def add_labels(
    df: pd.DataFrame,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """
    Create the forward return labels.

    Two labels are produced: the stock's own forward return and the benchmark's
    forward return over the same window. Their difference is the excess return
    the model is trained to predict.
    """
    out = df.copy()
    close = out["Adj Close"].astype(float)
    out["future_return"] = close.shift(-horizon) / close - 1.0

    if "benchmark_close" in out.columns:
        benchmark_close = out["benchmark_close"].astype(float)
        out["benchmark_future_return"] = benchmark_close.shift(-horizon) / benchmark_close - 1.0

    return out


def is_market_wide_feature(column: str) -> bool:
    """
    True for features whose value is identical for every ticker on a given date.

    Benchmark and macro columns describe the market state, not the stock. When
    the target is market-excess return they cannot contribute anything to a
    cross-sectional ranking -- but they do give the model a way to identify
    *which date* a sequence came from, and therefore to memorise that date's
    cross-sectional noise. Excluding them removes the memorisation channel
    without removing any usable signal.
    """
    return column.startswith("benchmark_") or column.startswith("macro_")


def get_feature_columns(df: pd.DataFrame, exclude_market_wide: bool = False) -> list[str]:
    excluded = {
        "Ticker",
        "future_return",
        "future_excess_return",
        "benchmark_future_return",
        "benchmark_close",
        "model_target",
        "target_scale",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        # Absolute price/volume levels are non-stationary and let a pooled model
        # identify tickers by scale instead of learning transferable relationships.
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_20",
        "ema_50",
        "ema_200",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "bb_upper_20",
        "bb_lower_20",
        "volume_sma_20",
        # Snapshot earnings values are not guaranteed point-in-time historical data.
        "eps_estimate",
        "reported_eps",
        "eps_surprise_pct",
        "days_to_earnings",
        "is_near_earnings",
    }
    columns = [
        col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if exclude_market_wide:
        columns = [col for col in columns if not is_market_wide_feature(col)]
    return columns
