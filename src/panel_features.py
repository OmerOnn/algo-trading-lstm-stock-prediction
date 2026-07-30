"""
Features that only exist once the per-ticker frames are pooled into a panel.

Three families live here, and the distinction between them drives how they are
allowed to be used:

1. **Sector features** — a stock's behaviour relative to its own sector rather
   than to the whole market. Sector composites are built from the universe
   itself (the equal-weight average of the sector's members on each date) rather
   than from sector ETFs. That is deliberate: XLC did not exist before June 2018
   and XLRE before October 2015, so an ETF-based sector return would be missing
   for a third of the sample and would silently delete every pre-2018 row for
   the communication-services names when incomplete rows are dropped. A
   composite of the actual universe is available from the first date, and it is
   the more precise benchmark for these specific stocks anyway.

2. **Market-state features** — breadth, dispersion, average correlation and
   volatility regime. These are identical for every ticker on a date, so they
   carry no cross-sectional ranking information *by themselves*, and worse, they
   let a sequence model recognise which date a window came from and memorise
   that date's noise. They are therefore prefixed ``marketstate_`` and excluded
   from the stock model's feature list, but they are **not** thrown away: the
   market-return model consumes them directly, and the interaction features
   below carry their information into the stock model in a form that varies
   across the cross-section.

3. **Cross-sectional ranks and regime interactions** — stock-specific by
   construction (a rank is relative to the other names on the same date; an
   interaction multiplies a stock-specific factor by a market-state scalar), so
   they survive the market-wide exclusion and are what let the market
   environment change how a stock-specific factor behaves.

Every value below is computed from information available strictly at or before
the row's own date. Cross-sectional statistics use contemporaneous *past*
returns, which is knowable in real time; nothing is shifted forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_STATE_PREFIX = "marketstate_"

# Features ranked cross-sectionally within each date. Chosen because a rank is
# more informative than a level for these: what matters is whether this stock has
# more momentum / more volatility / more turnover than its peers today, not the
# absolute number, which drifts with the regime.
CROSS_SECTIONAL_RANK_FEATURES = (
    "return_5d",
    "return_20d",
    "return_60d",
    "excess_return_20d",
    "residual_momentum_20d",
    "sector_relative_return_20d",
    "volatility_20d",
    "idiosyncratic_volatility_20d",
    "rsi_14",
    "price_to_sma_50",
    "market_beta_60d",
    "volume_ratio_20",
    "dollar_volume_zscore_60d",
)


def assign_sectors(
    df: pd.DataFrame,
    ticker_sectors: dict[str, str] | None,
) -> pd.DataFrame:
    """Attach a sector label per row. Unmapped tickers fall into ``Unclassified``."""
    out = df.copy()
    mapping = {str(k).upper(): str(v) for k, v in (ticker_sectors or {}).items()}
    out["sector"] = out["Ticker"].astype(str).str.upper().map(mapping).fillna("Unclassified")
    return out


def add_sector_features(
    df: pd.DataFrame,
    minimum_sector_members: int = 3,
    beta_window: int = 60,
) -> pd.DataFrame:
    """
    Add sector composite returns and each stock's relationship to its sector.

    A sector holding fewer than ``minimum_sector_members`` names on a date falls
    back to the whole-universe composite. Without that fallback a one-stock
    sector makes ``sector_relative_return`` identically zero and sector beta
    identically one, which is not a feature but a constant.
    """
    out = df.copy()
    if "sector" not in out.columns:
        out["sector"] = "Unclassified"

    # The panel index is a date with one row per ticker, so index labels repeat
    # ~100 times. Any per-ticker rolling result carrying that index back would be
    # aligned by label and silently scrambled. Everything below therefore works on
    # a unique positional index, sorted by (ticker, date), and the original index
    # is restored at the end.
    out["_date"] = pd.DatetimeIndex(out.index)
    out = out.sort_values(["Ticker", "_date"], kind="stable").reset_index(drop=True)

    # Equal-weight composite of the sector's members on each date, and of the
    # whole universe as the fallback.
    by_date_sector = out.groupby(["_date", "sector"])["return_1d"]
    sector_mean = by_date_sector.transform("mean")
    sector_size = by_date_sector.transform("count")
    universe_mean = out.groupby("_date")["return_1d"].transform("mean")

    thin_sector = sector_size < int(minimum_sector_members)
    out["sector_return_1d"] = np.where(thin_sector, universe_mean, sector_mean)
    out["sector_is_composite_fallback"] = thin_sector.astype(float)
    out["universe_return_1d"] = universe_mean

    def rolling_sum(column: str, window: int) -> pd.Series:
        # The groupby is re-created per call rather than cached: a cached
        # SeriesGroupBy is bound to the columns that existed when it was made,
        # and several of the columns below are derived from earlier ones.
        return out.groupby("Ticker")[column].transform(
            lambda series: series.astype(float).rolling(window, min_periods=window).sum()
        )

    # A sum of daily returns is the standard sector-composite convention and, at
    # these magnitudes, indistinguishable from compounding.
    out["sector_return_5d"] = rolling_sum("sector_return_1d", 5)
    out["sector_return_20d"] = rolling_sum("sector_return_1d", 20)
    out["sector_return_60d"] = rolling_sum("sector_return_1d", 60)
    out["sector_volatility_20d"] = out.groupby("Ticker")["sector_return_1d"].transform(
        lambda series: series.astype(float).rolling(20, min_periods=20).std()
    )

    # Sector-relative momentum: does this stock beat its own sector, not the
    # index? A semiconductor name up 8% while semis are up 10% is a laggard, and
    # a market-relative feature cannot see that.
    out["sector_relative_return_5d"] = out["return_5d"].astype(float) - out["sector_return_5d"]
    out["sector_relative_return_20d"] = out["return_20d"].astype(float) - out["sector_return_20d"]
    out["sector_relative_return_60d"] = out["return_60d"].astype(float) - out["sector_return_60d"]
    out["sector_relative_volatility_20d"] = out["volatility_20d"].astype(float) / out[
        "sector_volatility_20d"
    ].replace(0.0, np.nan)

    # Rolling sector beta, known at prediction time, plus the residual left over
    # once the sector move is removed.
    window = max(20, int(beta_window))
    minimum_periods = max(10, window // 2)

    # Beta from rolling moments rather than a grouped ``apply``: beta is
    # cov(stock, sector) / var(sector), and expressing both as population moments
    # lets the whole thing be built from ``transform`` calls that align
    # positionally. A grouped apply returning a Series would have to be realigned
    # by index, which is exactly the hazard this function is avoiding.
    out["_stock_x_sector"] = out["return_1d"].astype(float) * out["sector_return_1d"].astype(float)
    out["_sector_squared"] = out["sector_return_1d"].astype(float) ** 2

    def rolling_mean(column: str) -> pd.Series:
        return out.groupby("Ticker")[column].transform(
            lambda series: series.astype(float).rolling(window, min_periods=minimum_periods).mean()
        )

    mean_stock_x_sector = rolling_mean("_stock_x_sector")
    mean_sector_squared = rolling_mean("_sector_squared")
    mean_stock = rolling_mean("return_1d")
    mean_sector = rolling_mean("sector_return_1d")

    covariance = mean_stock_x_sector - mean_stock * mean_sector
    variance = mean_sector_squared - mean_sector**2
    out[f"sector_beta_{window}d"] = covariance / variance.replace(0.0, np.nan)
    out = out.drop(columns=["_stock_x_sector", "_sector_squared"])
    out["sector_residual_return_1d"] = out["return_1d"].astype(float) - out[
        f"sector_beta_{window}d"
    ] * out["sector_return_1d"]

    # Beta-neutral residual momentum: cumulative idiosyncratic return with the
    # market component already stripped out by market_beta_60d.
    if "idiosyncratic_return_1d" in out.columns:
        out["residual_momentum_20d"] = rolling_sum("idiosyncratic_return_1d", 20)
        out["residual_momentum_60d"] = rolling_sum("idiosyncratic_return_1d", 60)
    out["sector_residual_momentum_20d"] = rolling_sum("sector_residual_return_1d", 20)

    return out.set_index("_date").rename_axis(None).sort_index(kind="stable")


def add_market_state_features(
    df: pd.DataFrame,
    volatility_regime_window: int = 252,
) -> pd.DataFrame:
    """
    Add breadth, dispersion, average correlation and volatility-regime measures.

    All of these are constant across the cross-section on a given date, hence the
    ``marketstate_`` prefix that keeps them out of the stock model's direct
    feature list.
    """
    out = df.copy()
    out["_date"] = out.index

    # --- Breadth: how much of the universe is in an uptrend -------------------
    for window in (50, 200):
        column = f"price_to_sma_{window}"
        if column in out.columns:
            above = (out[column].astype(float) > 0).astype(float)
            out[f"{MARKET_STATE_PREFIX}pct_above_sma{window}"] = above.groupby(
                out["_date"]
            ).transform("mean")

    # --- Dispersion: how differently stocks are moving ------------------------
    out[f"{MARKET_STATE_PREFIX}return_dispersion_1d"] = out.groupby("_date")[
        "return_1d"
    ].transform("std")
    if "return_20d" in out.columns:
        out[f"{MARKET_STATE_PREFIX}return_dispersion_20d"] = out.groupby("_date")[
            "return_20d"
        ].transform("std")

    # --- Average pairwise correlation ----------------------------------------
    # Direct pairwise correlation over ~100 names on 5,000 dates is far too
    # expensive. The standard portfolio-variance identity gives the same quantity
    # in closed form for an equal-weight basket:
    #     var(basket) = w'Sigma w  =>  rho_bar = (var_basket - mean_var/N)
    #                                            / (mean_std^2 - mean_var/N)
    # with equal weights w = 1/N. It needs only per-stock trailing volatilities
    # and the trailing volatility of the equal-weight basket.
    if "volatility_20d" in out.columns:
        per_date = out.groupby("_date")
        mean_variance = per_date["volatility_20d"].transform(
            lambda series: float(np.nanmean(np.square(series.astype(float))))
        )
        mean_std = per_date["volatility_20d"].transform("mean")
        names = per_date["volatility_20d"].transform("count").clip(lower=2)

        basket = (
            out.groupby("_date")["universe_return_1d"].first()
            if "universe_return_1d" in out.columns
            else None
        )
        if basket is not None:
            basket_variance = basket.astype(float).rolling(20, min_periods=20).var()
            basket_variance = out["_date"].map(basket_variance)
            diversified = mean_variance / names
            denominator = (np.square(mean_std) - diversified).replace(0.0, np.nan)
            out[f"{MARKET_STATE_PREFIX}average_correlation"] = (
                (basket_variance - diversified) / denominator
            ).clip(-1.0, 1.0)

    # --- Volatility regime ---------------------------------------------------
    benchmark_volatility_column = (
        "benchmark_volatility_20d"
        if "benchmark_volatility_20d" in out.columns
        else f"{MARKET_STATE_PREFIX}return_dispersion_1d"
    )
    per_date_volatility = out.groupby("_date")[benchmark_volatility_column].first().astype(float)
    window = max(60, int(volatility_regime_window))
    trailing_median = per_date_volatility.rolling(window, min_periods=window // 4).median()
    trailing_std = per_date_volatility.rolling(window, min_periods=window // 4).std()

    # A z-score against the market's own trailing volatility distribution, so
    # "high volatility" means high for this market rather than high in absolute
    # terms, which drifts across two decades.
    regime_z = (per_date_volatility - trailing_median) / trailing_std.replace(0.0, np.nan)
    out[f"{MARKET_STATE_PREFIX}volatility_regime_z"] = out["_date"].map(regime_z).clip(-5.0, 5.0)
    out[f"{MARKET_STATE_PREFIX}high_volatility_regime"] = (
        out[f"{MARKET_STATE_PREFIX}volatility_regime_z"] > 0.5
    ).astype(float)

    short_volatility = per_date_volatility.rolling(20, min_periods=20).mean()
    long_volatility = per_date_volatility.rolling(60, min_periods=60).mean()
    out[f"{MARKET_STATE_PREFIX}volatility_ratio"] = out["_date"].map(
        short_volatility / long_volatility.replace(0.0, np.nan)
    )

    # --- Benchmark drawdown from its trailing peak ---------------------------
    if "benchmark_close" in out.columns:
        benchmark_close = out.groupby("_date")["benchmark_close"].first().astype(float)
        trailing_peak = benchmark_close.rolling(252, min_periods=60).max()
        out[f"{MARKET_STATE_PREFIX}benchmark_drawdown"] = out["_date"].map(
            benchmark_close / trailing_peak - 1.0
        )

    return out.drop(columns=["_date"])


def add_cross_sectional_ranks(
    df: pd.DataFrame,
    columns: tuple[str, ...] = CROSS_SECTIONAL_RANK_FEATURES,
) -> pd.DataFrame:
    """
    Add each stock's within-date percentile rank for selected features.

    Ranks are centred on zero and scaled to [-0.5, 0.5]. A rank is exactly the
    right representation for a cross-sectional model: it is scale-free, immune to
    the level drift that makes a raw momentum number mean different things in
    2009 and 2021, and robust to the outliers a fat-tailed panel produces.
    """
    out = df.copy()
    out["_date"] = out.index
    grouped = out.groupby("_date")
    for column in columns:
        if column not in out.columns:
            continue
        out[f"xs_rank_{column}"] = (
            grouped[column].rank(pct=True, method="average") - 0.5
        ).astype(float)
    return out.drop(columns=["_date"])


def add_regime_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiply stock-specific factors by market-state scalars.

    This is how market-wide information reaches the stock model without handing
    it a date fingerprint. The product varies across the cross-section (the
    stock-specific leg differs per name), so it cannot be used to identify a
    date, but it does let the model express "momentum works differently when
    dispersion is high" or "high beta is punished in a high-volatility regime" —
    which is precisely the conditional structure a single additive feature set
    cannot represent.
    """
    out = df.copy()
    regime = out.get(f"{MARKET_STATE_PREFIX}volatility_regime_z")
    dispersion = out.get(f"{MARKET_STATE_PREFIX}return_dispersion_20d")
    correlation = out.get(f"{MARKET_STATE_PREFIX}average_correlation")

    interactions = {
        "regime_x_momentum_20d": ("xs_rank_return_20d", regime),
        "regime_x_beta": ("market_beta_60d", regime),
        "regime_x_residual_momentum": ("xs_rank_residual_momentum_20d", regime),
        "dispersion_x_momentum_20d": ("xs_rank_return_20d", dispersion),
        "dispersion_x_volatility": ("xs_rank_volatility_20d", dispersion),
        "correlation_x_beta": ("market_beta_60d", correlation),
        "correlation_x_sector_relative": ("xs_rank_sector_relative_return_20d", correlation),
    }

    for name, (stock_column, market_series) in interactions.items():
        if stock_column not in out.columns or market_series is None:
            continue
        out[name] = out[stock_column].astype(float) * market_series.astype(float)
    return out


# Panel features whose informative content is their position relative to the
# stock's own recent history. Z-scored per ticker on a trailing window, exactly
# as the per-ticker feature stage does for its own columns.
PANEL_REGIME_NORMALIZED_FEATURES = (
    "sector_relative_return_20d",
    "sector_relative_return_60d",
    "sector_relative_volatility_20d",
    "residual_momentum_20d",
    "residual_momentum_60d",
    "sector_residual_momentum_20d",
    "sector_beta_60d",
    "amihud_illiquidity_20d",
    "high_low_range_20d",
    "dollar_volume_trend_20d",
)


def add_panel_regime_normalized_features(
    df: pd.DataFrame,
    window: int = 252,
    columns: tuple[str, ...] = PANEL_REGIME_NORMALIZED_FEATURES,
    clip: float = 5.0,
) -> pd.DataFrame:
    """
    Trailing per-ticker z-scores of panel features, computed after pooling.

    The per-ticker feature stage cannot do this because these columns do not
    exist until the panel has been assembled. Grouping by ticker keeps each
    z-score inside one stock's own history, and only past observations enter it.
    """
    out = df.copy()
    window = max(20, int(window))
    minimum_periods = max(20, window // 4)
    present = [column for column in columns if column in out.columns]
    if not present:
        return out

    # Same duplicate-index hazard as the sector stage: work positionally.
    out["_date"] = pd.DatetimeIndex(out.index)
    out = out.sort_values(["Ticker", "_date"], kind="stable").reset_index(drop=True)
    grouped = out.groupby("Ticker")

    for column in present:
        series = out[column].astype(float)
        mean = grouped[column].transform(
            lambda values: values.astype(float).rolling(window, min_periods=minimum_periods).mean()
        )
        std = grouped[column].transform(
            lambda values: values.astype(float).rolling(window, min_periods=minimum_periods).std()
        )
        out[f"{column}_z{window}"] = ((series - mean) / std.replace(0.0, np.nan)).clip(-clip, clip)

    return out.set_index("_date").rename_axis(None).sort_index(kind="stable")


def add_panel_features(
    df: pd.DataFrame,
    ticker_sectors: dict[str, str] | None = None,
    minimum_sector_members: int = 3,
    sector_beta_window: int = 60,
    volatility_regime_window: int = 252,
    regime_normalization_window: int = 252,
) -> pd.DataFrame:
    """Run the whole panel-level feature stage in dependency order."""
    out = assign_sectors(df, ticker_sectors)
    out = add_sector_features(
        out,
        minimum_sector_members=minimum_sector_members,
        beta_window=sector_beta_window,
    )
    out = add_market_state_features(out, volatility_regime_window=volatility_regime_window)
    out = add_panel_regime_normalized_features(out, window=regime_normalization_window)
    # Ranks must come after the sector and residual columns they rank, and the
    # interactions must come after the ranks they multiply.
    out = add_cross_sectional_ranks(out)
    out = add_regime_interactions(out)
    return out.replace([np.inf, -np.inf], np.nan)
