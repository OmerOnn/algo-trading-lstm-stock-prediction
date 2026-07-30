"""
Portfolio-level backtesting of a cross-sectional return forecast.

Why the previous backtest was not a fair test
--------------------------------------------
The signal backtest holds a full unit in every name whose forecast clears a
hurdle and cash otherwise. On this model that meant roughly 17–31% of capital was
invested, and its total return was then printed next to a 100%-invested
buy-and-hold number. That comparison is meaningless: a strategy that is
two-thirds in cash *should* return less, and its Sharpe ratio is flattered
because cash has no volatility.

What this module does instead
-----------------------------
1. **Fully invested top-k selection.** On each rebalance date the universe is
   ranked by forecast, the best ``k`` names are bought, and weights are
   renormalised so invested exposure sums to one. Now the strategy and the
   equal-weight universe are both 100% invested, so the difference between them
   is attributable to selection rather than to exposure.
2. **Costs from realised turnover.** Cost is charged on the weight actually
   traded, ``sum |w_new - w_old|``, not on a per-signal flat fee. A strategy that
   holds the same names pays nothing to keep holding them, and one that churns
   the whole book pays for the whole book.
3. **Every rebalance offset, not one arbitrary phase.** A 21-day rebalance can
   start on any of 21 offsets, and the choice moves the result materially. All
   offsets are evaluated and the mean, median, worst and dispersion are reported,
   so the headline number is not one lucky calendar alignment.
4. **Sector- and beta-neutral variants**, which answer whether the edge is real
   stock selection or a persistent bet on one sector or on high-beta names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class PortfolioConfig:
    """Construction rules for the fully invested cross-sectional portfolio."""

    top_k: int = 20
    # Used when top_k is not set explicitly: the top fraction of the universe.
    top_quantile: float = 0.20
    long_short: bool = False
    # Gross exposure cap for the long/short variant (1.0 long + 1.0 short = 2.0).
    maximum_gross_exposure: float = 2.0
    transaction_cost_pct: float = 0.001
    slippage_pct: float = 0.0005
    initial_cash: float = 10_000.0
    neutralize: str = "none"  # "none", "sector" or "beta"
    minimum_names_per_date: int = 10

    def round_trip_cost(self) -> float:
        """Cost of moving one unit of weight in and out again."""
        return float(self.transaction_cost_pct + self.slippage_pct)


def _select_weights(
    group: pd.DataFrame,
    cfg: PortfolioConfig,
) -> pd.Series:
    """
    Target weights for one rebalance date.

    Long-only weights sum to exactly 1. Long/short weights sum to 0 with a gross
    exposure of ``maximum_gross_exposure``, so it is a self-financing spread
    rather than a leveraged long.
    """
    count = len(group)
    k = int(cfg.top_k) if cfg.top_k else max(1, int(round(count * float(cfg.top_quantile))))
    k = max(1, min(k, count // 2 if cfg.long_short else count))

    ordered = group["score"].rank(ascending=False, method="first")
    weights = pd.Series(0.0, index=group.index, dtype=float)

    longs = ordered <= k
    weights[longs] = 1.0 / max(1, int(longs.sum()))

    if cfg.long_short:
        shorts = ordered > (count - k)
        half = float(cfg.maximum_gross_exposure) / 2.0
        weights[longs] = half / max(1, int(longs.sum()))
        weights[shorts] = -half / max(1, int(shorts.sum()))

    return weights


def _neutralize_scores(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Remove a systematic tilt from the forecast before ranking.

    Sector neutralisation de-means the score inside each sector, so the ranking
    can only express "best within its sector" and cannot express "technology
    over utilities". Beta neutralisation regresses the score on beta per date and
    keeps the residual, so it cannot express "high beta over low beta". If
    performance survives either, the edge is genuine stock selection.
    """
    out = frame.copy()
    if mode == "sector" and "sector" in out.columns:
        out["score"] = out["score"] - out.groupby(["date", "sector"])["score"].transform("mean")
    elif mode == "beta" and "beta" in out.columns:

        def residualise(group: pd.DataFrame) -> pd.Series:
            beta = group["beta"].astype(float).to_numpy()
            score = group["score"].astype(float).to_numpy()
            if len(group) < 5 or np.std(beta) < 1e-12:
                return pd.Series(score, index=group.index)
            design = np.column_stack([beta, np.ones(len(beta))])
            coefficients, *_ = np.linalg.lstsq(design, score, rcond=None)
            return pd.Series(score - design @ coefficients, index=group.index)

        out["score"] = out.groupby("date", group_keys=False).apply(residualise)
    return out


def run_top_k_backtest(
    frame: pd.DataFrame,
    cfg: PortfolioConfig,
    horizon: int,
    offset: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest one rebalance schedule.

    ``frame`` needs ``date``, ``ticker``, ``score`` (the forecast used for
    ranking) and ``forward_return`` (the realised return over the horizon), plus
    optional ``sector`` and ``beta`` columns for the neutral variants.

    ``offset`` selects which of the ``horizon`` possible rebalance phases to use.
    Rebalance dates are every ``horizon``-th trading date starting at ``offset``,
    so each period's forward return is realised before the next decision and no
    return is double-counted.
    """
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = _neutralize_scores(data, str(cfg.neutralize))

    all_dates = np.asarray(sorted(data["date"].unique()))
    horizon = max(1, int(horizon))
    rebalance_dates = all_dates[int(offset) :: horizon]
    data = data[data["date"].isin(set(rebalance_dates))]

    periods: list[dict] = []
    previous_weights: dict[str, float] = {}
    cost_per_unit = cfg.round_trip_cost()

    for date, group in data.groupby("date", sort=True):
        if len(group) < int(cfg.minimum_names_per_date):
            continue

        group = group.set_index("ticker")
        weights = _select_weights(group, cfg)

        # Turnover is the weight that actually had to change hands. Names held
        # from the previous period at the same weight cost nothing.
        tickers = set(weights.index) | set(previous_weights)
        turnover = float(
            sum(
                abs(float(weights.get(ticker, 0.0)) - float(previous_weights.get(ticker, 0.0)))
                for ticker in tickers
            )
        )
        cost = turnover * cost_per_unit

        forward = group["forward_return"].astype(float)
        gross_return = float((weights * forward).sum())
        universe_return = float(forward.mean())

        periods.append(
            {
                "date": date,
                "gross_return": gross_return,
                "transaction_cost": cost,
                "net_return": gross_return - cost,
                "universe_return": universe_return,
                "turnover": turnover,
                "names_held": int((weights != 0).sum()),
                "long_exposure": float(weights[weights > 0].sum()),
                "short_exposure": float(weights[weights < 0].sum()),
                "gross_exposure": float(weights.abs().sum()),
                "net_exposure": float(weights.sum()),
                "universe_size": int(len(group)),
            }
        )
        previous_weights = {ticker: float(value) for ticker, value in weights.items() if value != 0}

    if not periods:
        return pd.DataFrame(), {"rebalances": 0, "offset": int(offset)}

    history = pd.DataFrame(periods).sort_values("date").reset_index(drop=True)
    history["equity"] = cfg.initial_cash * (1.0 + history["net_return"]).cumprod()
    history["universe_equity"] = cfg.initial_cash * (1.0 + history["universe_return"]).cumprod()
    history["running_max"] = history["equity"].cummax()
    history["drawdown"] = history["equity"] / history["running_max"] - 1.0

    return history, summarise_portfolio(history, cfg, horizon, offset)


def summarise_portfolio(
    history: pd.DataFrame,
    cfg: PortfolioConfig,
    horizon: int,
    offset: int = 0,
) -> dict:
    """Risk and return statistics for one backtest run."""
    net = history["net_return"].astype(float)
    universe = history["universe_return"].astype(float)
    periods_per_year = TRADING_DAYS_PER_YEAR / max(1, int(horizon))
    periods = len(net)

    total_return = float(history["equity"].iloc[-1] / cfg.initial_cash - 1.0)
    universe_total = float(history["universe_equity"].iloc[-1] / cfg.initial_cash - 1.0)
    years = periods / periods_per_year if periods_per_year > 0 else 0.0

    def annualise(total: float) -> float:
        if years <= 0 or total <= -1.0:
            return 0.0
        return float((1.0 + total) ** (1.0 / years) - 1.0)

    volatility = float(net.std(ddof=1)) if periods > 1 else 0.0
    annual_volatility = volatility * float(np.sqrt(periods_per_year))
    downside = net[net < 0]
    downside_volatility = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0

    active = net - universe
    tracking_error = float(active.std(ddof=1)) if periods > 1 else 0.0

    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0 or not np.isfinite(denominator):
            return 0.0
        return float(numerator / denominator * np.sqrt(periods_per_year))

    return {
        "offset": int(offset),
        "rebalances": int(periods),
        "years": float(years),
        "total_return": total_return,
        "annualized_return": annualise(total_return),
        "universe_total_return": universe_total,
        "universe_annualized_return": annualise(universe_total),
        "excess_return_vs_universe": float(total_return - universe_total),
        "period_volatility": volatility,
        "annualized_volatility": annual_volatility,
        "sharpe_ratio": ratio(float(net.mean()), volatility),
        "sortino_ratio": ratio(float(net.mean()), downside_volatility),
        "information_ratio_vs_universe": ratio(float(active.mean()), tracking_error),
        "tracking_error_annualized": tracking_error * float(np.sqrt(periods_per_year)),
        "max_drawdown": float(history["drawdown"].min()),
        "hit_rate": float((net > 0).mean()),
        "hit_rate_vs_universe": float((active > 0).mean()),
        "average_turnover": float(history["turnover"].mean()),
        "annualized_turnover": float(history["turnover"].mean() * periods_per_year),
        "total_transaction_cost": float(history["transaction_cost"].sum()),
        "average_transaction_cost": float(history["transaction_cost"].mean()),
        "cost_drag_annualized": float(history["transaction_cost"].mean() * periods_per_year),
        "average_gross_exposure": float(history["gross_exposure"].mean()),
        "average_net_exposure": float(history["net_exposure"].mean()),
        "average_names_held": float(history["names_held"].mean()),
        "best_period_return": float(net.max()),
        "worst_period_return": float(net.min()),
        "neutralization": str(cfg.neutralize),
        "long_short": bool(cfg.long_short),
        "top_k": int(cfg.top_k),
    }


def run_all_offsets(
    frame: pd.DataFrame,
    cfg: PortfolioConfig,
    horizon: int,
) -> dict:
    """
    Evaluate every rebalance phase and report the distribution of outcomes.

    With a 21-day holding period there are 21 valid rebalance calendars. Reporting
    only one is reporting one draw from a distribution and calling it the result.
    The dispersion across offsets is itself a finding: a strategy whose Sharpe
    swings from 0.2 to 1.8 depending on which day of the month it trades has not
    demonstrated a robust edge, however good its best offset looks.
    """
    horizon = max(1, int(horizon))
    runs: list[dict] = []
    for offset in range(horizon):
        _, metrics = run_top_k_backtest(frame, cfg, horizon, offset=offset)
        if metrics.get("rebalances", 0) > 0:
            runs.append(metrics)

    if not runs:
        return {"offsets_evaluated": 0}

    def distribution(key: str) -> dict:
        values = np.asarray([run[key] for run in runs], dtype=float)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "worst": float(values.min()),
            "best": float(values.max()),
        }

    tracked = (
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "sortino_ratio",
        "information_ratio_vs_universe",
        "max_drawdown",
        "annualized_turnover",
        "cost_drag_annualized",
        "excess_return_vs_universe",
        "hit_rate",
    )
    return {
        "offsets_evaluated": len(runs),
        "horizon": horizon,
        "note": (
            "every rebalance phase of the horizon is evaluated; the spread across "
            "offsets shows how much of any single result is calendar luck"
        ),
        "distribution": {key: distribution(key) for key in tracked},
        "per_offset": runs,
    }


def regime_performance(
    history: pd.DataFrame,
    regime_series: pd.Series | None = None,
    blocks: int = 4,
) -> list[dict]:
    """
    Split performance by market regime.

    When a regime series is supplied, periods are grouped by whether the market
    was in its high- or low-volatility state on the decision date. Otherwise the
    timeline is split into consecutive equal blocks, which at least exposes a
    strategy that worked only in the first half of the sample.
    """
    if history.empty:
        return []

    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    if regime_series is not None and len(regime_series) > 0:
        aligned = pd.Series(regime_series)
        aligned.index = pd.to_datetime(aligned.index)
        frame["regime"] = frame["date"].map(aligned)
        frame["regime"] = np.where(
            frame["regime"].astype(float) > 0.5, "high_volatility", "low_volatility"
        )
    else:
        frame["regime"] = [
            f"block_{index + 1}"
            for index in np.repeat(
                np.arange(max(1, int(blocks))),
                int(np.ceil(len(frame) / max(1, int(blocks)))),
            )[: len(frame)]
        ]

    rows: list[dict] = []
    for regime, group in frame.groupby("regime", sort=True):
        net = group["net_return"].astype(float)
        universe = group["universe_return"].astype(float)
        rows.append(
            {
                "regime": str(regime),
                "periods": int(len(group)),
                "start_date": str(group["date"].min().date()),
                "end_date": str(group["date"].max().date()),
                "mean_net_return": float(net.mean()),
                "mean_universe_return": float(universe.mean()),
                "excess_vs_universe": float(net.mean() - universe.mean()),
                "hit_rate": float((net > 0).mean()),
                "volatility": float(net.std(ddof=1)) if len(net) > 1 else 0.0,
            }
        )
    return rows


def build_portfolio_frame(
    signal_df: pd.DataFrame,
    score_column: str = "predicted_return",
    return_column: str = "true_return",
    sector_map: dict[str, str] | None = None,
    beta_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Reshape a signal table into the columns the portfolio backtest expects."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(signal_df["date"]),
            "ticker": signal_df["ticker"].astype(str),
            "score": signal_df[score_column].astype(float),
            "forward_return": signal_df[return_column].astype(float),
        }
    )
    if sector_map:
        upper = {str(k).upper(): str(v) for k, v in sector_map.items()}
        frame["sector"] = frame["ticker"].str.upper().map(upper).fillna("Unclassified")
    if beta_series is not None:
        frame["beta"] = np.asarray(beta_series, dtype=float)
    return frame.dropna(subset=["score", "forward_return"]).reset_index(drop=True)
