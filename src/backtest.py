from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    transaction_cost_pct: float = 0.001
    slippage_pct: float = 0.0005
    allow_short: bool = False
    signal_threshold_multiplier: float = 1.0
    min_signal_edge: float = 0.0


def cost_aware_signal_threshold(cfg: BacktestConfig) -> float:
    round_trip_cost = 2.0 * (cfg.transaction_cost_pct + cfg.slippage_pct)
    return float(max(cfg.min_signal_edge, round_trip_cost * cfg.signal_threshold_multiplier))


def return_to_signal(predicted_return: float, threshold: float) -> str:
    if predicted_return >= threshold:
        return "BUY"
    if predicted_return <= -threshold:
        return "SELL"
    return "HOLD"


def signal_to_position(signal: str, cfg: BacktestConfig) -> int:
    if signal == "BUY":
        return 1
    if signal == "SELL":
        return -1 if cfg.allow_short else 0
    return 0


def build_signal_frame(
    metadata: list[dict],
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Create a clean signal dataframe from regression outputs."""
    threshold = cost_aware_signal_threshold(cfg)
    rows = []
    for i, meta in enumerate(metadata):
        predicted_value = float(predicted_return[i])
        true_value = float(true_return[i])
        predicted_signal = return_to_signal(predicted_value, threshold)
        true_signal = return_to_signal(true_value, threshold)
        rows.append(
            {
                "ticker": meta["ticker"],
                "date": pd.to_datetime(meta["date"]),
                "true_signal": true_signal,
                "predicted_signal": predicted_signal,
                "true_return": true_value,
                "predicted_return": predicted_value,
                "signal_threshold": threshold,
                "signal_strength": 0.0 if threshold <= 0 else abs(predicted_value) / threshold,
            }
        )
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


def backtest_signals(signal_df: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    """
    Backtest model signals using realized future returns.

    This is an event-based academic backtest. Each prediction is treated as one independent
    trade over the configured prediction horizon. It is intentionally simple and transparent.
    """
    if signal_df.empty:
        raise ValueError("signal_df is empty. Cannot run backtest.")

    df = signal_df.copy().sort_values("date")
    df["position"] = [signal_to_position(signal, cfg) for signal in df["predicted_signal"]]

    round_trip_cost = 2 * (cfg.transaction_cost_pct + cfg.slippage_pct)
    df["strategy_return"] = df["position"] * df["true_return"]
    df.loc[df["position"] != 0, "strategy_return"] -= round_trip_cost
    df["strategy_return"] = df["strategy_return"].fillna(0.0)

    # Equal-weight all active predictions by date.
    daily = (
        df.groupby("date")
        .agg(
            strategy_return=("strategy_return", "mean"),
            active_trades=("position", lambda x: int(np.sum(np.asarray(x) != 0))),
            average_signal_strength=("signal_strength", "mean"),
            average_true_return=("true_return", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )

    daily["equity"] = cfg.initial_cash * (1.0 + daily["strategy_return"]).cumprod()
    daily["buy_and_hold_equity"] = cfg.initial_cash * (1.0 + daily["average_true_return"]).cumprod()
    daily["running_max"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["running_max"] - 1.0

    returns = daily["strategy_return"].astype(float)
    total_return = daily["equity"].iloc[-1] / cfg.initial_cash - 1.0
    buy_hold_return = daily["buy_and_hold_equity"].iloc[-1] / cfg.initial_cash - 1.0
    avg_return = returns.mean()
    volatility = returns.std(ddof=0)
    sharpe = 0.0 if volatility == 0 or np.isnan(volatility) else float((avg_return / volatility) * np.sqrt(252))
    downside = returns[returns < 0].std(ddof=0)
    sortino = 0.0 if downside == 0 or np.isnan(downside) else float((avg_return / downside) * np.sqrt(252))

    active_returns = df.loc[df["position"] != 0, "strategy_return"].astype(float)
    excess_return_vs_buy_hold = total_return - buy_hold_return

    if len(active_returns) > 0:
        win_rate = float((active_returns > 0).mean())
        average_trade_return = float(active_returns.mean())
    else:
        win_rate = 0.0
        average_trade_return = 0.0

    metrics = {
        "initial_cash": float(cfg.initial_cash),
        "final_equity": float(daily["equity"].iloc[-1]),
        "total_return": float(total_return),
        "buy_and_hold_total_return": float(buy_hold_return),
        "excess_return_vs_buy_hold": float(excess_return_vs_buy_hold),
        "max_drawdown": float(daily["drawdown"].min()),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "average_period_return": float(avg_return),
        "best_period_return": float(returns.max()),
        "worst_period_return": float(returns.min()),
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "period_return_volatility": float(volatility),
        "number_of_prediction_dates": int(len(daily)),
        "number_of_active_trades": int(df["position"].ne(0).sum()),
        "trade_activation_rate": float(df["position"].ne(0).mean()),
    }
    return daily, metrics
