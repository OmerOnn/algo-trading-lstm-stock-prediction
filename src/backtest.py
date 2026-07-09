from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import ID_TO_CLASS


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    transaction_cost_pct: float = 0.001
    slippage_pct: float = 0.0005
    min_signal_confidence: float = 0.45
    allow_short: bool = False


def _signal_to_position(signal_id: int, confidence: float, cfg: BacktestConfig) -> int:
    """Convert model class output to target position: 1 long, 0 cash, -1 short."""
    if confidence < cfg.min_signal_confidence:
        return 0

    signal = ID_TO_CLASS[int(signal_id)]
    if signal == "BUY":
        return 1
    if signal == "SELL":
        return -1 if cfg.allow_short else 0
    return 0


def build_signal_frame(
    metadata: list[dict],
    true_class: np.ndarray,
    predicted_class: np.ndarray,
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Create a clean signal dataframe from model evaluation outputs."""
    rows = []
    for i, meta in enumerate(metadata):
        confidence = float(np.max(probabilities[i]))
        rows.append(
            {
                "ticker": meta["ticker"],
                "date": pd.to_datetime(meta["date"]),
                "true_signal": ID_TO_CLASS[int(true_class[i])],
                "predicted_signal": ID_TO_CLASS[int(predicted_class[i])],
                "true_return": float(true_return[i]),
                "predicted_return": float(predicted_return[i]),
                "prob_sell": float(probabilities[i][0]),
                "prob_hold": float(probabilities[i][1]),
                "prob_buy": float(probabilities[i][2]),
                "confidence": confidence,
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
    df["predicted_class_id"] = df["predicted_signal"].map({"SELL": 0, "HOLD": 1, "BUY": 2})
    df["position"] = [
        _signal_to_position(cls_id, confidence, cfg)
        for cls_id, confidence in zip(df["predicted_class_id"], df["confidence"])
    ]

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
            average_confidence=("confidence", "mean"),
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

    trade_returns = df.loc[df["position"] != 0, "strategy_return"].astype(float)
    win_rate = 0.0 if trade_returns.empty else float((trade_returns > 0).mean())
    average_trade_return = 0.0 if trade_returns.empty else float(trade_returns.mean())

    metrics = {
        "initial_cash": float(cfg.initial_cash),
        "final_equity": float(daily["equity"].iloc[-1]),
        "total_return": float(total_return),
        "buy_and_hold_total_return": float(buy_hold_return),
        "excess_return_vs_buy_hold": float(total_return - buy_hold_return),
        "max_drawdown": float(daily["drawdown"].min()),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "average_period_return": float(avg_return),
        "period_return_volatility": float(volatility),
        "number_of_prediction_dates": int(len(daily)),
        "number_of_active_trades": int(df["position"].ne(0).sum()),
        "trade_activation_rate": float(df["position"].ne(0).mean()),
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "best_period_return": float(returns.max()) if not returns.empty else 0.0,
        "worst_period_return": float(returns.min()) if not returns.empty else 0.0,
    }
    return daily, metrics
