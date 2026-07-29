from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.decision import (
    BUY,
    HOLD,
    SELL,
    DecisionConfig,
    decide_batch,
    direction_probability,
    edge_z_score,
    position_sizes,
    realised_signal,
)


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    transaction_cost_pct: float = 0.001
    slippage_pct: float = 0.0005
    allow_short: bool = False
    signal_threshold_multiplier: float = 1.0
    min_signal_edge: float = 0.0


def cost_aware_signal_threshold(cfg: BacktestConfig) -> float:
    """The smallest predicted return that can be profitable after round-trip costs."""
    round_trip_cost = 2.0 * (cfg.transaction_cost_pct + cfg.slippage_pct)
    return float(max(cfg.min_signal_edge, round_trip_cost * cfg.signal_threshold_multiplier))


def return_to_signal(predicted_return: float, threshold: float) -> str:
    """Plain point-forecast rule, kept for the no-uncertainty comparison path."""
    if predicted_return >= threshold:
        return BUY
    if predicted_return <= -threshold:
        return SELL
    return HOLD


def signal_to_position(signal: str, cfg: BacktestConfig) -> int:
    if signal == BUY:
        return 1
    if signal == SELL:
        return -1 if cfg.allow_short else 0
    return 0


def build_signal_frame(
    metadata: list[dict],
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    cfg: BacktestConfig,
    threshold: float | None = None,
    sigma: np.ndarray | None = None,
    decision_cfg: DecisionConfig | None = None,
) -> pd.DataFrame:
    """
    Turn regression output into a dated signal table.

    ``sigma`` is the calibrated forecast standard deviation. When it is absent
    the frame falls back to the plain point-forecast threshold rule, so the two
    decision rules can be compared on identical predictions.
    """
    resolved_threshold = cost_aware_signal_threshold(cfg) if threshold is None else float(threshold)
    predictions = np.asarray(predicted_return, dtype=float)
    truths = np.asarray(true_return, dtype=float)

    if decision_cfg is None:
        decision_cfg = DecisionConfig(
            rule="point",
            threshold=resolved_threshold,
            allow_short=cfg.allow_short,
        )
    else:
        decision_cfg = replace(
            decision_cfg,
            threshold=resolved_threshold,
            allow_short=cfg.allow_short,
        )

    if sigma is None:
        sigma_values = np.full(len(predictions), np.nan)
        effective_cfg = replace(decision_cfg, rule="point")
        signals = decide_batch(predictions, np.ones(len(predictions)), effective_cfg)
        positions = position_sizes(
            signals, predictions, np.ones(len(predictions)), replace(effective_cfg, position_sizing="binary")
        )
        z_scores = np.zeros(len(predictions))
        probabilities = np.full(len(predictions), np.nan)
    else:
        sigma_values = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
        signals = decide_batch(predictions, sigma_values, decision_cfg)
        positions = position_sizes(signals, predictions, sigma_values, decision_cfg)
        z_scores = edge_z_score(predictions, sigma_values, decision_cfg.threshold)
        probabilities = direction_probability(predictions, sigma_values)

    frame = pd.DataFrame(
        {
            "ticker": [meta["ticker"] for meta in metadata],
            "date": pd.to_datetime([meta["date"] for meta in metadata]),
            "true_signal": realised_signal(truths, resolved_threshold),
            "predicted_signal": signals,
            "true_return": truths,
            "predicted_return": predictions,
            "forecast_sigma": sigma_values,
            "lower_bound": predictions - sigma_values if sigma is not None else np.nan,
            "upper_bound": predictions + sigma_values if sigma is not None else np.nan,
            "direction_probability": probabilities,
            "edge_z_score": z_scores,
            "position": positions,
            "signal_threshold": resolved_threshold,
            "signal_strength": np.abs(predictions) / max(resolved_threshold, 1e-9),
        }
    )
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def backtest_signals(
    signal_df: pd.DataFrame,
    cfg: BacktestConfig,
    horizon: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest model signals using realised forward returns.

    This is an event-based academic backtest. Each prediction is one trade held
    for the full prediction horizon. Overlapping forward windows are removed so
    an h-day outcome is never compounded as if it were a daily return.
    """
    if signal_df.empty:
        raise ValueError("signal_df is empty. Cannot run backtest.")

    df = signal_df.copy().sort_values("date")
    horizon = max(1, int(horizon))

    unique_dates = np.asarray(sorted(pd.to_datetime(df["date"]).unique()))
    selected_dates = set(unique_dates[::horizon])
    df = df[pd.to_datetime(df["date"]).isin(selected_dates)].copy()

    if "position" not in df.columns:
        df["position"] = [signal_to_position(signal, cfg) for signal in df["predicted_signal"]]
    if not cfg.allow_short:
        df["position"] = df["position"].clip(lower=0.0)

    round_trip_cost = 2 * (cfg.transaction_cost_pct + cfg.slippage_pct)
    df["strategy_return"] = df["position"] * df["true_return"]
    # Costs scale with the traded exposure, so partial positions pay partial cost.
    df["strategy_return"] -= df["position"].abs() * round_trip_cost
    df["strategy_return"] = df["strategy_return"].fillna(0.0)

    daily = (
        df.groupby("date")
        .agg(
            strategy_return=("strategy_return", "mean"),
            active_trades=("position", lambda x: int(np.sum(np.asarray(x) != 0))),
            gross_exposure=("position", lambda x: float(np.mean(np.abs(np.asarray(x))))),
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
    benchmark_returns = daily["average_true_return"].astype(float)
    total_return = daily["equity"].iloc[-1] / cfg.initial_cash - 1.0
    buy_hold_return = daily["buy_and_hold_equity"].iloc[-1] / cfg.initial_cash - 1.0
    avg_return = returns.mean()
    volatility = returns.std(ddof=0)
    periods_per_year = 252.0 / horizon
    sharpe = 0.0 if volatility == 0 or np.isnan(volatility) else float(
        (avg_return / volatility) * np.sqrt(periods_per_year)
    )
    downside = returns[returns < 0].std(ddof=0)
    sortino = 0.0 if downside == 0 or np.isnan(downside) else float(
        (avg_return / downside) * np.sqrt(periods_per_year)
    )

    # Information ratio against the equal-weight universe, which is the honest
    # comparison for a long-only stock-selection strategy.
    active = returns - benchmark_returns
    tracking_error = active.std(ddof=0)
    information_ratio = 0.0 if tracking_error == 0 or np.isnan(tracking_error) else float(
        (active.mean() / tracking_error) * np.sqrt(periods_per_year)
    )

    active_returns = df.loc[df["position"] != 0, "strategy_return"].astype(float)
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
        "excess_return_vs_buy_hold": float(total_return - buy_hold_return),
        "max_drawdown": float(daily["drawdown"].min()),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "information_ratio_vs_universe": information_ratio,
        "average_period_return": float(avg_return),
        "best_period_return": float(returns.max()),
        "worst_period_return": float(returns.min()),
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "period_return_volatility": float(volatility),
        "number_of_prediction_dates": int(len(daily)),
        "number_of_active_trades": int(df["position"].ne(0).sum()),
        "trade_activation_rate": float(df["position"].ne(0).mean()),
        "average_gross_exposure": float(df["position"].abs().mean()),
        "prediction_horizon": horizon,
        "periods_per_year": float(periods_per_year),
        "overlapping_predictions_removed": True,
    }
    return daily, metrics


def tune_decision_config(
    metadata: list[dict],
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    sigma: np.ndarray | None,
    cfg: BacktestConfig,
    horizon: int,
    base_decision_cfg: DecisionConfig,
    threshold_quantiles: list[float] | None = None,
    z_candidates: list[float] | None = None,
    min_active_trades: int = 20,
) -> tuple[DecisionConfig, dict]:
    """
    Select the decision-rule parameters on validation data only.

    Search runs over the return hurdle (never below round-trip cost) and, for
    the risk-adjusted rule, over the minimum edge-per-risk. The objective is net
    non-overlapping validation Sharpe with a mild penalty for trading almost
    everything, which keeps the rule from degenerating into "always long".
    """
    floor = cost_aware_signal_threshold(cfg)
    absolute_predictions = np.abs(np.asarray(predicted_return, dtype=float))
    threshold_quantiles = threshold_quantiles or [0.0, 0.30, 0.50, 0.65, 0.80]
    thresholds = {floor}
    if len(absolute_predictions):
        thresholds.update(
            max(floor, float(np.quantile(absolute_predictions, quantile)))
            for quantile in threshold_quantiles
        )

    if sigma is None or base_decision_cfg.rule == "point":
        z_values = [0.0]
    else:
        z_values = z_candidates or [0.0, 0.05, 0.10, 0.20, 0.30, 0.45]

    best_cfg = replace(base_decision_cfg, threshold=floor, allow_short=cfg.allow_short)
    best_score = -np.inf
    best_metrics: dict = {}
    evaluated: list[dict] = []

    for threshold in sorted(thresholds):
        for min_z in z_values:
            candidate = replace(
                base_decision_cfg,
                threshold=float(threshold),
                min_z_score=float(min_z),
                allow_short=cfg.allow_short,
            )
            frame = build_signal_frame(
                metadata,
                true_return,
                predicted_return,
                cfg,
                threshold=threshold,
                sigma=sigma,
                decision_cfg=candidate,
            )
            _, metrics = backtest_signals(frame, cfg, horizon=horizon)
            evaluated.append(
                {
                    "threshold": float(threshold),
                    "min_z_score": float(min_z),
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "active_trades": metrics["number_of_active_trades"],
                    "activation_rate": metrics["trade_activation_rate"],
                }
            )
            if metrics["number_of_active_trades"] < int(min_active_trades):
                continue
            activation_penalty = max(0.0, metrics["trade_activation_rate"] - 0.60)
            score = float(metrics["sharpe_ratio"] - activation_penalty)
            if score > best_score:
                best_score = score
                best_cfg = candidate
                best_metrics = metrics

    if not best_metrics:
        fallback = build_signal_frame(
            metadata,
            true_return,
            predicted_return,
            cfg,
            threshold=floor,
            sigma=sigma,
            decision_cfg=best_cfg,
        )
        _, best_metrics = backtest_signals(fallback, cfg, horizon=horizon)

    tuning_report = {
        "selection_dataset": "validation",
        "objective": "non_overlapping_net_sharpe_minus_activation_penalty",
        "minimum_cost_threshold": floor,
        "selected": best_cfg.to_dict(),
        "candidates_evaluated": len(evaluated),
        "candidate_grid": evaluated,
        "validation_backtest_metrics": best_metrics,
    }
    return best_cfg, tuning_report


def tune_signal_threshold(
    metadata: list[dict],
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    cfg: BacktestConfig,
    horizon: int,
    quantiles: list[float] | None = None,
    min_active_trades: int = 20,
) -> tuple[float, dict]:
    """Backwards-compatible wrapper that tunes only the point-forecast threshold."""
    decision_cfg, report = tune_decision_config(
        metadata=metadata,
        true_return=true_return,
        predicted_return=predicted_return,
        sigma=None,
        cfg=cfg,
        horizon=horizon,
        base_decision_cfg=DecisionConfig(rule="point", allow_short=cfg.allow_short),
        threshold_quantiles=quantiles,
        min_active_trades=min_active_trades,
    )
    report["selected_threshold"] = decision_cfg.threshold
    return decision_cfg.threshold, report
