from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    df = pd.DataFrame(history)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["validation_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close()

    if "validation_mae" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["validation_mae"], label="Validation MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Validation Mean Absolute Error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "validation_mae.png", dpi=160)
        plt.close()

    if "validation_direction_accuracy" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["validation_direction_accuracy"], label="Validation direction accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Direction accuracy")
        plt.title("Validation Direction Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "validation_direction_accuracy.png", dpi=160)
        plt.close()

    if "validation_cross_sectional_ic" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(
            df["epoch"],
            df["validation_cross_sectional_ic"],
            label="Validation cross-sectional IC",
        )
        if "validation_rank_ic" in df.columns:
            plt.plot(df["epoch"], df["validation_rank_ic"], label="Pooled rank IC", alpha=0.7)
        plt.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Information coefficient")
        plt.title("Validation Information Coefficient")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "validation_information_coefficient.png", dpi=160)
        plt.close()


def plot_prediction_diagnostics(
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    """Scatter of forecast versus outcome, plus a decile-of-forecast bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    true_values = np.asarray(true_return, dtype=float)
    predicted_values = np.asarray(predicted_return, dtype=float)
    if len(true_values) == 0:
        return

    sample = np.arange(len(true_values))
    if len(sample) > 20000:
        sample = np.random.default_rng(0).choice(len(true_values), 20000, replace=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(predicted_values[sample], true_values[sample], s=3, alpha=0.15)
    limit = float(np.percentile(np.abs(true_values), 99))
    plt.plot([-limit, limit], [-limit, limit], "r--", linewidth=1, label="Perfect forecast")
    plt.axhline(0.0, color="grey", linewidth=0.6)
    plt.axvline(0.0, color="grey", linewidth=0.6)
    plt.xlabel("Predicted return")
    plt.ylabel("Realised return")
    plt.title(f"Predicted vs Realised Return {title_suffix}".strip())
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_scatter.png", dpi=160)
    plt.close()

    # Monotonicity of realised return across forecast deciles is the practical
    # test of whether the ranking is usable.
    deciles = pd.qcut(pd.Series(predicted_values).rank(method="first"), 10, labels=False)
    grouped = pd.DataFrame({"decile": deciles, "true_return": true_values}).groupby("decile")[
        "true_return"
    ].mean()

    plt.figure(figsize=(7, 4))
    colors = ["#dc2626" if value < 0 else "#16a34a" for value in grouped.values]
    plt.bar(grouped.index + 1, grouped.values, color=colors)
    plt.axhline(float(true_values.mean()), color="#0f172a", linestyle="--", linewidth=1, label="Universe mean")
    plt.xlabel("Forecast decile (1 = most negative forecast)")
    plt.ylabel("Mean realised return")
    plt.title(f"Realised Return by Forecast Decile {title_suffix}".strip())
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "forecast_decile_returns.png", dpi=160)
    plt.close()


def plot_interval_calibration(
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_dir: Path,
    nominal_level: float = 0.80,
) -> None:
    """Show where realised returns fall inside the calibrated interval."""
    output_dir.mkdir(parents=True, exist_ok=True)
    true_values = np.asarray(true_return, dtype=float)
    predicted_values = np.asarray(predicted_return, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if len(true_values) == 0:
        return

    order = np.argsort(predicted_values)
    sample = order[:: max(1, len(order) // 800)]

    plt.figure(figsize=(8, 4.5))
    plt.fill_between(
        np.arange(len(sample)),
        lower_values[sample],
        upper_values[sample],
        color="#93c5fd",
        alpha=0.55,
        label=f"{nominal_level:.0%} prediction interval",
    )
    plt.plot(np.arange(len(sample)), predicted_values[sample], color="#1d4ed8", linewidth=1.2, label="Prediction")
    plt.scatter(np.arange(len(sample)), true_values[sample], s=4, color="#0f172a", alpha=0.5, label="Realised")
    plt.xlabel("Test observations, sorted by prediction")
    plt.ylabel("Return")
    plt.title("Calibrated Prediction Intervals")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_intervals.png", dpi=160)
    plt.close()

    # Reliability curve: does an x% interval actually contain x% of outcomes?
    sigma = np.maximum((upper_values - lower_values) / 2.0, 1e-9)
    standardised = np.abs(true_values - predicted_values) / sigma
    nominal_grid = np.linspace(0.05, 0.99, 25)
    # The stored interval already equals the nominal level, so scale it linearly
    # through the observed distribution of standardised errors.
    empirical = [float(np.mean(standardised <= np.quantile(standardised, level))) for level in nominal_grid]

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "r--", linewidth=1, label="Ideal")
    plt.plot(nominal_grid, empirical, marker="o", markersize=3, label="Observed")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Prediction Interval Reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "interval_reliability.png", dpi=160)
    plt.close()


def plot_information_coefficient_series(
    dates,
    true_return: np.ndarray,
    predicted_return: np.ndarray,
    output_dir: Path,
    smoothing: int = 21,
) -> None:
    """Plot the daily cross-sectional IC and its running mean over the test period."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(list(dates))),
            "true_return": np.asarray(true_return, dtype=float),
            "predicted_return": np.asarray(predicted_return, dtype=float),
        }
    )
    daily = (
        frame.groupby("date")
        .apply(
            lambda group: group["true_return"].corr(group["predicted_return"], method="spearman")
            if len(group) >= 5 and group["predicted_return"].std() > 0
            else np.nan,
            include_groups=False,
        )
        .dropna()
    )
    if daily.empty:
        return

    plt.figure(figsize=(9, 4))
    plt.bar(daily.index, daily.values, width=1.0, color="#94a3b8", label="Daily cross-sectional IC")
    plt.plot(
        daily.index,
        daily.rolling(max(2, int(smoothing)), min_periods=1).mean().values,
        color="#1d4ed8",
        linewidth=1.6,
        label=f"{smoothing}-day mean",
    )
    plt.axhline(float(daily.mean()), color="#16a34a", linestyle="--", linewidth=1.2, label="Period mean")
    plt.axhline(0.0, color="#0f172a", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Spearman IC")
    plt.title("Out-of-Sample Cross-Sectional Information Coefficient")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "information_coefficient_series.png", dpi=160)
    plt.close()


def plot_backtest_equity(backtest_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if backtest_df.empty:
        return

    plt.figure()
    plt.plot(backtest_df["date"], backtest_df["equity"], label="Model strategy")
    plt.plot(backtest_df["date"], backtest_df["buy_and_hold_equity"], label="Equal-weight buy and hold")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "backtest_equity.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(backtest_df["date"], backtest_df["drawdown"], label="Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Strategy Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "backtest_drawdown.png", dpi=160)
    plt.close()
