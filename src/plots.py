from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
