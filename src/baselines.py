from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import CLASS_TO_ID


def add_sma_crossover_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple SMA 20/50 baseline prediction for comparison."""
    out = df.copy()
    buy = out["sma_20"] > out["sma_50"]
    sell = out["sma_20"] < out["sma_50"]
    out["baseline_sma_signal"] = np.select(
        [sell, buy],
        [CLASS_TO_ID["SELL"], CLASS_TO_ID["BUY"]],
        default=CLASS_TO_ID["HOLD"],
    )
    return out


def baseline_accuracy(df: pd.DataFrame) -> dict:
    """Compute simple baseline accuracies."""
    if "baseline_sma_signal" not in df.columns:
        df = add_sma_crossover_baseline(df)

    majority_label = int(df["signal_label"].mode().iloc[0])
    majority_acc = float((df["signal_label"] == majority_label).mean())
    sma_acc = float((df["signal_label"] == df["baseline_sma_signal"]).mean())

    return {
        "majority_class_accuracy": majority_acc,
        "sma_20_50_crossover_accuracy": sma_acc,
        "majority_class_id": majority_label,
    }
