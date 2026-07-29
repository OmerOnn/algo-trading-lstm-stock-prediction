"""
Temporal splitting for financial panels.

Two schemes are provided:

* a single purged chronological hold-out (fast, used for the shipped model), and
* purged walk-forward validation (slower, used to show the result is not an
  artefact of one arbitrary cut of the timeline).

Both apply a **purge** of ``horizon`` sessions between adjacent segments. Without
it, a training row dated just before the boundary carries a label built from
prices that fall inside the next segment, which leaks the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TemporalSplit:
    """One train / validation / test partition of the date axis."""

    index: int
    label: str
    train_dates: np.ndarray = field(repr=False)
    validation_dates: np.ndarray = field(repr=False)
    test_dates: np.ndarray = field(repr=False)

    def describe(self) -> dict:
        def bounds(dates: np.ndarray) -> dict:
            if len(dates) == 0:
                return {"start": None, "end": None, "dates": 0}
            return {
                "start": str(pd.Timestamp(dates[0]).date()),
                "end": str(pd.Timestamp(dates[-1]).date()),
                "dates": int(len(dates)),
            }

        return {
            "fold": self.index,
            "label": self.label,
            "train": bounds(self.train_dates),
            "validation": bounds(self.validation_dates),
            "test": bounds(self.test_dates),
        }

    def frames(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Slice a dated panel into the three partitions of this split."""
        index = pd.DatetimeIndex(df.index)
        train = df[index.isin(self.train_dates)].copy()
        validation = df[index.isin(self.validation_dates)].copy()
        test = df[index.isin(self.test_dates)].copy()
        return train, validation, test


def chronological_train_validation_test_split(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    purge_horizon: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single purged chronological hold-out split."""
    unique_dates = sorted(df.index.unique())
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates for chronological split.")

    train_idx = int(len(unique_dates) * train_ratio)
    validation_idx = int(len(unique_dates) * (train_ratio + validation_ratio))
    train_idx = max(1, min(train_idx, len(unique_dates) - 2))
    validation_idx = max(train_idx + 1, min(validation_idx, len(unique_dates) - 1))

    purge_horizon = max(0, int(purge_horizon))
    purged_train_end = max(1, train_idx - purge_horizon)
    purged_validation_end = max(train_idx + 1, validation_idx - purge_horizon)
    train_end_date = unique_dates[purged_train_end]
    validation_start_date = unique_dates[train_idx]
    validation_end_date = unique_dates[validation_idx]

    train_df = df[df.index < train_end_date].copy()
    validation_df = df[
        (df.index >= validation_start_date) & (df.index < unique_dates[purged_validation_end])
    ].copy()
    test_df = df[df.index >= validation_end_date].copy()
    return train_df, validation_df, test_df


def purged_walk_forward_splits(
    dates,
    folds: int = 3,
    initial_train_ratio: float = 0.50,
    validation_ratio: float = 0.15,
    purge_horizon: int = 0,
    expanding: bool = True,
) -> list[TemporalSplit]:
    """
    Build ``folds`` successive out-of-sample windows over the date axis.

    Each fold trains on everything available before its test window (expanding
    origin) or on a fixed-length trailing window (rolling origin), keeps the last
    ``validation_ratio`` of that history for model selection, and evaluates on
    the next unseen block. Purge gaps are inserted before validation and before
    test.
    """
    unique_dates = np.asarray(sorted(pd.DatetimeIndex(pd.Series(list(dates))).unique()))
    total = len(unique_dates)
    folds = max(1, int(folds))
    purge = max(0, int(purge_horizon))

    if total < 20 * folds:
        raise ValueError(
            f"Not enough unique dates ({total}) for {folds} walk-forward folds."
        )

    start_index = int(total * float(initial_train_ratio))
    test_block = max(1, (total - start_index) // folds)

    splits: list[TemporalSplit] = []
    for fold in range(folds):
        test_start = start_index + fold * test_block
        test_end = total if fold == folds - 1 else test_start + test_block
        if test_start >= total:
            break

        history_end = max(1, test_start - purge)
        validation_size = max(1, int(history_end * float(validation_ratio)))
        validation_start = max(1, history_end - validation_size)
        train_end = max(1, validation_start - purge)

        train_slice = unique_dates[:train_end]
        if not expanding:
            window = start_index - purge
            train_slice = train_slice[-max(1, window):]

        splits.append(
            TemporalSplit(
                index=fold + 1,
                label=f"fold_{fold + 1}",
                train_dates=train_slice,
                validation_dates=unique_dates[validation_start:history_end],
                test_dates=unique_dates[test_start:test_end],
            )
        )
    return splits
