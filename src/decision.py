from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


DEFAULT_HOLD_CLASS_ID = 1
DEFAULT_SELL_CLASS_ID = 0
DEFAULT_BUY_CLASS_ID = 2


def apply_class_thresholds(
    probabilities: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    """Convert class probabilities into SELL/HOLD/BUY ids with calibrated action thresholds."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_samples, 3)")

    thresholds = thresholds or {}
    sell_threshold = float(thresholds.get("sell", 0.5))
    buy_threshold = float(thresholds.get("buy", 0.5))

    sell_prob = probs[:, DEFAULT_SELL_CLASS_ID]
    buy_prob = probs[:, DEFAULT_BUY_CLASS_ID]

    predictions = np.full(len(probs), DEFAULT_HOLD_CLASS_ID, dtype=np.int64)
    sell_mask = sell_prob >= sell_threshold
    buy_mask = buy_prob >= buy_threshold

    sell_only = sell_mask & ~buy_mask
    buy_only = buy_mask & ~sell_mask
    both = sell_mask & buy_mask

    predictions[sell_only] = DEFAULT_SELL_CLASS_ID
    predictions[buy_only] = DEFAULT_BUY_CLASS_ID
    predictions[both] = np.where(
        sell_prob[both] >= buy_prob[both],
        DEFAULT_SELL_CLASS_ID,
        DEFAULT_BUY_CLASS_ID,
    )
    return predictions


def tune_class_thresholds(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    threshold_grid: list[float] | None = None,
    minimum_action_rate: float = 0.01,
) -> dict[str, float]:
    """Choose BUY/SELL probability thresholds on validation data to improve action precision."""
    probs = np.asarray(probabilities, dtype=float)
    labels = np.asarray(true_labels, dtype=np.int64)

    if threshold_grid is None:
        threshold_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    best_thresholds = {"sell": 0.5, "buy": 0.5}
    best_score = -np.inf
    fallback_score = -np.inf
    fallback_thresholds = best_thresholds

    for sell_threshold, buy_threshold in product(threshold_grid, repeat=2):
        thresholds = {"sell": float(sell_threshold), "buy": float(buy_threshold)}
        predictions = apply_class_thresholds(probs, thresholds)

        action_mask = predictions != DEFAULT_HOLD_CLASS_ID
        action_rate = float(np.mean(action_mask))

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=[DEFAULT_SELL_CLASS_ID, DEFAULT_HOLD_CLASS_ID, DEFAULT_BUY_CLASS_ID],
            zero_division=0,
        )

        sell_precision, hold_precision, buy_precision = precision
        sell_recall, hold_recall, buy_recall = recall
        sell_f1, hold_f1, buy_f1 = f1

        action_precision = (sell_precision + buy_precision) / 2.0
        action_recall = (sell_recall + buy_recall) / 2.0
        action_f1 = (sell_f1 + buy_f1) / 2.0
        macro_f1 = float(np.mean(f1))

        score = (
            0.45 * action_precision
            + 0.25 * action_f1
            + 0.20 * macro_f1
            + 0.10 * hold_precision
            - 0.10 * abs(action_rate - 0.10)
        )

        if score > fallback_score:
            fallback_score = score
            fallback_thresholds = thresholds

        if action_rate < minimum_action_rate:
            continue

        if action_recall == 0.0:
            continue

        if score > best_score:
            best_score = score
            best_thresholds = thresholds

    return best_thresholds if best_score > -np.inf else fallback_thresholds
