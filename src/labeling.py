from __future__ import annotations

import math


def resolve_label_thresholds(config: dict, horizon: int) -> tuple[float, float]:
    """Resolve BUY/SELL label thresholds for a specific horizon.

    By default, thresholds scale with sqrt(horizon / reference_horizon), which keeps
    short-horizon labels from becoming overwhelmingly HOLD while preserving symmetric
    BUY/SELL semantics.
    """
    buy_threshold = float(config["buy_threshold"])
    sell_threshold = float(config["sell_threshold"])

    threshold_cfg = config.get("threshold_scaling", {})
    enabled = bool(threshold_cfg.get("enabled", True))
    mode = str(threshold_cfg.get("mode", "sqrt_horizon")).lower().strip()

    if not enabled or mode == "fixed":
        return buy_threshold, sell_threshold

    reference_horizon = int(
        threshold_cfg.get(
            "reference_horizon",
            config.get("default_prediction_horizon", config.get("prediction_horizon", 21)),
        )
    )

    if reference_horizon <= 0 or horizon <= 0:
        return buy_threshold, sell_threshold

    if mode != "sqrt_horizon":
        raise ValueError("Unsupported threshold scaling mode. Use one of: fixed, sqrt_horizon")

    scale = math.sqrt(float(horizon) / float(reference_horizon))
    scaled_buy = buy_threshold * scale
    scaled_sell = sell_threshold * scale

    min_abs_threshold = float(threshold_cfg.get("min_abs_threshold", 0.005))
    max_abs_threshold = float(threshold_cfg.get("max_abs_threshold", 0.20))

    scaled_buy = min(max(abs(scaled_buy), min_abs_threshold), max_abs_threshold)
    scaled_sell = -min(max(abs(scaled_sell), min_abs_threshold), max_abs_threshold)
    return float(scaled_buy), float(scaled_sell)
