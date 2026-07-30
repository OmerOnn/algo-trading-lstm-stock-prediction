"""
Fit and evaluate the constrained LSTM / XGBoost blend.

Run after both trainers have completed a ``--walk-forward`` run, which is what
writes the out-of-fold prediction files this consumes:

```bash
python3 train_lstm.py    --horizon 21 --walk-forward
python3 train_xgboost.py --horizon 21 --walk-forward
python3 blend_models.py  --horizon 21
```

Weights are fitted **only** on purged walk-forward out-of-fold predictions, on
rows where both families produced a forecast. They are non-negative and sum to
one, and the blend is retained only if it beats the better standalone model on
the out-of-fold consistency score. The development holdout is then scored once
with the frozen weights, for reporting.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.blending import align_model_predictions, fit_blend_weights
from src.regression import cross_sectional_metrics, full_metrics
from src.training_common import ROOT, json_safe, load_config


MODEL_KEYS = {"lstm": "LSTM", "xgboost": "XGBoost"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend the LSTM and XGBoost forecasts.")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon.")
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=0.01,
        help="Out-of-fold score the blend must add over the best single model to be retained.",
    )
    return parser.parse_args()


def load_out_of_fold(horizon: int) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key, label in MODEL_KEYS.items():
        path = ROOT / "reports" / f"oof_predictions_{key}_h{horizon}.csv"
        if not path.exists():
            print(
                f"  missing {path.name} — run: python3 train_{key}.py "
                f"--horizon {horizon} --walk-forward"
            )
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frames[label] = frame
        print(f"  {label}: {len(frame):,} out-of-fold rows across {frame['fold'].nunique()} folds")
    return frames


def load_test_predictions(horizon: int) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key, label in MODEL_KEYS.items():
        name = f"test_predictions_h{horizon}.csv" if key == "lstm" else f"test_predictions_xgboost_h{horizon}.csv"
        path = ROOT / "reports" / name
        if path.exists():
            frames[label] = pd.read_csv(path, parse_dates=["date"])
    return frames


def folds_for_blending(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Build per-fold blending inputs from aligned rows.

    Alignment is an inner join on ``(ticker, date)``. The two families do not
    cover identical rows — the sequence model needs a full look-back window and
    so drops the first rows of every ticker — and lining them up positionally
    would compare one model's forecast for one stock against another's for a
    different stock.
    """
    per_fold: list[dict] = []
    folds = sorted(set.intersection(*[set(frame["fold"]) for frame in frames.values()]))
    for fold in folds:
        subset = {name: frame[frame["fold"] == fold] for name, frame in frames.items()}
        aligned = align_model_predictions(subset)
        if aligned.empty:
            continue
        per_fold.append(
            {
                "fold": int(fold),
                "true_return": aligned["true_return"].to_numpy(dtype=float),
                "dates": aligned["date"],
                "reference_prediction": float(
                    next(iter(subset.values()))["reference_prediction"].iloc[0]
                ),
                "predictions": {
                    name: aligned[f"prediction_{name}"].to_numpy(dtype=float) for name in subset
                },
                "rows": int(len(aligned)),
            }
        )
        print(f"  fold {fold}: {len(aligned):,} aligned rows")
    return per_fold


def main() -> None:
    args = parse_args()
    config = load_config()
    horizon = int(
        args.horizon
        if args.horizon is not None
        else config.get("default_prediction_horizon", 21)
    )

    print(f"Blending LSTM and XGBoost for horizon {horizon}")
    print("=" * 60)
    print("\nLoading out-of-fold predictions:")
    frames = load_out_of_fold(horizon)
    if len(frames) < 2:
        print("\nNeed out-of-fold predictions from both families to fit a blend. Nothing to do.")
        return

    print("\nAligning folds on (ticker, date):")
    per_fold = folds_for_blending(frames)
    if not per_fold:
        print("No overlapping rows between the two families. Nothing to blend.")
        return

    blend = fit_blend_weights(
        per_fold, horizon=horizon, minimum_improvement=float(args.minimum_improvement)
    )

    print("\nOut-of-fold score per standalone model (mean - std across folds):")
    for name, score in sorted(blend.per_model_scores.items()):
        print(f"  {name:<10} {score:+.4f}")
    print(f"\nFitted weights: {', '.join(f'{k}={v:.2f}' for k, v in blend.weights.items())}")
    print(f"Retained: {blend.retained}")
    print(f"Reason:  {blend.reason}")

    payload = {
        "horizon": horizon,
        "weights": blend.to_dict(),
        "constraints": "non-negative, sum to one, fitted on purged out-of-fold rows only",
        "folds": [
            {"fold": fold["fold"], "aligned_rows": fold["rows"]} for fold in per_fold
        ],
        "candidate_grid": getattr(blend, "candidate_grid", []),
    }

    # Score the frozen weights once on the development holdout, for reporting.
    test_frames = load_test_predictions(horizon)
    if len(test_frames) == 2:
        aligned = align_model_predictions(test_frames)
        if not aligned.empty:
            truth = aligned["true_return"].to_numpy(dtype=float)
            columns = {
                name: aligned[f"prediction_{name}"].to_numpy(dtype=float) for name in test_frames
            }
            blended = blend.apply(columns)
            reference = float(np.mean([fold["reference_prediction"] for fold in per_fold]))

            print(f"\nDevelopment holdout ({len(aligned):,} aligned rows):")
            holdout: dict[str, dict] = {}
            for name, values in list(columns.items()) + [("Blend", blended)]:
                metrics = full_metrics(
                    aligned["date"], truth, values, horizon, reference_prediction=reference
                )
                holdout[name] = {
                    "cross_sectional_ic": metrics["cross_sectional_ic"],
                    "icir": metrics["cross_sectional_icir"],
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "mse_skill_vs_historical_mean": metrics.get(
                        "mse_skill_vs_historical_mean", 0.0
                    ),
                    "direction_accuracy": metrics["direction_accuracy"],
                }
                print(
                    f"  {name:<10} IC {metrics['cross_sectional_ic']:+.4f} | "
                    f"ICIR {metrics['cross_sectional_icir']:+.3f} | "
                    f"MSE {metrics['mse']:.6f} | "
                    f"MSE skill {metrics.get('mse_skill_vs_historical_mean', 0.0):+.4f}"
                )
            payload["development_holdout"] = holdout
            payload["development_holdout_rows"] = int(len(aligned))
            payload["holdout_policy"] = (
                "weights were frozen from out-of-fold data before this was scored; "
                "the holdout is reported, never used to choose the weights"
            )

            aligned["prediction_blend"] = blended
            aligned.to_csv(ROOT / "reports" / f"blend_predictions_h{horizon}.csv", index=False)

    path = ROOT / "reports" / f"blend_h{horizon}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, indent=2)
    print(f"\nSaved blend report to: {path}")


if __name__ == "__main__":
    main()
