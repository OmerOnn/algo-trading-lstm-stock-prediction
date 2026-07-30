from __future__ import annotations

import argparse
import sys

from src.data_download import preload_training_data
from src.device import get_best_device
from src.training_common import get_horizons, load_config, parse_horizon_list
from train_lstm import train_models_for_horizons as train_lstm_models_for_horizons
from train_xgboost import resolve_xgboost_backend
from train_xgboost import train_models_for_horizons as train_xgboost_models_for_horizons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM and XGBoost models for one horizon or for all configured horizons."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Train only one prediction horizon. If omitted, trains all configured horizons.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run compare_results.py after both models finish for each trained horizon.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated horizons to train, for example 21,63.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run purged walk-forward validation for both model families.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run the documented XGBoost hyperparameter grid on purged folds.",
    )
    parser.add_argument(
        "--permutation-importance",
        action="store_true",
        help="Compute out-of-fold permutation importance and recommend a feature blocklist.",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="Fit the constrained LSTM/XGBoost blend after both families finish.",
    )
    return parser.parse_args()


def train_all_models(
    selected_horizon: int | None = None,
    selected_horizons: list[int] | None = None,
    compare: bool = False,
    walk_forward: bool = False,
    grid_search: bool = False,
    permutation_importance: bool = False,
    blend: bool = False,
) -> list[int]:
    config = load_config()
    horizons = get_horizons(config, selected_horizon, selected_horizons)
    device_info = get_best_device(config.get("device", "auto"))
    xgb_device, xgb_backend_message = resolve_xgboost_backend(config)

    print("Training workflow")
    print("-----------------")
    print(f"Horizons: {horizons}")
    print("Models: LSTM, XGBoost")
    print(f"Compare reports: {'enabled' if compare else 'disabled'}")
    print(f"Walk-forward validation: {'enabled' if walk_forward else 'disabled'}")
    print(
        f"LSTM backend: {device_info.accelerator} "
        f"({device_info.device}) - {device_info.device_name}"
    )
    if device_info.device.type == "mps":
        print("Apple GPU acceleration is active for LSTM training through PyTorch MPS.")
    elif device_info.device.type == "cuda":
        print("NVIDIA GPU acceleration is active for LSTM training through CUDA.")
    else:
        print("LSTM training is running on CPU because no supported GPU backend was selected.")
    print(f"XGBoost backend: {xgb_device.upper()}")
    print(xgb_backend_message)

    preload_training_data(
        tickers=config["tickers"],
        benchmark_ticker=config["benchmark_ticker"],
        start=config["start_date"],
        end=config["end_date"],
        macro_tickers=config.get("macro_tickers"),
        earnings_limit=int(config.get("earnings_history_limit", 100)),
        preload_earnings=bool(config.get("use_earnings_features", False)),
    )

    train_lstm_models_for_horizons(config, horizons, walk_forward=walk_forward)
    train_xgboost_models_for_horizons(
        config,
        horizons,
        walk_forward=walk_forward,
        grid_search=grid_search,
        permutation=permutation_importance,
    )

    if blend:
        # The blend needs out-of-fold predictions from both families, which only
        # a walk-forward run produces.
        from blend_models import main as blend_main

        for horizon in horizons:
            original_argv = sys.argv[:]
            try:
                sys.argv = ["blend_models.py", "--horizon", str(horizon)]
                blend_main()
            finally:
                sys.argv = original_argv

    if compare:
        from compare_results import main as compare_results_main

        for horizon in horizons:
            original_argv = sys.argv[:]
            try:
                sys.argv = ["compare_results.py", "--horizon", str(horizon)]
                compare_results_main()
            finally:
                sys.argv = original_argv

    print("\nFinished training all requested models.")
    print("Checkpoints are saved under models/ and reports/ with horizon-specific filenames.")
    return horizons


def main() -> None:
    args = parse_args()
    selected_horizons = parse_horizon_list(args.horizons)
    train_all_models(
        selected_horizon=args.horizon,
        selected_horizons=selected_horizons,
        compare=args.compare,
        walk_forward=bool(args.walk_forward),
        grid_search=bool(args.grid_search),
        permutation_importance=bool(args.permutation_importance),
        blend=bool(args.blend),
    )


if __name__ == "__main__":
    main()
