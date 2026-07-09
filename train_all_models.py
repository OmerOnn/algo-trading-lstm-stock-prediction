from __future__ import annotations

import argparse
<<<<<<< HEAD
import subprocess
import sys
from pathlib import Path

from src.device import get_best_device
from train import get_horizons, load_config
from train_xgboost import resolve_xgboost_backend


ROOT = Path(__file__).resolve().parent
=======

from train import get_horizons, load_config, train_models_for_horizons as train_lstm_models_for_horizons
from src.device import get_best_device
from src.data_download import preload_training_data
from train_xgboost import resolve_xgboost_backend
from train_xgboost import train_models_for_horizons as train_xgboost_models_for_horizons
>>>>>>> bc65240c7ac90ca33229b03fd61c50d107fdb64b


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
    return parser.parse_args()


<<<<<<< HEAD
def run_command(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


=======
>>>>>>> bc65240c7ac90ca33229b03fd61c50d107fdb64b
def train_all_models(selected_horizon: int | None = None, compare: bool = False) -> list[int]:
    config = load_config()
    horizons = get_horizons(config, selected_horizon)
    device_info = get_best_device(config.get("device", "auto"))
    xgb_device, xgb_backend_message = resolve_xgboost_backend(config)

    print("Training workflow")
    print("-----------------")
    print(f"Horizons: {horizons}")
    print("Models: LSTM, XGBoost")
    print(f"Compare reports: {'enabled' if compare else 'disabled'}")
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
<<<<<<< HEAD

    run_command(["train.py", *([] if selected_horizon is None else ["--horizon", str(selected_horizon)])])
    run_command(
        ["train_xgboost.py", *([] if selected_horizon is None else ["--horizon", str(selected_horizon)])]
    )

    if compare:
        for horizon in horizons:
            run_command(["compare_results.py", "--horizon", str(horizon)])
=======
    preload_training_data(
        tickers=config["tickers"],
        benchmark_ticker=config["benchmark_ticker"],
        start=config["start_date"],
        end=config["end_date"],
        macro_tickers=config.get("macro_tickers"),
        earnings_limit=int(config.get("earnings_history_limit", 100)),
    )

    train_lstm_models_for_horizons(config, horizons)
    train_xgboost_models_for_horizons(config, horizons)

    if compare:
        from compare_results import main as compare_results_main
        import sys

        for horizon in horizons:
            original_argv = sys.argv[:]
            try:
                sys.argv = ["compare_results.py", "--horizon", str(horizon)]
                compare_results_main()
            finally:
                sys.argv = original_argv
>>>>>>> bc65240c7ac90ca33229b03fd61c50d107fdb64b

    print("\nFinished training all requested models.")
    print("Checkpoints are saved under models/ and reports/ with horizon-specific filenames.")
    return horizons


def main() -> None:
    args = parse_args()
    train_all_models(selected_horizon=args.horizon, compare=args.compare)


if __name__ == "__main__":
    main()
