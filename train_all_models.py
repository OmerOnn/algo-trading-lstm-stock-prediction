from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train import get_horizons, load_config
from src.device import get_best_device
from train_xgboost import resolve_xgboost_backend


ROOT = Path(__file__).resolve().parent


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


def run_command(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


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

    run_command(["train.py", *([] if selected_horizon is None else ["--horizon", str(selected_horizon)])])
    run_command(
        ["train_xgboost.py", *([] if selected_horizon is None else ["--horizon", str(selected_horizon)])]
    )

    if compare:
        for horizon in horizons:
            run_command(["compare_results.py", "--horizon", str(horizon)])

    print("\nFinished training all requested models.")
    print("Checkpoints are saved under models/ and reports/ with horizon-specific filenames.")
    return horizons


def main() -> None:
    args = parse_args()
    train_all_models(selected_horizon=args.horizon, compare=args.compare)


if __name__ == "__main__":
    main()
