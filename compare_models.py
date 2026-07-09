from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM and XGBoost training and compare results.")
    parser.add_argument("--horizon", type=int, default=None, help="Horizon to train and compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command_args = ["python", "train.py"]
    xgb_args = ["python", "train_xgboost.py"]
    compare_args = ["python", "compare_results.py"]

    if args.horizon is not None:
        command_args.extend(["--horizon", str(args.horizon)])
        xgb_args.extend(["--horizon", str(args.horizon)])
        compare_args.extend(["--horizon", str(args.horizon)])

    subprocess.run(command_args, cwd=ROOT, check=True)
    subprocess.run(xgb_args, cwd=ROOT, check=True)
    subprocess.run(compare_args, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
