from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full LSTM vs XGBoost comparison workflow.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction horizon in trading days. Defaults to the configured default horizon.",
    )
    return parser.parse_args()


def load_default_horizon() -> int:
    import yaml

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return int(config.get("default_prediction_horizon", config.get("prediction_horizon", 10)))


def run_command(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    horizon = int(args.horizon) if args.horizon is not None else load_default_horizon()

    print(f"Running full model comparison for horizon={horizon}")
    run_command(["train.py", "--horizon", str(horizon)])
    run_command(["train_xgboost.py", "--horizon", str(horizon)])
    run_command(["compare_results.py", "--horizon", str(horizon)])

    print("\nFinished LSTM vs XGBoost comparison.")


if __name__ == "__main__":
    main()
