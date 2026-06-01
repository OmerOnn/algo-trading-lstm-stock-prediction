from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def main() -> None:
    """
    Simple experiment helper.

    It runs the same project twice: once with GRU and once with LSTM.
    Results are saved by train.py. Rename the reports folder between runs if you
    want to keep both full sets of plots and CSV files.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    original_model_type = config.get("model_type", "gru")

    for model_type in ["gru", "lstm"]:
        print(f"\nRunning experiment with model_type={model_type}")
        config["model_type"] = model_type
        config["model_output_path"] = f"models/stock_advanced_model_{model_type}.pt"
        config["metadata_output_path"] = f"models/model_metadata_{model_type}.json"
        config["metrics_output_path"] = f"reports/metrics_{model_type}.json"
        config["backtest_output_path"] = f"reports/backtest_results_{model_type}.csv"

        with open(CONFIG_PATH, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        subprocess.run(["python", "train.py"], cwd=ROOT, check=True)

    config["model_type"] = original_model_type
    config["model_output_path"] = "models/stock_advanced_model.pt"
    config["metadata_output_path"] = "models/model_metadata.json"
    config["metrics_output_path"] = "reports/metrics.json"
    config["backtest_output_path"] = "reports/backtest_results.csv"

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print("\nFinished GRU/LSTM comparison.")


if __name__ == "__main__":
    main()
