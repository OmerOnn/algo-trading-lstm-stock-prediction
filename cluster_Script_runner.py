"""Repository job launcher.

Edit RUN_PRESET, then run this file to submit the selected script through
JobManager.  This keeps the VS Code play-button workflow simple while still
submitting the real script as a SLURM job.
"""

from __future__ import annotations

import os

from utils.job_manager import JobManager
from utils.main_utils import create_or_clear_log_file

# need to change to your root on cluster -> need to clone the project there
PROJECT_ROOT = ""

RUN_PRESET = "train_all_models"

# python3 train_all_models.py \
#   --horizons 1,5,10,21,30,126,252,1260,2520 \
#   --walk-forward \
#   --compare

PRESETS = {
    "drl_hparam": {
        "main_path": os.path.join(PROJECT_ROOT, "train_all_models.py"),
        "args": "--horizons 1,5,10,21,30,126,252,1260,2520 --walk-forward --compare",
        "num_jobs": 1,
        "fixed_mem_gb": 128,
        "max_cpus_per_job": 64,
    },
}


def main() -> None:
    """Submit the selected preset as a SLURM job."""
    preset = PRESETS[RUN_PRESET]
    create_or_clear_log_file()

    print(f"[main] preset: {RUN_PRESET}", flush=True)
    print(f"[main] script: {preset['main_path']}", flush=True)
    print(f"[main] args: {preset['args']}", flush=True)

    manager = JobManager(
        main_path=preset["main_path"],
        num_jobs=int(preset["num_jobs"]),
        fixed_mem_gb=preset["fixed_mem_gb"],
        max_cpus_per_job=int(preset["max_cpus_per_job"]),
    )
    manager.args = str(preset["args"])
    manager.create_jobs()


if __name__ == "__main__":
    main()
