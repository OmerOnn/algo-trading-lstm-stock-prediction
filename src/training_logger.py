from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO


class TeeStream:
    """
    Writes output to multiple streams at the same time.

    This allows us to print logs to the terminal and also save them
    into a log file.
    """

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, message: str) -> int:
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def build_training_log_path(logs_dir: Path, horizons: list[int], prefix: str = "lstm") -> Path:
    """
    Creates a clean log filename.

    Example:
    logs/training_run_lstm_20260601_154233_h5_h10_h20_h30.log
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    horizons_part = "_".join(f"h{h}" for h in horizons)

    return logs_dir / f"training_run_{prefix}_{timestamp}_{horizons_part}.log"


@contextmanager
def training_log_context(logs_dir: Path, horizons: list[int], prefix: str = "lstm"):
    """
    Context manager that saves all terminal output into a log file.

    It captures:
    - print(...)
    - errors
    - warnings printed to stderr

    But it still shows everything in the terminal normally.
    """
    log_path = build_training_log_path(logs_dir, horizons, prefix=prefix)

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee_stdout = TeeStream(sys.__stdout__, log_file)
        tee_stderr = TeeStream(sys.__stderr__, log_file)

        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print("=" * 80)
            print("Training log started")
            print(f"Log file: {log_path}")
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            try:
                yield log_path
            finally:
                print("=" * 80)
                print(f"Training log finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Saved full log to: {log_path}")
                print("=" * 80)