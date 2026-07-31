from __future__ import annotations

import torch

from src.device import get_best_device, resolve_xgboost_backend


def main() -> None:
    print("PyTorch device check (LSTM)")
    print("===========================")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    info = get_best_device("auto")
    print(f"Selected device: {info.device}")
    print(f"Accelerator: {info.accelerator}")
    print(f"Device name: {info.device_name}")

    x = torch.tensor([1.0, 2.0, 3.0], device=info.device)
    y = x * 2
    print(f"Small tensor test: {y.cpu().tolist()}")

    print("\nXGBoost device check")
    print("====================")
    backend = resolve_xgboost_backend("auto")
    print(f"XGBoost version: {backend.xgboost_version}")
    print(f"Compiled with CUDA: {backend.cuda_build}")
    print(f"Compiled with OpenMP: {backend.openmp_build}")
    print(f"Selected device: {backend.device} ({backend.accelerator})")
    if backend.device == "cpu":
        print(f"CPU threads: {backend.threads}")
    print(backend.message)
    if info.device.type == "mps":
        print(
            "\nNote: the Apple GPU is used by PyTorch only. XGBoost's device parameter\n"
            "accepts cpu, cuda, and cuda:<ordinal> — it has no Metal/MPS backend, so\n"
            "there is no setting that would move XGBoost training onto this GPU."
        )


if __name__ == "__main__":
    main()
