from __future__ import annotations

import torch

from src.device import get_best_device


def main() -> None:
    print("PyTorch device check")
    print("====================")
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


if __name__ == "__main__":
    main()
