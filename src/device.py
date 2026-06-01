from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Small container describing the selected PyTorch device."""

    device: torch.device
    device_name: str
    accelerator: str


def get_best_device(preferred_device: str = "auto") -> DeviceInfo:
    """Return the best available device for PyTorch training/inference.

    Priority in auto mode:
    1. NVIDIA GPU through CUDA
    2. Apple Silicon GPU through MPS
    3. CPU fallback

    Parameters
    ----------
    preferred_device:
        "auto", "cuda", "mps", or "cpu".
    """
    preferred_device = str(preferred_device or "auto").lower().strip()
    allowed = {"auto", "cuda", "mps", "cpu"}
    if preferred_device not in allowed:
        raise ValueError(f"Invalid device '{preferred_device}'. Use one of: {sorted(allowed)}")

    if preferred_device in {"auto", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        return DeviceInfo(device=device, device_name=name, accelerator="CUDA / NVIDIA GPU")

    if preferred_device == "cuda":
        raise RuntimeError("CUDA was requested, but no CUDA-capable NVIDIA GPU is available to PyTorch.")

    if preferred_device in {"auto", "mps"} and torch.backends.mps.is_available():
        device = torch.device("mps")
        return DeviceInfo(device=device, device_name="Apple Silicon GPU", accelerator="MPS / Apple Metal GPU")

    if preferred_device == "mps":
        raise RuntimeError("MPS was requested, but Apple Metal GPU acceleration is not available to PyTorch.")

    device = torch.device("cpu")
    return DeviceInfo(device=device, device_name="CPU", accelerator="CPU")


def dataloader_device_kwargs(device: torch.device, num_workers: int = 0, pin_memory: bool = True) -> dict:
    """Return DataLoader options that are safe for the selected device.

    pin_memory is useful mainly when transferring batches from CPU to CUDA.
    It is disabled for CPU and MPS to avoid unnecessary overhead.
    """
    num_workers = int(num_workers or 0)
    kwargs = {"num_workers": num_workers}

    if device.type == "cuda":
        kwargs["pin_memory"] = bool(pin_memory)
        if num_workers > 0:
            kwargs["persistent_workers"] = True
    else:
        kwargs["pin_memory"] = False

    return kwargs


def move_batch_to_device(batch, device: torch.device):
    """Move a tuple/list of tensors to the selected device."""
    return tuple(item.to(device, non_blocking=(device.type == "cuda")) for item in batch)
