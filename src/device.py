from __future__ import annotations

import os
from dataclasses import dataclass, field

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
        "auto", "cuda", "mps", "metal", "mac", or "cpu".
    """
    preferred_device = str(preferred_device or "auto").lower().strip()
    aliases = {"metal": "mps", "mac": "mps", "apple": "mps"}
    preferred_device = aliases.get(preferred_device, preferred_device)
    allowed = {"auto", "cuda", "mps", "cpu"}
    if preferred_device not in allowed:
        raise ValueError("Invalid device. Use one of: auto, cuda, mps, metal, mac, apple, cpu")

    if preferred_device in {"auto", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        return DeviceInfo(device=device, device_name=name, accelerator="CUDA / NVIDIA GPU")

    if preferred_device == "cuda":
        raise RuntimeError("CUDA was requested, but no CUDA-capable NVIDIA GPU is available to PyTorch.")

    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    if preferred_device in {"auto", "mps"} and mps_available:
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


# ---------------------------------------------------------------------------
# XGBoost execution backend
# ---------------------------------------------------------------------------
#
# XGBoost's GPU support is CUDA-only. Its ``device`` parameter accepts exactly
# ``cpu``, ``cuda``, and ``cuda:<ordinal>``; anything else — including ``mps``
# and ``metal`` — is rejected by the C++ layer with an "Invalid argument for
# `device`" error. There is no Metal, MPS, or OpenCL tree-building backend in
# any XGBoost release, so on Apple Silicon the CPU ``hist`` builder is the only
# way to train, regardless of how capable the GPU is for PyTorch. The LSTM's
# use of MPS says nothing about what XGBoost can do: they share a machine, not
# a compute backend.
#
# The subtlety that motivates checking the *build* rather than the machine:
# ``device="cuda"`` on a wheel compiled without CUDA does not raise. It emits a
# warning and silently trains on CPU. Gating on ``torch.cuda.is_available()``
# alone therefore lets the trainer report "CUDA acceleration is active" while
# every tree is in fact being built by OpenMP threads. The authoritative signal
# is ``xgboost.build_info()["USE_CUDA"]``, which describes the binary that will
# actually run.


@dataclass(frozen=True)
class XGBoostBackend:
    """The resolved XGBoost execution backend and the evidence behind it."""

    device: str
    message: str
    xgboost_version: str = "not installed"
    cuda_build: bool = False
    openmp_build: bool = False
    threads: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def accelerator(self) -> str:
        return "CUDA / NVIDIA GPU" if self.device == "cuda" else "CPU"

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "accelerator": self.accelerator,
            "message": self.message,
            "xgboost_version": self.xgboost_version,
            "compiled_with_cuda": bool(self.cuda_build),
            "compiled_with_openmp": bool(self.openmp_build),
            "threads": int(self.threads),
            "notes": list(self.notes),
        }


def _xgboost_build_facts() -> tuple[str, bool, bool]:
    """(version, compiled_with_cuda, compiled_with_openmp) for the installed wheel."""
    try:
        import xgboost
    except ModuleNotFoundError:
        return "not installed", False, False

    version = str(getattr(xgboost, "__version__", "unknown"))
    build_info = getattr(xgboost, "build_info", None)
    if build_info is None:  # xgboost < 1.6 has no build_info()
        return version, False, True
    try:
        info = build_info()
    except Exception:
        return version, False, True
    return version, bool(info.get("USE_CUDA", False)), bool(info.get("USE_OPENMP", True))


def _effective_thread_count(n_jobs: int = -1) -> int:
    """How many threads XGBoost's OpenMP pool will actually use."""
    if int(n_jobs) > 0:
        return int(n_jobs)
    for variable in ("OMP_NUM_THREADS", "XGBOOST_NUM_THREADS"):
        value = os.environ.get(variable)
        if value and value.strip().isdigit():
            return int(value.strip())
    return int(os.cpu_count() or 1)


def resolve_xgboost_backend(preferred_device: str = "auto", n_jobs: int = -1) -> XGBoostBackend:
    """Resolve the execution backend XGBoost will genuinely use.

    Parameters
    ----------
    preferred_device:
        "auto", "cuda"/"gpu", "cpu", or one of the Apple aliases
        ("mps", "metal", "mac", "apple"). The Apple aliases resolve to CPU with
        an explanation rather than an error, because asking for the Mac GPU is a
        reasonable thing to try — it just is not something XGBoost implements.
    n_jobs:
        The ``n_jobs`` the trainer will pass to the estimator, used only to
        report the thread count truthfully.
    """
    requested = str(preferred_device or "auto").lower().strip()
    aliases = {"gpu": "cuda", "metal": "mps", "mac": "mps", "apple": "mps"}
    requested = aliases.get(requested, requested)
    allowed = {"auto", "cuda", "mps", "cpu"}
    if requested not in allowed:
        raise ValueError(
            "Invalid XGBoost device. Use one of: auto, cuda, gpu, cpu, mps, metal, mac, apple"
        )

    version, cuda_build, openmp_build = _xgboost_build_facts()
    threads = _effective_thread_count(n_jobs)
    cuda_visible = bool(torch.cuda.is_available())
    # A usable CUDA backend needs both: a binary that contains the GPU kernels
    # and a physical device to run them on. Either one alone means CPU.
    cuda_usable = cuda_build and cuda_visible

    def facts(*extra: str) -> list[str]:
        base = [
            f"xgboost {version} compiled with CUDA: {cuda_build}",
            f"xgboost compiled with OpenMP: {openmp_build}",
            f"CUDA device visible to PyTorch: {cuda_visible}",
            "XGBoost has no Metal/MPS backend; its device parameter accepts only "
            "cpu, cuda, and cuda:<ordinal>",
        ]
        base.extend(extra)
        return base

    def cpu(message: str, *extra: str) -> XGBoostBackend:
        if not openmp_build:
            extra = extra + (
                "this wheel was built without OpenMP, so training is single-threaded; "
                "installing xgboost from a wheel with OpenMP support would be a large speed-up",
            )
        return XGBoostBackend(
            device="cpu",
            message=message,
            xgboost_version=version,
            cuda_build=cuda_build,
            openmp_build=openmp_build,
            threads=threads,
            notes=facts(*extra),
        )

    if requested == "cuda":
        if not cuda_build:
            raise RuntimeError(
                "XGBoost CUDA was requested, but the installed xgboost "
                f"({version}) was compiled without CUDA support. On macOS this is "
                "expected: no CUDA-enabled XGBoost wheel exists for Apple Silicon, "
                "and Apple GPUs are not CUDA devices. Setting device='cuda' here "
                "would not fail loudly — XGBoost would warn and silently train on "
                "CPU. Use device: cpu or device: auto."
            )
        if not cuda_visible:
            raise RuntimeError(
                "XGBoost CUDA was requested, but no CUDA-capable NVIDIA GPU is available."
            )

    if cuda_usable and requested in {"auto", "cuda"}:
        return XGBoostBackend(
            device="cuda",
            message=(
                f"CUDA GPU acceleration is active for XGBoost ({torch.cuda.get_device_name(0)})."
            ),
            xgboost_version=version,
            cuda_build=cuda_build,
            openmp_build=openmp_build,
            threads=threads,
            notes=facts(),
        )

    if requested == "mps":
        return cpu(
            "Apple Metal/MPS is not an XGBoost backend, so XGBoost is running on CPU. "
            "PyTorch uses the Apple GPU for the LSTM, but XGBoost's only GPU backend "
            "is CUDA and it rejects device='mps' outright.",
            "the Apple GPU alias was requested and mapped to CPU rather than passed "
            "through, because XGBoost would raise on it",
        )

    if requested == "cpu":
        return cpu(f"XGBoost is configured to run on CPU using {threads} threads.")

    if not cuda_build:
        return cpu(
            "XGBoost is running on CPU: this build has no GPU support compiled in "
            f"(CPU hist builder, {threads} threads)."
        )
    return cpu(
        "No CUDA-capable NVIDIA GPU detected, so XGBoost is running on CPU "
        f"({threads} threads)."
    )
