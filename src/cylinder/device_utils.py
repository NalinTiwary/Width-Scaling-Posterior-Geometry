"""Device resolution and float64 defaults for CPU / CUDA."""

from __future__ import annotations

import os
from typing import Optional

import torch


def configure_dtype() -> torch.dtype:
    """Force float64 as the reference dtype everywhere."""
    torch.set_default_dtype(torch.float64)
    return torch.float64


def resolve_device(requested: str = "auto", *, require_float64: bool = True) -> torch.device:
    """
    Resolve `auto | cpu | cuda | mps` to a torch.device.

    auto: CUDA if available, else CPU.
    MPS is skipped when require_float64=True because the Metal backend
    does not support float64 (the protocol's reference dtype).
    """
    req = (requested or "auto").lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    if req == "mps":
        if require_float64:
            raise RuntimeError(
                "device=mps is incompatible with float64; use cuda or cpu"
            )
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("device=mps requested but MPS is not available")
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device request: {requested!r}")


def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize accelerators before timing."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    # MPS has no reliable public synchronize; no-op.


def device_metadata(requested: str, effective: torch.device) -> dict:
    """Record software / device provenance for run metadata."""
    info = {
        "requested_device": requested,
        "effective_device": str(effective),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "thread_count": int(torch.get_num_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "dtype": str(torch.get_default_dtype()),
    }
    if effective.type == "cuda":
        idx = effective.index or 0
        info["cuda_device_name"] = torch.cuda.get_device_name(idx)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info
