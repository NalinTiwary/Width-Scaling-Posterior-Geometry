"""Synchronized block timing. The timer does not draw random numbers."""

from __future__ import annotations

import platform
import time
from typing import Any, Callable

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware_record(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(device)
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": gpu,
        "dtype": "float64",
        "threads": int(torch.get_num_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def time_blocks(step: Callable[[], None], device: torch.device, *, transitions: int, blocks: int,
                min_seconds: float = 0.0, max_doublings: int = 6) -> dict[str, Any]:
    """Time `blocks` calls of `step`, which itself must run `transitions` kernel steps.

    When min_seconds > 0, the block length doubles until one timed block exceeds that
    duration or max_doublings is reached. Warm-up is the caller's responsibility and is
    not included here.
    """
    if transitions <= 0 or blocks <= 0:
        raise ValueError("transitions and blocks must be positive")
    length = int(transitions)
    doublings = 0
    records: list[dict[str, float]] = []
    while True:
        records = []
        for _ in range(blocks):
            synchronize(device)
            t0 = time.perf_counter()
            for _s in range(length):
                step()
            synchronize(device)
            dt = time.perf_counter() - t0
            records.append({"seconds": dt, "transitions": length})
        if min_seconds <= 0 or min(r["seconds"] for r in records) >= min_seconds or doublings >= max_doublings:
            break
        length *= 2
        doublings += 1
    seconds = [r["seconds"] for r in records]
    total_transitions = length * blocks
    total_seconds = float(sum(seconds))
    return {
        "blocks": records,
        "transitions_per_block": length,
        "doublings": doublings,
        "total_transitions": total_transitions,
        "total_seconds": total_seconds,
        "seconds_per_transition": total_seconds / total_transitions,
        "min_seconds": float(min(seconds)),
        "hardware": hardware_record(device),
        "met_duration_rule": bool(min_seconds <= 0 or min(seconds) >= min_seconds),
    }
