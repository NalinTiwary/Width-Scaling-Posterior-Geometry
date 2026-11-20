"""Local checks for the operations kit. These do not need a GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finished_finalists_and_time_zero_are_not_resubmitted():
    pending = _load(ROOT / "experiment_ops" / "pending.py", "pending_ops")
    out = ROOT / "results" / "revision_next"
    assert pending.complete(out, "finalists", "endpoint", "deep_m0032_r0", 8192)
    assert pending.complete(out, "finalists", "endpoint", "shallow_m4096_r2", 8192)
    assert pending.complete(out, "cold-zero", "endpoint", "deep_m0032_r0", 8192)
    assert pending.complete(out, "cold-zero", "endpoint", "shallow_m4096_r2", 8192)


def test_prior_start_accuracy_intervals_cover_a_known_shift():
    accuracy = _load(ROOT / "experiment_ops" / "prior_start_accuracy.py", "accuracy_ops").accuracy
    values = np.linspace(1.0, 3.0, 32)
    result = accuracy(values, reference=0.0, reference_sd=1.0, reference_mcse=0.0, seed=1)
    assert result["standardized_bias"] == pytest.approx(float(values.mean()))
    assert result["bias_lo"] < result["standardized_bias"] < result["bias_hi"]


def test_index_lists_expand_and_reject_reversed_ranges():
    pending = _load(ROOT / "experiment_ops" / "pending.py", "pending_ops_indices")
    assert pending.indices("0-5,7-11") == set(range(0, 6)) | set(range(7, 12))
    with pytest.raises(ValueError):
        pending.indices("5-1")
