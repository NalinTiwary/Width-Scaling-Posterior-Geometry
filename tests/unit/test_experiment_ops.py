"""Local checks for the operations kit. These do not need a GPU."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
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
    assert pending.indices("6-11") == set(range(6, 12))
    with pytest.raises(ValueError):
        pending.indices("5-1")


def test_retained_length_matches_the_requested_level():
    import sys
    sys.path.insert(0, str(ROOT / "a10_next"))
    import cluster
    assert sum(n for _, n in cluster.production_segments(1)) == 8192
    assert sum(n for _, n in cluster.production_segments(2)) == 16384
    assert sum(n for _, n in cluster.production_segments(3)) == 32768
    assert [name for name, _ in cluster.production_segments(3)] == [
        "production_1", "production_2", "production_3"]


def _pending(tmp_path, ledger_text, level):
    ledger = tmp_path / "active.csv"
    ledger.write_text(ledger_text)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "experiment_ops" / "pending.py"),
         "reference", "endpoint", "6-11", str(level),
         "--ledger", str(ledger), "--out", str(tmp_path / "empty")],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    return proc.stdout.strip()


def test_a_finished_level_one_row_does_not_block_a_longer_run(tmp_path):
    (tmp_path / "empty").mkdir()
    old = 'job,scope,ids,slurm_id\nreference,endpoint,"6,9",12605023\n'
    assert _pending(tmp_path, old, 3) == "6,7,8,9,10,11"
    longer = 'job,scope,ids,slurm_id,level\nreference,endpoint,"6,9",12605023,3\n'
    assert _pending(tmp_path, longer, 3) == "7,8,10,11"
