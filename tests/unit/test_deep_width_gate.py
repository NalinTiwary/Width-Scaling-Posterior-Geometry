"""The width-4096 Langevin job checks itself and continues in the same allocation."""
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from deep_width.core import PROBES, gate_decision, needs_inline_check, remaining_joint_steps
from deep_width.core import require_comparative_device, segments_for
from deep_width.run import continue_after_short_check, read_short_run, short_check_then_run


def _args(out, **kwargs):
    values = dict(action="run", kernel="pcnl", step=0.16, level=1, pilot_steps=16,
                  out=out, device="cpu")
    values.update(kwargs)
    return Namespace(**values)


def _write_short_run(path, steps, *, acceptance=0.4, seconds=8.0, failures=None, mean=1.0):
    path.mkdir(parents=True)
    (path / "production_clock.json").write_text(json.dumps({"pilot": seconds}))
    (path / f"run_{steps}.json").write_text(json.dumps({
        "counters": {"numerical_failures": failures if failures is not None else [0, 0, 0, 0]},
        "peak_cuda_reserved_bytes": None,
    }))
    frame = pd.DataFrame([
        dict(probe=probe, steps=steps, chains=4, acceptance=acceptance, mean=mean)
        for probe in PROBES
    ])
    frame.to_csv(path / f"metrics_{steps}.csv", index=False)


def test_colab_can_name_its_gpu_and_the_cluster_path_stays_on_a10():
    assert require_comparative_device("run", "cuda", "NVIDIA A10") == "NVIDIA A10"
    with pytest.raises(ValueError, match="A10"):
        require_comparative_device("run", "cuda", "Tesla T4")
    assert require_comparative_device("run", "cuda", "Tesla T4", allow_any_device=True) == "Tesla T4"
    with pytest.raises(ValueError, match="CUDA"):
        require_comparative_device("run", "cpu", "cpu", allow_any_device=True)


def test_only_width_4096_langevin_production_is_gated():
    assert needs_inline_check("run", "pcnl", 4096)
    assert not needs_inline_check("run", "pcnl", 256)
    assert not needs_inline_check("run", "slice", 4096)
    assert not needs_inline_check("pilot", "pcnl", 4096)


def test_low_acceptance_continues_and_a_broken_clock_stops():
    healthy = gate_decision(seconds_per_step=0.2, acceptance=0.01, failures=[0, 0, 0, 0],
                            remaining_steps=100, remaining_seconds=10_000,
                            peak_reserved_bytes=None, total_gpu_bytes=None)
    assert healthy["ok"]
    broken = gate_decision(seconds_per_step=None, acceptance=0.5, failures=[],
                           remaining_steps=100, remaining_seconds=10_000,
                           peak_reserved_bytes=None, total_gpu_bytes=None)
    assert not broken["ok"]


def test_reader_accepts_a_finished_short_run_and_rejects_a_nan(tmp_path):
    path = tmp_path / "pilot"
    _write_short_run(path, 16)
    info = read_short_run(path, 16)
    assert info["acceptance"] == 0.4
    assert info["seconds_per_step"] == 0.5
    bad = tmp_path / "bad"
    _write_short_run(bad, 16, mean=np.nan)
    with pytest.raises(SystemExit, match="GATE_STOP"):
        read_short_run(bad, 16)


def test_time_stop_does_not_start_production(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_sample(args, cfg, task):
        calls.append(args.action)
        if args.action == "pilot":
            _write_short_run(args.out / "pilot" / "pcnl_h_0.16" / task["id"], args.pilot_steps,
                             seconds=160.0)
        else:
            raise AssertionError("production started after a failed check")

    monkeypatch.setattr("deep_width.run.sample", fake_sample)
    monkeypatch.setattr("deep_width.run.slurm_remaining_seconds", lambda: 30)
    monkeypatch.setattr("deep_width.run.gpu_capacity_bytes", lambda device: None)
    task = {"id": "deep_m4096_r0", "m": 4096}
    with pytest.raises(SystemExit, match="Production was not started"):
        short_check_then_run(_args(tmp_path), None, task)
    assert calls == ["pilot"]
    assert not (tmp_path / "run").exists()
    assert "GATE_PASS" not in capsys.readouterr().out


def test_passing_check_continues_in_the_same_call(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_sample(args, cfg, task):
        calls.append(args.action)
        if args.action == "pilot":
            _write_short_run(args.out / "pilot" / "pcnl_h_0.16" / task["id"], args.pilot_steps,
                             seconds=1.6, acceptance=0.01)

    monkeypatch.setattr("deep_width.run.sample", fake_sample)
    monkeypatch.setattr("deep_width.run.slurm_remaining_seconds", lambda: 8 * 3600)
    monkeypatch.setattr("deep_width.run.gpu_capacity_bytes", lambda device: None)
    task = {"id": "deep_m4096_r0", "m": 4096}
    short_check_then_run(_args(tmp_path), None, task)
    assert calls == ["pilot", "run"]
    output = capsys.readouterr().out
    assert "GATE_PASS" in output
    assert "acceptance_low=true" in output


def test_finished_segment_is_not_projected_again():
    warmup, production = segments_for("pcnl", 2, 0.16)
    assert remaining_joint_steps(warmup, production, {"production_1": 12.0}) == 8192
    info = dict(seconds_per_step=100.0, acceptance=0.2, failures=[0, 0, 0, 0],
                peak_reserved_bytes=None)
    # Nothing left at level 1, so a slow rate cannot veto a completed cell.
    continue_after_short_check(
        _args(None, level=1), {"id": "deep_m4096_r0"}, info,
        remaining_seconds=1, total_gpu_bytes=None, ledger={"production_1": 12.0})
