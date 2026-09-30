"""Runbook §6.1 F: bitwise resume, incompatible checkpoints and corrupted traces."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from conftest import CPU, tiny_model

from bnn_geometry.chains import Interrupted, Trajectory, prior_draws
from bnn_geometry.ellipse import EllipticalSlice
from bnn_geometry.pcnl import PCNL
from bnn_geometry.probes import Observables, ProbeSpec
from bnn_geometry.randomness import ChainRNG
from bnn_geometry.storage import StorageError, read_trace_file, segment_path, write_trace_file

SPEC = ProbeSpec(train_idx=(0, 1, 2, 3, 4), test_idx=(0, 1), entropy_train_idx=(0, 2, 3, 4),
                 neurons=(0, 1, 2, 3), shallow_inputs=(0, 1, 2, 3))


def _traj(path, kind="ess", hashes=None, h=0.05, arch="deep"):
    model, _ = tiny_model(arch)
    obs = Observables(model, SPEC)
    rngs = [ChainRNG(100 + c, CPU) for c in range(3)]
    if kind == "ess":
        sampler = EllipticalSlice(model.V, model.theta0, model.sigma)
        rec = lambda st: obs.record(st.theta, st.V, st.cache)  # noqa: E731
    else:
        sampler = PCNL(model.V_and_grad, model.theta0, model.sigma, h)
        rec = lambda st: obs.record(st.theta, st.V, st.cache)  # noqa: E731
    tr = Trajectory(path, kind=kind, sampler=sampler, rngs=rngs, chains=[0, 1, 2],
                    hashes=hashes or {"target_hash": "t", "execution_hash": "e"}, record=rec,
                    names=obs.names, checkpoint_stride=7,
                    archive_fn=lambda th: {"S": th[:, 0] * 0 + 1.0})
    init_rngs = [ChainRNG(900 + c, CPU) for c in range(3)]
    return tr.open(lambda: prior_draws(model.theta0, model.sigma, init_rngs))


def _run_all(tr, stops=()):
    plan = [("burnin", 20, 0), ("calibration", 24, 8), ("production_s1", 33, 16)]
    stops = list(stops)
    for seg, n, a in plan:
        while True:
            try:
                tr.run_segment(seg, n, archive_stride=a, stop_after=stops.pop(0) if stops else None)
                break
            except Interrupted:
                tr = _reopen(tr)
    return tr


def _reopen(tr):
    return _traj(tr.path, tr.kind, tr.hashes)


@pytest.mark.parametrize("kind", ["ess", "pcnl"])
def test_resume_is_bitwise_identical(tmp_path, kind):
    a = _run_all(_traj(tmp_path / "a", kind))
    b = _run_all(_traj(tmp_path / "b", kind), stops=[5, 9, 30, 3])
    ra = a.read(["calibration", "production_s1"])
    rb = b.read(["calibration", "production_s1"])
    for ca, cb in zip(ra, rb):
        assert set(ca) == set(cb)
        for k in ca:
            assert np.array_equal(ca[k], cb[k]), k
    assert torch.equal(a.state.theta, b.state.theta)
    assert a.ck["counters"] == b.ck["counters"]
    np.testing.assert_array_equal(ra[0]["archive/draw"], [0, 8, 16, 24 + 0, 24 + 16, 24 + 32])


def test_incompatible_checkpoint_rejected(tmp_path):
    tr = _traj(tmp_path / "x")
    tr.run_segment("burnin", 10)
    with pytest.raises(StorageError):
        _traj(tmp_path / "x", hashes={"target_hash": "t", "execution_hash": "changed-step"})
    with pytest.raises(StorageError):
        _traj(tmp_path / "x", hashes={"target_hash": "other-data", "execution_hash": "e"})


def test_segment_order_and_length_enforced(tmp_path):
    tr = _traj(tmp_path / "x")
    with pytest.raises(Interrupted):
        tr.run_segment("burnin", 20, stop_after=3)
    tr = _reopen(tr)
    with pytest.raises(StorageError):
        tr.run_segment("calibration", 5)
    with pytest.raises(StorageError):
        tr.run_segment("burnin", 21)


def _corrupt(path, fn):
    arr, attrs = read_trace_file(path)
    arr, attrs = fn(dict(arr), dict(attrs))
    write_trace_file(path, arrays=arr, attrs=attrs)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "shuffled", "wrong_chain", "archive"])
def test_corrupted_segments_detected(tmp_path, damage):
    tr = _run_all(_traj(tmp_path / "x"))
    p = segment_path(tr.chain_dir(1), "calibration")

    def f(arr, attrs):
        d = arr["draw"].copy()
        if damage == "duplicate":
            d[5] = d[4]
        elif damage == "missing":
            arr = {k: (v if k.startswith("archive/") else np.delete(v, 3, axis=0)) for k, v in arr.items()}
            return arr, attrs
        elif damage == "shuffled":
            d[[2, 3]] = d[[3, 2]]
        elif damage == "wrong_chain":
            attrs["chain"] = 2
        elif damage == "archive":
            arr["archive/draw"] = arr["archive/draw"] + 1
        arr["draw"] = d
        return arr, attrs

    _corrupt(p, f)
    with pytest.raises(StorageError):
        tr.read(["calibration"])


def test_truncated_file_detected(tmp_path):
    tr = _run_all(_traj(tmp_path / "x"))
    p = segment_path(tr.chain_dir(0), "burnin")
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(StorageError):
        tr.read(["burnin"])


def test_stale_chunks_after_checkpoint_are_discarded(tmp_path):
    """A chunk written after the last restart record (job killed between the two writes) is redone."""
    ref = _run_all(_traj(tmp_path / "ref"))
    tr = _traj(tmp_path / "x")
    with pytest.raises(Interrupted):
        tr.run_segment("burnin", 20, stop_after=7)
    ck_before = (tmp_path / "x" / "checkpoint.pt").read_bytes()
    tr = _reopen(tr)
    with pytest.raises(Interrupted):
        tr.run_segment("burnin", 20, stop_after=7)
    (tmp_path / "x" / "checkpoint.pt").write_bytes(ck_before)   # restart record lost, chunk kept
    tr = _run_all(_reopen(tr))
    for ca, cb in zip(ref.read(["burnin", "production_s1"]), tr.read(["burnin", "production_s1"])):
        for k in ca:
            assert np.array_equal(ca[k], cb[k])
