"""IO helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cylinder.io import atomic_save_json, atomic_save_npz, validate_resume


def test_atomic_savez(tmp_path: Path):
    p = tmp_path / "obs.npz"
    atomic_save_npz(p, H=np.arange(6.0).reshape(2, 3), flag=np.array(1))
    z = np.load(p)
    assert z.files == ["H", "flag"] or set(z.files) == {"H", "flag"}
    assert z["H"].shape == (2, 3)


def test_validate_resume_mismatch():
    try:
        validate_resume({"m": 256, "sigma": 0.5}, {"m": 1024, "sigma": 0.5}, keys=["m", "sigma"])
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "m" in str(e)
