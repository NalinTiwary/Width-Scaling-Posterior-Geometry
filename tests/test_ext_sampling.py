"""Extension sampling: matched full-vs-reduced ESS (with negative control) and the target runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cylinder.ext import validation as V  # noqa: E402
from cylinder.ext.data import orth_replicate, save_npz  # noqa: E402
from cylinder.ext.problem import data_path  # noqa: E402
from cylinder.ext.run import is_complete, run_dir, run_ext_target  # noqa: E402
from cylinder.ext.targets import Target  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)


def test_matched_sampling_agrees_and_detects_wrong_reduction():
    kw = dict(n_chains=4, burnin=200, T=1000)
    ok = V.check_matched_sampling(**kw)
    assert ok["ok"], ok["max_abs_z"]
    bad = V.check_matched_sampling(perturb=1.25, **kw)
    assert not bad["ok"], bad["max_abs_z"]


@pytest.fixture()
def tiny_cfg(tmp_path):
    cfg = yaml.safe_load(open(ROOT / "config.ext.smoke.yaml"))
    cfg["artifacts_dir"] = str(tmp_path / "art")
    cfg["sampling"].update(n_chains=2, burnin=5, retained_initial=16)
    cfg["curvature"]["states_per_chain"] = 2
    ctl = cfg["controlled"]
    rep = orth_replicate(0, d=int(ctl["d"]), input_seed_base=int(ctl["input_seed_base"]),
                         label_seed_base=int(ctl["label_seed_base"]),
                         center_seed_base=int(ctl["center_seed_base"]), center_bank_rows=64,
                         teacher_scale=float(ctl["teacher_scale"]))
    art = Path(cfg["artifacts_dir"])
    save_npz(data_path(art, "orth", 0), rep)
    return cfg, art


def test_runner_writes_outputs_is_deterministic_and_resumes(tiny_cfg):
    cfg, art = tiny_cfg
    t = Target("orth", 64, 64, 0)
    meta = run_ext_target(cfg=cfg, target=t, artifacts=art, device_request="cpu")
    rd = run_dir(art, t)
    for fn in ("metadata.json", "observables.npz", "curvature.npz"):
        assert (rd / fn).exists()
    assert is_complete(art, t, 16) and not is_complete(art, t, 32)
    obs = dict(np.load(rd / "observables.npz"))
    cz = dict(np.load(rd / "curvature.npz"))
    assert obs["H"].shape == (2, 16) and np.isfinite(obs["H"]).all()
    assert np.allclose(obs["H_over_B"], obs["H"] / meta["theory"]["B"])
    assert cz["d_plus"].shape == (2, 2)
    assert (cz["d_minus"] <= cz["d_plus"] + 1e-12).all() and (cz["d_minus"] >= 0).all()
    assert float(cz["max_eigen_residual"].max()) < 1e-8

    skipped = run_ext_target(cfg=cfg, target=t, artifacts=art, device_request="cpu", skip_existing=True)
    assert "_buf" not in skipped

    run_ext_target(cfg=cfg, target=t, artifacts=art, device_request="cpu", overwrite=True)
    again = dict(np.load(rd / "observables.npz"))
    for k in ("H", "V", "f_sub", "projections"):
        np.testing.assert_array_equal(obs[k], again[k])

    run_ext_target(cfg=cfg, target=t, artifacts=art, device_request="cpu", n_retained=32, overwrite=True)
    longer = dict(np.load(rd / "observables.npz"))
    assert json.loads((rd / "metadata.json").read_text())["n_retained"] == 32
    np.testing.assert_array_equal(longer["H"][:, :16], obs["H"])
