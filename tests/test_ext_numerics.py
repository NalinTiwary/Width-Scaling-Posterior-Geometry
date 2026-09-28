"""Extension numerics: theorem table, frozen data, reduced coordinates, arrowhead solver, brackets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cylinder.ext import validation as V  # noqa: E402
from cylinder.ext.arrowhead import arrowhead_dense, arrowhead_min_eig  # noqa: E402
from cylinder.ext.data import orth_replicate  # noqa: E402
from cylinder.ext.targets import Target, all_targets, parse_name  # noqa: E402
from cylinder.ext.theory import EXT_PREFLIGHT, assert_ext_preflight, ext_bundle  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_theorem_table_matches_addendum():
    rows = assert_ext_preflight()
    assert len(rows) == len(EXT_PREFLIGHT)
    b = ext_bundle(128, 4096)
    assert b["coverage_floor"] == pytest.approx(1 - 1 / 4096)
    assert b["n_over_sqrt_m"] == pytest.approx(2.0)
    assert ext_bundle(128, 16384)["D_th"] < b["D_th"] < ext_bundle(128, 1024)["D_th"]


@pytest.fixture(scope="module")
def rep0():
    return orth_replicate(0, d=256, input_seed_base=1000, label_seed_base=2000, center_seed_base=3000,
                          center_bank_rows=64, teacher_scale=2.0)


def test_orth_bank_orthonormal_nested_and_deterministic(rep0):
    X = rep0["X_bank"]
    assert X.shape == (256, 256)
    assert np.abs(X @ X.T - np.eye(256)).max() < 1e-12
    again = orth_replicate(0, d=256, input_seed_base=1000, label_seed_base=2000, center_seed_base=3000,
                           center_bank_rows=64, teacher_scale=2.0)
    assert rep0["hash_X"] == again["hash_X"] and rep0["hash_y"] == again["hash_y"]
    y = rep0["y_all"]
    assert set(np.unique(y)) <= {0.0, 1.0}
    for n in (64, 128, 256):
        assert 0 < y[:n].sum() < n


def test_replicates_differ(rep0):
    r1 = orth_replicate(1, d=256, input_seed_base=1000, label_seed_base=2000, center_seed_base=3000,
                        center_bank_rows=64, teacher_scale=2.0)
    assert r1["hash_X"] != rep0["hash_X"] and r1["hash_U"] != rep0["hash_U"]


def test_target_order_and_names():
    import yaml

    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.ext.yaml"))
    ts = all_targets(cfg)
    assert len(ts) == 7 * 3 + 2 * 3
    assert ts[0].name == "orth_n128_m4096_r0"
    ns = [t.n for t in ts if t.kind == "orth"]
    assert ns.index(64) > ns.index(128) and ns.index(256) > ns.index(64)
    assert all(t.kind == "fmnist" for t in ts[21:])
    assert parse_name("fmnist_n256_m16384_r2") == Target("fmnist", 256, 16384, 2)
    assert [t.name for t in all_targets(cfg, pattern="orth_n64_")][0] == "orth_n64_m1024_r0"


def test_reduction_exact():
    r = V.check_reduction_exact(n_draws=10)
    assert r["ok"], r


@pytest.mark.parametrize("scale", [1e-12, 1e-3, 1.0, 1e3])
def test_arrowhead_matches_dense(scale):
    g = np.random.default_rng(7)
    n = 9
    c = torch.as_tensor(scale * g.standard_normal((50, n)))
    b = torch.as_tensor(g.standard_normal((50, n)))
    c[3, :4] = 0.0          # partially deflated
    b[4, :] = 0.5            # repeated poles
    lam, vec, resid = arrowhead_min_eig(c, b)
    ref = torch.linalg.eigvalsh(arrowhead_dense(c, b))[:, 0]
    assert torch.allclose(lam, ref, atol=1e-12 * max(1.0, scale), rtol=1e-12)
    assert float(resid.max()) < 1e-10 * max(1.0, scale)
    assert torch.allclose(vec.norm(dim=1), torch.ones(50))


def test_arrowhead_check():
    r = V.check_arrowhead(m=300)
    assert r["ok"], r


def test_hessian_decomposition():
    assert V.check_decomposition()["ok"]


def test_brackets_contain_dense_hessian_curvature():
    r = V.check_brackets(n_states=8)
    assert r["ok"], r


def test_dense_bracket_matches_pilot():
    assert V.check_dense_matches_pilot()["ok"]
