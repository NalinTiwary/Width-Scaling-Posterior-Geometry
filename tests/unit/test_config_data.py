"""Config schema, seed scheme, frozen data and centers (runbook §2.2, §2.4, §4, §6.1 B)."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from bnn_geometry import config as C
from bnn_geometry.data import data_summary, make_center_bank, make_replicate_data
from bnn_geometry.randomness import stream_seed


@pytest.fixture(scope="module")
def cfg():
    return C.load("configs/campaign.yaml")


def test_campaign_validates_and_has_24_targets(cfg):
    ts = C.targets(cfg)
    assert len(ts) == 24 == len({t.target_id for t in ts})
    assert sum(t.arch == "deep" for t in ts) == 12
    assert {t.p(32) for t in ts if t.arch == "shallow"} == {33 * m for m in (64, 256, 1024, 4096)}
    assert {t.p(32) for t in ts if t.arch == "deep"} == {m * m + 33 * m for m in (32, 64, 128, 256)}


@pytest.mark.parametrize("path,value", [
    (("model", "sigma"), 0.5),                  # a = 2.5 no longer > 2 sigma? (still valid) -> use variance check
    (("model", "biases"), True),
    (("domain", "spectral_cutoff_a"), 2.0),
    (("model", "shallow", "widths"), [256, 64]),
])
def test_invalid_configs_fail(cfg, path, value):
    bad = copy.deepcopy(cfg)
    node = bad
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(C.ConfigError):
        C.validate(bad)


def test_unknown_or_missing_key_fails(cfg):
    bad = copy.deepcopy(cfg)
    bad["model"]["temperature"] = 1.0
    with pytest.raises(C.ConfigError):
        C.validate(bad)
    bad = copy.deepcopy(cfg)
    del bad["probes"]["tilts"]
    with pytest.raises(C.ConfigError):
        C.validate(bad)


def test_seed_scheme_matches_spec_formula():
    s = "20260930|schema=1|rep=0|arch=shared|width=shared|chain=shared|stage=data|stream=teacher"
    expect = int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") % (2**63 - 1)
    assert stream_seed(20260930, rep=0, arch="shared", width="shared", chain="shared", stage="data",
                       stream="teacher") == expect
    assert stream_seed(20260930, rep=0, arch="deep", width=32, chain=0, stage="x", stream="a") != \
        stream_seed(20260930, rep=0, arch="deep", width=32, chain=1, stage="x", stream="a")


def test_data_and_centers(cfg):
    D = make_replicate_data(cfg, 0)
    assert D["X"].shape == (128, 32) and D["X_test"].shape == (1024, 32)
    assert np.abs(np.linalg.norm(D["X"], axis=1) - 1).max() < 1e-12
    assert np.abs(np.linalg.norm(D["X_test"], axis=1) - 1).max() < 1e-12
    assert set(np.unique(D["y"])) <= {0.0, 1.0}
    assert np.linalg.norm(D["teacher"]) > 1.0          # unnormalized N(0, I_32) teacher
    s = data_summary(D)
    assert s["rank"] <= 32 and len(s["XXT_nonzero_eigenvalues"]) == s["rank"]
    D2 = make_replicate_data(cfg, 0)
    assert all(np.array_equal(D[k], D2[k]) for k in D)
    D1 = make_replicate_data(cfg, 1)
    assert not np.array_equal(D["X"], D1["X"])
    bank = make_center_bank(cfg, 0)
    assert bank.shape == (4096, 32)
    assert not np.array_equal(bank, make_center_bank(cfg, 1))
    assert abs(bank.std() - 1.0) < 0.02
