"""Figure guide §9.4 calculation tests (vendored selfcheck + extras) and adapter/rendering checks."""

import json
import sys
import unittest

import numpy as np
import pandas as pd
import pytest

from bnn_geometry.paper_export import selection_rule
from bnn_geometry.paper_figures import analysis, render
from bnn_geometry.paper_figures.common import HERE, Layout, metrics
from bnn_geometry.paper_figures.prepare import InputError, load_loss_npz
from bnn_geometry import config as C

fm = metrics()


def test_vendored_selfcheck_suite():
    sys.path.insert(0, str(HERE))
    import selfcheck
    res = unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromModule(selfcheck))
    assert res.wasSuccessful() and res.testsRun == 7


def test_lag_zero_and_moderate_affine_invariance():
    x = np.cumsum(np.random.default_rng(3).normal(size=4096)) * 0.01 + np.random.default_rng(4).normal(size=4096)
    r = fm.acf_1d(x, 100)
    assert abs(r[0] - 1.0) <= 1e-12
    np.testing.assert_allclose(fm.acf_1d(250.0 * x + 3.0e4, 100), r, atol=1e-10, rtol=0)
    z = x - x.mean()
    direct = np.array([np.dot(z[:z.size - k], z[k:]) / np.dot(z, z) for k in range(101)])
    np.testing.assert_allclose(r, direct, atol=1e-10, rtol=0)


def test_replicate_acf_is_not_concatenation():
    x = np.random.default_rng(5).normal(size=(4, 3000)) + np.array([0.0, 5.0, -5.0, 10.0])[:, None]
    rep = fm.acf_replicate(x, 50, 2000)["replicate_acf"]
    concat = fm.acf_1d(x[:, -2000:].ravel(), 50)
    assert np.max(np.abs(rep - concat)) > 0.1
    np.testing.assert_allclose(rep, np.mean([fm.acf_1d(r[-2000:], 50) for r in x], axis=0), atol=1e-12)


def test_spectral_guard_and_quantiles():
    s = np.array([0.5, 0.9, 1.0 + 5e-11, 0.7, 1.2])
    out = fm.spectral_target(s)
    assert (out["n_inside"], out["n_outside"], out["n_ambiguous"]) == (3, 1, 1)
    np.testing.assert_allclose([out["q50"], out["q95"], out["q99"]],
                               np.quantile(s, [0.5, 0.95, 0.99], method="linear"))
    assert out["n_inside"] + out["n_outside"] + out["n_ambiguous"] == out["n_inspected"]


# ---- adapter ------------------------------------------------------------------------------------------
def _npz(path, t, V, *, h=0.01, window=None, n=128, stride=1, observable="summed_training_cross_entropy"):
    window = window or V.shape[1]
    start = np.full(4, 2001, dtype=np.int64)
    np.savez_compressed(
        path, V=V, accepted=np.ones_like(V, dtype=np.int8), chain_ids=np.arange(4), iteration_start=start,
        iteration_end=start + window - 1, save_stride=np.int64(stride), h=np.float64(h), architecture=np.str_(t.arch),
        L=np.int64(t.L), m=np.int64(t.m), n=np.int64(n), d=np.int64(32), sigma=np.float64(1.0),
        replicate_id=np.int64(t.rep), observable=np.str_(observable), contains_rejected_transitions=np.bool_(True),
        target_hash=np.str_("x" * 64), sampler_code_hash=np.str_("s" * 64), execution_hash=np.str_("e" * 64),
        source_run_ids=np.asarray([f"e/chain_{c}" for c in range(4)]), source_filenames=np.asarray(["f"]),
        selection_rule=np.str_(selection_rule(window)), retained_transitions_per_chain=np.int64(window))


def test_npz_metadata_checks(tmp_path):
    t = C.Target("deep", 32, 1)
    V = np.random.default_rng(0).normal(size=(4, 500))
    p = tmp_path / f"{t.target_id}.npz"
    _npz(p, t, V)
    assert load_loss_npz(p, t=t, h=0.01, window=500, n=128, spec=None)["V"].shape == (4, 500)
    for kw, msg in [({"h": 0.005}, "h="), ({"stride": 16}, "save_stride"), ({"n": 64}, "n="),
                    ({"observable": "U"}, "observable")]:
        _npz(p, t, V, **kw)
        with pytest.raises(InputError, match=msg):
            load_loss_npz(p, t=t, h=0.01, window=500, n=128, spec=None)
    _npz(p, t, V)
    with pytest.raises(InputError, match="shape"):
        load_loss_npz(p, t=t, h=0.01, window=400, n=128, spec=None)
    with pytest.raises(InputError, match="file name"):
        load_loss_npz(p, t=C.Target("deep", 64, 1), h=0.01, window=500, n=128, spec=None)


def _ar1(rng, phi, n):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi * phi) * e[i]
    return x


def test_K_extension_and_tables_on_slow_traces(tmp_path):
    cfg = {"campaign_id": "unit_test", "replicates": [0, 1, 2], "data": {"train_size": 128, "input_dimension": 32},
           "model": {"shallow": {"widths": [4, 8]}, "deep": {"widths": [4, 8]}, "sigma": 1.0}}
    L = Layout(tmp_path / "pf")
    L.mkdirs()
    L.traces.mkdir(parents=True)
    rng = np.random.default_rng(11)
    window = 8000
    for t in C.targets(cfg):
        phi = 0.9993 if t.m == 4 else 0.99
        V = np.stack([_ar1(rng, phi, window) for _ in range(4)]) + 50.0
        _npz(L.traces / f"{t.target_id}.npz", t, V, window=window)
    st = analysis.acf_tables(cfg, tmp_path, L, window=window)
    assert st["K"] > 400 and [h["K"] for h in st["K_history"]][0] == 400
    plot = pd.read_csv(L.tables / "acf_plot.csv")
    assert plot.lag.max() == st["K"]
    chain = pd.read_csv(L.tables / "acf_chain.csv.gz")
    assert len(chain) == 12 * 4 * (st["K"] + 1)
    assert np.allclose(chain[chain.lag == 0].rho, 1.0)
    con = pd.read_csv(L.tables / "acf_width_contrasts.csv")
    assert (con.delta < 0).all()  # faster-mixing widest width in every replicate at every reported lag
    diag = pd.read_csv(L.tables / "loss_window_diagnostics.csv")
    assert (diag.check_status == "fail").all()  # no campaign reference status in this synthetic root
    st["fixture"] = True
    (L.tables / "acf_settings.json").write_text(json.dumps(st))
    out = render.figure2(L.tables, L.figures)
    assert all(p.exists() and p.stat().st_size > 0 for p in out)


def test_figure1_renders_from_tables_only(tmp_path):
    rows = []
    for m in (32, 64, 128, 256):
        for q, base in (("0.50", 0.77), ("0.95", 0.81), ("0.99", 0.83)):
            rows.append({"m": m, "quantile": q, "posterior_median": base, "replicate_min": base - 0.002,
                         "replicate_max": base + 0.002, "prior": base})
    pd.DataFrame(rows).to_csv(tmp_path / "spectral_plot.csv", index=False)
    (tmp_path / "spectral_counts.json").write_text(json.dumps(
        {"n_inspected": 196608, "n_outside": 0, "n_ambiguous": 0, "fixture": True}))
    out = render.figure1(tmp_path, tmp_path / "fig")
    assert all(p.exists() for p in out)


def test_fixture_cannot_write_into_paper_dir():
    from bnn_geometry.paper_figures.build import _guard
    with pytest.raises(SystemExit):
        _guard({"campaign_id": "fixture3_x"}, C.ROOT / "results" / "final_geometry" / "paper_figures")
    _guard({"campaign_id": "final_geometry_v15_20260930"}, C.ROOT / "results" / "final_geometry" / "paper_figures")
