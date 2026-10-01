"""Figure-cleanup and S1/S2 supplement package (docs/paper_cleanup)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bnn_geometry.paper_cleanup.common import HISTORICAL_A, Layout, run_selfcheck
from bnn_geometry.paper_cleanup.main_figures import threshold_tables
from bnn_geometry.paper_cleanup.supplements import render_s1, render_s2, s1_analysis, s2_analysis
from bnn_geometry.paper_export import h_dir_name, sha256_file

ROOT = Path(__file__).resolve().parents[2] / "results" / "final_geometry"
WIDTHS = {"shallow": [64, 256, 1024, 4096], "deep": [32, 64, 128, 256]}
CFG = {"replicates": [0, 1, 2], "model": {a: {"widths": w} for a, w in WIDTHS.items()}}


def test_bundled_selfcheck_passes(tmp_path):
    assert run_selfcheck(tmp_path / "selfcheck.txt").splitlines()[-1].startswith("OK")


@pytest.mark.skipif(not (ROOT / "paper_cleanup" / "exports" / "spectral_states.csv.gz").exists(),
                    reason="committed cleanup export not present")
def test_committed_T_is_stored_S_times_a_exactly_once():
    T = pd.read_csv(ROOT / "paper_cleanup" / "exports" / "spectral_states.csv.gz")
    S = pd.read_csv(ROOT / "paper_figures" / "tables" / "spectral_states.csv.gz",
                    usecols=["m", "replicate_id", "chain_id", "original_draw_index", "S"])
    S = S.rename(columns={"m": "width", "replicate_id": "replicate", "chain_id": "chain", "original_draw_index": "draw"})
    d = T.merge(S, on=["width", "replicate", "chain", "draw"], validate="one_to_one")
    assert len(d) == len(T) == len(S)
    np.testing.assert_allclose(d["T"], HISTORICAL_A * d["S"], rtol=1e-15, atol=0)
    assert 1.8 < d["T"].median() < 2.0


def _counts(tmp_path, **override):
    L = Layout(tmp_path)
    L.mkdirs()
    rows = [{"width": m, "replicate": r, "threshold": 2.0, "n": 100, "n_above": 10, "n_at_or_below": 90,
             "fraction_above": 0.1, "n_near_threshold": 0, "maximum": 2.1} for m in (32, 256) for r in (0, 1, 2)]
    rows[0].update(override)
    pd.DataFrame(rows).to_csv(L.figures / "figure_1_spectral_domain_threshold_counts.csv", index=False)
    return L


def test_threshold_bookkeeping(tmp_path):
    t = threshold_tables(_counts(tmp_path / "ok"), 2.0)
    assert list(t.width) == [32, 256] and (t.n_above_r0 == 10).all() and (t.pct_above_median == 10).all()
    with pytest.raises(ValueError, match="bookkeeping"):
        threshold_tables(_counts(tmp_path / "bad", n_above=11), 2.0)
    with pytest.raises(ValueError, match="guard"):
        threshold_tables(_counts(tmp_path / "near", n_near_threshold=1), 2.0)


def _ar1(rng, phi, shape):
    x = np.empty(shape)
    x[..., 0] = rng.standard_normal(shape[:-1])
    e = rng.standard_normal(shape) * np.sqrt(1 - phi ** 2)
    for k in range(1, shape[-1]):
        x[..., k] = phi * x[..., k - 1] + e[..., k]
    return x


def _write(L, name, side, **arrays):
    L.traces.mkdir(parents=True, exist_ok=True)
    p = L.traces / name
    np.savez_compressed(p, **arrays)
    side["npz_sha256"] = sha256_file(p)
    p.with_suffix(".json").write_text(json.dumps(side))


def test_s1_s2_synthetic_width_ordering(tmp_path):
    """Wider = faster-decaying AR(1); the paired narrow-minus-wide differences must all be positive."""
    rng = np.random.default_rng(0)
    L = Layout(tmp_path / "pc")
    L.mkdirs()
    N = 4000
    for arch, ws in WIDTHS.items():
        for i, m in enumerate(ws):
            for r in CFG["replicates"]:
                z = _ar1(rng, 0.99 - 0.02 * i, (4, 8, N)).transpose(0, 2, 1)
                tid = f"{arch}_m{m:04d}_r{r}"
                _write(L, f"s1_{tid}.npz", {"iteration_start": 1, "iteration_end": N},
                       probabilities=1 / (1 + np.exp(-0.5 * z)), V=_ar1(rng, 0.9, (4, N)),
                       iteration=np.tile(np.arange(1, N + 1), (4, 1)), test_indices=np.arange(0, 1024, 128),
                       accepted=np.ones((4, N), bool))
    s1 = s1_analysis(CFG, L, window=N)
    assert s1["settings"]["K"] == 400
    rep = pd.read_csv(L.tables / "prediction_acf_replicates.csv")
    assert np.allclose(rep[rep.lag == 0].acf, 1.0) and len(rep) == 24 * 401
    d = pd.read_csv(L.tables / "prediction_width_differences.csv")
    m8 = d[d.test_index == "mean8"]
    assert len(m8) == 2 * 3 * 4 and (m8.difference > 0).all() and (m8[m8.lag <= 100].n_points_positive == 8).all()

    (tmp_path / "endpoint_decision_round2.json").write_text(json.dumps({"deep": {"final_h": 0.01}}))
    dur = 16.0
    for m, tau in ((32, 0.5), (256, 0.2)):
        for r in CFG["replicates"]:
            for h in (0.01, 0.005):
                n = int(round(dur / h))
                tid = f"deep_m{m:04d}_r{r}"
                _write(L, f"s2_{tid}_{h_dir_name(h)}.npz", {"iteration_start": 1, "iteration_end": n},
                       V=_ar1(rng, np.exp(-h / tau), (4, n)), accepted=np.ones((4, n), bool), h=np.float64(h),
                       iteration=np.tile(np.arange(1, n + 1), (4, 1)))
    s2 = s2_analysis(CFG, tmp_path, L, duration=dur)
    assert s2["settings"]["t_max"] == 2.0 and s2["settings"]["widths"] == [32, 256]
    sd = pd.read_csv(L.tables / "step_width_differences.csv")
    assert set(sd.lag[np.isclose(sd.h, 0.005)]) == {50, 100, 200, 400}
    assert (sd[sd.algorithmic_lag <= 0.5].difference > 0).all()
    for p in render_s1(CFG, L, 6.75) + render_s2(CFG, L, 6.75):
        assert p.exists() and p.stat().st_size > 0
    meta = json.loads((L.figures / "supplement_s2_step_size.json").read_text())
    assert meta["figure_inches"] == [6.75, 2.65]
