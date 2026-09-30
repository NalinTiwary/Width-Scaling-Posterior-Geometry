"""ArviZ diagnostic wrapper compatibility."""

from __future__ import annotations

import numpy as np

from cylinder.diagnostics import arviz_diagnostics, coverage_summary


def test_arviz_diagnostics_runs():
    rng = np.random.default_rng(0)
    arrays = {
        "H": rng.normal(size=(2, 60)),
        "V": rng.normal(size=(2, 60)),
        "p_probe": rng.uniform(size=(2, 60, 3)),
    }
    diag = arviz_diagnostics(arrays, head_quantiles=[0.95])
    assert np.isfinite(diag["rhat_max"])
    assert np.isfinite(diag["ess_bulk_min"])
    assert "0.95" in diag["quantile_ess_H"]


def test_arviz_diagnostics_unrounded():
    import arviz as az

    from cylinder.diagnostics import posterior_idata

    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 500)) + np.array([0.0, 0.02, -0.03, 0.05])[:, None]
    diag = arviz_diagnostics({"V": x}, head_quantiles=[])
    r = az.rhat(posterior_idata({"V": x}))
    r = r.posterior if hasattr(r, "posterior") else r
    e = az.ess(posterior_idata({"V": x}), method="bulk")
    e = e.posterior if hasattr(e, "posterior") else e
    assert diag["rhat_max"] == float(np.asarray(r["V"]))
    assert diag["ess_bulk_min"] == float(np.asarray(e["V"]))


def test_coverage_all_inside():
    inside = np.ones((4, 50), dtype=np.int8)
    cov = coverage_summary(inside)
    assert cov["all_inside"]
    assert cov["p_hat"] == 1.0
    assert not cov["mcse_estimable"]
