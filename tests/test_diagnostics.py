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


def test_coverage_all_inside():
    inside = np.ones((4, 50), dtype=np.int8)
    cov = coverage_summary(inside)
    assert cov["all_inside"]
    assert cov["p_hat"] == 1.0
    assert not cov["mcse_estimable"]
