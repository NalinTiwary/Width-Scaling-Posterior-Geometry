"""Runbook §6.2 known-distribution integration tests (CPU, float64).

Set BNN_CALIBRATION_OUT=<dir> to also write the rows to <dir>/calibration_results.csv (used by Appendix S1).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
torch.set_default_dtype(torch.float64)

from bnn_geometry import calibration as cal  # noqa: E402

ROWS: list[dict] = []


def _record(rows):
    ROWS.extend(rows)
    out = os.environ.get("BNN_CALIBRATION_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ROWS).to_csv(Path(out) / "calibration_results.csv", index=False)
    return rows


@pytest.mark.parametrize("sigma", [0.7, 1.0])
def test_ou_calibration(sigma):
    rows = _record(cal.ou_calibration(sigma))
    for r in rows:
        assert r["pass_acceptance"], r
        assert r["pass_mean"] and r["pass_var"], r
        # Runbook §6.2: a statistical failure is accepted only with an independent error audit, and the
        # coordinate mean must itself be unbiased.
        assert r["pass_tau"] or r["audit"].startswith("ordinary MC fluctuation"), r
        assert abs(r["tau_coordinate_mean"] - r["tau_exact_discrete"]) <= max(
            0.05 * r["tau_exact_discrete"], 3 * r["tau_independent_se"] / 4.0), r


@pytest.mark.parametrize("sigma", [0.7, 1.0])
def test_gaussian_entropy(sigma):
    for r in _record(cal.gaussian_entropy(sigma)):
        assert r["pass"], r


def test_quadratic_fixture():
    rows = _record(cal.quadratic_fixture())
    bad = [r for r in rows if not (r["pass_mean"] and r["pass_var"])]
    assert not bad, bad


def test_conditional_entropy():
    for r in _record(cal.conditional_entropy()):
        assert r["pass"], r
