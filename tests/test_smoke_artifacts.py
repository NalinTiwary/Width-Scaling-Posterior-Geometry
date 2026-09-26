"""End-to-end smoke checks for figure sidecars and theorem-aligned outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cylinder.theorem import PREFLIGHT, prior_cdf_H, theorem_bundle


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "artifacts_smoke"


@pytest.mark.skipif(not (SMOKE / "figure1_coverage.csv").exists(), reason="smoke artifacts not built")
def test_figure1_coverage_schema_and_floor():
    df = pd.read_csv(SMOKE / "figure1_coverage.csv")
    assert set(df["m"]) >= {256, 1024, 4096}
    assert df["coverage"].between(0.0, 1.0 + 1e-9).all()
    for m, g in df.groupby("m"):
        floor = PREFLIGHT[int(m)]["coverage_floor"]
        assert np.allclose(g["theorem_floor"], floor, rtol=0, atol=1e-6)
        # Short chains may exit; still expect high occupancy for this prior scale
        assert g["coverage"].min() >= 0.5


@pytest.mark.skipif(not (SMOKE / "figure2_curvature.csv").exists(), reason="smoke artifacts not built")
def test_figure2_curvature_brackets():
    df = pd.read_csv(SMOKE / "figure2_curvature.csv")
    assert set(df["m"]) >= {256, 1024, 4096}
    for _, row in df.iterrows():
        if row["n_inside"] == 0:
            continue
        assert row["q95_d_minus"] <= row["q95_d_plus"] + 1e-9
        assert row["q95_d_plus"] <= row["D_th"] + 1e-4
        assert row["max_eigen_residual"] < 1e-8


@pytest.mark.skipif(not (SMOKE / "figureS1_cutoff.csv").exists(), reason="smoke artifacts not built")
def test_figureS1_prior_overlay_matches_analytic():
    df = pd.read_csv(SMOKE / "figureS1_cutoff.csv")
    # Spot-check prior CDF column against closed form
    sub = df[(df["m"] == 256) & (df["seed"] == 0)].reset_index(drop=True)
    row = sub.iloc[len(sub) // 4]
    B = theorem_bundle(m=256)["B_m"]
    z = float(row["z"])
    expected = float(prior_cdf_H(z * B, m=256, b0=1.0, sigma=0.5))
    assert abs(float(row["cdf_prior"]) - expected) < 1e-8
    assert df["ecdf_posterior"].between(0.0, 1.0 + 1e-9).all()


@pytest.mark.skipif(not (SMOKE / "target_summary.csv").exists(), reason="smoke artifacts not built")
def test_target_summary_required_columns():
    df = pd.read_csv(SMOKE / "target_summary.csv")
    required = {
        "width", "p", "B_m", "D_th", "center_seed", "coverage",
        "exit_count", "H_over_B_q95", "mean_posterior_V", "mean_prior_V",
    }
    assert required.issubset(df.columns)
    assert len(df) >= 6  # 3 widths × 2 seeds


@pytest.mark.skipif(not (SMOKE / "figures" / "figure1_coverage.png").exists(), reason="figures not built")
def test_figure_pngs_nonempty():
    for name in (
        "figure1_coverage.png",
        "figure2_curvature.png",
        "figureS1_cutoff.png",
        "figureS2_efficiency.png",
    ):
        p = SMOKE / "figures" / name
        assert p.exists(), name
        assert p.stat().st_size > 5_000, name
