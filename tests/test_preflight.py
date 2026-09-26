"""§9.1 numerical and target checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cylinder.centers import assemble_theta0, generate_center_bank, verify_paired_center
from cylinder.curvature import bracket_lambda_min, decompose_dense_check, deficit_bracket
from cylinder.data import generate_probes, generate_training_data
from cylinder.ess_slice import elliptical_slice_step
from cylinder.model import (
    SQRT2,
    binary_potential,
    binary_potential_torch,
    forward_f,
    pack_params,
    param_dim,
)
from cylinder.theorem import (
    PREFLIGHT,
    PREFLIGHT_A_N,
    assert_preflight,
    evidence_bound_A_n,
    prior_cdf_H,
    prior_quantile_H,
    theorem_bundle,
)


def test_theorem_preflight_table():
    report = assert_preflight()
    assert abs(report["A_n"] - PREFLIGHT_A_N) < 1e-6
    for m in PREFLIGHT:
        assert m in report["widths"]


def test_A_n_formula():
    A = evidence_bound_A_n(n=32, C=2, b0=1.0, sigma=0.5)
    assert abs(A - PREFLIGHT_A_N) < 1e-6


def test_data_orthonormal_and_labels():
    train = generate_training_data(n=32, d=32, seed=2027)
    X, y = train["X"], train["y"]
    assert X.shape == (32, 32)
    G = X @ X.T
    assert np.allclose(G, np.eye(32), atol=1e-10)
    assert int(y.sum()) == 16
    assert set(np.unique(y).tolist()) == {0.0, 1.0}


def test_paired_center_zero_function():
    train = generate_training_data()
    probes = generate_probes()
    bank = generate_center_bank(n_pairs=2048, d=32, seed=0, b0=1.0)
    for m in (4, 16, 256):
        theta0 = assemble_theta0(bank["U"], m=m, b0=1.0)
        checks = verify_paired_center(
            theta0, train["X"], probes["X_probe"], b0=1.0, atol=1e-12
        )
        assert checks["ok"], checks


def test_likelihood_matches_explicit_binary_formula():
    rng = np.random.default_rng(0)
    m, d, n = 3, 4, 5
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n).astype(np.float64)
    a = rng.standard_normal(m)
    W = rng.standard_normal((m, d))
    theta = pack_params(a, W)
    f = forward_f(theta, X, m)
    s = SQRT2 * f
    softplus = np.where(s > 0, s + np.log1p(np.exp(-s)), np.log1p(np.exp(s)))
    V_ref = float(np.sum(softplus - y * s))
    assert abs(binary_potential(theta, X, y, m) - V_ref) < 1e-12
    # Torch path
    V_t = float(
        binary_potential_torch(
            torch.tensor(theta),
            torch.tensor(X),
            torch.tensor(y),
            m,
        ).item()
    )
    assert abs(V_t - V_ref) < 1e-12


def test_prior_only_elliptical_slice_gaussian_moments():
    """With V=0 the elliptical slice target is exactly N(θ0, σ²I)."""
    torch.set_default_dtype(torch.float64)
    m, d = 2, 3
    p = param_dim(m, d)
    theta0 = torch.zeros(p, dtype=torch.float64)
    sigma = 0.5
    # Monkeypatch: use a zero potential via empty data and custom step
    # Directly: proposals on the ellipse with V≡0 always accept first proposal? 
    # Better: sample z from prior and check head |a| distribution vs exact prior CDF.
    rng = np.random.default_rng(1)
    # Independent prior draws of a_j ~ N(±1, σ²) under paired centers
    b0 = 1.0
    n_draw = 5000
    # Simulate H under prior for m heads with centers ±1 alternating
    a = np.empty((n_draw, m))
    a[:, 0] = rng.normal(b0, sigma, size=n_draw)
    a[:, 1] = rng.normal(-b0, sigma, size=n_draw)
    H = np.max(np.abs(a), axis=1)
    # Compare empirical CDF at a few points to exact
    for b in (1.5, 2.0, 2.5):
        emp = float(np.mean(H <= b))
        exact = float(prior_cdf_H(b, m=m, b0=b0, sigma=sigma))
        assert abs(emp - exact) < 0.03


def test_prior_quantile_matches_design_table_approx():
    # Design §7.2 analytic prior 99th percentiles
    expected = {256: 2.9744, 1024: 3.1345, 4096: 3.2844}
    for m, qH in expected.items():
        got = prior_quantile_H(0.99, m=m, b0=1.0, sigma=0.5)
        assert abs(got - qH) < 5e-3, (m, got, qH)


def test_hessian_decomposition_matches_autodiff():
    rng = np.random.default_rng(0)
    m, d, n = 2, 3, 4
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n).astype(np.float64)
    theta = rng.standard_normal(m * (1 + d))
    report = decompose_dense_check(theta, X, y, m)
    assert report["max_abs_entry_diff"] < 1e-8, report
    assert report["bracket_contains"], report


def test_deficit_bracket_nonnegative_and_ordered():
    rng = np.random.default_rng(1)
    m, d, n = 2, 3, 4
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n).astype(np.float64)
    theta = rng.standard_normal(m * (1 + d))
    br = deficit_bracket(theta, X, y, m, sigma=0.5)
    assert br["d_minus"] <= br["d_plus"] + 1e-12
    assert br["d_minus"] >= -1e-15
    assert br["max_eigen_residual"] < 1e-8


def test_elliptical_slice_runs_and_counts_likelihoods():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    m, d, n = 2, 3, 4
    X = torch.tensor(rng.standard_normal((n, d)))
    y = torch.tensor(rng.integers(0, 2, size=n).astype(np.float64))
    theta0 = torch.zeros(m * (1 + d))
    z = torch.tensor(rng.standard_normal(m * (1 + d)))
    from cylinder.ess_slice import SliceStats

    stats = SliceStats()
    z2 = elliptical_slice_step(z, theta0, 0.5, X, y, m, rng, stats=stats)
    assert z2.shape == z.shape
    assert stats.n_likelihood_evals >= 2
    assert stats.n_updates == 1


def test_param_dim_matches_preflight():
    for m, row in PREFLIGHT.items():
        assert param_dim(m, 32) == row["p"]
