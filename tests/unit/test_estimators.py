"""Runbook §6.1 D (spectral membership) and E (estimators), plus diagnostics conventions."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bnn_geometry import diagnostics as dg
from bnn_geometry.entropy import TiltProblem, heldout_selection, q_ratio, standardization, tilt_terms
from bnn_geometry.relaxation import family_heldout, inefficiency, tau_hat, tau_ou
from bnn_geometry.spectral import INSIDE, OUTSIDE, UNRESOLVED, classify, normalized_norm

A = 2.5


# ---- D. spectral membership --------------------------------------------------------------------------
def _S(M):
    return float(normalized_norm(torch.as_tensor(M)[None], A)[0])


@pytest.mark.parametrize("top,expect", [(4.9, INSIDE), (5.0, UNRESOLVED), (5.1, OUTSIDE)])
def test_diagonal_membership(top, expect):
    M = np.diag([top, 1.0, 0.5, 0.1])            # m=4: boundary at a*sqrt(m)=5
    S = _S(M)
    assert abs(S - top / 5.0) < 1e-15
    mem, notes = classify(np.array([S]), M[None], A, 1e-10)
    assert mem[0] == expect
    if expect == UNRESOLVED:
        assert notes and notes[0]["reconstruction_residual"] < 1e-12


def test_nonnormal_uses_singular_value_not_eigenvalue():
    M = np.zeros((4, 4))
    M[0, 1] = 6.0                                 # nilpotent: all eigenvalues 0, largest singular value 6
    assert np.max(np.abs(np.linalg.eigvals(M))) == 0.0
    S = _S(M)
    assert abs(S - 6.0 / 5.0) < 1e-14
    assert classify(np.array([S]), M[None], A, 1e-10)[0][0] == OUTSIDE
    M2 = np.array([[1.0, 4.0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    assert max(abs(np.linalg.eigvals(M2))) < 5.0 * 0.99   # eigen-radius says inside ...
    assert abs(_S(M2) - np.linalg.svd(M2, compute_uv=False)[0] / 5.0) < 1e-14


# ---- E. estimators ----------------------------------------------------------------------------------
def _direct(f, gz, J, t):
    ins = J == 1
    e = np.exp(t * f[ins])
    r = e / e.mean()
    D = np.mean(r * np.log(r))
    I = t * t * np.mean(r * gz[ins])
    return D, I, 2 * D / I


@pytest.mark.parametrize("t", [-1.0, -0.5, 0.5, 1.0])
def test_entropy_matches_direct_weighted_sum(t):
    g = np.random.default_rng(3)
    f = g.standard_normal(5000) * 1.3 + 0.2
    gz = g.gamma(2.0, 1.0, 5000)
    J = (g.random(5000) < 0.7).astype(float)
    st = tilt_terms(f, gz, J, t)
    D, I, R = _direct(f, gz, J, t)
    assert abs(st["D"] - D) < 1e-12 and abs(st["I"] - I) < 1e-12 * I and abs(st["R"] - R) < 1e-11 * abs(R)
    assert abs(st["P"] - J.mean()) < 1e-15
    # shift invariance: adding a constant to every f changes nothing
    st2 = tilt_terms(f + 7.0, gz, J, t)
    assert abs(st2["R"] - st["R"]) < 1e-10


def test_constant_probe_is_degenerate_not_floored():
    mu, s, deg = standardization(np.full(100, 3.0), 1e-8)
    assert deg and s == 0.0
    st = tilt_terms(np.zeros(50), np.zeros(50), np.ones(50), 0.5)
    assert st["reason"] == "I=0" and math.isnan(st["R"])
    st = tilt_terms(np.zeros(50), np.ones(50), np.zeros(50), 0.5)
    assert st["reason"] == "P=0"


def test_outside_states_keep_time_positions():
    """Conditional weighting must keep zeros in the time series (P uses all states)."""
    g = np.random.default_rng(5)
    f = g.standard_normal(1000)
    J = (np.abs(f) < 1.0).astype(float)
    st = tilt_terms(f, np.ones(1000), J, 1.0)
    st_dropped = tilt_terms(f[J == 1], np.ones(int(J.sum())), np.ones(int(J.sum())), 1.0)
    assert abs(st["R"] - st_dropped["R"]) < 1e-12          # ratio identical ...
    assert st["P"] < 1.0 and st_dropped["P"] == 1.0         # ... but P, Z, B, C are over all states
    assert abs(st["Z"] - st_dropped["Z"] * st["P"]) < 1e-12


def _ar1(phi, n, chains, seed):
    g = np.random.default_rng(seed)
    x = np.empty((chains, n))
    x[:, 0] = g.standard_normal(chains)
    e = g.standard_normal((chains, n)) * math.sqrt(1 - phi**2)
    for i in range(1, n):
        x[:, i] = phi * x[:, i - 1] + e[:, i]
    return x


@pytest.mark.parametrize("phi", [0.5, 0.9])
def test_raw_iat_on_ar1(phi):
    x = _ar1(phi, 100_000, 4, 11)
    s = inefficiency(x)
    exact = (1 + phi) / (1 - phi)
    assert abs(s - exact) / exact < 0.05
    assert abs(tau_hat(x, 0.1) - 0.05 * s) < 1e-12        # factor one half


def test_rank_transform_changes_nonlinear_iat():
    phi, c = 0.9, 1.5
    x = _ar1(phi, 200_000, 4, 12)
    g = np.exp(c * x)
    k = np.arange(1, 2000)
    exact_raw = 1 + 2 * np.sum((np.exp(c * c * phi**k) - 1) / (np.exp(c * c) - 1))
    raw = inefficiency(g)
    ranked = inefficiency(dg._z_scale(g))
    assert abs(raw - exact_raw) / exact_raw < 0.10
    assert abs(ranked - raw) / raw > 0.3                   # rank-based IAT is a different quantity


def test_tau_ou_limit():
    assert abs(tau_ou(1e-6, 0.7) - 0.49) < 1e-9
    assert tau_ou(0.02 * 0.49, 0.7) > 0.49


def test_family_heldout_selects_on_other_fold():
    x_slow = _ar1(0.95, 20_000, 4, 1)
    x_fast = _ar1(0.2, 20_000, 4, 2)
    est, info = family_heldout({"slow": x_slow, "fast": x_fast}, ["slow", "fast"], 0.01, [[0, 1], [2, 3]])
    assert info["selected"] == ["slow", "slow"]
    assert abs(est - np.mean([tau_hat(x_slow[[2, 3]], 0.01), tau_hat(x_slow[[0, 1]], 0.01)])) < 1e-15


def test_heldout_entropy_selection():
    g = np.random.default_rng(0)
    F = [g.standard_normal((400, 2)) for _ in range(4)]
    for f in F:
        f[:, 1] = f[:, 1] ** 2                                # nonlinear probe: larger ratio
    G = [np.ones((400, 2)) for _ in range(4)]
    J = [np.ones(400) for _ in range(4)]
    tp = TiltProblem(F, G, J, np.zeros(2), np.ones(2), ["lin", "sq"])
    est, info = heldout_selection(tp, [(0, 1.0), (1, 1.0)], [[0, 1], [2, 3]])
    assert all(s[0] == 1 for s in info["selected"])


def test_static_q_uses_population_variance():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert q_ratio(v, np.full(4, 2.0)) == pytest.approx(np.var(v) / 2.0)


# ---- diagnostics conventions --------------------------------------------------------------------------
def test_block_indices_within_chain():
    rng = np.random.default_rng(0)
    idx = dg.block_indices(103, 16, 50, rng)
    assert idx.shape == (50, 103) and idx.min() >= 0 and idx.max() <= 102
    d = np.diff(idx[:, :16], axis=1)
    assert np.all(d == 1)


def test_bootstrap_iid_mean_mcse():
    g = np.random.default_rng(1)
    chains = [g.standard_normal(4096) for _ in range(4)]
    br = dg.bootstrap_scalar(lambda cs: float(np.concatenate(cs).mean()), chains, 1.0, reps=400, seed=7)
    assert abs(br.mcse - 1 / math.sqrt(4 * 4096)) / (1 / math.sqrt(4 * 4096)) < 0.15
    assert br.block == 16 and br.enough_blocks and br.stable


def test_constant_scalar_status():
    d = dg.scalar_diagnostics("c", np.ones((4, 100)))
    assert d.status == "constant" and math.isnan(d.ess_bulk)
    assert dg.rank_gate(d, rhat_max=1.01, bulk_min=1000, tail_min=400) == (True, "constant")
    d = dg.scalar_diagnostics("near", np.ones((4, 100)) + 1e-14 * np.arange(100))
    assert d.status == "ok"


def test_matches_installed_arviz():
    az = pytest.importorskip("arviz")
    x = _ar1(0.7, 3000, 4, 21)
    x = np.exp(x) + 0.1 * x
    assert abs(dg.rhat_rank(x) - float(az.rhat(x, method="rank"))) < 1e-10
    assert abs(dg.ess_bulk(x) - float(az.ess(x, method="bulk"))) < 1e-6 * dg.ess_bulk(x)
    assert abs(dg.geyer_ess(dg._split(x))[0] - float(az.ess(x, method="mean"))) < 1e-6 * dg.ess_bulk(x)
    try:
        tail = float(az.ess(x, method="tail"))
    except TypeError:
        tail = float(az.ess(x, method="tail", prob=(0.05, 0.95)))
    assert abs(dg.ess_tail(x) - tail) < 1e-6 * tail
