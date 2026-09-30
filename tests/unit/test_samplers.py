"""Runbook §6.1 C (sampler algebra) plus per-chain independence of the batched samplers."""

from __future__ import annotations

import numpy as np
import torch
from conftest import CPU, tiny_model

from bnn_geometry.ellipse import EllipticalSlice
from bnn_geometry.pcnl import PCNL, log_ratio_full, log_ratio_stable
from bnn_geometry.randomness import ChainRNG


def test_stable_ratio_equals_full_ratio_on_ten_pairs():
    g = np.random.default_rng(0)
    for _ in range(10):
        p = 7
        x, y = torch.as_tensor(g.standard_normal(p)), torch.as_tensor(g.standard_normal(p))
        gx, gy = torch.as_tensor(g.standard_normal(p)), torch.as_tensor(g.standard_normal(p))
        V, Vp = torch.as_tensor(g.standard_normal()), torch.as_tensor(g.standard_normal())
        h, sigma = float(g.uniform(1e-4, 0.5)), float(g.uniform(0.5, 1.5))
        a = float(log_ratio_stable(x, y, V, Vp, gx, gy, h, sigma))
        b = float(log_ratio_full(x, y, V, Vp, gx, gy, h, sigma))
        assert abs(a - b) <= 1e-9 + 1e-9 * abs(b)


def test_ratio_vanishes_when_V_is_zero():
    g = np.random.default_rng(1)
    for _ in range(10):
        x, y = torch.as_tensor(g.standard_normal(9)), torch.as_tensor(g.standard_normal(9))
        z = torch.zeros(9)
        for h in (1e-3, 0.02, 0.3):
            assert abs(float(log_ratio_full(x, y, 0.0, 0.0, z, z, h, 0.7))) < 1e-10
            assert abs(float(log_ratio_stable(x, y, 0.0, 0.0, z, z, h, 0.7))) < 1e-10


def test_omitting_reverse_proposal_is_detected():
    g = np.random.default_rng(2)
    x, y = torch.as_tensor(g.standard_normal(5)), torch.as_tensor(g.standard_normal(5))
    gx, gy = torch.as_tensor(g.standard_normal(5)), torch.as_tensor(g.standard_normal(5))
    V, Vp = 0.3, 1.1
    full = float(log_ratio_full(x, y, V, Vp, gx, gy, 0.05, 1.0))
    likelihood_only = V - Vp
    assert abs(full - likelihood_only) > 1e-3


def _rngs(seeds):
    return [ChainRNG(s, CPU) for s in seeds]


def test_batched_chains_match_single_chain_runs(arch):
    model, _ = tiny_model(arch)
    start = model.theta0 + torch.randn(3, model.lay.p, generator=torch.Generator().manual_seed(4))
    ess = EllipticalSlice(model.V, model.theta0, model.sigma)
    st, _ = ess.init_state(start)
    rngs = _rngs([11, 12, 13])
    for _ in range(25):
        ess.step(st, rngs)
    for c, s in enumerate([11, 12, 13]):
        st1, _ = ess.init_state(start[c:c + 1])
        r1 = _rngs([s])
        for _ in range(25):
            ess.step(st1, r1)
        assert torch.allclose(st1.theta[0], st.theta[c], atol=1e-12, rtol=0)
    pc = PCNL(model.V_and_grad, model.theta0, model.sigma, 0.05)
    st = pc.init_state(start)
    rngs = _rngs([21, 22, 23])
    accs = [pc.step(st, rngs)[0] for _ in range(30)]
    for c, s in enumerate([21, 22, 23]):
        st1 = pc.init_state(start[c:c + 1])
        r1 = _rngs([s])
        a1 = [pc.step(st1, r1)[0][0] for _ in range(30)]
        assert torch.allclose(st1.theta[0], st.theta[c], atol=1e-12, rtol=0)
        assert a1 == [a[c] for a in accs]


def test_rejection_keeps_exact_state():
    model, _ = tiny_model("shallow")
    pc = PCNL(model.V_and_grad, model.theta0, model.sigma, 5.0)   # huge step: frequent rejections
    st = pc.init_state(model.theta0.unsqueeze(0) + 0.1)
    rngs = _rngs([3])
    rejected = 0
    for _ in range(50):
        before = st.theta.clone()
        Vb, gb = st.V.clone(), st.g.clone()
        acc, _ = pc.step(st, rngs)
        if not acc[0]:
            rejected += 1
            assert torch.equal(before, st.theta) and torch.equal(Vb, st.V) and torch.equal(gb, st.g)
    assert rejected > 0


def test_prior_draw_variance_is_width_independent():
    from bnn_geometry.chains import prior_draws
    for m in (4, 16):
        model, _ = tiny_model("deep", m=m, sigma=0.7)
        th = prior_draws(model.theta0, model.sigma, _rngs(list(range(200))))
        z = (th - model.theta0) / 0.7
        assert abs(float(z.var()) - 1.0) < 0.05
