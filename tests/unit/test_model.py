"""Runbook §6.1 A (model and gradients) and B (architecture and centers)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import tiny_model

from bnn_geometry.model import Layout, loop_logits, loop_V, pack, prior_center, unpack
from bnn_geometry.probes import Observables, ProbeSpec


def spec_small():
    return ProbeSpec(train_idx=(0, 1, 2, 3, 4), test_idx=(0, 1), entropy_train_idx=(0, 2, 3, 4),
                     neurons=(0, 1, 2, 3), shallow_inputs=(0, 1, 2, 3))


def draws(model, k=3, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return model.theta0 + scale * torch.randn(k, model.lay.p, generator=g)


def test_forward_matches_literal_loop(arch):
    model, _ = tiny_model(arch)
    th = draws(model)
    f = model.logits(th).numpy()
    V = model.V(th)[0].numpy()
    for c in range(th.shape[0]):
        fl = loop_logits(th[c].numpy(), model.lay, model.X.numpy())
        assert np.allclose(f[c], fl, atol=1e-11, rtol=1e-10)
        assert abs(V[c] - loop_V(th[c].numpy(), model.lay, model.X.numpy(), model.y.numpy())) < 1e-10


def test_binary_loss_matches_centered_log_softmax(arch):
    model, _ = tiny_model(arch)
    th = draws(model)
    f = model.logits(th)
    logits = torch.stack([-f / math.sqrt(2), f / math.sqrt(2)], dim=-1)
    ls = torch.log_softmax(logits, dim=-1)
    y = model.y.long()
    V_ls = -(ls.gather(-1, y.expand(th.shape[0], -1).unsqueeze(-1)).squeeze(-1)).sum(-1)
    assert torch.allclose(model.V(th)[0], V_ls, atol=1e-12, rtol=1e-12)
    p1 = torch.softmax(logits, -1)[..., 1]
    assert torch.allclose(p1, torch.sigmoid(math.sqrt(2) * f), atol=1e-14)


def test_V_at_zero_logits_is_n_log2(arch):
    model, _ = tiny_model(arch)
    th = model.theta0.clone().unsqueeze(0)
    lay = model.lay
    a, b = lay.offsets["A"]
    th[:, a:b] = 0.0
    assert abs(float(model.V(th)[0]) - model.X.shape[0] * math.log(2)) < 1e-13


def test_duplicated_data_doubles_V_and_grad(arch):
    model, _ = tiny_model(arch)
    th = draws(model)
    V1, g1, _ = model.V_and_grad(th)
    model2, _ = tiny_model(arch)
    model2.X = torch.cat([model.X, model.X])
    model2.y = torch.cat([model.y, model.y])
    V2, g2, _ = model2.V_and_grad(th)
    assert torch.allclose(V2, 2 * V1, rtol=1e-13, atol=1e-13)
    assert torch.allclose(g2, 2 * g1, rtol=1e-12, atol=1e-13)


def test_analytic_gradient_matches_autograd_and_gradcheck(arch):
    model, _ = tiny_model(arch)
    th = draws(model).requires_grad_(True)
    V, g, _ = model.V_and_grad(th.detach())
    (ga,) = torch.autograd.grad(model.V(th)[0].sum(), th)
    assert torch.allclose(g, ga, atol=1e-12, rtol=1e-11)
    assert torch.autograd.gradcheck(lambda t: model.V(t)[0], (th,), eps=1e-6, atol=1e-7)


def _directional_ok(fun, grad_dot, x, directions):
    for v in directions:
        exact = grad_dot(v)
        ok = False
        for eps in (1e-4, 1e-5, 1e-6):
            fd = (fun(x + eps * v) - fun(x - eps * v)) / (2 * eps)
            err = abs(fd - exact)
            if (abs(exact) > 1e-7 and err / abs(exact) < 1e-5) or (abs(exact) <= 1e-7 and err < 1e-7):
                ok = True
                break
        if not ok:
            return False
    return True


@pytest.mark.parametrize("sigma", [0.7, 1.0])
def test_directional_differences_full_potential_nonzero_center(arch, sigma):
    model, _ = tiny_model(arch, sigma=sigma, center_scale=1.3)
    x = draws(model, k=1, seed=5)[0]
    g = torch.Generator().manual_seed(9)
    dirs = [v / v.norm() for v in torch.randn(5, model.lay.p, generator=g)]
    U = lambda t: float(model.U(t.unsqueeze(0))[0])  # noqa: E731
    V, gV, _ = model.V_and_grad(x.unsqueeze(0))
    gU = gV[0] + (x - model.theta0) / sigma**2
    assert _directional_ok(U, lambda v: float(gU @ v), x, dirs)


def test_every_probe_gradient(arch):
    model, _ = tiny_model(arch, center_scale=1.1)
    obs = Observables(model, spec_small())
    th = draws(model, k=2, seed=3)
    vals, gsq = obs.probe_values_and_grad_sq(th)
    names = spec_small().probe_names()
    assert vals.shape == (2, len(names)) == gsq.shape
    g = torch.Generator().manual_seed(11)
    dirs = [v / v.norm() for v in torch.randn(5, model.lay.p, generator=g)]
    for k in range(len(names)):
        def fun(t, k=k):
            return float(obs.probe_values_and_grad_sq(t.unsqueeze(0))[0][0, k])
        x = th[:1].detach().clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(_probe_fn(obs, x, k).sum(), x)
        grad = grad[0]
        assert abs(float((grad * grad).sum()) - float(gsq[0, k])) < 1e-10
        assert _directional_ok(fun, lambda v, grad=grad: float(grad @ v), th[0].detach(), dirs), names[k]
        assert torch.autograd.gradcheck(lambda t, k=k: _probe_fn(obs, t, k),
                                        (th[:1].detach().clone().requires_grad_(True),), eps=1e-6, atol=1e-7)
    assert torch.allclose(gsq[:, -2:], torch.ones(2, 2), atol=1e-12)   # unit linear controls


def _probe_fn(obs, t, k):
    model = obs.model
    f = model.logits(t)
    V = model.V_from_f(f)
    ints = obs.interactions(t)
    qA, qW = obs.linear_controls(t)
    probes = [V] + [f[:, i] for i in obs.spec.entropy_train_idx] + [ints[:, j] for j in range(4)] + [qA, qW]
    return probes[k]


def test_parameter_counts_and_packing_bitwise(arch):
    for m, d in ((4, 3), (64, 32)):
        lay = Layout(arch, m, d)
        assert lay.p == (33 * m if d == 32 else m + m * d) + (m * m if arch == "deep" else 0)
        th = torch.randn(3, lay.p)
        assert torch.equal(pack(unpack(th, lay), lay), th)


def test_every_block_trainable(arch):
    model, _ = tiny_model(arch)
    _, g, _ = model.V_and_grad(draws(model, k=1))
    for name, (a, b) in model.lay.offsets.items():
        assert float(g[0, a:b].abs().max()) > 0, name


def test_centers(arch):
    d = 3
    bank = np.random.default_rng(0).standard_normal((16, d))
    for m in (4, 8, 16):
        lay = Layout(arch, m, d)
        th0 = prior_center(lay, bank)
        blk = unpack(torch.as_tensor(th0).unsqueeze(0), lay)
        assert abs(float(blk["A"].norm()) - 1.0) < 1e-12
        assert np.array_equal(blk["W1"][0].numpy(), bank[:m])      # nested row bank
        if arch == "deep":
            assert float(blk["W2"].abs().max()) == 0.0
        assert np.array_equal(np.sign(blk["A"][0].numpy()), (-1.0) ** np.arange(m))
