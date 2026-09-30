"""Metropolis-adjusted Gaussian-preserving Langevin (pCNL) proposal with its exact reverse density (runbook §8).

For x = θ − θ0, g = ∇V(θ): η = e^{−h/σ²}, b = σ²(1−η), q_v = σ²(1−η²),
y = η x − b g + √q_v ξ,  θ' = θ0 + y. Batched over chains with per-chain RNG streams (ξ, then u).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .randomness import ChainRNG

# potential_grad(theta (k, p)) -> (V (k,), grad (k, p), cache (k, n) or None)
PotentialGrad = Callable[[torch.Tensor], "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]"]


@dataclass
class PCNLState:
    theta: torch.Tensor
    V: torch.Tensor
    g: torch.Tensor
    cache: "torch.Tensor | None"


def coefficients(h: float, sigma: float) -> tuple[float, float, float]:
    s2 = sigma * sigma
    one_m_eta = -math.expm1(-h / s2)
    one_m_eta2 = -math.expm1(-2.0 * h / s2)
    eta = 1.0 - one_m_eta
    return eta, s2 * one_m_eta, s2 * one_m_eta2


def log_ratio_stable(x, y, V, Vp, g, gp, h: float, sigma: float) -> torch.Tensor:
    """Canceled-form log acceptance ratio Λ (runbook §8.1), batched over the leading dimension."""
    eta, _, _ = coefficients(h, sigma)
    one_m_eta = -math.expm1(-h / sigma**2)
    t1 = ((y - eta * x) * g).sum(-1) - ((x - eta * y) * gp).sum(-1)
    return (V - Vp) + t1 / (1.0 + eta) + sigma**2 * one_m_eta / (2.0 * (1.0 + eta)) * (
        (g * g).sum(-1) - (gp * gp).sum(-1))


def log_ratio_full(x, y, V, Vp, g, gp, h: float, sigma: float) -> torch.Tensor:
    """−U(θ') + U(θ) + log q(θ|θ') − log q(θ'|θ) with explicit Gaussian densities (for testing)."""
    eta, b, qv = coefficients(h, sigma)
    U = V + (x * x).sum(-1) / (2 * sigma**2)
    Up = Vp + (y * y).sum(-1) / (2 * sigma**2)
    fwd = y - (eta * x - b * g)
    rev = x - (eta * y - b * gp)
    return -Up + U - (rev * rev).sum(-1) / (2 * qv) + (fwd * fwd).sum(-1) / (2 * qv)


class PCNL:
    def __init__(self, potential_grad: PotentialGrad, theta0: torch.Tensor, sigma: float, h: float):
        self.pg, self.theta0, self.sigma, self.h = potential_grad, theta0, float(sigma), float(h)
        self.eta, self.b, self.qv = coefficients(self.h, self.sigma)

    def init_state(self, theta: torch.Tensor) -> PCNLState:
        V, g, cache = self.pg(theta)
        return PCNLState(theta.clone(), V.clone(), g.clone(), None if cache is None else cache.clone())

    def step(self, st: PCNLState, rngs: list[ChainRNG]) -> tuple[np.ndarray, np.ndarray]:
        """One transition in place; returns (accepted (C,) bool, log ratio (C,)). One gradient per chain."""
        C, p = st.theta.shape
        dev = st.theta.device
        x = st.theta - self.theta0
        xi = torch.stack([torch.randn(p, generator=r.torch, device=dev, dtype=st.theta.dtype) for r in rngs])
        y = self.eta * x - self.b * st.g + math.sqrt(self.qv) * xi
        prop = self.theta0 + y
        Vp, gp, cp = self.pg(prop)
        lam = log_ratio_stable(x, y, st.V, Vp, st.g, gp, self.h, self.sigma)
        lam_np = lam.detach().cpu().numpy()
        logu = np.array([math.log(r.np.random()) for r in rngs])
        finite = np.isfinite(lam_np) & np.isfinite(Vp.detach().cpu().numpy())
        acc = finite & (logu < np.minimum(0.0, np.where(finite, lam_np, -np.inf)))
        if acc.any():
            sel = torch.as_tensor(np.flatnonzero(acc), device=dev)
            st.theta[sel] = prop[sel]
            st.V[sel] = Vp[sel]
            st.g[sel] = gp[sel]
            if st.cache is not None and cp is not None:
                st.cache[sel] = cp[sel]
        return acc, lam_np
