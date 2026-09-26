"""Preconditioned Crank–Nicolson cross-check (Cotter et al.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .device_utils import synchronize
from .model import binary_potential_torch, theta_from_z


@dataclass
class PCNStats:
    n_updates: int = 0
    n_accepted: int = 0
    n_likelihood_evals: int = 0
    wall_time_s: float = 0.0
    beta_final: float = 0.2
    beta_history: list[float] = field(default_factory=list)


def pcn_step(
    z: torch.Tensor,
    theta0: torch.Tensor,
    sigma: float,
    X: torch.Tensor,
    y: torch.Tensor,
    m: int,
    beta: float,
    rng: np.random.Generator,
    stats: Optional[PCNStats] = None,
) -> tuple[torch.Tensor, bool]:
    """
    z' = sqrt(1-β^2) z + β ξ,  ξ ~ N(0,I)
    α = min{1, exp(V(θ) - V(θ'))}   (likelihood-only MH ratio)
    """
    if stats is None:
        stats = PCNStats()
    p = z.numel()
    device = z.device
    xi = torch.as_tensor(rng.standard_normal(p), device=device, dtype=z.dtype)
    z_prop = float(np.sqrt(1.0 - beta**2)) * z + beta * xi

    theta = theta_from_z(z, theta0, sigma)
    theta_prop = theta_from_z(z_prop, theta0, sigma)
    V = float(binary_potential_torch(theta, X, y, m).item())
    Vp = float(binary_potential_torch(theta_prop, X, y, m).item())
    stats.n_likelihood_evals += 2

    log_alpha = V - Vp
    accept = False
    if log_alpha >= 0.0 or float(rng.uniform()) < float(np.exp(log_alpha)):
        z = z_prop
        accept = True
        stats.n_accepted += 1
    stats.n_updates += 1
    return z, accept


def run_pcn_chain(
    z0: torch.Tensor,
    theta0: torch.Tensor,
    sigma: float,
    X: torch.Tensor,
    y: torch.Tensor,
    m: int,
    n_burnin: int,
    n_retained: int,
    rng: np.random.Generator,
    beta0: float = 0.2,
    target_accept: float = 0.3,
    adapt_scale: float = 0.05,
    beta_clip: tuple[float, float] = (1e-3, 0.999),
    likelihood_budget: Optional[int] = None,
    stats: Optional[PCNStats] = None,
) -> tuple[torch.Tensor, PCNStats, bool]:
    """
    Adapt log β during burn-in:

      log β_{t+1} = log β_t + 0.05 (t+10)^{-1/2} (I_accept - 0.3)

    then freeze β. Retain repeated states after rejections.
    """
    if stats is None:
        stats = PCNStats()
    import time

    z = z0.clone()
    beta = float(beta0)
    synchronize(z.device)
    t0 = time.perf_counter()
    hit_budget = False

    def over_budget() -> bool:
        return likelihood_budget is not None and stats.n_likelihood_evals >= likelihood_budget

    for t in range(n_burnin):
        if over_budget():
            hit_budget = True
            break
        z, accepted = pcn_step(
            z, theta0, sigma, X, y, m, beta, rng, stats=stats
        )
        log_beta = np.log(beta)
        log_beta = log_beta + adapt_scale * ((t + 10) ** -0.5) * (
            float(accepted) - target_accept
        )
        beta = float(np.clip(np.exp(log_beta), beta_clip[0], beta_clip[1]))
        stats.beta_history.append(beta)

    stats.beta_final = beta
    retained = []
    for _ in range(n_retained):
        if over_budget():
            hit_budget = True
            break
        z, _ = pcn_step(z, theta0, sigma, X, y, m, beta, rng, stats=stats)
        retained.append(z.detach().cpu().clone())

    synchronize(z.device)
    stats.wall_time_s += time.perf_counter() - t0
    if retained:
        Z = torch.stack(retained, dim=0)
    else:
        Z = torch.empty((0, z.numel()), dtype=z.dtype)
    return Z, stats, hit_budget
