"""Elliptical slice sampling in standardized coordinates (Murray et al., 2010)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .device_utils import synchronize
from .model import binary_potential_torch, theta_from_z


@dataclass
class SliceStats:
    n_updates: int = 0
    n_likelihood_evals: int = 0
    n_bracket_shrinks: int = 0
    wall_time_s: float = 0.0


@dataclass
class SliceState:
    z: torch.Tensor
    stats: SliceStats = field(default_factory=SliceStats)


def elliptical_slice_step(
    z: torch.Tensor,
    theta0: torch.Tensor,
    sigma: float,
    X: torch.Tensor,
    y: torch.Tensor,
    m: int,
    rng: np.random.Generator,
    stats: Optional[SliceStats] = None,
    max_shrinks: int = 1_000_000,
) -> torch.Tensor:
    """
    One elliptical slice update on z ~ N(0,I) * exp(-V(θ0+σz)).

    Never rejects for leaving the cylinder; membership is postprocessing.
    Counts every likelihood evaluation, including rejected angle proposals.
    """
    if stats is None:
        stats = SliceStats()

    p = z.numel()
    device = z.device
    nu = torch.as_tensor(rng.standard_normal(p), device=device, dtype=z.dtype)

    def neg_V_of_z(zz: torch.Tensor) -> float:
        theta = theta_from_z(zz, theta0, sigma)
        val = -float(binary_potential_torch(theta, X, y, m).item())
        stats.n_likelihood_evals += 1
        return val

    # Current height threshold
    log_y = neg_V_of_z(z)
    u = float(rng.uniform(0.0, 1.0))
    # log u + (-V)  <=>  threshold on -V
    h = log_y + float(np.log(u))

    # Full-angle bracket as in the original algorithm
    alpha = float(rng.uniform(0.0, 2.0 * np.pi))
    bracket_lo = alpha - 2.0 * np.pi
    bracket_hi = alpha

    for _ in range(max_shrinks):
        cos_a = math_cos(alpha)
        sin_a = math_sin(alpha)
        z_prop = z * cos_a + nu * sin_a
        if neg_V_of_z(z_prop) >= h:
            stats.n_updates += 1
            return z_prop
        # Shrink bracket around 0 according to the sign of alpha
        if alpha < 0.0:
            bracket_lo = alpha
        else:
            bracket_hi = alpha
        alpha = float(rng.uniform(bracket_lo, bracket_hi))
        stats.n_bracket_shrinks += 1

    raise RuntimeError("Elliptical slice sampling exceeded max bracket shrinks")


def math_cos(a: float) -> float:
    return float(np.cos(a))


def math_sin(a: float) -> float:
    return float(np.sin(a))


def run_elliptical_slice_chain(
    z0: torch.Tensor,
    theta0: torch.Tensor,
    sigma: float,
    X: torch.Tensor,
    y: torch.Tensor,
    m: int,
    n_burnin: int,
    n_retained: int,
    rng: np.random.Generator,
    record_fn: Optional[Callable[[torch.Tensor, int, str], None]] = None,
    likelihood_budget: Optional[int] = None,
    stats: Optional[SliceStats] = None,
) -> tuple[torch.Tensor, SliceStats, bool]:
    """
    Run burn-in + retained elliptical slice updates.

    record_fn(z, index, phase) is called after each retained update with
    phase in {"burnin","retained"} — burnin calls use index in burn-in range
    only if record_fn wants them; by default we only record retained.

    Returns (z_final, stats, hit_budget).
    """
    if stats is None:
        stats = SliceStats()
    z = z0.clone()
    import time

    synchronize(z.device)
    t0 = time.perf_counter()
    hit_budget = False

    def over_budget() -> bool:
        return likelihood_budget is not None and stats.n_likelihood_evals >= likelihood_budget

    for i in range(n_burnin):
        if over_budget():
            hit_budget = True
            break
        z = elliptical_slice_step(z, theta0, sigma, X, y, m, rng, stats=stats)

    retained = []
    for i in range(n_retained):
        if over_budget():
            hit_budget = True
            break
        z = elliptical_slice_step(z, theta0, sigma, X, y, m, rng, stats=stats)
        retained.append(z.detach().cpu().clone())
        if record_fn is not None:
            record_fn(z, i, "retained")

    synchronize(z.device)
    stats.wall_time_s += time.perf_counter() - t0
    if retained:
        Z = torch.stack(retained, dim=0)
    else:
        Z = torch.empty((0, z.numel()), dtype=z.dtype)
    return Z, stats, hit_budget
