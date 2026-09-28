"""
Elliptical slice sampling (Murray, Adams & MacKay 2010) on u ~ N(0, I) · exp(ℓ(u)).

Same algorithm as `cylinder.ess_slice`, but generic in the log-likelihood and with the
current state's log-likelihood carried between updates (no re-evaluation). The auxiliary
Gaussian direction is drawn with a seeded torch.Generator on the sampling device, since
drawing millions of normals per update on the host dominates cost at p ≈ 4·10⁶.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch


@dataclass
class ESSStats:
    n_updates: int = 0
    n_likelihood_evals: int = 0
    n_bracket_shrinks: int = 0


def ess_step(
    u: torch.Tensor,
    loglik_u: float,
    loglik: Callable[[torch.Tensor], float],
    rng: np.random.Generator,
    stats: ESSStats,
    gen: Optional[torch.Generator] = None,
    max_shrinks: int = 100_000,
) -> tuple[torch.Tensor, float]:
    """One update. Never rejects for leaving the cylinder. Returns (u_new, ℓ(u_new))."""
    if gen is None:
        nu = torch.as_tensor(rng.standard_normal(u.numel()), dtype=u.dtype, device=u.device)
    else:
        nu = torch.randn(u.shape, generator=gen, dtype=u.dtype, device=u.device)
    h = loglik_u + math.log(rng.uniform(0.0, 1.0))
    alpha = rng.uniform(0.0, 2.0 * math.pi)
    lo, hi = alpha - 2.0 * math.pi, alpha
    for _ in range(max_shrinks):
        prop = u * math.cos(alpha) + nu * math.sin(alpha)
        ll = loglik(prop)
        stats.n_likelihood_evals += 1
        if ll >= h:
            stats.n_updates += 1
            return prop, ll
        if alpha < 0.0:
            lo = alpha
        else:
            hi = alpha
        alpha = rng.uniform(lo, hi)
        stats.n_bracket_shrinks += 1
    raise RuntimeError("elliptical slice sampling exceeded max bracket shrinks")
