"""Centered full-state elliptical slice sampling around the Gaussian prior center (runbook §7).

Batched over chains: each chain draws from its own ChainRNG in a fixed order (ζ, u, α, shrink angles), so a
chain's trajectory does not depend on which other chains share the batch. Only chains still searching are
evaluated; every attempted likelihood evaluation is counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .randomness import ChainRNG

# potential(theta (k, p)) -> (V (k,), cache (k, n) or None)
Potential = Callable[[torch.Tensor], "tuple[torch.Tensor, torch.Tensor | None]"]


class SamplerFailure(RuntimeError):
    pass


@dataclass
class ESSState:
    theta: torch.Tensor          # (C, p)
    V: torch.Tensor              # (C,)
    cache: "torch.Tensor | None"  # (C, n) logits at theta, or None


class EllipticalSlice:
    def __init__(self, potential: Potential, theta0: torch.Tensor, sigma: float, guard: int = 10_000):
        self.potential, self.theta0, self.sigma, self.guard = potential, theta0, float(sigma), int(guard)

    def init_state(self, theta: torch.Tensor) -> tuple[ESSState, np.ndarray]:
        V, cache = self.potential(theta)
        if not torch.isfinite(V).all():
            raise SamplerFailure("non-finite V at initialization")
        return ESSState(theta.clone(), V.clone(), None if cache is None else cache.clone()), \
            np.ones(theta.shape[0], dtype=np.int64)

    def step(self, st: ESSState, rngs: list[ChainRNG]) -> np.ndarray:
        """One transition for every chain in place. Returns likelihood evaluations per chain."""
        C, p = st.theta.shape
        dev = st.theta.device
        x = st.theta - self.theta0
        zeta = torch.stack([torch.randn(p, generator=r.torch, device=dev, dtype=st.theta.dtype) for r in rngs])
        zeta.mul_(self.sigma)
        logy = np.empty(C)
        alpha, lo, hi = np.empty(C), np.empty(C), np.empty(C)
        Vcur = st.V.detach().cpu().numpy()
        for c, r in enumerate(rngs):
            logy[c] = -Vcur[c] + math.log(r.np.random())
            alpha[c] = r.np.uniform(0.0, 2.0 * math.pi)
            lo[c], hi[c] = alpha[c] - 2.0 * math.pi, alpha[c]
        active = np.ones(C, dtype=bool)
        evals = np.zeros(C, dtype=np.int64)
        while active.any():
            idx = np.flatnonzero(active)
            it = torch.as_tensor(idx, device=dev)
            a = torch.as_tensor(alpha[idx], device=dev, dtype=st.theta.dtype).unsqueeze(1)
            prop = self.theta0 + x[it] * torch.cos(a) + zeta[it] * torch.sin(a)
            Vp, cp = self.potential(prop)
            Vp_np = Vp.detach().cpu().numpy()
            evals[idx] += 1
            if not np.isfinite(Vp_np).all():
                raise SamplerFailure("non-finite V during an elliptical slice search")
            acc = -Vp_np >= logy[idx]
            if acc.any():
                sel = torch.as_tensor(np.flatnonzero(acc), device=dev)
                tgt = torch.as_tensor(idx[acc], device=dev)
                st.theta[tgt] = prop[sel]
                st.V[tgt] = Vp[sel]
                if st.cache is not None and cp is not None:
                    st.cache[tgt] = cp[sel]
                active[idx[acc]] = False
            for k in np.flatnonzero(~acc):
                c = idx[k]
                if alpha[c] < 0.0:
                    lo[c] = alpha[c]
                else:
                    hi[c] = alpha[c]
                if evals[c] >= self.guard:
                    raise SamplerFailure(f"chain {c}: bracket guard of {self.guard} evaluations reached")
                alpha[c] = rngs[c].np.uniform(lo[c], hi[c])
        return evals
