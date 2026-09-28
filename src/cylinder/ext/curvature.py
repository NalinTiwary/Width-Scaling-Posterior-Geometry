"""
Brackets d₋ ≤ d_H ≤ d₊ for d_H = σ² [−λ_min(∇²V)]₊ (addendum §5).

∇²V = Jᵀ diag(h) J + R with block-diagonal residual R = blockdiag(R_j).
Lower eigenvalue bound ℓ = min_j λ_min(R_j) (since Jᵀ h J ⪰ 0), so d₊ = σ²[−ℓ]₊.
Upper bound u = λ_min(Eᵀ ∇²V E) for an orthonormal E (Rayleigh–Ritz), so d₋ = σ²[−u]₊.

* orth (reduced coordinates): R_j is an arrowhead; E = minimum eigenvectors of the
  ``n_blocks`` worst distinct neuron blocks (disjoint supports ⇒ orthonormal).
  Directions orthogonal to the data (w_⊥) have zero likelihood curvature, so d_H of the
  full network equals d_H computed in the reduced coordinates.
* dense (fmnist, full coordinates): the pilot rule — dense (1+d)×(1+d) blocks and E the
  K = n+1 lowest eigenvectors of R — without materializing the full Jacobian.

In both cases EᵀRE is evaluated exactly (not assumed diagonal), so u is a true Rayleigh bound.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..model import SQRT2
from .arrowhead import arrowhead_min_eig
from .problem import Problem, preacts


def _likelihood_terms(theta: torch.Tensor, prob: Problem):
    a, pre = preacts(theta, prob)  # (m,), (m, n)
    phi = torch.tanh(pre)
    phi_p = 1.0 - phi * phi
    phi_pp = -2.0 * phi * phi_p
    f = (a @ phi) / math.sqrt(prob.m)
    p = torch.sigmoid(SQRT2 * f)
    q = SQRT2 * (p - prob.y)
    h = 2.0 * p * (1.0 - p)
    return a, phi, phi_p, phi_pp, q, h


def _finish(lam_sel: torch.Tensor, G_R: torch.Tensor, JE: torch.Tensor, h: torch.Tensor,
            ell: float, sigma: float, block_resid: float, K: int) -> dict[str, Any]:
    H_E = G_R + JE.T @ (h[:, None] * JE)
    H_E = 0.5 * (H_E + H_E.T)
    evals, evecs = torch.linalg.eigh(H_E)
    u = float(evals[0])
    r = H_E @ evecs[:, 0] - evals[0] * evecs[:, 0]
    u_resid = float(torch.linalg.vector_norm(r) / max(1.0, abs(u)))
    return {
        "ell": ell,
        "u": u,
        "d_minus": sigma**2 * max(0.0, -u),
        "d_plus": sigma**2 * max(0.0, -ell),
        "K": K,
        "max_eigen_residual": max(block_resid, u_resid),
    }


@torch.no_grad()
def bracket_orth(theta: torch.Tensor, prob: Problem, n_blocks: int = 32, iters: int = 100) -> dict[str, Any]:
    if prob.X is not None:
        raise ValueError("bracket_orth requires reduced (identity-input) coordinates")
    m = prob.m
    rs = 1.0 / math.sqrt(m)
    a, phi, phi_p, phi_pp, q, h = _likelihood_terms(theta, prob)
    c = (q[None, :] * phi_p) * rs  # (m, n)
    b = (a[:, None] * q[None, :] * phi_pp) * rs
    lam, vec, resid = arrowhead_min_eig(c, b, iters=iters)
    ell = float(lam.min())

    K = min(n_blocks, m)
    idx = torch.topk(lam, K, largest=False).indices
    V = vec[idx]  # (K, n+1)
    v0, vt = V[:, :1], V[:, 1:]
    # Exact Rayleigh quotients vᵀ R_j v (cross-block terms vanish).
    lam_sel = 2.0 * v0[:, 0] * (c[idx] * vt).sum(dim=1) + (b[idx] * vt * vt).sum(dim=1)
    JE = ((phi[idx] * v0 + a[idx, None] * phi_p[idx] * vt) * rs).T  # (n, K)
    return _finish(lam_sel, torch.diag(lam_sel), JE, h, ell, prob.sigma, float(resid.max()), K)


@torch.no_grad()
def bracket_dense(theta: torch.Tensor, prob: Problem, K: int | None = None) -> dict[str, Any]:
    if prob.X is None:
        raise ValueError("bracket_dense requires explicit inputs")
    X = prob.X
    n, d = X.shape
    m = prob.m
    rs = 1.0 / math.sqrt(m)
    if K is None:
        K = n + 1
    a, phi, phi_p, phi_pp, q, h = _likelihood_terms(theta, prob)  # phi: (m, n)

    Cm = (phi_p * q[None, :]) @ X  # (m, d): c_j = Xᵀ(q ⊙ φ'_j)
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, d * d)
    D = ((phi_pp * q[None, :]) @ XX).reshape(m, d, d) * a[:, None, None]
    R = torch.zeros((m, 1 + d, 1 + d), dtype=theta.dtype, device=theta.device)
    R[:, 0, 1:] = Cm
    R[:, 1:, 0] = Cm
    R[:, 1:, 1:] = D
    R *= rs
    R = 0.5 * (R + R.transpose(1, 2))

    evals, evecs = torch.linalg.eigh(R)
    recon = R @ evecs - evecs * evals[:, None, :]
    block_resid = torch.linalg.matrix_norm(recon) / torch.clamp(
        torch.linalg.vector_norm(evals, dim=1), min=1.0
    )
    flat = evals.reshape(-1)
    K_use = min(K, flat.numel())
    sel = torch.topk(flat, K_use, largest=False).indices
    blk = sel // (1 + d)
    col = sel % (1 + d)
    ell = float(flat.min())
    V = evecs[blk, :, col]  # (K, 1+d)

    RV = torch.bmm(R[blk], V[:, :, None])[:, :, 0]
    same = (blk[:, None] == blk[None, :]).to(theta.dtype)
    G_R = same * (V @ RV.T)
    G_R = 0.5 * (G_R + G_R.T)
    XV = V[:, 1:] @ X.T  # (K, n)
    JE = ((phi[blk] * V[:, :1] + a[blk, None] * phi_p[blk] * XV) * rs).T
    lam_sel = torch.diagonal(G_R)
    return _finish(lam_sel, G_R, JE, h, ell, prob.sigma, float(block_resid.max()), K_use)


def bracket(theta: torch.Tensor, prob: Problem, cfg_curv: dict[str, Any]) -> dict[str, Any]:
    if prob.X is None:
        return bracket_orth(
            theta,
            prob,
            n_blocks=int(cfg_curv.get("upper_bracket_blocks", 32)),
            iters=int(cfg_curv.get("bisection_iters", 100)),
        )
    return bracket_dense(theta, prob)
