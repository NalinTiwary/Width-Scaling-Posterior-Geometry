"""
Batched smallest eigenpair of symmetric arrowhead matrices (addendum §5)

    A_j = [[0, c_jᵀ], [c_j, diag(b_j)]],   c_j, b_j ∈ R^n,  j = 1..m.

Couplings with |c_ij| ≤ rtol · scale_j are deflated: their b_ij are exact eigenvalues
(eigenvector e_i).  For the coupled part, the smallest eigenvalue is the unique root of

    g(λ) = −λ − Σ_i c_i² / (b_i − λ)

below min(0, smallest coupled pole); g is strictly decreasing there, g(lo) ≥ 0 at
lo = min(0, min b) − ‖c‖ and g → −∞ at the pole, so a fixed number of bisection steps
brackets it to machine precision in O(n m) work per step.  The eigenvector is
∝ (1, −c_i / (b_i − λ)).  Residuals ‖A v − λ v‖ are evaluated on the undeflated matrix.
"""

from __future__ import annotations

import torch


def arrowhead_min_eig(
    c: torch.Tensor,
    b: torch.Tensor,
    iters: int = 100,
    rtol: float = 1e-14,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    c, b: (m, n).  Returns (lam (m,), vec (m, n+1) unit-norm, rel_residual (m,)).
    """
    m, n = c.shape
    dt, dev = c.dtype, c.device
    inf = torch.tensor(float("inf"), dtype=dt, device=dev)

    cnorm_full = torch.linalg.vector_norm(c, dim=1)
    scale = torch.maximum(
        torch.ones_like(cnorm_full),
        torch.maximum(b.abs().amax(dim=1), cnorm_full),
    )
    coupled = c.abs() > rtol * scale[:, None]
    cd = torch.where(coupled, c, torch.zeros_like(c))
    c2 = cd * cd
    has_c = coupled.any(dim=1)

    # Solve in μ = λ − s, shifted to the smallest coupled pole s, so the critical gap
    # b_k − λ = −μ keeps full relative precision when the root hugs the pole (LAPACK dlaed4).
    pole = torch.where(coupled, b, inf).amin(dim=1)
    s = torch.where(has_c, pole, torch.zeros_like(pole))
    bs = b - s[:, None]
    hi = torch.minimum(-s, torch.zeros_like(s))  # min(pole, 0) − s
    lo = (
        torch.minimum(b.amin(dim=1), torch.zeros_like(s))
        - torch.linalg.vector_norm(cd, dim=1)
        - s
    )
    lo = lo - 1e-300 - 4 * torch.finfo(dt).eps * (lo.abs() + s.abs())
    hi = torch.where(has_c, hi, torch.zeros_like(hi))
    lo = torch.where(has_c, lo, torch.zeros_like(lo))

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        denom = torch.where(coupled, bs - mid[:, None], torch.ones_like(b))
        g = -(s + mid) - (c2 / denom).sum(dim=1)
        above = g > 0  # root lies to the right of mid
        lo = torch.where(above, mid, lo)
        hi = torch.where(above, hi, mid)
    mu = 0.5 * (lo + hi)
    mu = torch.where(mu >= 0, lo, mu)

    # Candidate from the coupled part (or the decoupled head coordinate, eigenvalue 0).
    lam_c = torch.where(has_c, s + mu, torch.zeros_like(mu))
    unc_b = torch.where(coupled, inf, b)
    lam_u = unc_b.amin(dim=1)
    arg_u = unc_b.argmin(dim=1)
    use_unc = lam_u < lam_c
    lam = torch.where(use_unc, lam_u, lam_c)

    denom = torch.where(coupled, bs - mu[:, None], torch.ones_like(b))
    tail = torch.where(coupled, -cd / denom, torch.zeros_like(b))
    v = torch.cat([torch.ones((m, 1), dtype=dt, device=dev), tail], dim=1)
    v = v / torch.linalg.vector_norm(v, dim=1, keepdim=True)
    e_unc = torch.zeros((m, n + 1), dtype=dt, device=dev)
    e_unc[torch.arange(m, device=dev), arg_u + 1] = 1.0
    vec = torch.where(use_unc[:, None], e_unc, v)

    # Residual on the original (undeflated) arrowhead.
    Av0 = (c * vec[:, 1:]).sum(dim=1)
    Avt = c * vec[:, :1] + b * vec[:, 1:]
    r = torch.cat([(Av0 - lam * vec[:, 0])[:, None], Avt - lam[:, None] * vec[:, 1:]], dim=1)
    resid = torch.linalg.vector_norm(r, dim=1) / torch.clamp(lam.abs(), min=1.0)
    return lam, vec, resid


def arrowhead_dense(c: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Materialize (m, n+1, n+1) arrowheads — for validation only."""
    m, n = c.shape
    A = torch.zeros((m, n + 1, n + 1), dtype=c.dtype, device=c.device)
    A[:, 0, 1:] = c
    A[:, 1:, 0] = c
    idx = torch.arange(1, n + 1, device=c.device)
    A[:, idx, idx] = b
    return A
