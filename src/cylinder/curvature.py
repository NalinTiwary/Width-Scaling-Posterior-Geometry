"""
Block-diagonal residual Hessian brackets for d_H (design §6.2).

∇²V = Jᵀ diag(h) J + R, with R = blockdiag(R_1,...,R_m),
each R_j a (1+d)×(1+d) block. Bracket λ_min(∇²V) via the residual
spectrum and a Rayleigh–Ritz projection with the K = n+1 lowest
eigenvectors of R.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg
from scipy.special import expit

from .model import SQRT2, unpack_params


def _phi_derivatives(pre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tanh and derivatives. pre: (n, m)."""
    phi = np.tanh(pre)
    phi_p = 1.0 - phi**2
    phi_pp = -2.0 * phi * phi_p
    return phi, phi_p, phi_pp


def residual_blocks(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    m: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build all R_j as an (m, 1+d, 1+d) array, and return q, h, φ', a.

      R_j = m^{-1/2} [[0, c_jᵀ], [c_j, D_j]]
      c_j = Xᵀ(q ⊙ φ'_j),   D_j = a_j Xᵀ diag(q ⊙ φ''_j) X
      q_i = √2 (p_i - y_i),  h_i = 2 p_i (1-p_i),  p_i = sigmoid(√2 f_i).
    """
    n, d = X.shape
    a, W = unpack_params(theta, m, d)
    pre = X @ W.T  # (n, m)
    phi, phi_p, phi_pp = _phi_derivatives(pre)
    f = (phi @ a) / np.sqrt(m)
    p = expit(SQRT2 * f)
    q = SQRT2 * (p - y)
    h = 2.0 * p * (1.0 - p)

    C = (X.T @ (q[:, None] * phi_p)).T  # (m, d), row j = c_j
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, d * d)
    D = ((q[:, None] * phi_pp).T @ XX).reshape(m, d, d) * a[:, None, None]

    R = np.zeros((m, 1 + d, 1 + d), dtype=np.float64)
    R[:, 0, 1:] = C
    R[:, 1:, 0] = C
    R[:, 1:, 1:] = D
    R /= np.sqrt(m)
    R = 0.5 * (R + R.transpose(0, 2, 1))
    return R, q, h, phi_p, a


def jacobian_blocks(
    theta: np.ndarray,
    X: np.ndarray,
    m: int,
    phi_p: np.ndarray | None = None,
) -> np.ndarray:
    """
    J_j = m^{-1/2} [ φ_j , a_j diag(φ'_j) X ], returned as (m, n, 1+d).
    """
    n, d = X.shape
    a, W = unpack_params(theta, m, d)
    phi = np.tanh(X @ W.T)
    if phi_p is None:
        phi_p = 1.0 - phi**2
    J = np.empty((m, n, 1 + d), dtype=np.float64)
    J[:, :, 0] = phi.T
    J[:, :, 1:] = (a[:, None] * phi_p.T)[:, :, None] * X[None, :, :]
    J /= np.sqrt(m)
    return J


def bracket_lambda_min(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    m: int,
    K: int | None = None,
) -> dict[str, Any]:
    """
    Return ℓ ≤ λ_min(∇²V) ≤ u  (σ² normalization applied by the caller).

      1. Diagonalize every R_j (batched)
      2. ℓ = smallest eigenvalue over all blocks
      3. E = the K = n+1 lowest eigenvectors of R, embedded block-sparsely
      4. H_E = diag(λ_E) + (JE)ᵀ diag(h) (JE),  u = λ_min(H_E)
    """
    n, d = X.shape
    if K is None:
        K = n + 1
    R, q, h, phi_p, a = residual_blocks(theta, X, y, m)
    J = jacobian_blocks(theta, X, m, phi_p=phi_p)

    evals, evecs = np.linalg.eigh(R)  # (m, 1+d), (m, 1+d, 1+d); columns are eigenvectors
    recon = R @ evecs - evecs * evals[:, None, :]
    block_resid = np.linalg.norm(recon, axis=(1, 2)) / np.maximum(
        1.0, np.linalg.norm(evals, axis=1)
    )
    max_resid = float(block_resid.max())

    flat = evals.ravel()
    K_use = min(K, flat.size)
    sel = np.argpartition(flat, K_use - 1)[:K_use]
    sel = sel[np.argsort(flat[sel])]
    blk = sel // (1 + d)
    col = sel % (1 + d)
    lam = flat[sel]
    ell = float(lam[0])

    V_sel = evecs[blk, :, col]  # (K, 1+d) local eigenvectors
    JE = np.einsum("knp,kp->nk", J[blk], V_sel)  # (n, K)

    H_E = np.diag(lam) + (JE.T * h) @ JE
    H_E = 0.5 * (H_E + H_E.T)
    u_evals, u_vecs = linalg.eigh(H_E)
    u = float(u_evals[0])
    u_recon = H_E @ u_vecs[:, 0] - u_evals[0] * u_vecs[:, 0]
    u_rel = float(np.linalg.norm(u_recon) / max(1.0, abs(u_evals[0])))
    max_resid = max(max_resid, u_rel)

    Jv = J[blk[0]] @ V_sel[0]
    upper_simple = ell + float(np.sum(h * (Jv**2)))

    return {
        "ell": ell,
        "u": u,
        "upper_simple": upper_simple,
        "K": K_use,
        "max_eigen_residual": max_resid,
        "h": h,
        "inside_bracket_ok": ell <= u + 1e-10,
    }


def deficit_bracket(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    m: int,
    sigma: float,
    K: int | None = None,
) -> dict[str, Any]:
    """
    d_- = σ² [-u]₊  ≤  d_H  ≤  d_+ = σ² [-ℓ]₊
    """
    br = bracket_lambda_min(theta, X, y, m, K=K)
    d_minus = float((sigma**2) * max(0.0, -br["u"]))
    d_plus = float((sigma**2) * max(0.0, -br["ell"]))
    br.update({"d_minus": d_minus, "d_plus": d_plus, "sigma": sigma})
    return br


def dense_hessian_autodiff_torch(
    theta_np: np.ndarray,
    X_np: np.ndarray,
    y_np: np.ndarray,
    m: int,
) -> np.ndarray:
    """Dense ∇²V via PyTorch autograd — for tiny networks only."""
    import torch

    from .model import binary_potential_torch

    theta = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    X = torch.tensor(X_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)
    V = binary_potential_torch(theta, X, y, m)
    g = torch.autograd.grad(V, theta, create_graph=True)[0]
    p = theta.numel()
    H = torch.zeros(p, p, dtype=torch.float64)
    for i in range(p):
        gi = torch.autograd.grad(g[i], theta, retain_graph=True)[0]
        H[i] = gi
    return H.detach().numpy()


def decompose_dense_check(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    m: int,
) -> dict[str, Any]:
    """
    Compare dense AD Hessian with Jᵀ diag(h) J + R on a tiny net.
    """
    n, d = X.shape
    R, q, h, phi_p, a = residual_blocks(theta, X, y, m)
    J_blocks = jacobian_blocks(theta, X, m, phi_p=phi_p)

    p = m * (1 + d)
    R_dense = np.zeros((p, p), dtype=np.float64)
    J = np.zeros((n, p), dtype=np.float64)
    for j in range(m):
        sl = slice(j * (1 + d), (j + 1) * (1 + d))
        R_dense[sl, sl] = R[j]
        J[:, sl] = J_blocks[j]
    H_decomp = J.T @ (h[:, None] * J) + R_dense
    H_ad = dense_hessian_autodiff_torch(theta, X, y, m)
    max_abs = float(np.max(np.abs(H_decomp - H_ad)))
    br = bracket_lambda_min(theta, X, y, m)
    lam_min_ad = float(linalg.eigvalsh(H_ad)[0])
    return {
        "max_abs_entry_diff": max_abs,
        "lam_min_ad": lam_min_ad,
        "ell": br["ell"],
        "u": br["u"],
        "bracket_contains": br["ell"] - 1e-8 <= lam_min_ad <= br["u"] + 1e-8,
    }
