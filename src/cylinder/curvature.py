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
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build R_j blocks and return also q, h, phi', phi'' needed for J.

    q_i = √2 (p_i - y_i),  h_i = 2 p_i (1-p_i),  p_i = sigmoid(√2 f_i).
    """
    n, d = X.shape
    a, W = unpack_params(theta, m, d)
    pre = X @ W.T  # (n, m)
    phi, phi_p, phi_pp = _phi_derivatives(pre)
    f = (phi @ a) / np.sqrt(m)
    s = SQRT2 * f
    p = 1.0 / (1.0 + np.exp(-s))
    q = SQRT2 * (p - y)
    h = 2.0 * p * (1.0 - p)

    inv_sqrt_m = 1.0 / np.sqrt(m)
    blocks: list[np.ndarray] = []
    for j in range(m):
        # c_j = X^T (q ⊙ φ'_j)
        c_j = X.T @ (q * phi_p[:, j])
        # D_j = a_j X^T diag(q ⊙ φ''_j) X
        wq = q * phi_pp[:, j]
        D_j = a[j] * (X.T * wq) @ X
        R = np.zeros((1 + d, 1 + d), dtype=np.float64)
        R[0, 1:] = c_j
        R[1:, 0] = c_j
        R[1:, 1:] = D_j
        R *= inv_sqrt_m
        # Symmetrize numerically
        R = 0.5 * (R + R.T)
        blocks.append(R)
    return blocks, q, h, phi_p, a


def jacobian_blocks(
    theta: np.ndarray,
    X: np.ndarray,
    m: int,
    phi_p: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    J_j = m^{-1/2} [ φ_j , a_j diag(φ'_j) X ]  shape (n, 1+d).
    """
    n, d = X.shape
    a, W = unpack_params(theta, m, d)
    pre = X @ W.T
    phi = np.tanh(pre)
    if phi_p is None:
        phi_p = 1.0 - phi**2
    inv_sqrt_m = 1.0 / np.sqrt(m)
    blocks = []
    for j in range(m):
        J_j = np.empty((n, 1 + d), dtype=np.float64)
        J_j[:, 0] = phi[:, j]
        J_j[:, 1:] = a[j] * (phi_p[:, j][:, None] * X)
        J_j *= inv_sqrt_m
        blocks.append(J_j)
    return blocks


def eigen_residual(
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Symmetric eigh; return eigenvalues ascending, vectors, max residual."""
    evals, evecs = linalg.eigh(R)
    # Residual: ||R V - V Λ|| / max(1, ||Λ||)
    recon = R @ evecs - evecs * evals
    rel = float(np.linalg.norm(recon) / max(1.0, float(np.linalg.norm(evals))))
    return evals, evecs, rel


def bracket_lambda_min(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    m: int,
    K: int | None = None,
) -> dict[str, Any]:
    """
    Return ℓ ≤ λ_min(∇²V) ≤ u and d± = σ²[-·]₊  (σ applied by caller).

    Steps:
      1. Diagonalize each R_j
      2. ℓ = min eigenvalue across blocks
      3. Take K = n+1 lowest eigenvectors of R (embedded sparsely)
      4. H_E = diag(λ) + (JE)^T diag(h) (JE)
      5. u = λ_min(H_E)
    """
    n, d = X.shape
    if K is None:
        K = n + 1
    blocks, q, h, phi_p, a = residual_blocks(theta, X, y, m)
    J_blocks = jacobian_blocks(theta, X, m, phi_p=phi_p)

    # Collect all eigenpairs
    all_evals = []
    all_meta = []  # (block_index, local_vector)
    max_resid = 0.0
    for j, R in enumerate(blocks):
        evals, evecs, rel = eigen_residual(R)
        max_resid = max(max_resid, rel)
        for k in range(1 + d):
            all_evals.append(float(evals[k]))
            all_meta.append((j, evecs[:, k].copy()))

    all_evals_arr = np.asarray(all_evals, dtype=np.float64)
    order = np.argsort(all_evals_arr)
    ell = float(all_evals_arr[order[0]])

    # Select K lowest
    K_use = min(K, len(order))
    sel = order[:K_use]
    lam = all_evals_arr[sel]

    # JE: (n, K) — apply selected block vectors through J_j
    JE = np.zeros((n, K_use), dtype=np.float64)
    for col, idx in enumerate(sel):
        j, vloc = all_meta[idx]
        JE[:, col] = J_blocks[j] @ vloc

    H_E = np.diag(lam) + (JE.T * h) @ JE
    H_E = 0.5 * (H_E + H_E.T)
    u_evals, u_vecs = linalg.eigh(H_E)
    u = float(u_evals[0])
    # Residual of projected eigenpair
    u_recon = H_E @ u_vecs[:, 0] - u_evals[0] * u_vecs[:, 0]
    u_rel = float(
        np.linalg.norm(u_recon) / max(1.0, abs(u_evals[0]))
    )
    max_resid = max(max_resid, u_rel)

    # Simple one-block check for diagnostics
    j_worst = int(all_meta[order[0]][0])
    v_worst = all_meta[order[0]][1]
    Jv = J_blocks[j_worst] @ v_worst
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
    blocks, q, h, phi_p, a = residual_blocks(theta, X, y, m)
    J_blocks = jacobian_blocks(theta, X, m, phi_p=phi_p)

    p = m * (1 + d)
    R_dense = np.zeros((p, p), dtype=np.float64)
    J = np.zeros((n, p), dtype=np.float64)
    for j in range(m):
        sl = slice(j * (1 + d), (j + 1) * (1 + d))
        R_dense[sl, sl] = blocks[j]
        J[:, sl] = J_blocks[j]
    H_decomp = J.T @ (h[:, None] * J) + R_dense
    H_ad = dense_hessian_autodiff_torch(theta, X, y, m)
    max_abs = float(np.max(np.abs(H_decomp - H_ad)))
    br = bracket_lambda_min(theta, X, y, m)
    evals_ad = linalg.eigvalsh(H_ad)
    lam_min_ad = float(evals_ad[0])
    return {
        "max_abs_entry_diff": max_abs,
        "lam_min_ad": lam_min_ad,
        "ell": br["ell"],
        "u": br["u"],
        "bracket_contains": br["ell"] - 1e-8 <= lam_min_ad <= br["u"] + 1e-8,
    }
