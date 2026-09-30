"""Spectral norm of the later hidden matrix, event classification with a guard band (runbook §2.5, §12)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.linalg
import torch

from .model import Layout, unpack

INSIDE, OUTSIDE, UNRESOLVED = 1, 0, -1


def normalized_norm(W2: torch.Tensor, a: float) -> torch.Tensor:
    """S = σ_max(W2)/(a√m) for W2 (B, m, m) via full float64 singular values (never eigenvalues)."""
    m = W2.shape[-1]
    sv = torch.linalg.svdvals(W2.to(torch.float64))
    return sv[..., 0] / (a * math.sqrt(m))


def state_S(theta: torch.Tensor, lay: Layout, a: float) -> torch.Tensor:
    return normalized_norm(unpack(theta, lay)["W2"], a)


def independent_S(W2: np.ndarray, a: float) -> tuple[float, dict[str, float]]:
    """CPU LAPACK gesvd with reconstruction/orthogonality residuals (independent of the torch backend)."""
    W2 = np.asarray(W2, dtype=np.float64)
    U, s, Vt = scipy.linalg.svd(W2, lapack_driver="gesvd")
    m = W2.shape[-1]
    recon = float(np.linalg.norm(U @ np.diag(s) @ Vt - W2) / max(np.linalg.norm(W2), 1e-300))
    orth = float(max(np.linalg.norm(U.T @ U - np.eye(m)), np.linalg.norm(Vt @ Vt.T - np.eye(m))))
    return float(s[0] / (a * math.sqrt(m))), {"reconstruction_residual": recon, "orthogonality_residual": orth}


def classify(S: np.ndarray, W2: np.ndarray, a: float, guard: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Membership per state: 1 inside, 0 outside, -1 numerically unresolved. W2 (B, m, m) numpy."""
    S = np.asarray(S, dtype=np.float64)
    mem = np.where(S < 1.0 - guard, INSIDE, np.where(S > 1.0 + guard, OUTSIDE, UNRESOLVED))
    notes = []
    for i in np.flatnonzero(mem == UNRESOLVED):
        s2, res = independent_S(W2[i], a)
        m2 = INSIDE if s2 < 1.0 - guard else OUTSIDE if s2 > 1.0 + guard else UNRESOLVED
        mem[i] = m2
        notes.append({"index": int(i), "S_primary": float(S[i]), "S_independent": s2, **res, "membership": int(m2)})
    return mem, notes


def backend_check(W2: np.ndarray, S: np.ndarray, a: float, idx: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for i in idx:
        s2, res = independent_S(W2[i], a)
        out.append({"index": int(i), "S_primary": float(S[i]), "S_independent": s2,
                    "rel_diff": abs(S[i] - s2) / max(abs(s2), 1e-300), **res})
    return out


def check_indices(n: int, k: int) -> np.ndarray:
    """k fixed, evenly spaced state indices out of n (deterministic)."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))
