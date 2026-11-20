"""Uncentered input Gram spectrum. chi = lambda_max(X^T X) = ||X||_op^2."""

from __future__ import annotations

import numpy as np


def gram_spectrum(X: np.ndarray, *, atol: float = 1e-8) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be (n, d)")
    singular = np.linalg.svd(X, compute_uv=False)
    chi_svd = float(singular[0] ** 2)
    ev = np.linalg.eigvalsh(X.T @ X)
    chi_gram = float(ev[-1])
    scale = max(1.0, chi_svd)
    if abs(chi_svd - chi_gram) > atol * scale:
        raise ValueError(f"SVD and Gram eigenvalues disagree: {chi_svd} vs {chi_gram}")
    norms = np.linalg.norm(X, axis=1)
    return {
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "chi": chi_svd,
        "chi_gram": chi_gram,
        "op_norm": float(singular[0]),
        "row_norm_min": float(norms.min()),
        "row_norm_max": float(norms.max()),
        "leading_eigenvalues": [float(v) for v in singular[: min(8, singular.size)] ** 2],
    }
