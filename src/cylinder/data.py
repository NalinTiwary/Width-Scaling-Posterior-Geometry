"""Deterministic orthogonal training data and fixed predictive probes."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _qr_positive_diagonal(G: np.ndarray) -> np.ndarray:
    """Economy QR with column signs fixed so diag(R) > 0. Returns Q."""
    Q, R = np.linalg.qr(G, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return Q


def generate_training_data(
    n: int = 32,
    d: int = 32,
    seed: int = 2027,
) -> dict[str, Any]:
    """
    Design §3.2:
      1. G ~ N(0,1)^{n x d} with PCG64(seed)
      2. G = QR with positive R diagonal; X = Q^T  (so rows of X are orthonormal)
      3. teacher v_* ~ N(0, I_d)
      4. y_i = 1 for the n/2 largest x_i^T v_*, else 0
    """
    if n != d:
        raise ValueError("This protocol freezes n = d for orthogonal rows")
    rng = np.random.Generator(np.random.PCG64(seed))
    G = rng.standard_normal(size=(n, d))
    Q = _qr_positive_diagonal(G)
    X = Q.T.astype(np.float64)  # shape (n, d), orthonormal rows
    v_star = rng.standard_normal(size=(d,)).astype(np.float64)
    scores = X @ v_star
    order = np.argsort(scores)
    y = np.zeros(n, dtype=np.float64)
    y[order[n // 2 :]] = 1.0

    # Sanity: unit rows and M2 = M4 = 1 under orthogonality.
    row_norms = np.linalg.norm(X, axis=1)
    if not np.allclose(row_norms, 1.0, atol=1e-10):
        raise RuntimeError("Training rows are not unit norm")

    return {
        "X": X,
        "y": y,
        "v_star": v_star,
        "scores": scores,
        "seed": seed,
        "n": n,
        "d": d,
        "numpy_bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "hash_X": sha256_array(X),
        "hash_y": sha256_array(y),
        "hash_v_star": sha256_array(v_star),
    }


def generate_probes(
    d: int = 32,
    n_probes: int = 8,
    seed: int = 2028,
) -> dict[str, Any]:
    """Eight fixed unit-norm Gaussian inputs (seed 2028)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    raw = rng.standard_normal(size=(n_probes, d))
    X_probe = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float64)
    return {
        "X_probe": X_probe,
        "seed": seed,
        "hash_X_probe": sha256_array(X_probe),
    }


def generate_projections(
    p: int,
    n_proj: int = 8,
    seed: int = 2029,
) -> dict[str, Any]:
    """
    Fixed unit-vector projections of z for each parameter dimension p.
    Generate before sampling; do not select based on ESS.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    raw = rng.standard_normal(size=(n_proj, p))
    U = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float64)
    return {
        "U": U,
        "seed": seed,
        "p": p,
        "hash_U": sha256_array(U),
    }


def save_data_bundle(
    path: str,
    train: dict[str, Any],
    probes: dict[str, Any],
) -> None:
    np.savez_compressed(
        path,
        X=train["X"],
        y=train["y"],
        v_star=train["v_star"],
        X_probe=probes["X_probe"],
        seed=train["seed"],
        probe_seed=probes["seed"],
        hash_X=train["hash_X"],
        hash_y=train["hash_y"],
        hash_v_star=train["hash_v_star"],
        hash_X_probe=probes["hash_X_probe"],
        numpy_version=train["numpy_version"],
        numpy_bit_generator=train["numpy_bit_generator"],
    )


def load_data_bundle(path: str) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    # Decode scalar string hashes stored as 0-d arrays.
    for key in list(out):
        if isinstance(out[key], np.ndarray) and out[key].shape == ():
            out[key] = out[key].item()
    return out
