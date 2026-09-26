"""Nested paired prior centers (design §3.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import sha256_array
from .model import pack_params, forward_f


def generate_center_bank(
    n_pairs: int = 2048,
    d: int = 32,
    seed: int = 0,
    b0: float = 1.0,
) -> dict[str, Any]:
    """
    Generate n_pairs independent hidden-row centers u_j ~ N(0, I_d).
    Width m uses the first m/2 pairs with head centers ±b0.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    U = rng.standard_normal(size=(n_pairs, d)).astype(np.float64)
    return {
        "U": U,
        "seed": seed,
        "n_pairs": n_pairs,
        "d": d,
        "b0": float(b0),
        "hash_U": sha256_array(U),
    }


def assemble_theta0(
    U_bank: np.ndarray,
    m: int,
    b0: float = 1.0,
) -> np.ndarray:
    """
    Assemble θ0 for width m from the first m/2 pairs:

      (a_{j,0}, w_{j,0}) = (+b0, u_j)
      (a_{j+m/2,0}, w_{j+m/2,0}) = (-b0, u_j)

    for j = 0..m/2-1. Returns flat parameter vector of length m*(1+d).
    """
    if m % 2 != 0:
        raise ValueError("Width m must be even for paired centers")
    n_pairs = m // 2
    if U_bank.shape[0] < n_pairs:
        raise ValueError(
            f"Center bank has {U_bank.shape[0]} pairs; need {n_pairs} for m={m}"
        )
    d = U_bank.shape[1]
    a = np.empty(m, dtype=np.float64)
    W = np.empty((m, d), dtype=np.float64)
    a[:n_pairs] = b0
    a[n_pairs:] = -b0
    W[:n_pairs] = U_bank[:n_pairs]
    W[n_pairs:] = U_bank[:n_pairs]
    return pack_params(a, W)


def max_head_center_norm(a0: np.ndarray) -> float:
    return float(np.max(np.abs(a0)))


def verify_paired_center(
    theta0: np.ndarray,
    X: np.ndarray,
    X_probe: np.ndarray | None = None,
    b0: float = 1.0,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Check f_θ0 = 0 on train/probe and b0 magnitude."""
    m = theta0.size // (1 + X.shape[1])
    f_train = forward_f(theta0, X, m)
    checks = {
        "f_train_max_abs": float(np.max(np.abs(f_train))),
        "b0_empirical": max_head_center_norm(theta0[0 :: 1 + X.shape[1]]),
        "ok_train": bool(np.max(np.abs(f_train)) <= atol),
    }
    if X_probe is not None:
        f_probe = forward_f(theta0, X_probe, m)
        checks["f_probe_max_abs"] = float(np.max(np.abs(f_probe)))
        checks["ok_probe"] = bool(np.max(np.abs(f_probe)) <= atol)
    checks["ok_b0"] = bool(abs(checks["b0_empirical"] - b0) <= atol)
    checks["ok"] = checks["ok_train"] and checks["ok_b0"] and checks.get("ok_probe", True)
    return checks


def save_centers(path: str, bank: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        U=bank["U"],
        seed=bank["seed"],
        n_pairs=bank["n_pairs"],
        d=bank["d"],
        b0=bank["b0"],
        hash_U=bank["hash_U"],
    )


def load_centers(path: str) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for key in list(out):
        if isinstance(out[key], np.ndarray) and out[key].shape == ():
            out[key] = out[key].item()
    return out
