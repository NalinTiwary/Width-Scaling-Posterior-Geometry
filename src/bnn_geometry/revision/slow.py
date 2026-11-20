"""Search for a slow linear combination of a fixed probe bank.

Rules fixed before looking at outcomes: drop exact duplicate columns, drop covariance
eigenvalues below 1e-6 of the largest, whiten on a calibration sample, maximize the
symmetrized lag-k covariance on one chain fold, and score the frozen direction on the other.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.linalg import eigh

from .efficiency import chain_centered_acf, scalar_rate

RIDGE_REL = 1e-6
LAG = 50


def _unique_columns(X: np.ndarray, decimals: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rounded = np.round(X, decimals)
    _, idx = np.unique(rounded.T, axis=0, return_index=True)
    idx = np.sort(idx)
    return X[:, idx], idx


def whiten(calibration: np.ndarray, *, rel: float = RIDGE_REL) -> dict[str, Any]:
    """calibration is (N, P). Returns the retained basis and the map z = (x - mean) @ basis."""
    X = np.asarray(calibration, dtype=np.float64)
    X, kept = _unique_columns(X)
    mu = X.mean(axis=0)
    xc = X - mu
    cov = (xc.T @ xc) / max(X.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    thresh = rel * float(evals[0]) if evals.size and evals[0] > 0 else 0.0
    mask = evals >= thresh
    if not np.any(mask):
        raise ValueError("calibration covariance has no eigenvalue above the relative floor")
    ev, U = evals[mask], evecs[:, mask]
    basis = U / np.sqrt(ev)
    return {"mean": mu, "basis": basis, "kept_input_columns": kept, "eigenvalues": ev, "threshold": thresh}


def apply_whiten(X: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    cols = spec["kept_input_columns"]
    return (np.asarray(X, dtype=np.float64)[:, cols] - spec["mean"]) @ spec["basis"]


def symmetrized_lag_cov(chains: Sequence[np.ndarray], lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pooled, chain-centered lag covariance and the matching zero-lag covariance. chains are (N, P)."""
    num_k = None
    num_0 = None
    den_k = 0
    den_0 = 0
    for Z in chains:
        Z = np.asarray(Z, dtype=np.float64)
        if Z.shape[0] <= lag + 1:
            raise ValueError("chain shorter than the lag")
        X = Z - Z.mean(axis=0)
        gk = X[:-lag].T @ X[lag:]
        g0 = X.T @ X
        num_k = gk if num_k is None else num_k + gk
        num_0 = g0 if num_0 is None else num_0 + g0
        den_k += Z.shape[0] - lag
        den_0 += Z.shape[0]
    lag_c = num_k / den_k
    zero = num_0 / den_0
    return 0.5 * (lag_c + lag_c.T), 0.5 * (zero + zero.T)


def top_direction(lag_cov: np.ndarray, zero_cov: np.ndarray) -> np.ndarray:
    evals, evecs = eigh(lag_cov, zero_cov, check_finite=True)
    w = np.real(evecs[:, int(np.argmax(evals))])
    pivot = int(np.argmax(np.abs(w)))
    if w[pivot] < 0:
        w = -w
    norm = np.linalg.norm(w)
    return w / norm


def combination_series(chains: Sequence[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Return (C, N) scores."""
    return np.stack([(np.asarray(Z) @ w) for Z in chains], axis=0)


def search_slow(calibration: np.ndarray, fit_chains: Sequence[np.ndarray], eval_chains: Sequence[np.ndarray],
                *, lag: int = LAG, names: Sequence[str] | None = None) -> dict[str, Any]:
    spec = whiten(calibration)
    fit_z = [apply_whiten(Z, spec) for Z in fit_chains]
    eval_z = [apply_whiten(Z, spec) for Z in eval_chains]
    lag_c, zero = symmetrized_lag_cov(fit_z, lag)
    w = top_direction(lag_c, zero)
    series = combination_series(eval_z, w)
    rate = scalar_rate(series)
    acf = np.mean([chain_centered_acf(row, lag) for row in series], axis=0)
    # Also report the worst single original probe on the evaluation fold, before whitening.
    worst = None
    if names is not None:
        scores = []
        for j, name in enumerate(names):
            cols = np.stack([np.asarray(Z)[:, j] for Z in eval_chains], axis=0)
            if np.std(cols) == 0:
                continue
            ac = np.mean([chain_centered_acf(row, lag) for row in cols], axis=0)
            scores.append((float(ac[lag]), name))
        if scores:
            worst = max(scores, key=lambda t: t[0])
    return {"weights_whitened": w, "retained_rank": int(w.size), "lag": lag,
            "eval_acf_at_lag": float(acf[lag]), "eval_rate": rate,
            "worst_probe": None if worst is None else {"name": worst[1], "acf_at_lag": worst[0]},
            "whitening_eigenvalues": spec["eigenvalues"]}


def both_folds(calibration: np.ndarray, chains: Sequence[np.ndarray], *, lag: int = LAG,
               names: Sequence[str] | None = None) -> dict[str, Any]:
    """chains has four entries. Fit on 0-1 and evaluate on 2-3, then swap."""
    chains = list(chains)
    if len(chains) != 4:
        raise ValueError("the prespecified fold swap uses four chains")
    a = search_slow(calibration, chains[:2], chains[2:], lag=lag, names=names)
    b = search_slow(calibration, chains[2:], chains[:2], lag=lag, names=names)
    return {"fit_01_eval_23": a, "fit_23_eval_01": b,
            "held_out_acf": [a["eval_acf_at_lag"], b["eval_acf_at_lag"]]}
