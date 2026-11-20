"""Loss-CDF prior/posterior discrepancy.

For this likelihood-tilted Gaussian, ||pi - gamma||_TV = sup_c (F_pi(c) - F_gamma(c)).
A threshold chosen on one fold and scored on the other is a fixed-event lower bound.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _cdf_at(samples: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(samples) <= threshold))


def plug_in_sup(v_post: np.ndarray, v_prior: np.ndarray) -> dict[str, float]:
    """Supremum of the empirical CDF gap. Thresholds are the pooled sample values."""
    post = np.asarray(v_post, dtype=np.float64).ravel()
    prior = np.asarray(v_prior, dtype=np.float64).ravel()
    grid = np.unique(np.concatenate([post, prior]))
    gaps = np.array([_cdf_at(post, c) - _cdf_at(prior, c) for c in grid])
    k = int(np.argmax(gaps))
    return {"sup": float(gaps[k]), "threshold": float(grid[k]), "n_post": int(post.size), "n_prior": int(prior.size)}


def held_out_gap(v_post_fit: np.ndarray, v_prior_fit: np.ndarray, v_post_eval: np.ndarray,
                 v_prior_eval: np.ndarray) -> dict[str, float]:
    chosen = plug_in_sup(v_post_fit, v_prior_fit)
    c = chosen["threshold"]
    gap = _cdf_at(v_post_eval, c) - _cdf_at(v_prior_eval, c)
    return {"threshold": c, "fit_sup": chosen["sup"], "held_out_gap": float(gap)}


def two_fold_discrepancy(v_post: np.ndarray, v_prior: np.ndarray) -> dict[str, Any]:
    """v_post is (chains, draws) or a flat vector split in half. v_prior is split in half."""
    post = np.asarray(v_post, dtype=np.float64)
    prior = np.asarray(v_prior, dtype=np.float64).ravel()
    if post.ndim == 1:
        mid = post.size // 2
        folds_p = [post[:mid], post[mid:]]
    else:
        if post.shape[0] < 2:
            raise ValueError("need at least two posterior chains or a flat vector")
        half = post.shape[0] // 2
        folds_p = [post[:half].ravel(), post[half:].ravel()]
    mid = prior.size // 2
    folds_g = [prior[:mid], prior[mid:]]
    one = held_out_gap(folds_p[0], folds_g[0], folds_p[1], folds_g[1])
    two = held_out_gap(folds_p[1], folds_g[1], folds_p[0], folds_g[0])
    plug = plug_in_sup(post, prior)
    return {"fold_0_on_1": one, "fold_1_on_0": two,
            "held_out_mean": float(0.5 * (one["held_out_gap"] + two["held_out_gap"])),
            "plug_in_sup": plug["sup"], "plug_in_threshold": plug["threshold"]}


def two_point_tv(weights: tuple[float, float] = (1.0, 1.0 / 3.0),
                  prior: tuple[float, float] = (0.5, 0.5)) -> dict[str, float]:
    """Finite toy: prior atoms and likelihood weights. TV is the loss-CDF gap."""
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(prior, dtype=np.float64)
    if np.any(w <= 0) or abs(g.sum() - 1.0) > 1e-15:
        raise ValueError("weights must be positive and the prior must sum to 1")
    mass = g * w
    Z = float(mass.sum())
    pi = mass / Z
    tv = 0.5 * float(np.abs(pi - g).sum())
    # V = -log w, so the likelihood factor is e^{-V}.
    V = -np.log(w)
    # Population CDFs on the ordered atoms, evaluated at each atom's loss.
    order = np.argsort(V)
    gap = 0.0
    best = float(V[order[0]])
    for c in V:
        fp = float(pi[V <= c].sum())
        fg = float(g[V <= c].sum())
        if fp - fg >= gap:
            gap = fp - fg
            best = float(c)
    return {"Z": Z, "tv": tv, "cdf_gap": gap, "threshold": best,
            "posterior": pi.tolist(), "V": V.tolist()}


def simulate_gaussian_losses(n_post: int, n_prior: int, *, shift: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(shift, 1.0, size=(2, n_post // 2)), rng.normal(0.0, 1.0, size=n_prior)


def band_supremum(v_post: np.ndarray, v_prior: np.ndarray, *, reps: int = 200, seed: int = 0,
                  block: Optional[int] = None) -> dict[str, float]:
    """Chain-block resample of posterior losses and iid resample of prior losses.

    The returned interval is the range of suprema of the resampled CDF gaps. It is a
    descriptive uncertainty band for the plug-in supremum, not a formal confidence set
    when the block length is shorter than the dependence.
    """
    post = np.asarray(v_post, dtype=np.float64)
    prior = np.asarray(v_prior, dtype=np.float64).ravel()
    if post.ndim == 1:
        post = post[None, :]
    rng = np.random.default_rng(seed)
    C, N = post.shape
    b = int(block or max(1, N // 20))
    b = min(b, N)
    sups = np.empty(reps)
    for i in range(reps):
        pieces = []
        for c in range(C):
            starts = rng.integers(0, N - b + 1, size=int(np.ceil(N / b)))
            idx = (starts[:, None] + np.arange(b)[None, :]).ravel()[:N]
            pieces.append(post[c, idx])
        pr = prior[rng.integers(0, prior.size, size=prior.size)]
        sups[i] = plug_in_sup(np.concatenate(pieces), pr)["sup"]
    return {"mean": float(sups.mean()), "low": float(np.quantile(sups, 0.025)),
            "high": float(np.quantile(sups, 0.975)), "reps": reps, "block": b}
