"""Physical-time relaxation diagnostics (runbook §10).

τ̂_g = (h/2) · N_total / ESS_mean,g on unranked scalar traces; the held-out family diagnostic selects the largest
τ̂ on one chain fold and evaluates it on the other, averaged over both fold orders.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import numpy as np

from .diagnostics import bootstrap_scalar, ess_mean_raw, is_constant


def tau_ou(h: float, sigma: float) -> float:
    e = math.exp(-h / sigma**2)
    return h / 2.0 * (1.0 + e) / (-math.expm1(-h / sigma**2))


def inefficiency(x: np.ndarray) -> float:
    """Statistical inefficiency ŝ = N_total / ESS_mean in transitions (nan if constant)."""
    ess, _ = ess_mean_raw(x)
    return x.size / ess if np.isfinite(ess) and ess > 0 else float("nan")


def tau_hat(x: np.ndarray, h: float) -> float:
    return 0.5 * h * inefficiency(x)


def family_heldout(traces: dict[str, np.ndarray], names: Sequence[str], h: float,
                   folds: Sequence[Sequence[int]]) -> tuple[float, dict[str, Any]]:
    """traces[name] is (C, N). Returns (held-out estimate, info with selected ids per fold order)."""
    a, b = list(folds[0]), list(folds[1])
    est, sel = [], []
    for s, e in ((a, b), (b, a)):
        cand = {g: tau_hat(traces[g][s], h) for g in names if not is_constant(traces[g])}
        cand = {g: v for g, v in cand.items() if np.isfinite(v)}
        if not cand:
            return float("nan"), {"selected": [None, None]}
        g = max(cand, key=lambda k: (cand[k], k))
        sel.append(g)
        est.append(tau_hat(traces[g][e], h))
    return float(np.mean(est)), {"selected": sel, "fold_estimates": est}


def analyze_family(traces: dict[str, np.ndarray], names: Sequence[str], h: float, folds: Sequence[Sequence[int]],
                   *, reps: int, seed: int, multiplier: float, min_blocks: int, doubled_min_blocks: int,
                   stability: float, percentiles: Sequence[float]) -> dict[str, Any]:
    names = list(names)
    per_chain = [np.stack([traces[g][c] for g in names], axis=1) for c in range(traces[names[0]].shape[0])]

    def stat(chs: list[np.ndarray]):
        tr = {g: np.stack([ch[:, k] for ch in chs]) for k, g in enumerate(names)}
        return family_heldout(tr, names, h, folds)

    ineff = [inefficiency(traces[g]) for g in names]
    iat = float(np.nanmax(ineff)) if np.any(np.isfinite(ineff)) else float("nan")
    br = bootstrap_scalar(stat, per_chain, iat, reps=reps, seed=seed, multiplier=multiplier, min_blocks=min_blocks,
                          doubled_min_blocks=doubled_min_blocks, stability=stability, percentiles=percentiles)
    freq = Counter()
    for inf in br.replicate_info or []:
        if inf:
            for g in inf["selected"]:
                freq[g] += 1
    tot = sum(freq.values()) or 1
    return {"estimate": br.estimate, "mcse": br.mcse, "mc_low": br.low, "mc_high": br.high,
            "block_length": br.block, "mcse_doubled_block": br.mcse_doubled, "block_stable": br.stable,
            "enough_blocks": br.enough_blocks, "boot_reason": br.reason,
            "selected_fold_A_to_B": (br.info or {}).get("selected", [None, None])[0],
            "selected_fold_B_to_A": (br.info or {}).get("selected", [None, None])[1],
            "selection_frequencies": {g: freq[g] / tot for g in sorted(freq)},
            "max_tau_full_pool": 0.5 * h * iat if np.isfinite(iat) else float("nan"),
            "replicates": br.replicates}
