"""ESS per transition, break-even cost ratios, and the step-selection rule.

Bulk ESS is the rank-normalized loss ESS already stored in the archive. Raw-mean ESS is a
different estimand and is computed from traces by the existing multichain estimator.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from ..diagnostics import ess_bulk, ess_mean_raw, ess_tail, is_constant, rhat_rank, scalar_diagnostics
from ..paper_cleanup.supplement_metrics import acf_1d
from ..relaxation import inefficiency


TRANSITIONS_S1 = 4 * 102_400


def ess_per_transition(ess: float, transitions: int) -> float:
    if not np.isfinite(ess) or transitions <= 0:
        return float("nan")
    return float(ess) / float(transitions)


def break_even_ratio(wide_ess: float, narrow_ess: float) -> float:
    """Wide ESS/transition over narrow ESS/transition. The transition count cancels when it matches."""
    if not (np.isfinite(wide_ess) and np.isfinite(narrow_ess)) or narrow_ess == 0.0:
        return float("nan")
    return float(wide_ess) / float(narrow_ess)


def wins_on_time(cost_ratio: float, ess_ratio: float) -> Optional[bool]:
    """The wider endpoint wins on ESS/second when its cost ratio is strictly below its ESS ratio."""
    if not (np.isfinite(cost_ratio) and np.isfinite(ess_ratio)):
        return None
    return bool(cost_ratio < ess_ratio)


def timed_width_comparison(narrow_blocks: Sequence[tuple[float, int]], wide_blocks: Sequence[tuple[float, int]],
                           ess_ratio: float) -> dict[str, Any]:
    """Compare two widths from their own synchronized blocks.

    Each block is ``(elapsed_seconds, transitions)``. Seconds per transition is the total
    elapsed time divided by the total transition count, so the two widths may use different
    block lengths. The reported ratio is

        (wide training-loss bulk ESS/s) / (narrow training-loss bulk ESS/s)
        = ess_ratio / (wide seconds per transition / narrow seconds per transition).

    A value above one favors the wide model. The comparison is inconclusive when the range
    of block rates contains the break-even cost ratio.
    """
    def rates(blocks: Sequence[tuple[float, int]]) -> tuple[np.ndarray, float, float, int]:
        if len(blocks) == 0:
            raise ValueError("a width needs at least one timed block")
        seconds = np.asarray([float(b[0]) for b in blocks], dtype=np.float64)
        counts = np.asarray([int(b[1]) for b in blocks], dtype=np.float64)
        if np.any(counts <= 0) or np.any(seconds < 0) or not np.isfinite(seconds).all():
            raise ValueError("block times and transition counts must be finite and positive")
        per = seconds / counts
        return per, float(seconds.sum()), int(counts.sum()), float(seconds.sum() / counts.sum())

    narrow_rates, narrow_seconds, narrow_count, narrow_spt = rates(narrow_blocks)
    wide_rates, wide_seconds, wide_count, wide_spt = rates(wide_blocks)
    cost = wide_spt / narrow_spt
    if not np.isfinite(ess_ratio) or ess_ratio <= 0:
        raise ValueError("ess_ratio must be a positive finite break-even threshold")
    cost_low = float(wide_rates.min() / narrow_rates.max())
    cost_high = float(wide_rates.max() / narrow_rates.min())
    spans = bool(cost_low <= ess_ratio <= cost_high)
    ratio = float(ess_ratio / cost)
    if spans or not np.isfinite(ratio):
        decision = "inconclusive"
        favors_wide = None
    elif ratio > 1.0:
        decision, favors_wide = "wide", True
    else:
        decision, favors_wide = "narrow", False
    return {
        "narrow_elapsed_seconds": narrow_seconds,
        "narrow_transitions": narrow_count,
        "wide_elapsed_seconds": wide_seconds,
        "wide_transitions": wide_count,
        "seconds_per_transition_narrow": narrow_spt,
        "seconds_per_transition_wide": wide_spt,
        "cost_ratio_wide_over_narrow": float(cost),
        "cost_ratio_low": cost_low,
        "cost_ratio_high": cost_high,
        "ess_ratio": float(ess_ratio),
        "ess_per_second_ratio": ratio,
        "ess_per_second_ratio_low": float(ess_ratio / cost_high),
        "ess_per_second_ratio_high": float(ess_ratio / cost_low),
        "spans_break_even": spans,
        "decision": decision,
        "favors_wide": favors_wide,
    }


def chain_centered_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """rho(k) = sum_{t<N-k} (g_t-bar)(g_{t+k}-bar) / sum_{t<N} (g_t-bar)^2."""
    return acf_1d(np.asarray(values, dtype=np.float64), max_lag)


def explicit_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Direct sum, used to check the FFT implementation on short series."""
    x = np.asarray(values, dtype=np.float64)
    z = x - x.mean()
    denom = float(np.dot(z, z))
    if denom <= 0.0:
        raise ValueError("constant series")
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = float(np.dot(z[: x.size - k], z[k:])) / denom
    return out


def scalar_rate(chains: np.ndarray, *, seconds_per_transition: Optional[float] = None) -> dict[str, Any]:
    """chains is (C, N). Constant series are undefined rather than given an ESS of zero."""
    x = np.asarray(chains, dtype=np.float64)
    if is_constant(x) or not np.isfinite(x).all():
        return {"status": "undefined" if is_constant(x) else "nonfinite", "ess_mean": None, "ess_bulk": None,
                "ess_tail": None, "tau": None, "ess_per_transition": None, "ess_per_second": None,
                "seconds_per_effective_draw": None}
    ess, lag = ess_mean_raw(x)
    tau = float(x.size / ess) if ess > 0 else float("nan")
    per_t = ess_per_transition(ess, x.size)
    per_s = None
    sec_draw = None
    if seconds_per_transition is not None and np.isfinite(seconds_per_transition) and per_t > 0:
        per_s = per_t / seconds_per_transition
        sec_draw = tau * seconds_per_transition
    diag = scalar_diagnostics("g", x)
    return {"status": "ok", "ess_mean": ess, "truncation_lag": lag, "ess_bulk": ess_bulk(x),
            "ess_tail": ess_tail(x), "tau": tau, "rhat": rhat_rank(x), "mean": diag.mean, "sd": diag.sd,
            "mcse": diag.mcse, "ess_per_transition": per_t, "ess_per_second": per_s,
            "seconds_per_effective_draw": sec_draw, "n_total": int(x.size),
            "estimator": "geyer_initial_positive_monotone"}


def warmup_steps(h: float, sigma: float, *, time_over_sigma2: float = 20.0) -> int:
    """Discard ceil(time_over_sigma2 * sigma^2 / h) transitions before scoring a pilot."""
    if h <= 0 or sigma <= 0:
        raise ValueError("h and sigma must be positive")
    return int(math.ceil(time_over_sigma2 * sigma ** 2 / h))


def pilot_score(loss_ess_per_s: float, prediction_ess_per_s: Sequence[float]) -> float:
    """Smaller of loss raw-mean ESS/s and the median of the prediction raw-mean ESS/s values."""
    preds = [float(v) for v in prediction_ess_per_s if np.isfinite(v)]
    if not preds or not np.isfinite(loss_ess_per_s):
        return float("nan")
    return float(min(loss_ess_per_s, float(np.median(preds))))


def select_step(candidates: Sequence[tuple[float, float]], *, tie: float = 0.05) -> dict[str, Any]:
    """Pick one step. Higher score is better. A tie within `tie` of the best score favors the smaller step.

    If the winner is an endpoint of the supplied grid, ``boundary`` is true and ``suggest_h`` is one
    further halving or doubling. The caller runs that extra point at most once.
    """
    rows = [(float(h), float(s)) for h, s in candidates]
    eligible = [(h, s) for h, s in rows if np.isfinite(s)]
    failures = [h for h, s in rows if not np.isfinite(s)]
    if not eligible:
        return {"winner": None, "status": "no_eligible_step", "failures": failures, "grid": rows}
    best = max(s for _, s in eligible)
    near = [(h, s) for h, s in eligible if s >= best * (1.0 - tie)]
    winner = min(near, key=lambda hs: hs[0])
    ordered = sorted(h for h, _ in rows)
    boundary = winner[0] in (ordered[0], ordered[-1])
    suggest = None
    if boundary and winner[0] == ordered[0]:
        suggest = winner[0] / 2.0
    elif boundary and winner[0] == ordered[-1]:
        suggest = winner[0] * 2.0
    return {"winner": winner[0], "score": winner[1], "status": "boundary" if boundary else "interior",
            "boundary": boundary, "suggest_h": suggest, "tied": [h for h, _ in near],
            "failures": failures, "grid": rows, "tie_rule": tie}


def median_across_replicates(scores_by_rep: Sequence[float]) -> float:
    vals = [float(v) for v in scores_by_rep if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def ar1_tau(phi: float) -> float:
    return (1.0 + phi) / (1.0 - phi)


def iid_or_ar_check(phi: float, n: int = 20000, chains: int = 4, seed: int = 0) -> dict[str, float]:
    """A unit-level check that raw-mean inefficiency tracks the AR(1) constant, and that iid is near 1."""
    rng = np.random.default_rng(seed)
    x = np.empty((chains, n))
    x[:, 0] = rng.standard_normal(chains)
    if phi == 0.0:
        x[:, 1:] = rng.standard_normal((chains, n - 1))
    else:
        noise = rng.standard_normal((chains, n - 1)) * math.sqrt(1.0 - phi ** 2)
        for t in range(1, n):
            x[:, t] = phi * x[:, t - 1] + noise[:, t - 1]
    return {"phi": phi, "tau_hat": float(inefficiency(x)), "tau_exact": 1.0 if phi == 0.0 else ar1_tau(phi)}
