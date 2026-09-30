"""Rank diagnostics, raw-mean ESS, batch-means MCSE and the within-chain moving-block bootstrap (runbook §7.2, §9).

The rank-normalized split R-hat and bulk/tail ESS follow Vehtari et al. (2021) with the ArviZ conventions
(z-scale with (r-3/8)/(S+1/4), split chains, folded R-hat, Geyer initial positive + monotone sequence); they are
implemented here so that the laptop (ArviZ 1.x) and cluster (ArviZ 0.17) environments give identical numbers.
``tests/unit/test_diagnostics.py`` checks agreement with the installed ArviZ.

``ess_mean_raw`` is the chain-aware ESS of the *unranked, unsplit* chains used for physical relaxation times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.stats import norm, rankdata

QUANTILE_METHOD = "linear"


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance along the last axis for each row of x (C, N)."""
    n = x.shape[-1]
    xc = x - x.mean(axis=-1, keepdims=True)
    m = next_fast_len(2 * n)
    f = rfft(xc, n=m, axis=-1)
    return irfft(f * np.conj(f), n=m, axis=-1)[..., :n] / n


def geyer_ess(x: np.ndarray) -> tuple[float, int]:
    """Multichain ESS (ArviZ ``_ess``) of x (C, N); returns (ESS, truncation lag)."""
    x = np.asarray(x, dtype=np.float64)
    C, n = x.shape
    if n < 4 or not np.all(np.isfinite(x)):
        return float("nan"), 0
    acov = _autocov(x)
    chain_mean = x.mean(axis=1)
    mean_var = acov[:, 0].mean() * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n
    if C > 1:
        var_plus += np.var(chain_mean, ddof=1)
    if not var_plus > 0:
        return float("nan"), 0
    rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # pairs P_k = rho[2k] + rho[2k+1], computed while 2k+1 <= n-2 (ArviZ loop bound t < n-3)
    kmax = (n - 2) // 2
    P = rho[0:2 * kmax + 1:2][:kmax + 1] + rho[1:2 * kmax + 2:2][:kmax + 1]
    neg = np.flatnonzero(P[1:] <= 0.0)
    K = int(neg[0]) + 1 if neg.size else int(len(P) - 1)
    Pm = np.minimum.accumulate(P[:K])
    extra = rho[2 * K] if 2 * K < n and rho[2 * K] > 0 else 0.0
    tau = -1.0 + 2.0 * Pm.sum() + extra
    tau = max(tau, 1.0 / math.log10(C * n))
    return float(C * n / tau), int(2 * K)


def _split(x: np.ndarray) -> np.ndarray:
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]], axis=0)


def _z_scale(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, method="average").reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size + 0.25))


def _rhat(x: np.ndarray) -> float:
    n = x.shape[1]
    between = n * np.var(x.mean(axis=1), ddof=1)
    within = np.mean(np.var(x, axis=1, ddof=1))
    return float(np.sqrt((between / within + n - 1) / n))


def is_constant(x: np.ndarray) -> bool:
    return bool(np.all(x == x.flat[0]))


def rhat_rank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if is_constant(x):
        return float("nan")
    s = _split(x)
    bulk = _rhat(_z_scale(s))
    fold = _rhat(_z_scale(_split(np.abs(x - np.median(x)))))
    return max(bulk, fold)


def ess_bulk(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if is_constant(x):
        return float("nan")
    return geyer_ess(_z_scale(_split(x)))[0]


def ess_tail(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if is_constant(x):
        return float("nan")
    q05, q95 = np.quantile(x, [0.05, 0.95], method=QUANTILE_METHOD)
    e = []
    for ind in (x <= q05, x <= q95):
        ind = ind.astype(np.float64)
        e.append(geyer_ess(_split(ind))[0] if not is_constant(ind) else float("nan"))
    return float(np.nanmin(e)) if np.any(np.isfinite(e)) else float("nan")


def ess_mean_raw(x: np.ndarray) -> tuple[float, int]:
    """Chain-aware raw-value ESS for the mean (no ranking, no splitting)."""
    x = np.asarray(x, dtype=np.float64)
    if is_constant(x):
        return float("nan"), 0
    return geyer_ess(x)


def batch_means_mcse(x: np.ndarray) -> float:
    """Chain-aware batch-means MCSE of the pooled mean of x (C, N); batches never cross chains."""
    x = np.asarray(x, dtype=np.float64)
    C, n = x.shape
    b = max(1, int(math.floor(math.sqrt(n))))
    a = n // b
    if a < 2:
        return float("nan")
    bm = x[:, :a * b].reshape(C, a, b).mean(axis=2)
    var = np.var(bm, ddof=1) * b
    return float(math.sqrt(var / (C * n)))


@dataclass
class ScalarDiag:
    name: str
    status: str                 # ok | constant | nonfinite
    rhat: float
    ess_bulk: float
    ess_tail: float
    ess_mean: float
    lag_window: int
    mean: float
    sd: float
    mcse: float
    drift_first: float
    drift_second: float
    drift_mcse: float
    drift_pass: bool

    def row(self) -> dict[str, Any]:
        return dict(self.__dict__)


def scalar_diagnostics(name: str, x: np.ndarray, *, mcse_mult: float = 3.0, sd_allow: float = 0.05) -> ScalarDiag:
    x = np.asarray(x, dtype=np.float64)
    nan = float("nan")
    if not np.all(np.isfinite(x)):
        return ScalarDiag(name, "nonfinite", nan, nan, nan, nan, 0, nan, nan, nan, nan, nan, nan, False)
    if is_constant(x):
        v = float(x.flat[0])
        return ScalarDiag(name, "constant", nan, nan, nan, nan, 0, v, 0.0, nan, v, v, nan, True)
    n = x.shape[1]
    h = n // 2
    a, b = x[:, :h], x[:, n - h:]
    ma, mb = float(a.mean()), float(b.mean())
    dm = math.hypot(batch_means_mcse(a), batch_means_mcse(b))
    sd = float(x.std(ddof=1))
    em, lag = ess_mean_raw(x)
    drift_pass = bool(abs(ma - mb) <= mcse_mult * dm + sd_allow * sd) if np.isfinite(dm) else False
    return ScalarDiag(name, "ok", rhat_rank(x), ess_bulk(x), ess_tail(x), em, lag, float(x.mean()), sd,
                      batch_means_mcse(x), ma, mb, dm, drift_pass)


def rank_gate(d: ScalarDiag, *, rhat_max: float, bulk_min: float, tail_min: float) -> tuple[bool, str]:
    if d.status == "constant":
        return True, "constant"
    if d.status != "ok":
        return False, d.status
    reasons = []
    if not d.rhat < rhat_max:
        reasons.append(f"rhat={d.rhat:.4f}")
    if not d.ess_bulk >= bulk_min:
        reasons.append(f"ess_bulk={d.ess_bulk:.0f}")
    if not d.ess_tail >= tail_min:
        reasons.append(f"ess_tail={d.ess_tail:.0f}")
    return (not reasons), ";".join(reasons)


# ---- moving-block bootstrap ------------------------------------------------------------------------
def next_pow2(x: float) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(1.0, x)))))


def block_length(iat_steps: float, multiplier: float = 10.0) -> int:
    return next_pow2(multiplier * max(1.0, iat_steps))


def block_indices(n: int, b: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """(reps, n) indices: uniform block starts in [0, n-b], contiguous blocks, truncated to n."""
    nb = -(-n // b)
    starts = rng.integers(0, n - b + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(reps, nb * b)
    return idx[:, :n]


def bootstrap(stat: Callable[[list[np.ndarray]], Any], chains: Sequence[np.ndarray], b: int, reps: int,
              rng: np.random.Generator) -> list[Any]:
    """Recompute ``stat`` on ``reps`` within-chain block resamples of ``chains`` (each (N_c, ...) array)."""
    idx = [block_indices(c.shape[0], b, reps, rng) for c in chains]
    return [stat([c[ix[r]] for c, ix in zip(chains, idx)]) for r in range(reps)]


@dataclass
class BootResult:
    estimate: float
    mcse: float
    low: float
    high: float
    block: int
    mcse_doubled: float
    stable: Optional[bool]
    enough_blocks: bool
    replicates: np.ndarray
    reason: str = ""
    info: Any = None
    replicate_info: Optional[list[Any]] = None

    def row(self, prefix: str = "") -> dict[str, Any]:
        return {f"{prefix}estimate": self.estimate, f"{prefix}mcse": self.mcse, f"{prefix}mc_low": self.low,
                f"{prefix}mc_high": self.high, f"{prefix}block_length": self.block,
                f"{prefix}mcse_doubled_block": self.mcse_doubled, f"{prefix}block_stable": self.stable}


def _split_info(v: Any) -> tuple[float, Any]:
    if isinstance(v, tuple):
        return float(v[0]), v[1]
    return float(v), None


def bootstrap_scalar(stat: Callable[[list[np.ndarray]], Any], chains: Sequence[np.ndarray], iat_steps: float,
                     *, reps: int, seed: int, multiplier: float = 10.0, min_blocks: int = 20,
                     doubled_min_blocks: int = 10, stability: float = 0.30,
                     percentiles: Sequence[float] = (2.5, 97.5)) -> BootResult:
    """Bootstrap MCSE/interval of a scalar statistic with the block-length rule and doubled-block check.

    ``stat`` returns a float, or ``(float, info)`` where info (e.g. selected probe ids) is collected per replicate.
    """
    est, info = _split_info(stat(list(chains)))
    n_min = min(c.shape[0] for c in chains)
    b = block_length(iat_steps, multiplier)
    enough = n_min // b >= min_blocks
    nan = float("nan")
    if not np.isfinite(iat_steps) or b > n_min:
        return BootResult(est, nan, nan, nan, b, nan, None, False, np.zeros(0), "block_exceeds_chain", info, [])
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))
    out = [_split_info(v) for v in bootstrap(stat, chains, b, reps, rng)]
    reps1 = np.array([v for v, _ in out], dtype=np.float64)
    rinfo = [i for _, i in out]
    ok = np.isfinite(reps1)
    mcse = float(np.std(reps1[ok], ddof=1)) if ok.sum() > 1 else nan
    lo, hi = (np.percentile(reps1[ok], percentiles, method=QUANTILE_METHOD) if ok.any() else (nan, nan))
    stable, m2 = None, nan
    if n_min // (2 * b) >= doubled_min_blocks:
        rng2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 2])))
        reps2 = np.array([_split_info(v)[0] for v in bootstrap(stat, chains, 2 * b, reps, rng2)],
                         dtype=np.float64)
        ok2 = np.isfinite(reps2)
        m2 = float(np.std(reps2[ok2], ddof=1)) if ok2.sum() > 1 else nan
        stable = bool(np.isfinite(mcse) and np.isfinite(m2) and mcse > 0 and abs(m2 - mcse) / mcse <= stability) \
            or bool(mcse == 0 and m2 == 0)
    reason = "" if enough else "fewer_than_min_blocks"
    if int((~ok).sum()):
        reason = (reason + ";" if reason else "") + f"nonfinite_replicates={int((~ok).sum())}"
    return BootResult(est, mcse, float(lo), float(hi), b, m2, stable, enough, reps1, reason, info, rinfo)


def bootstrap_vector(stat: Callable[[list[np.ndarray]], np.ndarray], chains: Sequence[np.ndarray],
                     iat_steps: float, *, reps: int, seed: int, multiplier: float = 10.0, min_blocks: int = 20,
                     doubled_min_blocks: int = 10, stability: float = 0.30,
                     percentiles: Sequence[float] = (2.5, 97.5)) -> dict[str, Any]:
    """Componentwise bootstrap MCSE/intervals for a vector statistic; same block rules as bootstrap_scalar."""
    est = np.asarray(stat(list(chains)), dtype=np.float64)
    k = est.shape[0]
    n_min = min(c.shape[0] for c in chains)
    b = block_length(iat_steps, multiplier)
    nanv = np.full(k, np.nan)
    res = {"estimate": est, "mcse": nanv.copy(), "low": nanv.copy(), "high": nanv.copy(), "block": b,
           "mcse_doubled": nanv.copy(), "stable": np.zeros(k, dtype=bool), "enough_blocks": n_min // b >= min_blocks,
           "replicates": np.zeros((0, k))}
    if not np.isfinite(iat_steps) or b > n_min:
        res["enough_blocks"] = False
        return res
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))
    r1 = np.asarray(bootstrap(stat, chains, b, reps, rng), dtype=np.float64)
    res["replicates"] = r1
    for j in range(k):
        v = r1[:, j][np.isfinite(r1[:, j])]
        if v.size > 1:
            res["mcse"][j] = np.std(v, ddof=1)
            res["low"][j], res["high"][j] = np.percentile(v, percentiles, method=QUANTILE_METHOD)
    if n_min // (2 * b) >= doubled_min_blocks:
        rng2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 2])))
        r2 = np.asarray(bootstrap(stat, chains, 2 * b, reps, rng2), dtype=np.float64)
        for j in range(k):
            v = r2[:, j][np.isfinite(r2[:, j])]
            if v.size > 1:
                res["mcse_doubled"][j] = np.std(v, ddof=1)
        m1, m2 = res["mcse"], res["mcse_doubled"]
        with np.errstate(invalid="ignore", divide="ignore"):
            res["stable"] = (np.abs(m2 - m1) <= stability * m1) & np.isfinite(m1) & np.isfinite(m2)
    return res


def rel(mcse: float, est: float) -> float:
    return abs(mcse / est) if est not in (0.0,) and np.isfinite(est) and np.isfinite(mcse) else float("inf")
