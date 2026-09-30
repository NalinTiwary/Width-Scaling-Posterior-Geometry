"""Entropy–Fisher tilt ratios with event weighting, held-out family selection, and static PI ratios
(runbook §11, §13.3).

Inputs are per-chain, time-ordered arrays over the selected archived reference states:
``F[c]`` (N_c, 11) raw probe values, ``G[c]`` (N_c, 11) raw squared gradient norms, ``J[c]`` (N_c,) the event
indicator (all ones for shallow targets). Outside states stay in the sequence with zero weight.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import numpy as np

from .diagnostics import bootstrap_scalar, bootstrap_vector, ess_mean_raw, is_constant, rhat_rank

NEG_D_TOL = 1e-12


def standardization(calib: np.ndarray, rel_threshold: float) -> tuple[float, float, bool]:
    """Pooled mean and sample SD (N-1) of a calibration trace; degenerate if s < thr·max(1,|μ|)."""
    x = np.asarray(calib, dtype=np.float64).ravel()
    mu, s = float(x.mean()), float(x.std(ddof=1))
    return mu, s, bool(s < rel_threshold * max(1.0, abs(mu)))


def tilt_terms(fz: np.ndarray, gz: np.ndarray, J: np.ndarray, t: float) -> dict[str, float]:
    """P, Z, B, C, D, I, R over the supplied (pooled) states; shift c = max over inside states of t·f."""
    inside = J == 1
    out = {"P": float(J.mean()) if J.size else 0.0, "n_inside": int(inside.sum()), "n_states": int(J.size)}
    nan = float("nan")
    if not inside.any():
        return {**out, "Z": 0.0, "B": nan, "C": nan, "D": nan, "I": nan, "R": nan, "max_weight": nan,
                "concentration": 0.0, "underflows": 0, "shift": nan, "reason": "P=0"}
    tf = t * fz
    c = float(tf[inside].max())
    u = np.where(inside, tf - c, 0.0)
    w = np.where(inside, np.exp(u), 0.0)
    Z, B, C = float(w.mean()), float((w * u).mean()), float((w * gz).mean())
    P = out["P"]
    D = B / Z - math.log(Z / P)
    I = t * t * C / Z
    reason = ""
    if not (np.isfinite(D) and np.isfinite(I)):
        reason = "nonfinite"
    elif I == 0.0:
        reason = "I=0"
    R = 2.0 * D / I if reason == "" else nan
    sw = w.sum()
    return {**out, "Z": Z, "B": B, "C": C, "D": D, "I": I, "R": R, "max_weight": float(w.max() / sw),
            "concentration": float(sw * sw / (w * w).sum()), "underflows": int(((w == 0) & inside).sum()),
            "shift": c, "reason": reason}


def _pool(arrs: Sequence[np.ndarray], chains: Sequence[int]) -> np.ndarray:
    return np.concatenate([arrs[c] for c in chains], axis=0)


def _integrands(fz: np.ndarray, gz: np.ndarray, J: np.ndarray, t: float, c: float) -> tuple[np.ndarray, np.ndarray]:
    inside = J == 1
    w = np.where(inside, np.exp(np.where(inside, t * fz - c, 0.0)), 0.0)
    return w, w * gz


class TiltProblem:
    """All probes of one target standardized, with chain structure kept."""

    def __init__(self, F: list[np.ndarray], G: list[np.ndarray], J: list[np.ndarray], mu: np.ndarray, s: np.ndarray,
                 names: list[str]):
        self.names = names
        self.Fz = [(f - mu) / s for f in F]
        self.Gz = [g / (s * s) for g in G]
        self.J = J
        self.C = len(F)

    def stats(self, k: int, t: float, chains: Sequence[int], Fz=None, Gz=None, J=None) -> dict[str, float]:
        Fz, Gz, J = Fz or self.Fz, Gz or self.Gz, J or self.J
        return tilt_terms(_pool(Fz, chains)[:, k], _pool(Gz, chains)[:, k], _pool(J, chains), t)


def candidate_fold(tp: TiltProblem, k: int, t: float, fold: Sequence[int], *, gates: dict[str, Any],
                   boot: dict[str, Any], seed: int) -> dict[str, Any]:
    """Estimate and gate one (probe, tilt) on one chain fold."""
    st = tp.stats(k, t, fold)
    row: dict[str, Any] = {**st}
    fails = []
    inside_pc = [int((tp.J[c] == 1).sum()) for c in fold]
    row["inside_per_chain"] = inside_pc
    if min(inside_pc) < gates["inside_states_per_chain_min"]:
        fails.append("inside_states")
    if st["reason"]:
        fails.append(st["reason"])
        row.update(validity=False, failure_reason=";".join(fails))
        return row
    if st["D"] < -NEG_D_TOL:
        raise ArithmeticError(f"negative entropy D={st['D']} beyond roundoff: implementation failure")
    if abs(st["D"]) <= NEG_D_TOL:
        fails.append("numerically_zero_D")
    if st["max_weight"] > gates["max_normalized_tilt_weight"]:
        fails.append("max_weight")
    if st["concentration"] < gates["iid_weight_concentration_per_fold_min"]:
        fails.append("weight_concentration")
    c = st["shift"]
    wn, wf = zip(*[_integrands(tp.Fz[ch][:, k], tp.Gz[ch][:, k], tp.J[ch], t, c) for ch in fold])
    ess_n = ess_mean_raw(np.stack(wn))[0] if not is_constant(np.stack(wn)) else float("inf")
    ess_f = ess_mean_raw(np.stack(wf))[0] if not is_constant(np.stack(wf)) else float("inf")
    row["ess_normalizer"], row["ess_fisher"] = ess_n, ess_f
    if not (ess_n >= gates["normalizer_and_fisher_raw_mean_ess_per_fold_min"]
            and ess_f >= gates["normalizer_and_fisher_raw_mean_ess_per_fold_min"]):
        fails.append("integrand_ess")
    n_c = np.stack(wn).shape[1]
    iat = max(np.stack(wn).size / ess_n if np.isfinite(ess_n) else 1.0,
              np.stack(wf).size / ess_f if np.isfinite(ess_f) else 1.0)

    def stat(chs):
        f = [ch[:, 0] for ch in chs]
        g = [ch[:, 1] for ch in chs]
        j = [ch[:, 2] for ch in chs]
        s2 = tilt_terms(np.concatenate(f), np.concatenate(g), np.concatenate(j), t)
        return np.array([s2["D"], s2["I"], s2["R"]])

    data = [np.stack([tp.Fz[ch][:, k], tp.Gz[ch][:, k], tp.J[ch]], axis=1) for ch in fold]
    br = bootstrap_vector(stat, data, iat, seed=seed, **boot)
    for j, nm in enumerate(("D", "I", "R")):
        row[f"{nm}_mcse"], row[f"{nm}_mc_low"], row[f"{nm}_mc_high"] = br["mcse"][j], br["low"][j], br["high"][j]
        row[f"{nm}_mcse_doubled_block"] = br["mcse_doubled"][j]
        est = st[nm]
        relv = abs(br["mcse"][j] / est) if est != 0 and np.isfinite(br["mcse"][j]) else float("inf")
        row[f"{nm}_rel_mcse"] = relv
        thr = gates[{"D": "entropy_relative_mcse_max", "I": "fisher_relative_mcse_max",
                     "R": "ratio_relative_mcse_max"}[nm]]
        if not relv <= thr:
            fails.append(f"{nm}_rel_mcse")
    row["block_length"] = br["block"]
    row["block_stable"] = bool(np.all(br["stable"]))
    if not br["enough_blocks"]:
        fails.append("fewer_than_min_blocks")
    if not row["block_stable"]:
        fails.append("block_instability")
    row["n_per_chain"] = n_c
    row.update(validity=not fails, failure_reason=";".join(fails))
    return row


def all_chain_rhat(tp: TiltProblem, k: int, t: float, c: float) -> tuple[float, float]:
    wn, wf = zip(*[_integrands(tp.Fz[ch][:, k], tp.Gz[ch][:, k], tp.J[ch], t, c) for ch in range(tp.C)])
    n = min(len(x) for x in wn)
    a, b = np.stack([x[:n] for x in wn]), np.stack([x[:n] for x in wf])
    return (rhat_rank(a) if not is_constant(a) else float("nan"),
            rhat_rank(b) if not is_constant(b) else float("nan"))


def heldout_selection(tp: TiltProblem, cands: list[tuple[int, float]], folds, Fz=None, Gz=None, J=None):
    a, b = list(folds[0]), list(folds[1])
    est, sel = [], []
    for s, e in ((a, b), (b, a)):
        vals = {}
        for (k, t) in cands:
            r = tp.stats(k, t, s, Fz, Gz, J)["R"]
            if np.isfinite(r):
                vals[(k, t)] = r
        if not vals:
            return float("nan"), {"selected": [None, None]}
        best = max(vals, key=lambda kt: (vals[kt], kt))
        sel.append(best)
        est.append(tp.stats(best[0], best[1], e, Fz, Gz, J)["R"])
    return float(np.mean(est)), {"selected": sel, "fold_estimates": est}


def family_entropy(tp: TiltProblem, cands: list[tuple[int, float]], folds, iat: float, *, boot: dict[str, Any],
                   seed: int) -> dict[str, Any]:
    ks = sorted({k for k, _ in cands})
    data = [np.concatenate([tp.Fz[c][:, ks], tp.Gz[c][:, ks], tp.J[c][:, None]], axis=1) for c in range(tp.C)]
    kk = len(ks)
    remap = {k: i for i, k in enumerate(ks)}
    sub = TiltProblem.__new__(TiltProblem)
    sub.C, sub.names = tp.C, [tp.names[k] for k in ks]
    rc = [(remap[k], t) for k, t in cands]

    def stat(chs):
        sub.Fz = [ch[:, :kk] for ch in chs]
        sub.Gz = [ch[:, kk:2 * kk] for ch in chs]
        sub.J = [ch[:, 2 * kk] for ch in chs]
        return heldout_selection(sub, rc, folds)

    br = bootstrap_scalar(stat, data, iat, seed=seed, **boot)
    freq = Counter()
    for inf in br.replicate_info or []:
        for s in (inf or {}).get("selected", []):
            if s is not None:
                freq[f"{sub.names[s[0]]}@t={s[1]:+g}"] += 1
    tot = sum(freq.values()) or 1
    sel = (br.info or {}).get("selected", [None, None])
    lab = [None if s is None else f"{sub.names[s[0]]}@t={s[1]:+g}" for s in sel]
    return {"estimate": br.estimate, "mcse": br.mcse, "mc_low": br.low, "mc_high": br.high,
            "block_length": br.block, "mcse_doubled_block": br.mcse_doubled, "block_stable": br.stable,
            "enough_blocks": br.enough_blocks, "selected_fold_A_to_B": lab[0], "selected_fold_B_to_A": lab[1],
            "selection_frequencies": {g: freq[g] / tot for g in sorted(freq)}, "replicates": br.replicates}


# ---- static PI ratios --------------------------------------------------------------------------------
def q_ratio(v: np.ndarray, g: np.ndarray) -> float:
    den = float(g.mean())
    return float(np.var(v) / den) if den > 0 else float("nan")


def static_probe(F: list[np.ndarray], G: list[np.ndarray], k: int, *, boot: dict[str, Any], seed: int) -> dict[str, Any]:
    v = np.stack([f[:, k] for f in F]) if len({len(f) for f in F}) == 1 else None
    x = [np.stack([f[:, k], g[:, k]], axis=1) for f, g in zip(F, G)]
    est_iat = 1.0
    if v is not None and not is_constant(v):
        e1 = ess_mean_raw(v)[0]
        e2 = ess_mean_raw(np.stack([g[:, k] for g in G]))[0] if not is_constant(np.stack([g[:, k] for g in G])) else v.size
        est_iat = max(v.size / e1, v.size / e2)

    def stat(chs):
        a = np.concatenate(chs)
        return q_ratio(a[:, 0], a[:, 1])

    br = bootstrap_scalar(stat, x, est_iat, seed=seed, **boot)
    return {"estimate": br.estimate, "mcse": br.mcse, "mc_low": br.low, "mc_high": br.high,
            "block_length": br.block, "mcse_doubled_block": br.mcse_doubled, "block_stable": br.stable,
            "enough_blocks": br.enough_blocks, "iat_states": est_iat, "replicates": br.replicates}


def static_family(F: list[np.ndarray], G: list[np.ndarray], ks: list[int], names: list[str], folds, iat: float, *,
                  boot: dict[str, Any], seed: int) -> dict[str, Any]:
    a, b = list(folds[0]), list(folds[1])
    x = [np.concatenate([f[:, ks], g[:, ks]], axis=1) for f, g in zip(F, G)]
    kk = len(ks)

    def sel_eval(chs):
        est, sel = [], []
        for s, e in ((a, b), (b, a)):
            ps = np.concatenate([chs[c] for c in s])
            qs = [q_ratio(ps[:, j], ps[:, kk + j]) for j in range(kk)]
            if not np.any(np.isfinite(qs)):
                return float("nan"), {"selected": [None, None]}
            j = int(np.nanargmax(qs))
            pe = np.concatenate([chs[c] for c in e])
            sel.append(j)
            est.append(q_ratio(pe[:, j], pe[:, kk + j]))
        return float(np.mean(est)), {"selected": sel}

    br = bootstrap_scalar(sel_eval, x, iat, seed=seed, **boot)
    freq = Counter()
    for inf in br.replicate_info or []:
        for j in (inf or {}).get("selected", []):
            if j is not None:
                freq[names[ks[j]]] += 1
    tot = sum(freq.values()) or 1
    sel = (br.info or {}).get("selected", [None, None])
    return {"estimate": br.estimate, "mcse": br.mcse, "mc_low": br.low, "mc_high": br.high,
            "block_length": br.block, "mcse_doubled_block": br.mcse_doubled, "block_stable": br.stable,
            "enough_blocks": br.enough_blocks,
            "selected_fold_A_to_B": None if sel[0] is None else names[ks[sel[0]]],
            "selected_fold_B_to_A": None if sel[1] is None else names[ks[sel[1]]],
            "selection_frequencies": {g: freq[g] / tot for g in sorted(freq)}, "replicates": br.replicates}
