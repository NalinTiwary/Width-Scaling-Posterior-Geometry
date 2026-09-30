"""Reference sampling stage machine with diagnostic, spectral and static-estimate gates (runbook §7, §11, §12)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from . import diagnostics as dg
from .chains import Trajectory
from .context import TargetContext
from .ellipse import SamplerFailure
from .entropy import (TiltProblem, all_chain_rhat, candidate_fold, family_entropy, standardization, static_family,
                      static_probe)
from .spectral import INSIDE, OUTSIDE, UNRESOLVED, backend_check, check_indices, classify
from .storage import atomic_write_json, clean_json, read_json


def boot_kw(cfg: dict[str, Any]) -> dict[str, Any]:
    u = cfg["uncertainty"]
    return {"reps": int(u["bootstrap_replicates"]), "multiplier": float(u["block_integrated_time_multiplier"]),
            "min_blocks": int(u["nonoverlapping_blocks_per_chain_min"]),
            "doubled_min_blocks": int(u["doubled_block_min_blocks"]),
            "stability": float(u["doubled_block_mcse_relative_difference_max"]),
            "percentiles": tuple(float(x) for x in u["interval_percentiles"])}


def production_segments(k: int) -> list[str]:
    return [f"production_s{i}" for i in range(1, k + 1)]


def completed_production(tr: Trajectory) -> list[str]:
    segs, k = [], 1
    while f"production_s{k}" in tr.ck["done"]:
        segs.append(f"production_s{k}")
        k += 1
    return segs


def _budget_hit(tr: Trajectory) -> bool:
    return tr.ck.get("status") == "reference_budget_exhausted"


# ---- calibration constants ----------------------------------------------------------------------------
def calibration_constants(ctx: TargetContext, tr: Trajectory) -> dict[str, Any]:
    p = ctx.dir / "calibration.json"
    if p.exists():
        return read_json(p)
    dat = tr.read(["calibration"], keys=["scalars", "draw"])
    thr = float(ctx.cfg["probes"]["degenerate_sd_relative_threshold"])
    out = {"source": "reference calibration segment (all chains, pooled)", "n_per_chain": int(dat[0]["draw"].shape[0]),
           "probes": {}}
    for name in ctx.spec.probe_names():
        j = ctx.names.index(name)
        x = np.stack([d["scalars"][:, j] for d in dat])
        mu, s, deg = standardization(x, thr)
        out["probes"][name] = {"mean": mu, "sd": s, "degenerate": deg}
    atomic_write_json(p, clean_json(out))
    return out


# ---- spectral ----------------------------------------------------------------------------------------
def spectral_eval(ctx: TargetContext, tr: Trajectory, segs: list[str]) -> dict[str, Any]:
    cfg = ctx.cfg
    sp = cfg["spectral"]
    guard = float(sp["normalized_boundary_guard"])
    dat = tr.read(segs, keys=["archive/S", "archive/draw", "draw"])
    S = [d["archive/S"] for d in dat]
    draws = [d["archive/draw"] for d in dat]
    mem, notes = [], []
    for ci, c in enumerate(tr.chains):
        band = np.flatnonzero(np.abs(S[ci] - 1.0) <= guard)
        m = np.where(S[ci] < 1.0 - guard, INSIDE, np.where(S[ci] > 1.0 + guard, OUTSIDE, UNRESOLVED))
        if band.size:
            rows, _ = tr.archive_rows(c, segs, band)
            W2 = rows[:, ctx.lay.offsets["W2"][0]:].reshape(-1, ctx.t.m, ctx.t.m)
            mb, nb = classify(S[ci][band], W2, ctx.a, guard)
            m[band] = mb
            notes += [{"chain": c, "archive_index": int(band[n["index"]]), **n} for n in nb]
        mem.append(m)
    # independent backend check on 20 fixed states spread over the chains' archives
    k = int(sp["independent_backend_check_states_per_target"])
    per = [check_indices(len(S[i]), -(-k // len(tr.chains))) for i in range(len(tr.chains))]
    checks = []
    for ci, c in enumerate(tr.chains):
        rows, _ = tr.archive_rows(c, segs, per[ci])
        W2 = rows[:, ctx.lay.offsets["W2"][0]:].reshape(-1, ctx.t.m, ctx.t.m)
        for r in backend_check(W2, S[ci][per[ci]], ctx.a, np.arange(len(per[ci]))):
            r["index"] = int(per[ci][r["index"]])
            checks.append({"chain": c, **r})
    checks = checks[:k]
    tol = float(sp["independent_backend_relative_tolerance"])
    backend_ok = all(r["rel_diff"] < tol for r in checks)
    Sa = np.stack(S)
    sdiag = dg.scalar_diagnostics("S", Sa)
    ind = np.stack([(m == INSIDE).astype(float) for m in mem])
    idiag = dg.scalar_diagnostics("inside_indicator", ind)
    return {"S": S, "draws": draws, "membership": mem, "guard_notes": notes, "backend_checks": checks,
            "backend_ok": backend_ok, "S_diag": sdiag.row(), "indicator_diag": idiag.row(),
            "indicator_status": ("not_estimable_zero_exits" if idiag.status == "constant" else idiag.status)}


def spectral_summary(ctx: TargetContext, sp_eval: dict[str, Any], stage: str) -> dict[str, Any]:
    cfg = ctx.cfg
    qs = [float(q) for q in cfg["spectral"]["posterior_quantiles"]]
    S, mem, chains = sp_eval["S"], sp_eval["membership"], ctx.chains
    ids = ctx.ids(law="full_posterior_archive", stage=stage, execution_hash=ctx.ref_exec_hash())
    rows = []
    for ci, c in enumerate(chains):
        m = mem[ci]
        row = {**ids, "chain_id": c, "first_draw": int(sp_eval["draws"][ci][0]), "last_draw": int(sp_eval["draws"][ci][-1]),
               "saved_stride": int(cfg["reference"]["parameter_archive_stride"]), "a": ctx.a,
               "inspected_states": int(m.size), "inside": int((m == INSIDE).sum()), "exits": int((m == OUTSIDE).sum()),
               "unresolved": int((m == UNRESOLVED).sum()), "occupancy": float((m == INSIDE).mean())}
        for q in qs:
            row[f"S_q{q:g}"] = float(np.quantile(S[ci], q, method=dg.QUANTILE_METHOD))
        rows.append(row)
    allm = np.concatenate(mem)
    allS = np.concatenate(S)
    exits = int((allm == OUTSIDE).sum())
    unres = int((allm == UNRESOLVED).sum())
    chains_with_exits = sum(int((m == OUTSIDE).sum() > 0) for m in mem)
    tgt = {**ids, "chain_id": "all", "a": ctx.a, "inspected_states": int(allm.size), "inside": int((allm == INSIDE).sum()),
           "exits": exits, "unresolved": unres, "chains_with_exits": chains_with_exits,
           "occupancy": float((allm == INSIDE).mean()),
           "occupancy_low_numerical": float((allm == INSIDE).mean()),
           "occupancy_high_numerical": float(((allm == INSIDE) | (allm == UNRESOLVED)).mean()),
           "S_rhat": sp_eval["S_diag"]["rhat"], "S_ess_bulk": sp_eval["S_diag"]["ess_bulk"],
           "S_ess_tail": sp_eval["S_diag"]["ess_tail"], "indicator_status": sp_eval["indicator_status"],
           "backend_check_ok": sp_eval["backend_ok"],
           "backend_max_rel_diff": max((r["rel_diff"] for r in sp_eval["backend_checks"]), default=float("nan"))}
    iat = float(allS.size / sp_eval["S_diag"]["ess_mean"]) if sp_eval["S_diag"]["ess_mean"] else 1.0
    bk = boot_kw(cfg)
    for q in qs:
        br = dg.bootstrap_scalar(lambda cs, q=q: float(np.quantile(np.concatenate(cs), q, method=dg.QUANTILE_METHOD)),
                                 S, iat, seed=ctx.seed("shared", stage, f"bootstrap_S_q{q:g}"), **bk)
        tgt.update({f"S_q{q:g}": br.estimate, f"S_q{q:g}_mcse": br.mcse, f"S_q{q:g}_mc_low": br.low,
                    f"S_q{q:g}_mc_high": br.high, f"S_q{q:g}_block_stable": br.stable})
    if exits == 0:
        tgt["occupancy_uncertainty"] = f"0 exits / {allm.size} inspected archived states"
    elif exits >= int(cfg["spectral"]["occupancy_interval_min_exits"]) and \
            chains_with_exits >= int(cfg["spectral"]["occupancy_interval_min_chains_with_exits"]):
        ind = [(m == INSIDE).astype(float) for m in mem]
        ie = dg.ess_mean_raw(np.stack(ind))[0]
        br = dg.bootstrap_scalar(lambda cs: float(np.concatenate(cs).mean()), ind, allm.size / ie if ie else 1.0,
                                 seed=ctx.seed("shared", stage, "bootstrap_occupancy"), **bk)
        tgt.update({"occupancy_mcse": br.mcse, "occupancy_mc_low": br.low, "occupancy_mc_high": br.high,
                    "occupancy_uncertainty": "chain-aware block bootstrap"})
    else:
        tgt["occupancy_uncertainty"] = "rare-event uncertainty unresolved"
    return {"per_chain": rows, "target": tgt}


# ---- static estimates ----------------------------------------------------------------------------------
class StaticCache:
    """Probe values and squared gradient norms keyed by (chain, cumulative production draw id)."""

    def __init__(self, path: Path, thash: str):
        self.path, self.thash = path, thash
        self.data: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if str(z["target_hash"]) != thash:
                raise RuntimeError(f"{path}: static cache belongs to a different target")
            for c, d, v, g in zip(z["chain"], z["draw"], z["values"], z["gsq"]):
                self.data.setdefault(int(c), {})[int(d)] = (v, g)

    def __len__(self) -> int:
        return sum(len(v) for v in self.data.values())

    def save(self) -> None:
        ch, dr, va, gs = [], [], [], []
        for c in sorted(self.data):
            for d in sorted(self.data[c]):
                ch.append(c)
                dr.append(d)
                va.append(self.data[c][d][0])
                gs.append(self.data[c][d][1])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp.npz")
        np.savez(tmp, target_hash=np.array(self.thash), chain=np.array(ch, dtype=np.int64),
                 draw=np.array(dr, dtype=np.int64), values=np.array(va).reshape(-1, 11),
                 gsq=np.array(gs).reshape(-1, 11))
        tmp.replace(self.path)


def compute_probe_batch(ctx: TargetContext, states: np.ndarray, batch: int) -> tuple[np.ndarray, np.ndarray]:
    vals, gsq = [], []
    for i in range(0, states.shape[0], batch):
        th = torch.as_tensor(states[i:i + batch], device=ctx.device)
        v, g = ctx.obs.probe_values_and_grad_sq(th)
        vals.append(v.detach().cpu().numpy())
        gsq.append(g.detach().cpu().numpy())
    return np.concatenate(vals), np.concatenate(gsq)


def select_indices(M: int, per_chain: int) -> np.ndarray:
    stride = M // per_chain
    return np.arange(per_chain, dtype=np.int64) * stride


def static_eval(ctx: TargetContext, tr: Trajectory, segs: list[str], stage: str, ref_ok: bool,
                spec_eval: Optional[dict[str, Any]], calib: dict[str, Any]) -> dict[str, Any]:
    cfg = ctx.cfg
    st = cfg["static"]
    M = tr.archive_count(segs)
    cache = StaticCache(ctx.dir / "static_cache.npz", ctx.thash)
    batch = 64 if ctx.lay.p < 20000 else 16
    pnames = ctx.spec.probe_names()
    last = None
    t0 = time.time()
    arch_draws = [d["archive/draw"] for d in tr.read(segs, keys=["archive/draw"])]
    scal = [d["scalars"] for d in tr.read(segs, keys=["scalars"])]
    for per_chain in st["selected_states_per_chain_stages"]:
        per_chain = int(per_chain)
        if per_chain > M:
            break
        idx = select_indices(M, per_chain)
        F, G, J, D = [], [], [], []
        consistency = 0.0
        for ci, c in enumerate(tr.chains):
            dat_draw = arch_draws[ci][idx]
            need = [i for i, d in zip(idx, dat_draw) if int(d) not in cache.data.get(c, {})]
            if need:
                rows, dr = tr.archive_rows(c, segs, np.array(need))
                v, g = compute_probe_batch(ctx, rows, batch)
                for dd, vv, gg in zip(dr, v, g):
                    cache.data.setdefault(c, {})[int(dd)] = (vv, gg)
                cache.save()
            F.append(np.stack([cache.data[c][int(d)][0] for d in dat_draw]))
            G.append(np.stack([cache.data[c][int(d)][1] for d in dat_draw]))
            D.append(dat_draw)
            if spec_eval is not None:
                pos = np.searchsorted(spec_eval["draws"][ci], dat_draw)
                J.append((spec_eval["membership"][ci][pos] == INSIDE).astype(float))
                unresolved = int((spec_eval["membership"][ci][pos] == UNRESOLVED).sum())
            else:
                J.append(np.ones(per_chain))
                unresolved = 0
            sc = scal[ci][dat_draw]
            for j, nm in enumerate(pnames):
                consistency = max(consistency, float(np.max(np.abs(sc[:, ctx.names.index(nm)] - F[-1][:, j]))))
        res = static_estimates(ctx, F, G, J, calib, stage=f"{stage}|sel={per_chain}", ref_ok=ref_ok,
                               membership_unresolved=unresolved)
        res.update({"per_chain_selected": per_chain, "selected_indices": idx.tolist(),
                    "selected_draws": [d.tolist() for d in D], "archive_states_per_chain": M,
                    "probe_trace_consistency_max_abs": consistency,
                    "probe_gradient_evals_final_subset": 11 * per_chain * len(tr.chains),
                    "probe_gradient_evals_cumulative": 11 * len(cache), "static_wall_s": time.time() - t0})
        np.savez(ctx.dir / "analysis" / "static_inputs.npz", **{f"F_{c}": F[i] for i, c in enumerate(tr.chains)},
                 **{f"G_{c}": G[i] for i, c in enumerate(tr.chains)},
                 **{f"Gz_{c}": G[i] / np.array([calib["probes"][n]["sd"] ** 2 or np.nan for n in pnames])
                    for i, c in enumerate(tr.chains)},
                 **{f"J_{c}": J[i] for i, c in enumerate(tr.chains)},
                 **{f"draw_{c}": D[i] for i, c in enumerate(tr.chains)})
        last = res
        if res["static_pass"]:
            break
    if last is None:
        return {"static_pass": False, "reason": f"archive has {M} states per chain < first selection stage"}
    return last


def static_estimates(ctx: TargetContext, F, G, J, calib, *, stage: str, ref_ok: bool,
                     membership_unresolved: int = 0) -> dict[str, Any]:
    cfg = ctx.cfg
    gates = cfg["static_gates"]
    bk = boot_kw(cfg)
    pnames = ctx.spec.probe_names()
    mu = np.array([calib["probes"][n]["mean"] for n in pnames])
    sd = np.array([calib["probes"][n]["sd"] for n in pnames])
    deg = [bool(calib["probes"][n]["degenerate"]) for n in pnames]
    sd_safe = np.where(np.array(deg), 1.0, sd)
    tp = TiltProblem(F, G, J, mu, sd_safe, pnames)
    tilts = [float(t) for t in cfg["probes"]["tilts"]]
    fams = ctx.spec.static_families()
    controls = ["q_A", "q_W"]
    law = ctx.law_static
    ids = ctx.ids(law=law, stage=stage, execution_hash=ctx.ref_exec_hash())
    rhat_max = float(gates["selected_integrand_rhat_max_exclusive"])
    probe_rows, cand_ok, cand_iat = [], {}, {}
    for k, name in enumerate(pnames):
        family = next((f for f, ns in fams.items() if name in ns), "control")
        for t in tilts:
            row = {**ids, "probe": name, "family": family, "t": t, "calibration_mean": mu[k],
                   "calibration_sd": sd[k], "degenerate": deg[k]}
            if deg[k]:
                row.update(validity=False, failure_reason="degenerate_probe")
                probe_rows.append(row)
                cand_ok[(k, t)] = False
                continue
            pooled = tp.stats(k, t, list(range(tp.C)))
            row.update({f"pooled_{x}": pooled[x] for x in ("P", "Z", "B", "C", "D", "I", "R", "max_weight",
                                                           "concentration", "underflows", "n_inside", "n_states")})
            fails = []
            for fi, fold in enumerate(ctx.folds):
                fr = candidate_fold(tp, k, t, fold, gates=gates, boot=bk,
                                    seed=ctx.seed("shared", stage, f"bootstrap_entropy|{name}|t={t!r}|fold={fi}"))
                row.update({f"fold{fi}_{x}": v for x, v in fr.items()})
                if not fr["validity"]:
                    fails.append(f"fold{fi}:{fr['failure_reason']}")
            ra, rb = all_chain_rhat(tp, k, t, pooled["shift"]) if np.isfinite(pooled.get("shift", np.nan)) else (np.nan, np.nan)
            row["rhat_normalizer"], row["rhat_fisher"] = ra, rb
            if (np.isfinite(ra) and not ra < rhat_max) or (np.isfinite(rb) and not rb < rhat_max):
                fails.append("integrand_rhat")
            if membership_unresolved:
                fails.append(f"membership_unresolved={membership_unresolved}")
            if not ref_ok:
                fails.append("reference_not_passed")
            row["estimate"] = pooled["R"]
            row["mcse"] = float(np.hypot(row.get("fold0_R_mcse", np.nan), row.get("fold1_R_mcse", np.nan)) / 2.0)
            row["mcse_method"] = "fold bootstrap MCSEs combined as sqrt(m0^2+m1^2)/2 (pooled all-chain estimate)"
            row["estimate_over_sigma2"] = pooled["R"] / ctx.sigma**2
            row["n_draws_used"] = pooled["n_states"]
            row["validity"] = not fails
            row["failure_reason"] = ";".join(fails)
            gate_fail = [f for f in fails if f != "reference_not_passed"]
            cand_ok[(k, t)] = not gate_fail
            iats = [row.get(f"fold{fi}_n_per_chain", 0) * 2 / row.get(f"fold{fi}_ess_normalizer", np.inf)
                    for fi in range(2)] + [row.get(f"fold{fi}_n_per_chain", 0) * 2 / row.get(f"fold{fi}_ess_fisher", np.inf)
                                           for fi in range(2)]
            cand_iat[(k, t)] = float(np.nanmax([x for x in iats if np.isfinite(x)] or [1.0]))
            probe_rows.append(row)
    fam_rows = []
    boot_store = {}
    for fam, members in fams.items():
        ks = [pnames.index(n) for n in members]
        cands = [(k, t) for k in ks for t in tilts if not deg[k]]
        failing = [f"{pnames[k]}@t={t:+g}" for (k, t) in [(k, t) for k in ks for t in tilts] if not cand_ok[(k, t)]]
        row = {**ids, "family": fam, "candidates": len(ks) * len(tilts), "degenerate_probes": [pnames[k] for k in ks if deg[k]],
               "failing_candidates": failing}
        if cands:
            iat = max(cand_iat.get(c, 1.0) for c in cands)
            fr = family_entropy(tp, cands, ctx.folds, iat, boot=bk, seed=ctx.seed("shared", stage, f"bootstrap_entropy_family|{fam}"))
            boot_store[f"entropy_{fam}"] = fr.pop("replicates")
            row.update(fr)
        else:
            row.update(estimate=float("nan"))
        reasons = []
        if not ref_ok:
            reasons.append("reference_not_passed")
        if failing:
            reasons.append("incomplete:" + ",".join(failing))
        if row["degenerate_probes"]:
            reasons.append("degenerate:" + ",".join(row["degenerate_probes"]))
        if membership_unresolved:
            reasons.append("membership_unresolved")
        row["n_draws_used"] = int(sum(len(j) for j in J))
        row["validity"] = not reasons
        row["status"] = "valid" if not reasons else ("incomplete" if failing or row["degenerate_probes"] else "unresolved")
        row["failure_reason"] = ";".join(reasons)
        fam_rows.append(row)
    static_pass = all(cand_ok[(k, t)] for k, n in enumerate(pnames) if n not in controls and not deg[k] for t in tilts)
    # static PI on the unrestricted posterior (all selected states, including deep states outside G_a)
    pi_ids = ctx.ids(law="full_posterior", stage=stage, execution_hash=ctx.ref_exec_hash())
    thr = float(cfg["static_gates"]["ratio_relative_mcse_max"])
    pi_rows, pi_iat = [], {}
    for k, name in enumerate(pnames):
        family = next((f for f, ns in fams.items() if name in ns), "control")
        pr = static_probe(F, G, k, boot=bk, seed=ctx.seed("shared", stage, f"bootstrap_pi|{name}"))
        boot_store[f"pi_probe_{name}"] = pr.pop("replicates")
        pi_iat[k] = pr["iat_states"]
        reasons = []
        if not dg.rel(pr["mcse"], pr["estimate"]) <= thr:
            reasons.append("rel_mcse")
        if not pr["block_stable"]:
            reasons.append("block_instability")
        if not pr["enough_blocks"]:
            reasons.append("fewer_than_min_blocks")
        if not ref_ok:
            reasons.append("reference_not_passed")
        pi_rows.append({**pi_ids, "probe": name, "family": family, "kind": "probe", **pr,
                        "estimate_over_sigma2": pr["estimate"] / ctx.sigma**2,
                        "n_draws_used": int(sum(len(f) for f in F)), "validity": not reasons,
                        "failure_reason": ";".join(reasons)})
    for fam, members in fams.items():
        ks = [pnames.index(n) for n in members]
        iat = max(pi_iat[k] for k in ks)
        fr = static_family(F, G, ks, pnames, ctx.folds, iat, boot=bk, seed=ctx.seed("shared", stage, f"bootstrap_pi_family|{fam}"))
        boot_store[f"pi_{fam}"] = fr.pop("replicates")
        members_ok = all(r["validity"] or r["failure_reason"] == "reference_not_passed"
                         for r in pi_rows if r["probe"] in members)
        reasons = []
        if not dg.rel(fr["mcse"], fr["estimate"]) <= thr:
            reasons.append("rel_mcse")
        if not fr["block_stable"]:
            reasons.append("block_instability")
        if not members_ok:
            reasons.append("member_probe_invalid")
        if not ref_ok:
            reasons.append("reference_not_passed")
        pi_rows.append({**pi_ids, "probe": None, "family": fam, "kind": "family", **fr,
                        "estimate_over_sigma2": fr["estimate"] / ctx.sigma**2,
                        "n_draws_used": int(sum(len(f) for f in F)), "validity": not reasons,
                        "failure_reason": ";".join(reasons)})
    np.savez(ctx.dir / "analysis" / "boot_static.npz", **{k: np.asarray(v) for k, v in boot_store.items()})
    return {"static_pass": bool(static_pass), "entropy_probes": probe_rows, "entropy_families": fam_rows,
            "static_pi": pi_rows}


# ---- stage evaluation ---------------------------------------------------------------------------------
def evaluate_stage(ctx: TargetContext, tr: Trajectory, k: int, *, final: bool) -> dict[str, Any]:
    cfg = ctx.cfg
    name = f"reference_stage_{k}"
    prev = ctx.load_analysis(name)
    if prev is not None and prev.get("execution_hash") == ctx.ref_exec_hash() and (prev.get("final") or not final):
        return prev
    t0 = time.time()
    segs = production_segments(k)
    stage = f"reference_production_{k}"
    g = cfg["reference_gates"]
    dat = tr.read(segs, keys=["scalars", "draw", "evals"])
    N = int(dat[0]["draw"].shape[0])
    ids = ctx.ids(law="full_posterior", stage=stage, execution_hash=ctx.ref_exec_hash())
    rows, fails = [], []
    for j, nm in enumerate(ctx.names):
        x = np.stack([d["scalars"][:, j] for d in dat])
        d = dg.scalar_diagnostics(nm, x, mcse_mult=float(g["split_mean_mcse_multiplier"]),
                                  sd_allow=float(g["split_mean_reference_sd_allowance"]))
        ok, why = dg.rank_gate(d, rhat_max=float(g["rhat_max_exclusive"]), bulk_min=float(g["bulk_ess_min"]),
                               tail_min=float(g["tail_ess_min"]))
        if not d.drift_pass:
            ok, why = False, (why + ";" if why and why != "constant" else "") + "split_half_drift"
        rows.append({**ids, "observable": nm, **d.row(), "gate_pass": ok, "gate_reason": why,
                     "first_draw": 0, "last_draw": N - 1, "saved_stride": 1})
        if not ok:
            fails.append(f"{nm}:{why}")
    partial = any(tr.ck["done"][s]["partial"] for s in segs)
    if partial:
        fails.append("partial_stage_budget_exhausted")
    spec_eval, spec_sum = None, None
    if ctx.t.arch == "deep":
        spec_eval = spectral_eval(ctx, tr, segs)
        sd = dg.ScalarDiag(**spec_eval["S_diag"])
        ok, why = dg.rank_gate(sd, rhat_max=float(g["rhat_max_exclusive"]), bulk_min=float(g["bulk_ess_min"]),
                               tail_min=float(g["tail_ess_min"]))
        rows.append({**ids, "observable": "S_archive", **spec_eval["S_diag"], "gate_pass": ok, "gate_reason": why,
                     "first_draw": 0, "last_draw": N - 1, "saved_stride": int(cfg["reference"]["parameter_archive_stride"])})
        if not ok:
            fails.append(f"S:{why}")
        if not spec_eval["backend_ok"]:
            fails.append("svd_backend_check")
    ref_ok = not fails
    calib = calibration_constants(ctx, tr)
    static = None
    if ref_ok or final:
        static = static_eval(ctx, tr, segs, stage, ref_ok, spec_eval, calib)
    if spec_eval is not None:
        spec_sum = spectral_summary(ctx, spec_eval, stage)
        np.savez(ctx.dir / "analysis" / "spectral_states.npz",
                 **{f"S_{c}": spec_eval["S"][i] for i, c in enumerate(tr.chains)},
                 **{f"draw_{c}": spec_eval["draws"][i] for i, c in enumerate(tr.chains)},
                 **{f"membership_{c}": spec_eval["membership"][i] for i, c in enumerate(tr.chains)})
    res = {"execution_hash": ctx.ref_exec_hash(), "stage": k, "final": final, "production_per_chain": N,
           "reference_pass": ref_ok, "reference_failures": fails, "diagnostics": rows,
           "static_pass": bool(static and static["static_pass"]), "static": static,
           "spectral": None if spec_eval is None else {"summary": spec_sum, "guard_notes": spec_eval["guard_notes"],
                                                       "backend_checks": spec_eval["backend_checks"]},
           "eval_wall_s": time.time() - t0}
    ctx.save_analysis(name, res)
    return res


# ---- runner ------------------------------------------------------------------------------------------
def run_reference(ctx: TargetContext, *, stop_after: Optional[int] = None) -> dict[str, Any]:
    cfg = ctx.cfg
    r = cfg["reference"]
    ctx.write_spec()
    tr = ctx.open_reference()
    res = None
    try:
        for seg, n, a in (("burnin", r["burn_in_transitions"], 0),
                          ("calibration", r["calibration_transitions"], r["calibration_archive_stride"]),
                          ("separation", r["separation_transitions"], 0)):
            tr.run_segment(seg, int(n), archive_stride=int(a), stop_after=stop_after)
            if _budget_hit(tr):
                break
        if not _budget_hit(tr):
            prev = 0
            stages = [int(s) for s in r["cumulative_production_stages"]]
            for k, total in enumerate(stages, 1):
                tr.run_segment(f"production_s{k}", total - prev, archive_stride=int(r["parameter_archive_stride"]),
                               stop_after=stop_after)
                prev = total
                final = k == len(stages) or _budget_hit(tr)
                res = evaluate_stage(ctx, tr, k, final=final)
                if final or (res["reference_pass"] and res["static_pass"]):
                    break
    except SamplerFailure as exc:
        st = ctx.update_status(reference={"status": "reference_error", "error": str(exc),
                                          "counters": tr.ck.get("counters")})
        return st
    counters = tr.ck["counters"]
    if res is None:
        status = "reference_unresolved"
    else:
        status = "reference_pass" if res["reference_pass"] else "reference_unresolved"
    info = {"status": status, "budget_exhausted": _budget_hit(tr),
            "stopping_stage": None if res is None else res["stage"],
            "production_per_chain": None if res is None else res["production_per_chain"],
            "static_pass": None if res is None else res["static_pass"],
            "likelihood_evals_per_chain": counters["evals"], "transitions_per_chain": counters["transitions"],
            "segments": tr.ck["done"], "wall_s": tr.ck["elapsed_s"], "execution_hash": ctx.ref_exec_hash(),
            "device": str(ctx.device)}
    return ctx.update_status(reference=info)


def reference_final_state(ctx: TargetContext) -> torch.Tensor:
    tr = ctx.open_reference()
    return tr.state.theta.detach().clone()
