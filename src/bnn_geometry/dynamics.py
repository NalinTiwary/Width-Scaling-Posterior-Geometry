"""Dynamics: step calibration, production with duration stages, endpoint step-halving and one refinement
(runbook §8, §10)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import config as C
from . import diagnostics as dg
from .chains import Trajectory
from .context import TargetContext, h_id
from .reference import boot_kw, production_segments, reference_final_state
from .relaxation import analyze_family, inefficiency, tau_hat
from .storage import atomic_write_json, clean_json, read_json


def max_streak(acc: np.ndarray) -> int:
    best = cur = 0
    for a in acc:
        cur = 0 if a else cur + 1
        best = max(best, cur)
    return best


def retained_segments(k: int) -> list[str]:
    return [f"stage_{i}" for i in range(1, k + 1)]


# ---- step calibration -------------------------------------------------------------------------------
def step_calibration(ctx: TargetContext) -> dict[str, Any]:
    prev = ctx.load_analysis("step_calibration")
    if prev is not None and prev.get("complete"):
        return prev
    cfg, s2 = ctx.cfg, ctx.sigma**2
    d = cfg["dynamics"]
    ref = reference_final_state(ctx)
    rows, selected = [], None
    for hr in d["h_over_sigma_squared_candidates"]:
        h = float(hr) * s2
        tr = ctx.open_dynamics(h, ref, calibration=True)
        tr.run_segment("discard", C.n_steps(float(d["calibration_discard_time_over_sigma_squared"]) * s2, h))
        tr.run_segment("measure", C.n_steps(float(d["calibration_measure_time_over_sigma_squared"]) * s2, h))
        both = tr.read(["discard", "measure"], keys=["logratio"])[0]
        meas = tr.read(["measure"], keys=["accepted", "logratio"])[0]
        acc = float(meas["accepted"].mean()) if meas["accepted"].size else float("nan")
        streak = max_streak(meas["accepted"])
        nf = int((~np.isfinite(both["logratio"])).sum())
        partial = any(r["partial"] for r in tr.ck["done"].values())
        adm = (nf == 0 and acc >= float(d["calibration_acceptance_min"])
               and streak <= int(d["calibration_max_rejection_streak"]) and not partial)
        rows.append({"h": h, "h_over_sigma2": float(hr), "h_hex": float(h).hex(), "acceptance": acc,
                     "max_rejection_streak": streak, "numerical_failures": nf, "partial": partial,
                     "measured_transitions": int(meas["accepted"].size), "admissible": adm,
                     "grad_evals": tr.grad_evals()})
        if adm:
            selected = h
            break
        if tr.budget_exhausted():
            break
    res = {"complete": True, "candidates": rows, "selected_h": selected,
           "status": "calibrated" if selected is not None else "dynamics_calibration_failed",
           "execution_hash_reference": ctx.ref_exec_hash()}
    ctx.save_analysis("step_calibration", res)
    ctx.update_status(step_calibration={"status": res["status"], "selected_h": selected})
    return res


def select_steps(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    out = {}
    for arch in ("shallow", "deep"):
        per, failed = {}, []
        for t in C.targets(cfg):
            if t.arch != arch:
                continue
            p = root / "targets" / t.target_id / "analysis" / "step_calibration.json"
            sel = read_json(p)["selected_h"] if p.exists() else None
            per[t.target_id] = sel
            if sel is None:
                failed.append(t.target_id)
        ok = [v for v in per.values() if v is not None]
        out[arch] = {"h": min(ok) if ok else None, "per_target": per, "calibration_failed": failed,
                     "rule": "minimum admissible candidate across the architecture's targets"}
    atomic_write_json(root / "dynamics_step.json", clean_json(out))
    return out


# ---- analysis of one dynamics trajectory at a retained stage -----------------------------------------
_REF_CACHE: dict[str, Any] = {}


def reference_pool(ctx: TargetContext) -> dict[str, np.ndarray]:
    key = ctx.t.target_id
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    tr = ctx.open_reference()
    k = (ctx.status().get("reference") or {}).get("stopping_stage")
    if not k:
        return {}
    dat = tr.read(production_segments(int(k)), keys=["scalars"])
    pool = {nm: np.stack([d["scalars"][:, j] for d in dat]) for j, nm in enumerate(ctx.names)}
    _REF_CACHE[key] = pool
    return pool


def analyze_dynamics(ctx: TargetContext, tr: Trajectory, h: float, k: int) -> dict[str, Any]:
    name = f"dynamics_{h_id(h)}_stage_{k}"
    prev = ctx.load_analysis(name)
    ex = ctx.dyn_exec_hash(h)
    if prev is not None and prev.get("execution_hash") == ex:
        return prev
    t0 = time.time()
    cfg = ctx.cfg
    g = cfg["dynamics_gates"]
    segs = retained_segments(k)
    dat = tr.read(segs, keys=["scalars", "accepted", "logratio"])
    N = int(dat[0]["draw"].shape[0])
    T = N * h
    stage = f"dynamics_{h_id(h)}_stage_{k}"
    ids = ctx.ids(law="full_posterior", stage=stage, execution_hash=ex)
    acc = [float(d["accepted"].mean()) for d in dat]
    streaks = [max_streak(d["accepted"]) for d in dat]
    acc_fail = min(acc) < float(g["production_acceptance_per_chain_min"]) or \
        max(streaks) > int(g["production_max_rejection_streak"])
    X = {nm: np.stack([d["scalars"][:, j] for d in dat]) for j, nm in enumerate(ctx.names)}
    diag_rows, rank_fail = [], []
    for nm, x in X.items():
        d = dg.scalar_diagnostics(nm, x)
        ok, why = dg.rank_gate(d, rhat_max=float(g["rhat_max_exclusive"]), bulk_min=float(g["bulk_ess_min"]),
                               tail_min=float(g["tail_ess_min"]))
        diag_rows.append({**ids, "observable": nm, **d.row(), "gate_pass": ok, "gate_reason": why,
                          "first_draw": 0, "last_draw": N - 1, "saved_stride": 1})
        if not ok:
            rank_fail.append(f"{nm}:{why}")
    fams = ctx.spec.families()
    probe_rows = []
    for fam, members in list(fams.items()) + [("control", ["q_A", "q_W", "U"])]:
        for nm in members:
            x = X[nm]
            ess, lag = dg.ess_mean_raw(x)
            probe_rows.append({**ids, "observable": nm, "family": fam, "estimate": tau_hat(x, h),
                               "estimate_over_sigma2": tau_hat(x, h) / ctx.sigma**2, "ess_mean_raw": ess,
                               "lag_window": lag, "h": h, "T_per_chain": T, "n_draws_used": int(x.size),
                               "constant": dg.is_constant(x)})
    bk = boot_kw(cfg)
    fam_rows, boots = [], {}
    for fam, members in fams.items():
        fr = analyze_family({m: X[m] for m in members}, members, h, ctx.folds,
                            seed=ctx.seed("shared", stage, f"bootstrap_relaxation|{fam}"), **bk)
        boots[fam] = fr.pop("replicates")
        reasons = []
        if not dg.rel(fr["mcse"], fr["estimate"]) <= float(g["family_relative_mcse_max"]):
            reasons.append("rel_mcse")
        if not fr["block_stable"]:
            reasons.append("block_instability")
        if not fr["enough_blocks"]:
            reasons.append("fewer_than_min_blocks")
        if not T >= float(g["duration_over_max_integrated_time_min"]) * fr["max_tau_full_pool"]:
            reasons.append("duration_lt_100_tau")
        fam_rows.append({**ids, "family": fam, **fr, "estimate_over_sigma2": fr["estimate"] / ctx.sigma**2,
                         "mc_low_over_sigma2": fr["mc_low"] / ctx.sigma**2,
                         "mc_high_over_sigma2": fr["mc_high"] / ctx.sigma**2,
                         "h": h, "T_per_chain": T, "n_draws_used": int(N * len(dat)),
                         "precision_pass": not reasons, "precision_reasons": ";".join(reasons)})
    ref = reference_pool(ctx)
    comp_rows, comp_fail = [], []
    for nm in ["V"] + fams["train_logit"] + fams["test_probability"]:
        x = X[nm]
        if nm not in ref:
            comp_fail.append(f"{nm}:no_reference")
            continue
        r = ref[nm]
        md, mr = float(x.mean()), float(r.mean())
        ed, er = dg.batch_means_mcse(x), dg.batch_means_mcse(r)
        sd_r = float(r.std(ddof=1))
        thr = float(g["reference_mean_mcse_multiplier"]) * np.hypot(ed, er) + float(g["reference_mean_sd_allowance"]) * sd_r
        ok = abs(md - mr) <= thr
        comp_rows.append({"observable": nm, "dynamics_mean": md, "reference_mean": mr, "diff": md - mr,
                          "dynamics_mcse": ed, "reference_mcse": er, "reference_sd": sd_r, "threshold": thr,
                          "pass": ok})
        if not ok:
            comp_fail.append(nm)
    ref_status = (ctx.status().get("reference") or {}).get("status")
    stage_pass = not rank_fail and not acc_fail and not comp_fail and all(r["precision_pass"] for r in fam_rows)
    for r in fam_rows:
        reasons = [x for x in [r["precision_reasons"]] if x]
        if rank_fail:
            reasons.append("rank_diagnostics")
        if acc_fail:
            reasons.append("acceptance_or_streak")
        if comp_fail:
            reasons.append("reference_mean_screen")
        if ref_status != "reference_pass":
            reasons.append(f"reference:{ref_status}")
        r["validity"] = not reasons
        r["failure_reason"] = ";".join(reasons)
    np.savez(ctx.dir / "analysis" / f"boot_{name}.npz", **{f: np.asarray(v) for f, v in boots.items()})
    res = {"execution_hash": ex, "h": h, "stage": k, "N_per_chain": N, "T_per_chain": T,
           "acceptance_per_chain": acc, "max_rejection_streak_per_chain": streaks, "acceptance_fail": acc_fail,
           "rank_failures": rank_fail, "reference_comparison": comp_rows, "reference_comparison_failures": comp_fail,
           "stage_pass": stage_pass, "families": fam_rows, "probes": probe_rows, "diagnostics": diag_rows,
           "grad_evals_trajectory": tr.grad_evals(), "wall_s_trajectory": tr.ck["elapsed_s"],
           "analysis_wall_s": time.time() - t0, "budget_exhausted": tr.budget_exhausted(),
           "reference_status": ref_status}
    ctx.save_analysis(name, res)
    return res


# ---- production / endpoint runs ----------------------------------------------------------------------
def run_dynamics(ctx: TargetContext, h: float, *, max_stage: Optional[int] = None) -> dict[str, Any]:
    """Production logic (extend until the stage passes) or, with ``max_stage``, a fixed-duration comparison run."""
    cfg, s2 = ctx.cfg, ctx.sigma**2
    d = cfg["dynamics"]
    ref = reference_final_state(ctx)
    tr = ctx.open_dynamics(h, ref)
    tr.run_segment("discard", C.n_steps(float(d["production_discard_time_over_sigma_squared"]) * s2, h))
    prev, res, k = 0, None, 0
    Ts = [float(x) * s2 for x in d["cumulative_retained_time_over_sigma_squared"]]
    for k, T in enumerate(Ts, 1):
        if max_stage is not None and k > max_stage:
            k -= 1
            break
        N = C.n_steps(T, h)
        rec = tr.run_segment(f"stage_{k}", N - prev)
        prev = N
        res = analyze_dynamics(ctx, tr, h, k)
        if rec["partial"] or tr.budget_exhausted():
            break
        if max_stage is None and (res["stage_pass"] or res["acceptance_fail"]):
            break
    summary = {"h": h, "h_id": h_id(h), "final_stage": k, "stage_pass": bool(res and res["stage_pass"]),
               "acceptance_fail": bool(res and res["acceptance_fail"]), "budget_exhausted": tr.budget_exhausted(),
               "T_per_chain": None if res is None else res["T_per_chain"], "mode": "fixed" if max_stage else "production",
               "grad_evals_run": tr.grad_evals()}
    runs = ctx.load_analysis("dynamics_runs") or {}
    runs[h_id(h)] = summary
    ctx.save_analysis("dynamics_runs", runs)
    return summary


def production_stage(ctx: TargetContext, h: Optional[float]) -> dict[str, Any]:
    """Array stage C: base production at the architecture's step, plus the h/2 run at endpoint widths."""
    if h is None:
        return ctx.update_status(dynamics={"status": "dynamics_calibration_failed_architecture"})
    base = run_dynamics(ctx, h)
    out = {"production": base}
    if C.is_endpoint(ctx.cfg, ctx.t):
        out["half_step"] = run_dynamics(ctx, h / 2.0, max_stage=base["final_stage"])
    return ctx.update_status(dynamics={"round": 1, "h": h, **out})


def refinement_stage(ctx: TargetContext, decision: dict[str, Any]) -> dict[str, Any]:
    arch = decision.get(ctx.t.arch) or {}
    if not arch.get("refine"):
        return ctx.update_status(refinement={"refined": False})
    h2 = float(arch["h"]) / 2.0
    base = run_dynamics(ctx, h2)
    out = {"production": base}
    if C.is_endpoint(ctx.cfg, ctx.t):
        out["half_step"] = run_dynamics(ctx, h2 / 2.0, max_stage=base["final_stage"])
    return ctx.update_status(refinement={"refined": True, "h": h2, **out})


# ---- endpoint decision --------------------------------------------------------------------------------
def _final(ctx_dir: Path, h: float) -> Optional[dict[str, Any]]:
    p = ctx_dir / "analysis" / "dynamics_runs.json"
    if not p.exists():
        return None
    runs = read_json(p)
    s = runs.get(h_id(h))
    if not s or not s.get("final_stage"):
        return None
    q = ctx_dir / "analysis" / f"dynamics_{h_id(h)}_stage_{s['final_stage']}.json"
    return read_json(q) if q.exists() else None


def endpoint_decision(cfg: dict[str, Any], root: Path, round_: int) -> dict[str, Any]:
    steps = read_json(root / "dynamics_step.json")
    prev = read_json(root / "endpoint_decision_round1.json") if round_ == 2 else None
    g, rel_max = cfg["dynamics_gates"], float(cfg["dynamics"]["step_validation_relative_difference_max"])
    out = {}
    for arch in ("shallow", "deep"):
        h0 = steps[arch]["h"]
        if h0 is None:
            out[arch] = {"h": None, "pass": False, "refine": False, "status": "dynamics_calibration_failed"}
            continue
        if round_ == 2 and not (prev or {}).get(arch, {}).get("refine"):
            out[arch] = {**prev[arch], "round": 2, "note": "no refinement was required"}
            continue
        h = h0 if round_ == 1 else h0 / 2.0
        rows, fails = [], []
        prod_fail = []
        for t in C.targets(cfg):
            if t.arch != arch:
                continue
            tdir = root / "targets" / t.target_id
            a = _final(tdir, h)
            if a is None:
                prod_fail.append(f"{t.target_id}:missing_production")
                continue
            if a["acceptance_fail"]:
                prod_fail.append(f"{t.target_id}:acceptance_or_streak")
            if not C.is_endpoint(cfg, t):
                continue
            b = _final(tdir, h / 2.0)
            if b is None or b["stage"] != a["stage"]:
                fails.append(f"{t.target_id}:missing_half_step")
                continue
            fa = {r["family"]: r for r in a["families"]}
            fb = {r["family"]: r for r in b["families"]}
            for fam in fa:
                ea, eb = fa[fam]["estimate"], fb[fam]["estimate"]
                avg = 0.5 * (ea + eb)
                rd = abs(ea - eb) / avg if avg else float("inf")
                ok = bool(fa[fam]["precision_pass"] and fb[fam]["precision_pass"] and rd <= rel_max)
                rows.append({"target_id": t.target_id, "replicate": t.rep, "m": t.m, "kind": "family", "family": fam,
                             "h": h, "h_half": h / 2.0, "estimate_h": ea, "estimate_h_half": eb,
                             "mcse_h": fa[fam]["mcse"], "mcse_h_half": fb[fam]["mcse"],
                             "relative_difference": rd, "threshold": rel_max,
                             "precision_h": fa[fam]["precision_pass"], "precision_h_half": fb[fam]["precision_pass"],
                             "pass": ok, "T_per_chain": a["T_per_chain"], "T_per_chain_h_half": b["T_per_chain"]})
                if not ok:
                    fails.append(f"{t.target_id}:{fam}")
            ref = {r["observable"]: r for r in a["reference_comparison"]}
            refb = {r["observable"]: r for r in b["reference_comparison"]}
            for nm in ["V"] + [f"train_logit_{i}" for i in cfg["probes"]["train_indices"]]:
                if nm not in ref or nm not in refb:
                    continue
                diff = ref[nm]["dynamics_mean"] - refb[nm]["dynamics_mean"]
                thr = 3.0 * np.hypot(ref[nm]["dynamics_mcse"], refb[nm]["dynamics_mcse"]) + 0.05 * ref[nm]["reference_sd"]
                ok = abs(diff) <= thr
                rows.append({"target_id": t.target_id, "replicate": t.rep, "m": t.m, "kind": "mean", "family": nm,
                             "h": h, "h_half": h / 2.0, "estimate_h": ref[nm]["dynamics_mean"],
                             "estimate_h_half": refb[nm]["dynamics_mean"], "mcse_h": ref[nm]["dynamics_mcse"],
                             "mcse_h_half": refb[nm]["dynamics_mcse"], "difference": diff, "threshold": thr,
                             "pass": bool(ok)})
                if not ok:
                    fails.append(f"{t.target_id}:{nm}")
        passed = not fails and not prod_fail
        refine = (round_ == 1 and not passed and int(cfg["dynamics"]["architecture_wide_refinements_max"]) >= 1)
        out[arch] = {"round": round_, "h": h, "pass": passed, "refine": refine, "endpoint_failures": fails,
                     "production_failures": prod_fail, "comparisons": rows,
                     "final_h": h if round_ == 2 or passed else h / 2.0,
                     "status": ("validated" if passed else "refine" if refine else "dynamics_unresolved")}
    atomic_write_json(root / f"endpoint_decision_round{round_}.json", clean_json(out))
    return out
