"""Analysis: validated source tables / loss windows -> every computed table the renderer and text use.

Spectral (guide §3, §9.2) and loss ACF (guide §5, §6, §9.3). Calculations go through the vendored
``figure_metrics`` helpers; this module adds only the project-specific checks, bookkeeping and the fixed display
rules (K extension, fixed-lag contrasts, selected-window sampling checks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import config as C
from .. import diagnostics as dg
from .common import ANALYSIS_SCOPE, OBSERVABLE, Layout, metrics, style
from .prepare import InputError, load_loss_npz

QUANTS = [("q50", "0.50"), ("q95", "0.95"), ("q99", "0.99")]


# ---- Figure 1 ---------------------------------------------------------------------------------------
def spectral_tables(cfg: dict[str, Any], root: Path, L: Layout) -> dict[str, Any]:
    fm = metrics()
    sty = style()["spectral"]
    states = pd.read_csv(L.tables / "spectral_states.csv.gz")
    prior = pd.read_csv(L.tables / "spectral_prior_states.csv")
    rows = []
    for (m, r), d in states.groupby(["m", "replicate_id"], sort=True):
        s = fm.spectral_target(d.S.to_numpy(), guard=float(cfg["spectral"]["normalized_boundary_guard"]))
        rows.append({"m": int(m), "replicate_id": int(r), "target_id": d.target_id.iloc[0],
                     "n_chains": int(d.chain_id.nunique()), "reference_status": d.reference_status.iloc[0], **s})
    rep = pd.DataFrame(rows)
    cols = ["m", "replicate_id", "n_inspected", "n_inside", "n_outside", "n_ambiguous", "inside_fraction_lower",
            "inside_fraction_upper", "q50", "q95", "q99", "max_S", "target_id", "n_chains", "reference_status"]
    rep = rep[cols]

    # cross-check against the campaign's own pooled per-target summary (same states, same quantile convention)
    summ = root / "tables" / "spectral_summary.csv"
    xcheck = []
    if summ.exists():
        cs = pd.read_csv(summ)
        cs = cs[cs.chain_id.astype(str) == "all"].set_index("target_id")
        for r in rep.itertuples():
            c = cs.loc[r.target_id]
            dq = max(abs(r.q50 - c["S_q0.5"]), abs(r.q95 - c["S_q0.95"]), abs(r.q99 - c["S_q0.99"]))
            ok = dq <= 1e-12 and int(c["inspected_states"]) == r.n_inspected and int(c["exits"]) == r.n_outside
            xcheck.append({"target_id": r.target_id, "max_abs_quantile_diff": dq, "counts_match": ok,
                           "campaign_backend_check_ok": bool(c.get("backend_check_ok", False)),
                           "campaign_backend_max_rel_diff": c.get("backend_max_rel_diff"),
                           "campaign_S_rhat": c.get("S_rhat"), "campaign_S_ess_bulk": c.get("S_ess_bulk")})
        if not all(x["counts_match"] for x in xcheck):
            raise InputError("recomputed spectral quantiles/counts disagree with the campaign summary")
    rep.to_csv(L.tables / "spectral_replicate_summary.csv", index=False)
    pd.DataFrame(xcheck).to_csv(L.tables / "spectral_crosscheck.csv", index=False)

    pr = []
    for m, d in prior.groupby("m", sort=True):
        q = np.quantile(d.S.to_numpy(), [0.5, 0.95, 0.99], method=sty["quantile_method"])
        pr.append({"m": int(m), "n_iid_prior": int(len(d)), "seed": int(d.seed.iloc[0]), "q50": q[0], "q95": q[1],
                   "q99": q[2], "normalization": d.normalization.iloc[0], "n_outside": int((d.S > 1).sum())})
    prs = pd.DataFrame(pr)
    prs.to_csv(L.tables / "spectral_prior_summary.csv", index=False)

    plot = []
    n_rep = len(cfg["replicates"])
    for m, d in rep.groupby("m", sort=True):
        if len(d) != n_rep:
            raise InputError(f"m={m}: {len(d)} replicates, expected {n_rep}")
        agg = fm.spectral_replicates(d.to_dict("records")) if n_rep == 3 else None
        for key, qlab in QUANTS:
            vals = d[key].to_numpy()
            plot.append({"m": int(m), "quantile": qlab,
                         "posterior_median": agg[key]["median"] if agg else float(np.median(vals)),
                         "replicate_min": agg[key]["replicate_min"] if agg else float(vals.min()),
                         "replicate_max": agg[key]["replicate_max"] if agg else float(vals.max()),
                         **{f"replicate_{int(rr)}": float(v) for rr, v in zip(d.replicate_id, vals)},
                         "prior": float(prs.set_index("m").loc[m, key])})
    pd.DataFrame(plot).to_csv(L.tables / "spectral_plot.csv", index=False)
    counts = {"n_inspected": int(rep.n_inspected.sum()), "n_outside": int(rep.n_outside.sum()),
              "n_ambiguous": int(rep.n_ambiguous.sum()), "n_targets": int(len(rep)),
              "widths": sorted(int(m) for m in rep.m.unique()), "n": int(cfg["data"]["train_size"]),
              "a": float(cfg["domain"]["spectral_cutoff_a"]), "per_target_inspected": sorted(set(rep.n_inspected)),
              "campaign_id": cfg["campaign_id"], "fixture": cfg["campaign_id"].startswith("fixture"),
              "targets_without_reference_pass": sorted(rep[rep.reference_status != "reference_pass"].target_id)}
    (L.tables / "spectral_counts.json").write_text(json.dumps(counts, indent=2))
    return counts


# ---- Figure 2 ---------------------------------------------------------------------------------------
def _reference_status(root: Path, tid: str) -> str:
    p = root / "targets" / tid / "status.json"
    return ((json.loads(p.read_text()).get("reference") or {}).get("status")) if p.exists() else "missing"


def load_windows(cfg: dict[str, Any], root: Path, L: Layout, *, window: int) -> dict[str, dict[str, Any]]:
    sty = style()["acf"]
    out = {}
    for t in C.targets(cfg):
        p = L.traces / f"{t.target_id}.npz"
        if not p.exists():
            raise InputError(f"{p} missing: run `python -m bnn_geometry export-loss-traces` on the cluster (§10.3)")
        sp = root / "targets" / t.target_id / "spec.json"
        spec = json.loads(sp.read_text()) if sp.exists() else None
        out[t.target_id] = {"t": t, **load_loss_npz(p, t=t, h=float(sty["h"]), window=window,
                                                     n=int(cfg["data"]["train_size"]), spec=spec)}
    if len({w["sampler_code_hash"] for w in out.values()}) != 1:
        raise InputError("the selected traces were produced by different sampler implementations")
    return out


def acf_tables(cfg: dict[str, Any], root: Path, L: Layout, *, window: int) -> dict[str, Any]:
    fm = metrics()
    sty = style()["acf"]
    W = load_windows(cfg, root, L, window=window)
    reps = [int(r) for r in cfg["replicates"]]
    archs = {a: [int(m) for m in cfg["model"][a]["widths"]] for a in ("shallow", "deep")}

    def run(K: int) -> dict[str, dict[str, Any]]:
        return {tid: fm.acf_replicate(w["V"], K, window) for tid, w in W.items()}

    def medians(res: dict[str, dict[str, Any]]) -> dict[tuple[str, int], dict[str, np.ndarray]]:
        agg = {}
        for a, ms in archs.items():
            for m in ms:
                curves = np.stack([res[C.Target(a, m, r).target_id]["replicate_acf"] for r in reps])
                agg[(a, m)] = fm.aggregate_replicates(curves) if len(reps) == 3 else {
                    "median": np.median(curves, 0), "replicate_min": curves.min(0), "replicate_max": curves.max(0)}
        return agg

    seq = [int(sty["initial_max_lag"])] + [int(k) for k in sty["extension_max_lags"]]
    thr = float(sty["extension_absolute_endpoint_threshold"])
    history = []
    for K in seq:
        if K >= window:
            history.append({"K": K, "skipped": f"window {window} too short"})
            break
        res = run(K)
        agg = medians(res)
        ends = {f"{a}_m{m}": float(v["median"][K]) for (a, m), v in agg.items()}
        history.append({"K": K, "median_at_K": ends})
        if max(abs(v) for v in ends.values()) <= thr:
            break
    K = history[-1]["K"] if "skipped" not in history[-1] else history[-2]["K"]
    unresolved_at_end = max(abs(v) for v in history[-1 if "skipped" not in history[-1] else -2]["median_at_K"].values()) > thr

    lags = np.arange(K + 1)
    chain_rows, rep_rows, plot_rows, fixed_rows = [], [], [], []
    for tid, w in W.items():
        t, r = w["t"], res[tid]
        for c in range(4):
            chain_rows.append(pd.DataFrame({
                "architecture": t.arch, "m": t.m, "replicate": t.rep, "chain": c, "lag": lags,
                "rho": r["chain_acf"][c], "N": window, "h": float(sty["h"]),
                "iteration_start": int(w["iteration_start"][c]), "iteration_end": int(w["iteration_end"][c]),
                "source_run": w["source_run_ids"][c]}))
        rep_rows.append(pd.DataFrame({"architecture": t.arch, "m": t.m, "replicate": t.rep, "lag": lags,
                                      "rho": r["replicate_acf"]}))
        for k in sty["fixed_result_lags"]:
            if k <= K:
                fixed_rows.append({"architecture": t.arch, "m": t.m, "replicate": t.rep, "lag": int(k),
                                   "rho_replicate_mean": float(r["replicate_acf"][k])})
    for (a, m), v in agg.items():
        plot_rows.append(pd.DataFrame({"architecture": a, "m": m, "lag": lags, "median": v["median"],
                                       "replicate_min": v["replicate_min"], "replicate_max": v["replicate_max"]}))
    pd.concat(chain_rows).to_csv(L.tables / "acf_chain.csv.gz", index=False)
    pd.concat(rep_rows).to_csv(L.tables / "acf_replicate.csv", index=False)
    pd.concat(plot_rows).to_csv(L.tables / "acf_plot.csv", index=False)
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(L.tables / "acf_fixed_lags.csv", index=False)

    contrasts = []
    for a, ms in archs.items():
        lo, hi = min(ms), max(ms)
        for k in sty["fixed_result_lags"]:
            if k > K:
                continue
            for r in reps:
                rl = float(res[C.Target(a, lo, r).target_id]["replicate_acf"][k])
                rh = float(res[C.Target(a, hi, r).target_id]["replicate_acf"][k])
                contrasts.append({"architecture": a, "replicate": r, "lag": int(k), "m_min": lo, "m_max": hi,
                                  "rho_m_min": rl, "rho_m_max": rh, "delta": rh - rl})
    pd.DataFrame(contrasts).to_csv(L.tables / "acf_width_contrasts.csv", index=False)

    diag = window_diagnostics(cfg, root, W, window=window)
    diag.to_csv(L.tables / "loss_window_diagnostics.csv", index=False)
    failed = diag[diag.check_status != "pass"]
    settings = {
        "analysis_scope": ANALYSIS_SCOPE, "observable": OBSERVABLE, "h": float(sty["h"]), "window": window,
        "stride": 1, "K": int(K), "K_history": history, "endpoint_threshold": thr,
        "loss_remains_correlated_at_max_K": bool(unresolved_at_end), "replicates": reps, "widths": archs,
        "n": int(cfg["data"]["train_size"]), "chains_per_replicate": 4,
        "acf_convention": sty["acf_convention"], "within_replicate": sty["within_replicate"],
        "across_replicate": sty["across_replicate"], "band": sty["band"],
        "fixed_step_loss_acf_status": "pass" if failed.empty else "sampling_unresolved",
        "failed_targets": failed[["target_id", "check_reasons"]].to_dict("records"),
        "previous_continuous_time_status_preserved": "dynamics_unresolved (original audit unchanged)",
        "campaign_id": cfg["campaign_id"], "fixture": cfg["campaign_id"].startswith("fixture"),
        "sampler_code_hash": next(iter(W.values()))["sampler_code_hash"]}
    (L.tables / "acf_settings.json").write_text(json.dumps(settings, indent=2))
    return settings


def window_diagnostics(cfg: dict[str, Any], root: Path, W: dict[str, dict[str, Any]], *, window: int) -> pd.DataFrame:
    """Selected-window checks on V only (guide §5.3); not the old integrated-time/entropy/step-halving gates."""
    rows = []
    for tid, w in W.items():
        V = w["V"]
        d = dg.scalar_diagnostics("V", V)
        reasons = []
        if V.shape != (4, window):
            reasons.append("shape")
        if np.any(np.ptp(V, axis=1) == 0):
            reasons.append("constant_chain")
        if not (d.rhat < 1.01):
            reasons.append(f"rhat={d.rhat:.4f}")
        if not (d.ess_bulk >= 1000):
            reasons.append(f"ess_bulk={d.ess_bulk:.0f}")
        ref = _reference_status(root, tid)
        if ref != "reference_pass":
            reasons.append(f"reference:{ref}")
        acc = w["accepted"].mean(axis=1) if w["accepted"] is not None else np.full(4, np.nan)
        # split-half mean shift per chain in units of its batch-means MCSE (visual-integrity flag, not a gate)
        half = window // 2
        drift = [abs(V[c, :half].mean() - V[c, half:].mean()) /
                 max(np.hypot(dg.batch_means_mcse(V[c:c + 1, :half]), dg.batch_means_mcse(V[c:c + 1, half:])), 1e-300)
                 for c in range(4)]
        rows.append({"target_id": tid, "architecture": w["t"].arch, "m": w["t"].m, "replicate": w["t"].rep,
                     "V_rhat": d.rhat, "V_ess_bulk": d.ess_bulk, "V_ess_tail": d.ess_tail, "V_mean": d.mean,
                     "V_sd": d.sd, "acceptance_window_mean": float(np.mean(acc)),
                     "acceptance_window_min_chain": float(np.min(acc)), "max_split_half_shift_mcse": float(max(drift)),
                     "drift_flag": bool(max(drift) > 4.0), "reference_status": ref, "N_per_chain": window,
                     "retained_per_chain": w["retained_per_chain"], "iteration_start": int(w["iteration_start"][0]),
                     "iteration_end": int(w["iteration_end"][0]), "npz_sha256": w["sha256"],
                     "check_status": "pass" if not reasons else "fail", "check_reasons": ";".join(reasons),
                     "analysis_scope": ANALYSIS_SCOPE})
    return pd.DataFrame(rows)


def appendix_table(cfg: dict[str, Any], root: Path, L: Layout, export_log: Optional[dict] = None) -> pd.DataFrame:
    """One row per target (guide §11.3)."""
    diag = pd.read_csv(L.tables / "loss_window_diagnostics.csv").set_index("target_id")
    spec = pd.read_csv(L.tables / "spectral_replicate_summary.csv").set_index("target_id")
    pp = root / "tables" / "predictive_scores.csv"
    pred = pd.read_csv(pp).set_index("target_id") if pp.exists() else None
    logs = {e["target_id"]: e for e in (export_log or {}).get("exported", [])}
    sty = style()["acf"]
    rows = []
    for t in C.targets(cfg):
        d = diag.loc[t.target_id]
        r = {"architecture": t.arch, "m": t.m, "replicate": t.rep,
             "reference_status": d.reference_status, "V_rhat": d.V_rhat, "V_ess_bulk": d.V_ess_bulk,
             "h": float(sty["h"]), "selected_transitions_per_chain": int(d.N_per_chain),
             "acceptance_fraction": d.acceptance_window_mean,
             "spectral_inspected": int(spec.loc[t.target_id, "n_inspected"]) if t.target_id in spec.index else None,
             "spectral_exits": int(spec.loc[t.target_id, "n_outside"]) if t.target_id in spec.index else None,
             "predictive_improvement_prior_minus_posterior_nls": float(pred.loc[t.target_id, "improvement"])
             if pred is not None and t.target_id in pred.index else None,
             "predictive_improvement_mcse": float(pred.loc[t.target_id, "improvement_mcse"])
             if pred is not None and t.target_id in pred.index else None,
             "loss_acf_check": d.check_status,
             "source_status": logs.get(t.target_id, {}).get("source_status", "recovered_existing_trace")}
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(L.tables / "appendix_table.csv", index=False)
    return df
