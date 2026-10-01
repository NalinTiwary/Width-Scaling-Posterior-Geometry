"""S1/S2 analysis, tables and figures from the exported supplement traces (supplementary_experiments.md §4-§8).

ACFs use the vendored ``supplement_metrics`` (per-chain, per-point normalization, zero-padded FFT, unadjusted);
diagnostics use the project's rank-normalized split R-hat / bulk / tail ESS. No sampling, fitting or integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import config as C
from .. import diagnostics as dg
from ..paper_export import h_dir_name, sha256_file
from .common import PROTOCOL_GATES, S1_LAGS, S2_DURATION, S2_TIMES, WINDOW, Layout, clean_figures, \
    supplement_metrics, write_json

K_STEPS = (400, 800, 1600, 3200)
T_STEPS = (2.0, 4.0, 8.0)


def _diag_row(name: str, x: np.ndarray, **ids) -> dict[str, Any]:
    d = dg.scalar_diagnostics(name, x)
    half = x.shape[1] // 2
    shift = [abs(x[c, :half].mean() - x[c, half:].mean()) /
             max(np.hypot(dg.batch_means_mcse(x[c:c + 1, :half]), dg.batch_means_mcse(x[c:c + 1, half:])), 1e-300)
             for c in range(x.shape[0])]
    reasons = []
    if d.status != "ok":
        reasons.append(d.status)
    if not d.rhat < PROTOCOL_GATES["rhat_max_exclusive"]:
        reasons.append(f"rhat={d.rhat:.4f}")
    if not d.ess_bulk >= PROTOCOL_GATES["bulk_ess_min"]:
        reasons.append(f"ess_bulk={d.ess_bulk:.0f}")
    if not d.ess_tail >= PROTOCOL_GATES["tail_ess_min"]:
        reasons.append(f"ess_tail={d.ess_tail:.0f}")
    if np.any(x.std(axis=1) < 1e-10):
        reasons.append("numerically_degenerate")
    return {**ids, "observable": name, "rhat": d.rhat, "ess_bulk": d.ess_bulk, "ess_tail": d.ess_tail,
            "mean": float(x.mean()), "sd": float(x.std()),
            "first_half_mean": float(x[:, :half].mean()), "second_half_mean": float(x[:, half:].mean()),
            "max_split_half_shift_mcse": float(max(shift)), "drift_flag": bool(max(shift) > 4.0),
            "valid": not reasons, "reasons": ";".join(reasons)}


def _streak(a: np.ndarray) -> int:
    best = cur = 0
    for v in a:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


# ---- S1 -----------------------------------------------------------------------------------------------
def s1_analysis(cfg: dict[str, Any], L: Layout, *, window: int = WINDOW) -> dict[str, Any]:
    sm = supplement_metrics()
    reps = [int(r) for r in cfg["replicates"]]
    per_point, rep_rows, diag, half_rows = [], [], [], []
    curves, point_curves = {}, {}
    K = K_STEPS[0]
    data = {}
    for t in C.targets(cfg):
        p = L.traces / f"s1_{t.target_id}.npz"
        z = np.load(p, allow_pickle=False)
        side = json.loads(p.with_suffix(".json").read_text())
        P, V, J = z["probabilities"], z["V"], [int(j) for j in z["test_indices"]]
        if P.shape != (4, window, len(J)) or V.shape != (4, window):
            raise ValueError(f"{t.target_id}: unexpected S1 shapes {P.shape} {V.shape}")
        if not np.all(np.diff(z["iteration"], axis=1) == 1):
            raise ValueError(f"{t.target_id}: iteration ids not consecutive")
        if side["npz_sha256"] != sha256_file(p):
            raise ValueError(f"{t.target_id}: S1 trace hash differs from its export record")
        data[t.target_id] = (t, P, V, J, z["accepted"])
        ids = {"target_id": t.target_id, "architecture": t.arch, "m": t.m, "replicate": t.rep, "h": 0.01,
               "window": window, "iteration_start": side["iteration_start"], "iteration_end": side["iteration_end"]}
        acc = z["accepted"]
        common = {"acceptance": float(acc.mean()), "max_rejection_streak": max(_streak(a) for a in acc)}
        diag.append({**_diag_row("V", V, **ids), **common})
        for j, jj in enumerate(J):
            diag.append({**_diag_row(f"test_prob_{jj}", P[:, :, j], **ids), **common})
    # lag horizon: the same K for both architectures, extended while any median mean8 |rho(K)| > 0.10
    hist = []
    for K in K_STEPS:
        curves.clear(); point_curves.clear()
        for tid, (t, P, V, J, _) in data.items():
            r = sm.probability_acfs(P, K)
            curves[t.arch, t.m, t.rep] = r["mean8"]
            point_curves[t.arch, t.m, t.rep] = r["per_point"]
        ends = {}
        for arch in ("shallow", "deep"):
            for m in cfg["model"][arch]["widths"]:
                ends[f"{arch}_m{m}"] = float(np.median([curves[arch, int(m), r][K] for r in reps]))
        hist.append({"K": K, "median_mean8_at_K": ends})
        if max(abs(v) for v in ends.values()) <= 0.10:
            break
    for (arch, m, r), c in curves.items():
        J = data[C.Target(arch, m, r).target_id][3]
        for j, jj in enumerate(J):
            pc = point_curves[arch, m, r][j]
            per_point += [{"architecture": arch, "width": m, "replicate": r, "test_index": jj, "lag": k,
                           "acf": float(pc[k])} for k in range(K + 1)]
        rep_rows += [{"architecture": arch, "width": m, "replicate": r, "lag": k, "acf": float(c[k])}
                     for k in range(K + 1)]
    pd.DataFrame(per_point).to_csv(L.tables / "prediction_acf_per_point.csv.gz", index=False)
    pd.DataFrame(rep_rows).to_csv(L.tables / "prediction_acf_replicates.csv", index=False)
    # paired narrow-minus-wide differences (positive = lower autocorrelation at the widest width)
    diffs = []
    for arch in ("shallow", "deep"):
        ws = [int(m) for m in cfg["model"][arch]["widths"]]
        lo, hi = ws[0], ws[-1]
        for r in reps:
            J = data[C.Target(arch, lo, r).target_id][3]
            for k in S1_LAGS:
                if k > K:
                    continue
                pos = 0
                for j, jj in enumerate(J):
                    a, b = point_curves[arch, lo, r][j][k], point_curves[arch, hi, r][j][k]
                    pos += int(a - b > 0)
                    diffs.append({"architecture": arch, "replicate": r, "test_index": str(jj), "lag": k,
                                  "width_narrow": lo, "width_wide": hi, "acf_narrow": float(a), "acf_wide": float(b),
                                  "difference": float(a - b)})
                a, b = curves[arch, lo, r][k], curves[arch, hi, r][k]
                diffs.append({"architecture": arch, "replicate": r, "test_index": "mean8", "lag": k, "width_narrow": lo,
                              "width_wide": hi, "acf_narrow": float(a), "acf_wide": float(b), "difference": float(a - b),
                              "n_points_positive": pos, "n_points": len(J)})
    dfr = pd.DataFrame(diffs)
    for c in ("n_points_positive", "n_points"):
        dfr[c] = dfr[c].astype("Int64")
    dfr.to_csv(L.tables / "prediction_width_differences.csv", index=False)
    dd = pd.DataFrame(diag)
    settings = {"K": K, "K_history": hist, "window": window, "h": 0.01, "endpoint_threshold": 0.10,
                "remains_correlated_at_max_K": bool(K == K_STEPS[-1] and max(abs(v) for v in hist[-1]
                                                    ["median_mean8_at_K"].values()) > 0.10),
                "estimator": "per chain and test point: mean-subtracted unadjusted ACF (zero-padded FFT); mean over "
                             "4 chains then 8 points per replicate; pointwise median/min/max over 3 replicates",
                "all_series_valid": bool(dd.valid.all()), "n_series": int(len(dd)), "n_invalid": int((~dd.valid).sum()),
                "n_drift_flags": int(dd.drift_flag.sum())}
    write_json(L.tables / "prediction_acf_settings.json", settings)
    return {"settings": settings, "diagnostics": dd}


# ---- S2 -----------------------------------------------------------------------------------------------
def s2_analysis(cfg: dict[str, Any], root: Path, L: Layout, *, duration: float = S2_DURATION) -> dict[str, Any]:
    from .supplement_export import deep_final_h
    sm = supplement_metrics()
    reps = [int(r) for r in cfg["replicates"]]
    h0 = deep_final_h(root)
    steps = [h0, h0 / 2.0]
    lo, hi = C.endpoint_widths(cfg, "deep")
    series, diag = {}, []
    for m in (lo, hi):
        for r in reps:
            t = C.Target("deep", m, r)
            for h in steps:
                p = L.traces / f"s2_{t.target_id}_{h_dir_name(h)}.npz"
                z = np.load(p, allow_pickle=False)
                side = json.loads(p.with_suffix(".json").read_text())
                if side["npz_sha256"] != sha256_file(p):
                    raise ValueError(f"{p.name}: hash differs from its export record")
                V = sm.duration_window(z["V"], h, duration)
                if V.shape[1] != z["V"].shape[1]:
                    raise ValueError(f"{p.name}: exported window is not exactly the matched duration")
                series[m, r, h] = V
                acc = z["accepted"]
                diag.append({**_diag_row("V", V, target_id=t.target_id, architecture="deep", m=m, replicate=r, h=h,
                                         duration=duration, transitions=int(V.shape[1]),
                                         iteration_start=side["iteration_start"], iteration_end=side["iteration_end"]),
                             "acceptance": float(acc.mean()), "max_rejection_streak": max(_streak(a) for a in acc)})
    hist = []
    for tmax in T_STEPS:
        curves = {}
        for (m, r, h), V in series.items():
            kmax = int(sm.exact_lag_indices([tmax], h, V.shape[1] - 1)[0])
            curves[m, r, h] = sm.loss_acfs(V, kmax)["mean4"]
        ends = {f"m{m}_h{h:g}": float(np.median([curves[m, r, h][-1] for r in reps])) for m in (lo, hi) for h in steps}
        hist.append({"t_max": tmax, "median_at_t_max": ends})
        if max(abs(v) for v in ends.values()) <= 0.10:
            break
    rows = []
    for (m, r, h), c in curves.items():
        rows += [{"width": m, "replicate": r, "h": h, "lag": k, "algorithmic_lag": k * h, "acf": float(c[k])}
                 for k in range(len(c))]
    pd.DataFrame(rows).to_csv(L.tables / "step_acf_replicates.csv", index=False)
    diffs = []
    for h in steps:
        for tt in S2_TIMES:
            if tt > tmax:
                continue
            k = int(sm.exact_lag_indices([tt], h, len(curves[lo, reps[0], h]) - 1)[0])
            for r in reps:
                a, b = curves[lo, r, h][k], curves[hi, r, h][k]
                diffs.append({"replicate": r, "h": h, "algorithmic_lag": tt, "lag": k, "width_narrow": lo,
                              "width_wide": hi, "acf_narrow": float(a), "acf_wide": float(b), "difference": float(a - b)})
    pd.DataFrame(diffs).to_csv(L.tables / "step_width_differences.csv", index=False)
    dd = pd.DataFrame(diag)
    settings = {"t_max": tmax, "t_history": hist, "duration": duration, "steps": steps, "widths": [lo, hi],
                "all_series_valid": bool(dd.valid.all()), "n_invalid": int((~dd.valid).sum()),
                "n_drift_flags": int(dd.drift_flag.sum())}
    write_json(L.tables / "step_acf_settings.json", settings)
    return {"settings": settings, "diagnostics": dd}


# ---- rendering (cleanup-guide style) -------------------------------------------------------------------
def _style(cf, cfg, arch, m):
    """Paper colour/marker for width m; fixture widths borrow the style of the same position."""
    if m in cf.WIDTH_STYLE:
        return cf.WIDTH_STYLE[m]
    ws = [int(w) for w in cfg["model"][arch]["widths"]]
    return cf.WIDTH_STYLE[cf.WIDTHS[arch][ws.index(m)]]


def _handles(cf, cfg, arch, widths, size):
    return [cf.legend_handle(*_style(cf, cfg, arch, m), size=size) for m in widths]


def render_s1(cfg, L: Layout, width_in: float) -> list[Path]:
    cf = clean_figures()
    st = json.loads((L.tables / "prediction_acf_settings.json").read_text())
    K = int(st["K"])
    d = pd.read_csv(L.tables / "prediction_acf_replicates.csv")
    lower = -0.08
    fig, axes = cf.base_figure(width_in)
    x = np.arange(K + 1)
    ymin = 0.0
    for ax, arch, title, center in zip(axes, ("shallow", "deep"), ("(a) Shallow", "(b) Deep"), (0.3075, 0.7725)):
        ax.set_title(title, loc="left", pad=6)
        ws = [int(m) for m in cfg["model"][arch]["widths"]]
        for m in ws:
            color, marker = _style(cf, cfg, arch, m)
            arr = np.stack([d[(d.architecture == arch) & (d.width == m) & (d.replicate == r)].sort_values("lag").acf
                            .to_numpy() for r in cfg["replicates"]])
            ymin = min(ymin, float(arr.min()))
            ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=0.10, linewidth=0, zorder=1)
            ax.plot(x, np.median(arr, axis=0), color=color, linewidth=1.6, marker=marker, markersize=3.2,
                    markeredgewidth=0.5, markevery=list(range(K // 8, K + 1, K // 8)), zorder=3)
        ax.axhline(0, color="#777777", linewidth=0.65, zorder=2)
        ax.set_xlim(0, K)
        ax.set_xticks(np.linspace(0, K, 5))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Lag (sampler iterations)")
        ordered = [ws[i] for i in (0, 2, 1, 3)] if len(ws) == 4 else ws
        fig.legend(_handles(cf, cfg, arch, ordered, 3.2), [f"$m={m}$" for m in ordered], ncols=2, loc="center",
                   bbox_to_anchor=(center, 0.075), frameon=False, columnspacing=1.3, handletextpad=0.5)
    if ymin < lower:
        lower = float(np.floor((ymin - 0.02) / 0.05) * 0.05)
    for ax in axes:
        ax.set_ylim(lower, 1.02)
    axes[0].set_ylabel("Mean prediction autocorrelation")
    cf.check_legends(fig, 2)
    meta = {"display_lags": [0, K], "ylim": [lower, 1.02], "source": "tables/prediction_acf_replicates.csv",
            "aggregation": "pointwise median and min/max of 3 replicate curves (each: mean over 8 points of "
                           "4-chain-mean ACFs)"}
    out = L.figures / "supplement_s1_prediction_acf"
    cf.write_outputs(fig, out, str(L.tables / "prediction_acf_replicates.csv"), meta)
    return [out.with_suffix(".pdf"), out.with_suffix(".png")]


def render_s2(cfg, L: Layout, width_in: float) -> list[Path]:
    cf = clean_figures()
    st = json.loads((L.tables / "step_acf_settings.json").read_text())
    tmax, steps, (lo, hi) = float(st["t_max"]), st["steps"], st["widths"]
    d = pd.read_csv(L.tables / "step_acf_replicates.csv")
    fig, axes = cf.base_figure(width_in)
    ymin = 0.0
    for ax, h, title in zip(axes, steps, (f"(a) $h={steps[0]:g}$", f"(b) $h={steps[1]:g}$")):
        ax.set_title(title, loc="left", pad=6)
        for m in (lo, hi):
            color, marker = _style(cf, cfg, "deep", m)
            g = d[(d.width == m) & np.isclose(d.h, h)]
            arr = np.stack([g[g.replicate == r].sort_values("lag").acf.to_numpy() for r in cfg["replicates"]])
            t = g[g.replicate == cfg["replicates"][0]].sort_values("lag").algorithmic_lag.to_numpy()
            ymin = min(ymin, float(arr.min()))
            n = len(t) - 1
            ax.fill_between(t, arr.min(0), arr.max(0), color=color, alpha=0.10, linewidth=0, zorder=1)
            ax.plot(t, np.median(arr, axis=0), color=color, linewidth=1.6, marker=marker, markersize=3.2,
                    markeredgewidth=0.5, markevery=list(range(n // 8, n + 1, n // 8)), zorder=3)
        ax.axhline(0, color="#777777", linewidth=0.65, zorder=2)
        ax.set_xlim(0, tmax)
        ax.set_xticks(np.linspace(0, tmax, 5))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Algorithmic lag $kh$")
    lower = -0.08 if ymin >= -0.08 else float(np.floor((ymin - 0.02) / 0.05) * 0.05)
    for ax in axes:
        ax.set_ylim(lower, 1.02)
    axes[0].set_ylabel("Loss autocorrelation")
    fig.legend(_handles(cf, cfg, "deep", [lo, hi], 3.2), [f"$m={lo}$", f"$m={hi}$"], ncols=2, loc="center",
               bbox_to_anchor=(0.54, 0.075), frameon=False, columnspacing=1.3, handletextpad=0.5)
    cf.check_legends(fig, 1)
    meta = {"display_algorithmic_lag": [0, tmax], "ylim": [lower, 1.02], "source": "tables/step_acf_replicates.csv",
            "aggregation": "pointwise median and min/max of 3 replicate four-chain-mean loss ACFs"}
    out = L.figures / "supplement_s2_step_size"
    cf.write_outputs(fig, out, str(L.tables / "step_acf_replicates.csv"), meta)
    return [out.with_suffix(".pdf"), out.with_suffix(".png")]


# ---- appendix table, diagnostics, recovery log ----------------------------------------------------------
def appendix_table(cfg, root: Path, L: Layout, s1_diag: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reference-pool predictive improvement and h=0.01 dynamics-window diagnostics, labelled separately."""
    pf = root / "paper_figures" / "tables"
    loss = pd.read_csv(pf / "loss_window_diagnostics.csv")
    pred = pd.read_csv(root / "tables" / "predictive_scores.csv")
    rows = []
    for t in C.targets(cfg):
        pr = pred[pred.target_id == t.target_id].iloc[0]
        lw = loss[loss.target_id == t.target_id].iloc[0]
        row = {"target_id": t.target_id, "architecture": t.arch, "m": t.m, "replicate": t.rep,
               "reference_predictive_improvement": float(pr["improvement"]),
               "reference_predictive_improvement_mcse": float(pr["mcse"]),
               "dynamics_loss_rhat": float(lw.V_rhat), "dynamics_loss_ess_bulk": float(lw.V_ess_bulk)}
        if s1_diag is not None:
            g = s1_diag[(s1_diag.target_id == t.target_id) & s1_diag.observable.str.startswith("test_prob_")]
            row.update({"dynamics_prob_rhat_max": float(g.rhat.max()),
                        "dynamics_prob_ess_bulk_min": float(g.ess_bulk.min()),
                        "dynamics_prob_ess_tail_min": float(g.ess_tail.min())})
        rows.append(row)
    full = pd.DataFrame(rows)
    full.to_csv(L.tables / "appendix_diagnostics_per_target.csv", index=False)
    agg = []
    for (arch, m), g in full.groupby(["architecture", "m"], sort=False):
        pi = g.reference_predictive_improvement
        r = {"architecture": arch, "m": int(m), "reference_predictive_improvement_median": float(pi.median()),
             "reference_predictive_improvement_min": float(pi.min()),
             "reference_predictive_improvement_max": float(pi.max()),
             "dynamics_loss_rhat_max": float(g.dynamics_loss_rhat.max()),
             "dynamics_loss_ess_bulk_min": float(g.dynamics_loss_ess_bulk.min())}
        if s1_diag is not None:
            r.update({"dynamics_prob_rhat_max": float(g.dynamics_prob_rhat_max.max()),
                      "dynamics_prob_ess_bulk_min": float(g.dynamics_prob_ess_bulk_min.min())})
        agg.append(r)
    t8 = pd.DataFrame(agg)
    t8.to_csv(L.tables / "appendix_diagnostics.csv", index=False)
    return t8


def build_supplements(cfg, root: Path, L: Layout, *, width_in: float, window: int = WINDOW,
                      duration: float = S2_DURATION, parts=("s1", "s2")) -> dict[str, Any]:
    L.mkdirs()
    out, diags = {}, []
    s1 = s1_analysis(cfg, L, window=window) if "s1" in parts else None
    s2 = s2_analysis(cfg, root, L, duration=duration) if "s2" in parts else None
    if s1:
        diags.append(s1["diagnostics"].assign(analysis="S1_prediction_acf"))
        out["s1"] = s1["settings"]
        out["s1_figures"] = [str(p) for p in render_s1(cfg, L, width_in)]
    if s2:
        diags.append(s2["diagnostics"].assign(analysis="S2_step_size"))
        out["s2"] = s2["settings"]
        out["s2_figures"] = [str(p) for p in render_s2(cfg, L, width_in)]
    if diags:
        pd.concat(diags, ignore_index=True).to_csv(L.tables / "supplement_diagnostics.csv", index=False)
    appendix_table(cfg, root, L, s1["diagnostics"] if s1 else None)
    log_p = L.out / "supplement_export_log.json"
    exp = json.loads(log_p.read_text()) if log_p.exists() else {}
    rec = {"policy": "reuse only; no posterior sampling, replay or forward recomputation of intermediate states",
           "entries": [{"analysis": "S1", "target_id": e["target_id"], "h": e["h"], "status": e["status"],
                        "window": [e["iteration_start"], e["iteration_end"]]} for e in exp.get("s1", [])] +
                      [{"analysis": "S2", "target_id": e["target_id"], "h": e["h"], "status": e["status"],
                        "window": [e["iteration_start"], e["iteration_end"]]} for e in exp.get("s2", [])],
           "new_trajectories": 0, "export_failures": exp.get("failures", [])}
    write_json(L.out / "recovery_log.json", rec)
    return out
