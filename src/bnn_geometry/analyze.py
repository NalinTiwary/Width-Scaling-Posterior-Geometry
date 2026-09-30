"""Campaign tables from the per-target analysis records (runbook §16). Figures read only these tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd

from . import config as C
from . import diagnostics as dg
from .context import h_id
from .storage import atomic_write_json, clean_json, read_json

TABLES = ("target_audit", "reference_diagnostics", "dynamics_diagnostics", "relaxation_probes",
          "relaxation_families", "entropy_probes", "entropy_families", "static_pi", "spectral_states",
          "spectral_summary", "predictive_scores", "cost_and_counts", "exclusions", "endpoint_comparisons",
          "width_ratios")


def _j(p: Path) -> Optional[Any]:
    return read_json(p) if p.exists() else None


def _flat(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, default=str)
    return v


def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{k: _flat(v) for k, v in r.items()} for r in rows])


def final_steps(root: Path) -> dict[str, Any]:
    r2 = _j(root / "endpoint_decision_round2.json")
    r1 = _j(root / "endpoint_decision_round1.json")
    st = _j(root / "dynamics_step.json") or {}
    out = {}
    for arch in ("shallow", "deep"):
        d = (r2 or {}).get(arch) or (r1 or {}).get(arch) or {}
        h = d.get("final_h", (st.get(arch) or {}).get("h"))
        out[arch] = {"h": h, "status": d.get("status", "not_evaluated"),
                     "refined": bool((r1 or {}).get(arch, {}).get("refine")),
                     "initial_h": (st.get(arch) or {}).get("h")}
    return out


def _dyn_final(tdir: Path, h: Optional[float]) -> Optional[dict[str, Any]]:
    if h is None:
        return None
    runs = _j(tdir / "analysis" / "dynamics_runs.json") or {}
    s = runs.get(h_id(h))
    if not s or not s.get("final_stage"):
        return None
    return _j(tdir / "analysis" / f"dynamics_{h_id(h)}_stage_{s['final_stage']}.json")


def _boot(tdir: Path, name: str, key: str) -> Optional[np.ndarray]:
    p = tdir / "analysis" / name
    if not p.exists():
        return None
    z = np.load(p)
    return z[key] if key in z.files else None


def analyze(cfg: dict[str, Any], root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    steps = final_steps(root)
    tabs: dict[str, list[dict[str, Any]]] = {k: [] for k in TABLES}
    sigma = float(cfg["model"]["sigma"])
    for t in C.targets(cfg):
        tdir = root / "targets" / t.target_id
        spec = _j(tdir / "spec.json") or {}
        status = _j(tdir / "status.json") or {}
        ref = status.get("reference") or {}
        k = ref.get("stopping_stage")
        rs = _j(tdir / "analysis" / f"reference_stage_{k}.json") if k else None
        errs = _j(tdir / "errors.json") or []
        base = {"campaign_id": cfg["campaign_id"], "target_id": t.target_id, "replicate": t.rep,
                "architecture": t.arch, "L": t.L, "m": t.m, "p": t.p(int(cfg["data"]["input_dimension"])),
                "n": int(cfg["data"]["train_size"]), "d": int(cfg["data"]["input_dimension"]), "sigma": sigma,
                "target_hash": spec.get("target_hash")}
        # reference diagnostics
        if rs:
            tabs["reference_diagnostics"] += rs["diagnostics"]
        # static entropy / PI
        static = (rs or {}).get("static") or {}
        tabs["entropy_probes"] += static.get("entropy_probes", [])
        tabs["entropy_families"] += static.get("entropy_families", [])
        tabs["static_pi"] += static.get("static_pi", [])
        # spectral
        spec_sum = ((rs or {}).get("spectral") or {}).get("summary")
        if spec_sum:
            tabs["spectral_summary"] += spec_sum["per_chain"] + [spec_sum["target"]]
            z = np.load(tdir / "analysis" / "spectral_states.npz")
            for c in cfg["chain_ids"]:
                S, dr, mem = z[f"S_{c}"], z[f"draw_{c}"], z[f"membership_{c}"]
                guard = float(cfg["spectral"]["normalized_boundary_guard"])
                for s_, d_, m_ in zip(S, dr, mem):
                    tabs["spectral_states"].append({**base, "execution_hash": ref.get("execution_hash"), "chain_id": c,
                                                    "draw": int(d_), "S": float(s_), "membership": int(m_),
                                                    "near_boundary": bool(abs(s_ - 1.0) <= guard),
                                                    "backend": "torch.linalg.svdvals(float64)"})
        # dynamics
        fh = steps[t.arch]["h"]
        dyn = _dyn_final(tdir, fh)
        runs = _j(tdir / "analysis" / "dynamics_runs.json") or {}
        for hid, summ in runs.items():
            a = _j(tdir / "analysis" / f"dynamics_{hid}_stage_{summ['final_stage']}.json") if summ.get("final_stage") else None
            if not a:
                continue
            role = "production_final" if fh is not None and hid == h_id(fh) else "validation_or_superseded"
            for r in a["probes"]:
                tabs["relaxation_probes"].append({**r, "role": role})
            for r in a["diagnostics"]:
                tabs["dynamics_diagnostics"].append({**r, "role": role})
            for r in a["families"]:
                row = {**r, "role": role, "architecture_dynamics_status": steps[t.arch]["status"],
                       "acceptance_min": min(a["acceptance_per_chain"]),
                       "max_rejection_streak": max(a["max_rejection_streak_per_chain"]),
                       "grad_evals_target_total": summ.get("grad_evals_target_total")}
                if role == "production_final" and steps[t.arch]["status"] not in ("validated",):
                    row["validity"] = False
                    row["failure_reason"] = (row.get("failure_reason") or "") + \
                        f";architecture_dynamics_{steps[t.arch]['status']}"
                tabs["relaxation_families"].append(row)
        # predictive
        pr = _j(tdir / "analysis" / "predictive.json")
        if pr:
            tabs["predictive_scores"].append({**base, **pr,
                                              "validity": pr.get("status") == "ok" and ref.get("status") == "reference_pass",
                                              "estimate": pr.get("improvement"), "mcse": pr.get("improvement_mcse"),
                                              "mc_low": pr.get("improvement_mc_low"), "mc_high": pr.get("improvement_mc_high")})
        # costs
        tabs["cost_and_counts"].append({
            **base, "reference_status": ref.get("status"), "reference_production_per_chain": ref.get("production_per_chain"),
            "reference_transitions_per_chain": _flat(ref.get("transitions_per_chain")),
            "reference_likelihood_evals_per_chain": _flat(ref.get("likelihood_evals_per_chain")),
            "reference_wall_s": ref.get("wall_s"), "device": ref.get("device"),
            "dynamics_grad_evals_total": max([s.get("grad_evals_target_total") or 0 for s in runs.values()] or [0]),
            "probe_gradient_evals_cumulative": static.get("probe_gradient_evals_cumulative"),
            "probe_gradient_evals_final_subset": static.get("probe_gradient_evals_final_subset"),
            "static_selected_per_chain": static.get("per_chain_selected"),
            "dynamics_final_T_per_chain": None if dyn is None else dyn["T_per_chain"],
            "dynamics_final_N_per_chain": None if dyn is None else dyn["N_per_chain"],
            "dynamics_final_h": fh, "dynamics_wall_s": None if dyn is None else dyn["wall_s_trajectory"]})
        # target audit
        ent_fams = static.get("entropy_families", [])
        tabs["target_audit"].append({
            **base, "head_center_norm": spec.get("head_center_norm"), "reference_status": ref.get("status"),
            "reference_budget_exhausted": ref.get("budget_exhausted"),
            "step_calibration": (status.get("step_calibration") or {}).get("status"),
            "dynamics_final_h": fh, "dynamics_architecture_status": steps[t.arch]["status"],
            "dynamics_stage_pass": None if dyn is None else dyn["stage_pass"],
            "entropy_families_valid": sum(1 for f in ent_fams if f.get("validity")),
            "entropy_families_total": len(ent_fams),
            "static_pass": ref.get("static_pass"),
            "spectral_inspected": None if not spec_sum else spec_sum["target"]["inspected_states"],
            "spectral_exits": None if not spec_sum else spec_sum["target"]["exits"],
            "predictive_status": None if not pr else pr.get("status"),
            "errors": len(errs), "law_static": "conditional_G_2.5" if t.arch == "deep" else "full_posterior"})
        for e in errs:
            tabs["exclusions"].append({**base, "experiment": e["stage"], "reason": e["error"]})
        if ref.get("status") != "reference_pass":
            tabs["exclusions"].append({**base, "experiment": "reference", "reason": ref.get("status") or "missing"})
    # endpoint comparisons
    for rnd in (1, 2):
        d = _j(root / f"endpoint_decision_round{rnd}.json") or {}
        for arch, v in d.items():
            for r in v.get("comparisons", []):
                tabs["endpoint_comparisons"].append({"round": rnd, "architecture": arch, **r})
    # exclusions for invalid main-result rows
    for tab, exp in (("relaxation_families", "relaxation"), ("entropy_families", "entropy"), ("static_pi", "static_pi")):
        for r in tabs[tab]:
            if tab == "relaxation_families" and r.get("role") != "production_final":
                continue
            if tab == "static_pi" and r.get("kind") != "family":
                continue
            if not r.get("validity"):
                tabs["exclusions"].append({"campaign_id": cfg["campaign_id"], "target_id": r["target_id"],
                                           "experiment": f"{exp}:{r.get('family')}", "reason": r.get("failure_reason")})
    tabs["width_ratios"] = width_ratios(cfg, root, steps, tabs)
    out = {k: _df(v) for k, v in tabs.items()}
    (root / "tables").mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        df.to_csv(root / "tables" / f"{k}.csv", index=False)
    consolidate_prior_predictive(cfg, root)
    atomic_write_json(root / "tables" / "analysis_metadata.json",
                      clean_json({"quantile_method": dg.QUANTILE_METHOD, "final_steps": steps,
                                  "code_revision": C.code_revision(), "tables": list(out)}))
    return out


def width_ratios(cfg, root: Path, steps, tabs) -> list[dict[str, Any]]:
    """Widest/narrowest family ratios per replicate, with the ratio of independent bootstrap replicates."""
    rows = []
    pct = [float(x) for x in cfg["uncertainty"]["interval_percentiles"]]

    def add(exp, arch, rep, fam, a, b, ba, bb, va, vb, law):
        r = b / a if a else float("nan")
        row = {"experiment": exp, "architecture": arch, "replicate": rep, "family": fam, "law": law,
               "narrow_estimate": a, "wide_estimate": b, "estimate": r, "validity": bool(va and vb)}
        if ba is not None and bb is not None and len(ba) and len(bb):
            n = min(len(ba), len(bb))
            rr = bb[:n] / ba[:n]
            rr = rr[np.isfinite(rr)]
            if rr.size > 1:
                row.update(mcse=float(rr.std(ddof=1)), mc_low=float(np.percentile(rr, pct[0], method="linear")),
                           mc_high=float(np.percentile(rr, pct[1], method="linear")))
        rows.append(row)

    for arch in ("shallow", "deep"):
        lo, hi = C.endpoint_widths(cfg, arch)
        for rep in cfg["replicates"]:
            tl, th = C.Target(arch, lo, int(rep)).target_id, C.Target(arch, hi, int(rep)).target_id
            fh = steps[arch]["h"]
            if fh is not None:
                for fam in ("loss", "train_logit", "test_probability", "interaction"):
                    ra = [r for r in tabs["relaxation_families"] if r["target_id"] == tl and r["family"] == fam
                          and r["role"] == "production_final"]
                    rb = [r for r in tabs["relaxation_families"] if r["target_id"] == th and r["family"] == fam
                          and r["role"] == "production_final"]
                    if ra and rb:
                        sa = _j(root / "targets" / tl / "analysis" / "dynamics_runs.json")[h_id(fh)]["final_stage"]
                        sb = _j(root / "targets" / th / "analysis" / "dynamics_runs.json")[h_id(fh)]["final_stage"]
                        add("relaxation", arch, rep, fam, ra[0]["estimate"], rb[0]["estimate"],
                            _boot(root / "targets" / tl, f"boot_dynamics_{h_id(fh)}_stage_{sa}.npz", fam),
                            _boot(root / "targets" / th, f"boot_dynamics_{h_id(fh)}_stage_{sb}.npz", fam),
                            ra[0]["validity"], rb[0]["validity"], "full_posterior")
            for fam in ("loss", "train_logit", "interaction"):
                law = "conditional_G_2.5" if arch == "deep" else "full_posterior"
                ra = [r for r in tabs["entropy_families"] if r["target_id"] == tl and r["family"] == fam]
                rb = [r for r in tabs["entropy_families"] if r["target_id"] == th and r["family"] == fam]
                if ra and rb:
                    add("entropy", arch, rep, fam, ra[0].get("estimate"), rb[0].get("estimate"),
                        _boot(root / "targets" / tl, "boot_static.npz", f"entropy_{fam}"),
                        _boot(root / "targets" / th, "boot_static.npz", f"entropy_{fam}"),
                        ra[0]["validity"], rb[0]["validity"], law)
                pa = [r for r in tabs["static_pi"] if r["target_id"] == tl and r["family"] == fam and r["kind"] == "family"]
                pb = [r for r in tabs["static_pi"] if r["target_id"] == th and r["family"] == fam and r["kind"] == "family"]
                if pa and pb:
                    add("static_pi", arch, rep, fam, pa[0]["estimate"], pb[0]["estimate"],
                        _boot(root / "targets" / tl, "boot_static.npz", f"pi_{fam}"),
                        _boot(root / "targets" / th, "boot_static.npz", f"pi_{fam}"),
                        pa[0]["validity"], pb[0]["validity"], "full_posterior")
    return rows


def consolidate_prior_predictive(cfg, root: Path) -> None:
    src = root / "controls" / "prior_predictive"
    if not src.exists():
        return
    out = root / "controls" / "prior_predictive.h5"
    tmp = out.with_name(f".{out.name}.tmp")
    with h5py.File(tmp, "w") as f:
        for p in sorted(src.glob("*.h5")):
            with h5py.File(p, "r") as g:
                grp = f.create_group(p.stem)
                grp.create_dataset("log_prob", data=g["log_prob"][()])
                for k, v in g.attrs.items():
                    grp.attrs[k] = v
    tmp.replace(out)
