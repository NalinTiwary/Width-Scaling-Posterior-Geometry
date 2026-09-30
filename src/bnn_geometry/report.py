"""SUMMARY.md, strict audit and export (runbook §16.3, §18, §15)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as C
from .figures import CaptionError, check_caption, load
from .storage import StorageError, atomic_write_json, clean_json, read_json

FIGURES = ("main_1_relaxation", "main_2_entropy", "main_3_spectral", "appendix_s1_validation",
           "appendix_s2_static_checks")


def _b(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).lower() in ("true", "1", "1.0"))


def _du(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# ---- SUMMARY ----------------------------------------------------------------------------------------------
def write_summary(cfg: dict[str, Any], root: Path) -> str:
    ta = load(root, "target_audit")
    rf = load(root, "relaxation_families")
    ef = load(root, "entropy_families")
    ss = load(root, "spectral_summary")
    wr = load(root, "width_ratios")
    ps = load(root, "predictive_scores")
    cc = load(root, "cost_and_counts")
    meta = read_json(root / "tables" / "analysis_metadata.json")
    steps = meta["final_steps"]
    caps = {f: read_json(root / "figures" / f"{f}.json")["caption"] for f in FIGURES
            if (root / "figures" / f"{f}.json").exists()}
    n_t = len(C.targets(cfg))
    ref_pass = int((ta["reference_status"] == "reference_pass").sum()) if len(ta) else 0
    prod = rf[rf["role"] == "production_final"] if len(rf) else rf
    dyn_ok = int(prod.groupby("target_id")["validity"].apply(lambda s: _b(s).all()).sum()) if len(prod) else 0
    ent_ok = int(ef.groupby("target_id")["validity"].apply(lambda s: _b(s).all()).sum()) if len(ef) else 0
    sst = ss[ss["chain_id"].astype(str) == "all"] if len(ss) else ss
    deep_n = sum(1 for t in C.targets(cfg) if t.arch == "deep")
    spec_ok = int((_b(sst["backend_check_ok"]) & (sst["S_rhat"] < float(cfg["reference_gates"]["rhat_max_exclusive"]))).sum()) \
        if len(sst) else 0
    lines = [f"# Final geometry campaign summary ({cfg['campaign_id']})", ""]
    if "fixture" in cfg:
        lines += ["**FIXTURE — not a scientific result.**", ""]
    lines += ["## 1. Completion", "",
              f"- Intended targets: {n_t}; reference_pass: {ref_pass}; dynamics (all four families valid at the final "
              f"step): {dyn_ok}; entropy (all three families valid): {ent_ok}; spectral (deep, S R-hat and SVD backend "
              f"checks pass): {spec_ok} of {deep_n}.",
              f"- Dynamics step status: shallow {steps['shallow']['status']} (h={steps['shallow']['h']}), deep "
              f"{steps['deep']['status']} (h={steps['deep']['h']}); refinement used: "
              f"shallow={steps['shallow']['refined']}, deep={steps['deep']['refined']}.", "",
              "## 2. Configuration", "",
              f"- Shallow widths {cfg['model']['shallow']['widths']}, deep widths {cfg['model']['deep']['widths']}, "
              f"n={cfg['data']['train_size']}, d={cfg['data']['input_dimension']}, sigma={cfg['model']['sigma']}, "
              f"head-center norm {ta['head_center_norm'].round(12).unique().tolist() if len(ta) else 'n/a'}, final "
              f"conditional domain ||W2||op/sqrt(m) <= {cfg['domain']['spectral_cutoff_a']} (deep entropy only).", "",
              "## 3. Main figures", ""]
    for f in FIGURES[:3]:
        lines.append(f"- **{f}**: {caps.get(f, 'not generated')}")
    lines += ["", "## 4. Widest/narrowest ratios (per replicate; MC interval; validity)", "",
              "| experiment | arch | family | replicate | ratio | MC interval | valid | law |", "|---|---|---|---|---|---|---|---|"]
    for r in wr.itertuples() if len(wr) else []:
        iv = f"[{getattr(r, 'mc_low', float('nan')):.3g}, {getattr(r, 'mc_high', float('nan')):.3g}]"
        lines.append(f"| {r.experiment} | {r.architecture} | {r.family} | {r.replicate} | {r.estimate:.3g} | {iv} | "
                     f"{r.validity} | {r.law} |")
    lines += ["", "## 5. Spectral domain", ""]
    for _, r in (sst.iterrows() if len(sst) else []):
        unc = str(r["occupancy_uncertainty"])
        unc = "" if "inspected" in unc else f"; {unc}"
        lines.append(f"- {r['target_id']}: {int(r['exits'])} exits / {int(r['inspected_states'])} inspected archived "
                     f"states; S median {r.get('S_q0.5', float('nan')):.3f}, q95 {r.get('S_q0.95', float('nan')):.3f}, "
                     f"q99 {r.get('S_q0.99', float('nan')):.3f}{unc}.")
    pr = root / "controls" / "prior_spectral.csv"
    if pr.exists():
        for _, r in pd.read_csv(pr).iterrows():
            lines.append(f"- prior m={int(r['m'])}: S median {r['S_q0.5']:.3f}, q95 {r['S_q0.95']:.3f}, "
                         f"q99 {r['S_q0.99']:.3f} ({int(r['n_matrices'])} iid matrices, one reference shared by all "
                         "replicates).")
    calp = root / "tests" / "calibration_results.csv"
    lines += ["", "## 6. Numerical checks", ""]
    if calp.exists():
        cal = pd.read_csv(calp)
        passcols = [c for c in cal.columns if c.startswith("pass")]
        fails = cal[~cal[passcols].fillna(True).astype(bool).all(axis=1)]
        lines.append(f"- Known-distribution calibration rows: {len(cal)}; rows with a failed statistical check: "
                     f"{len(fails)} (audits: {sorted(set(fails.get('audit', pd.Series(dtype=str)).dropna()))}).")
    ep = load(root, "endpoint_comparisons")
    if len(ep):
        lines.append(f"- Endpoint step comparisons: {int(_b(ep['pass']).sum())} of {len(ep)} rows pass "
                     "(tables/endpoint_comparisons.csv).")
    if len(ta):
        be = ta[_b(ta["reference_budget_exhausted"].fillna(False))]["target_id"].tolist()
        lines.append(f"- Reference budget exhausted: {be or 'none'}. Code fixes during the campaign: see "
                     "manifest.json (code_fixes).")
    lines += ["", "## 7. Predictive check", ""]
    for r in ps.itertuples() if len(ps) else []:
        imp = r.improvement if isinstance(r.improvement, float) else float("nan")
        lines.append(f"- {r.target_id}: NLS(prior) - NLS(posterior) = {imp:.4f} nats/point "
                     f"[{getattr(r, 'improvement_mc_low', float('nan')):.4f}, {getattr(r, 'improvement_mc_high', float('nan')):.4f}]")
    lines += ["", "## 8. Compute and storage", ""]
    if len(cc):
        lines.append(f"- Dynamics gradient evaluations (all targets): {int(cc['dynamics_grad_evals_total'].fillna(0).sum()):,}; "
                     f"reference wall time {cc['reference_wall_s'].fillna(0).sum() / 3600:.2f} h; "
                     f"dynamics wall time {cc['dynamics_wall_s'].fillna(0).sum() / 3600:.2f} h.")
    lines.append(f"- Storage under {root.name}: {_du(root) / 1e9:.2f} GB.")
    lines.append("- Tables: tables/*.csv; figures: figures/*; audit: audit.json; captions: captions.tex.")
    text = "\n".join(lines) + "\n"
    (root / "SUMMARY.md").write_text(text)
    return text


# ---- audit ------------------------------------------------------------------------------------------------
def audit(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    from .context import TargetContext  # noqa: F401  (import check only)
    from .data import load_replicate
    checks: list[dict[str, Any]] = []

    def chk(name: str, ok: bool, detail: Any = "", publication: bool = False) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail, "publication_only": publication})

    tg = C.targets(cfg)
    ta = load(root, "target_audit")
    n_set = len(cfg["model"]["shallow"]["widths"]) + len(cfg["model"]["deep"]["widths"])
    chk("settings_and_targets", len(ta) == len(tg) and len({(t.arch, t.m) for t in tg}) == n_set
        and set(ta.get("target_id", [])) == {t.target_id for t in tg}, f"{len(ta)} rows for {len(tg)} targets")
    d = int(cfg["data"]["input_dimension"])
    bad = []
    for t in tg:
        sp = root / "targets" / t.target_id / "spec.json"
        if not sp.exists():
            bad.append(f"{t.target_id}:no_spec")
            continue
        s = read_json(sp)
        D, bank, dh = load_replicate(root, t.rep)
        if s["layout"]["p"] != t.p(d) or abs(s["head_center_norm"] - 1.0) > 1e-12 or s["sigma"] != cfg["model"]["sigma"] \
                or s["data_hashes"] != dh or s["m"] != t.m:
            bad.append(t.target_id)
    chk("target_identity_and_pairing", not bad, bad)
    # trace integrity and counts
    from .chains import Trajectory  # noqa: F401
    from .reference import production_segments
    probs = []
    for t in tg:
        st_p = root / "targets" / t.target_id / "status.json"
        if not st_p.exists():
            probs.append(f"{t.target_id}:no_status")
            continue
        ref = read_json(st_p).get("reference") or {}
        k = ref.get("stopping_stage")
        if not k:
            continue
        try:
            from .context import TargetContext
            ctx = TargetContext(cfg, t, root, device=None)
            tr = ctx.reference_traj()
            import torch
            from .storage import torch_load
            tr.ck = torch_load(tr.ck_path)
            dat = tr.read(production_segments(int(k)), keys=["draw"])
            n = [int(x["draw"].shape[0]) for x in dat]
            if len(set(n)) != 1 or n[0] != ref.get("production_per_chain"):
                probs.append(f"{t.target_id}:count_mismatch {n} vs {ref.get('production_per_chain')}")
        except (StorageError, OSError, KeyError) as exc:
            probs.append(f"{t.target_id}:{exc}")
    chk("reference_chain_boundaries_and_counts", not probs, probs)
    sst = load(root, "spectral_states")
    ssum = load(root, "spectral_summary")
    if len(ssum):
        tgt = ssum[ssum["chain_id"].astype(str) == "all"]
        mism = [r.target_id for r in tgt.itertuples()
                if int(r.inspected_states) != int((sst["target_id"] == r.target_id).sum())]
        chk("inspection_counts_equal_svd_rows", not mism, mism)
        zero = tgt[tgt["exits"] == 0]
        zbad = [r.target_id for r in zero.itertuples() if "occupancy_mcse" in tgt and pd.notna(getattr(r, "occupancy_mcse", np.nan))]
        chk("zero_exit_rows_without_interval", not zbad, zbad)
    ef = load(root, "entropy_families")
    if len(ef):
        wrong = ef[(ef["architecture"] == "deep") & (ef["law"] != "conditional_G_2.5")]
        wrong2 = ef[(ef["architecture"] == "shallow") & (ef["law"] != "full_posterior")]
        chk("entropy_law_labels", len(wrong) == 0 and len(wrong2) == 0, len(wrong) + len(wrong2))
    rf = load(root, "relaxation_families")
    if len(rf):
        chk("relaxation_unrestricted_law", (rf["law"] == "full_posterior").all())
        chk("relaxation_factor_half_convention", True, "tau = (h/2) N_total / ESS_mean (relaxation.tau_hat)")
    for tab in ("relaxation_families", "entropy_families", "static_pi", "predictive_scores"):
        df = load(root, tab)
        chk(f"validity_column:{tab}", len(df) == 0 or "validity" in df, tab)
    ep = load(root, "endpoint_comparisons")
    if len(ep):
        eqT = True
        for r in ep[ep["kind"] == "family"].itertuples():
            tdir = root / "targets" / r.target_id / "analysis"
            from .context import h_id
            runs = read_json(tdir / "dynamics_runs.json")
            a, b = runs.get(h_id(r.h)), runs.get(h_id(r.h_half))
            if not a or not b or abs(a["T_per_chain"] - b["T_per_chain"]) > max(r.h, 1e-12) * 1.0001:
                eqT = False
        chk("step_comparisons_equal_physical_duration", eqT)
    missing = [f"{f}.{e}" for f in FIGURES for e in ("pdf", "png", "csv", "json") if not (root / "figures" / f"{f}.{e}").exists()]
    chk("figures_present", not missing, missing)
    capok = True
    for f in FIGURES:
        p = root / "figures" / f"{f}.json"
        if p.exists():
            try:
                check_caption(read_json(p)["caption"])
            except CaptionError:
                capok = False
    chk("captions_without_placeholders", capok and (root / "captions.tex").exists())
    for tab in ("target_audit", "reference_diagnostics", "relaxation_families", "entropy_families", "static_pi",
                "spectral_summary", "predictive_scores", "cost_and_counts", "exclusions"):
        chk(f"table_present:{tab}", (root / "tables" / f"{tab}.csv").exists())
    chk("regeneration_command", True, "python -m bnn_geometry figures --config <config> (reads tables only)")
    # publication readiness: every main result valid
    pub = []
    if len(rf):
        prod = rf[rf["role"] == "production_final"]
        pub.append(("relaxation_all_valid", bool(_b(prod["validity"]).all()) if len(prod) else False))
    pub.append(("entropy_all_valid", bool(_b(ef["validity"]).all()) if len(ef) else False))
    pub.append(("all_references_pass", bool((ta["reference_status"] == "reference_pass").all()) if len(ta) else False))
    for name, ok in pub:
        chk(name, ok, publication=True)
    arch_ok = all(c["pass"] for c in checks if not c["publication_only"])
    pub_ok = arch_ok and all(c["pass"] for c in checks if c["publication_only"]) and "fixture" not in cfg
    out = {"archival_pass": arch_ok, "publication_ready": pub_ok, "fixture": "fixture" in cfg, "checks": checks}
    atomic_write_json(root / "audit.json", clean_json(out))
    return out


# ---- export -----------------------------------------------------------------------------------------------
def environment() -> dict[str, Any]:
    import numpy
    import scipy
    import torch
    env = {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__,
           "scipy": scipy.__version__, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "cuda_version": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
           "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    try:
        import arviz
        env["arviz"] = arviz.__version__
    except ImportError:
        env["arviz"] = None
    try:
        import h5py
        env["h5py"] = h5py.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    try:
        env["blas"] = numpy.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})  # type: ignore
    except TypeError:
        env["blas"] = "numpy<1.25: see environment.freeze.txt"
    try:
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        env["nvidia_smi"] = None
    return env


def write_environment(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "environment.json", clean_json(environment()))
    frz = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (root / "environment.freeze.txt").write_text(frz)
    (root / "source_revision.txt").write_text(C.code_revision() + "\n" + "sampler_code_hash " + C.sampler_code_hash() + "\n")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def export(cfg: dict[str, Any], cfg_path: Path, root: Path, *, heavy: bool = True) -> Path:
    shutil.copyfile(cfg_path, root / "campaign.yaml")
    write_environment(root)
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "checksums.sha256" or p.name.startswith("."):
            continue
        if p.name == "checkpoint.pt" or ".chunk" in p.name:
            continue
        if not heavy and p.suffix == ".h5" and ("reference" in p.parts or "dynamics" in p.parts):
            continue
        files.append(p)
    lines = [f"{sha_file(p)}  {p.relative_to(root)}" for p in files]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n")
    return root / "checksums.sha256"
