#!/usr/bin/env python3
"""
Export lightweight, git-trackable results (no .npz payloads).

Copies the summary CSV/JSON sidecars, figures and captions, per-run
metadata/diagnostics, and a gzipped scalar trace per run into
`results_dir` (config) or --out, then writes manifest.json and SUMMARY.md.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cylinder.config import artifacts_root, load_config, results_root  # noqa: E402
from cylinder.io import atomic_save_json, run_dir_name  # noqa: E402

SIDECARS = (
    "target_summary.csv",
    "figure1_coverage.csv",
    "figure2_curvature.csv",
    "figureS1_cutoff.csv",
    "figureS2_efficiency.csv",
    "curvature.csv",
    "curvature_summary.json",
    "extension_summary.json",
    "pcn_crosscheck.json",
    "captions.md",
)
RUN_FILES = ("metadata.json", "diagnostics.json")
TRACE_KEYS = ("H", "H_over_B", "inside", "V", "z_sq_over_p", "mean_sq_head")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_trace(obs_path: Path, out_path: Path) -> int:
    """Per-draw scalar observables as chain,t,<keys>; returns retained T."""
    obs = np.load(obs_path)
    T = int(np.isfinite(obs["H"]).all(axis=0).sum())
    C = obs["H"].shape[0]
    keys = [k for k in TRACE_KEYS if k in obs.files]
    cols = [obs[k][:, :T] for k in keys]
    probe = obs["p_probe"][:, :T] if "p_probe" in obs.files else None
    n_probe = 0 if probe is None else probe.shape[-1]
    header = ["chain", "t", *keys, *[f"p_probe_{k}" for k in range(n_probe)]]
    with gzip.open(out_path, "wt", newline="", encoding="utf-8", compresslevel=9) as f:
        w = csv.writer(f)
        w.writerow(header)
        for c in range(C):
            for t in range(T):
                row = [c, t]
                for k, col in zip(keys, cols):
                    v = col[c, t]
                    row.append(int(v) if k == "inside" else f"{v:.10g}")
                if probe is not None:
                    row.extend(f"{v:.8g}" for v in probe[c, t])
                w.writerow(row)
    return T


def expected_targets(cfg: dict) -> list[tuple[int, int, str]]:
    out = [(int(m), int(s), "ess") for m in cfg["widths"] for s in cfg["prior"]["center_seeds"]]
    pc = cfg.get("pcn_crosscheck")
    if pc:
        out.append((int(pc["width"]), int(pc["center_seed"]), "pcn"))
    return out


def _fmt(v, nd: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "—"
    return f"{x:.{nd}g}"


def write_summary_md(out: Path, targets: list[dict], manifest: dict) -> None:
    lines = [
        "# Cylinder experiment results",
        "",
        f"Generated {manifest['generated_utc']} on `{manifest['hostname']}`"
        + (f" (SLURM job {manifest['slurm_job_id']})" if manifest.get("slurm_job_id") else "")
        + f", commit `{(manifest.get('git_commit') or 'unknown')[:10]}`"
        + (" (dirty tree)" if manifest.get("git_dirty") else "")
        + f", config `{manifest['config']}`.",
        "",
        "## Target status",
        "",
        "| target | status | T | diag pass | coverage | exits | R̂ max | bulk ESS min | hit budget |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in targets:
        lines.append(
            f"| {t['name']} | {t['status']} | {t.get('T', '—')} | {t.get('diagnostics_pass', '—')} "
            f"| {_fmt(t.get('coverage'))} | {t.get('exit_count', '—')} | {_fmt(t.get('rhat_max'))} "
            f"| {_fmt(t.get('ess_bulk_min'))} | {t.get('hit_budget', '—')} |"
        )

    ts = out / "target_summary.csv"
    if ts.exists():
        with open(ts, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        lines += [
            "",
            "## Theorem checks (ESS targets)",
            "",
            "| m | seed | coverage | H/B_m q95 | H/B_m q99 | q95 d₋ | q95 d₊ | D_th | d₊/D_th | E[V] post | E[V] prior |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            try:
                ratio = float(r["q95_d_plus"]) / float(r["D_th"])
            except (KeyError, ValueError, ZeroDivisionError):
                ratio = float("nan")
            lines.append(
                f"| {r['width']} | {r['center_seed']} | {_fmt(r['coverage'])} "
                f"| {_fmt(r['H_over_B_q95'])} | {_fmt(r['H_over_B_q99'])} "
                f"| {_fmt(r.get('q95_d_minus'))} | {_fmt(r.get('q95_d_plus'))} "
                f"| {_fmt(r['D_th'])} | {_fmt(ratio, 3)} "
                f"| {_fmt(r['mean_posterior_V'])} | {_fmt(r['mean_prior_V'])} |"
            )

    pcn = out / "pcn_crosscheck.json"
    if pcn.exists():
        cross = json.loads(pcn.read_text())
        lines += ["", "## pCN cross-check (m=1024, seed 0)", ""]
        if cross.get("status") != "ok":
            lines.append(f"Status: {cross.get('status')}")
        else:
            lines += ["| quantity | ESS | pCN | diff | combined SE | >3 SE |", "|---|---|---|---|---|---|"]
            for name, c in cross["comparisons"].items():
                if "ess_mean" in c:
                    lines.append(
                        f"| {name} | {_fmt(c['ess_mean'], 6)} | {_fmt(c['pcn_mean'], 6)} "
                        f"| {_fmt(c['diff'], 3)} | {_fmt(c['combined_se'], 3)} | {c['flag_gt_3se']} |"
                    )
                else:
                    lines.append(
                        f"| {name} | {_fmt(c['ess'], 6)} | {_fmt(c['pcn'], 6)} | {_fmt(c['diff'], 3)} | | |"
                    )

    figs = sorted((out / "figures").glob("*.png")) if (out / "figures").exists() else []
    if figs:
        lines += ["", "## Figures", ""]
        lines += [f"![{p.stem}](figures/{p.name})" for p in figs]

    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="Override results_dir from config")
    ap.add_argument("--no-traces", action="store_true", help="Skip per-run traces.csv.gz")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    artifacts = artifacts_root(cfg, ROOT)
    out = Path(args.out).resolve() if args.out else results_root(cfg, ROOT)
    out.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in SIDECARS:
        src = artifacts / name
        if src.exists():
            shutil.copy2(src, out / name)
            copied.append(name)

    fig_src = artifacts / "figures"
    if fig_src.exists():
        (out / "figures").mkdir(exist_ok=True)
        for p in sorted(fig_src.iterdir()):
            if p.suffix.lower() in (".png", ".pdf"):
                shutil.copy2(p, out / "figures" / p.name)
                copied.append(f"figures/{p.name}")

    shutil.copy2(cfg_path, out / "config_used.yaml")

    targets: list[dict] = []
    data_hashes: dict[str, str] = {}
    for m, seed, kernel in expected_targets(cfg):
        name = run_dir_name(m, seed, kernel)
        run_dir = artifacts / "runs" / name
        rec: dict = {"name": name, "m": m, "seed": seed, "kernel": kernel}
        if not (run_dir / "metadata.json").exists():
            rec["status"] = "missing"
            targets.append(rec)
            continue
        dst = out / "runs" / name
        dst.mkdir(parents=True, exist_ok=True)
        for fn in RUN_FILES:
            if (run_dir / fn).exists():
                shutil.copy2(run_dir / fn, dst / fn)
        meta = json.loads((run_dir / "metadata.json").read_text())
        for k in ("data_hash_X", "data_hash_y", "data_hash_X_probe"):
            if k in meta:
                data_hashes.setdefault(k, meta[k])
        rec.update(
            status="done",
            T=meta.get("n_retained"),
            device=meta.get("effective_device"),
            wall_time_s=meta.get("wall_time_s"),
            likelihood_evals=meta.get("likelihood_evals"),
            hit_budget=meta.get("hit_likelihood_budget"),
        )
        if (run_dir / "diagnostics.json").exists():
            diag = json.loads((run_dir / "diagnostics.json").read_text())
            cov = diag.get("coverage", {})
            dd = diag.get("diagnostics", {})
            rec.update(
                diagnostics_pass=diag.get("pass"),
                unresolved=diag.get("unresolved", False),
                coverage=cov.get("p_hat"),
                exit_count=cov.get("exit_count"),
                rhat_max=dd.get("rhat_max"),
                ess_bulk_min=dd.get("ess_bulk_min"),
                ess_tail_min=dd.get("ess_tail_min"),
            )
        if not args.no_traces and (run_dir / "observables.npz").exists():
            rec["trace_T"] = write_trace(run_dir / "observables.npz", dst / "traces.csv.gz")
        targets.append(rec)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "config": str(cfg_path.relative_to(ROOT)) if cfg_path.is_relative_to(ROOT) else str(cfg_path),
        "config_sha256": _sha256(cfg_path),
        "artifacts_dir": str(artifacts),
        "data_hashes": data_hashes,
        "n_targets_expected": len(targets),
        "n_targets_done": sum(t["status"] == "done" for t in targets),
        "targets": targets,
        "files": copied,
    }
    atomic_save_json(out / "manifest.json", manifest)
    write_summary_md(out, targets, manifest)
    print(
        f"Exported results to {out}: {manifest['n_targets_done']}/{manifest['n_targets_expected']} "
        f"targets, {len(copied)} sidecar/figure files"
    )
    missing = [t["name"] for t in targets if t["status"] != "done"]
    if missing:
        print("Missing targets: " + ", ".join(missing))


if __name__ == "__main__":
    main()
