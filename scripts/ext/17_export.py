#!/usr/bin/env python3
"""
Export lightweight, git-trackable extension results (no .npz) to results_dir:
tables, figure sidecars, figures, captions, validation/profile/data summaries, per-target
metadata + diagnostics + per-state curvature CSV + gzipped scalar traces, manifest.json and
SUMMARY.md.
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.ext.cli import ROOT, base_parser, load  # noqa: E402
from cylinder.ext.run import run_dir  # noqa: E402
from cylinder.ext.targets import all_targets  # noqa: E402
from cylinder.io import atomic_save_json  # noqa: E402

SIDECARS = (
    "target_summary.csv", "setting_summary.csv", "matched_diagonal.csv",
    "figure1_coverage.csv", "figure2_curvature.csv", "figureA_realdata.csv", "figureB_cutoff.csv",
    "extension_summary.json", "validation.json", "profile.json", "data_summary.json", "captions.md",
)


def _git(*a: str) -> str | None:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fmt(v, nd: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{x:.{nd}g}" if np.isfinite(x) else "—"


def write_traces(rd: Path, dst: Path) -> None:
    with np.load(rd / "observables.npz") as z:
        HB, inside, Vv, f0 = z["H_over_B"], z["inside"], z["V"], z["f_sub"][:, :, 0]
    T = int(np.isfinite(HB).all(axis=0).sum())
    with gzip.open(dst / "traces.csv.gz", "wt", newline="", encoding="utf-8", compresslevel=6) as f:
        w = csv.writer(f)
        w.writerow(["chain", "t", "H_over_B", "inside", "V", "f_sub_0"])
        for c in range(HB.shape[0]):
            w.writerows(
                (c, t, f"{h:.8g}", int(i), f"{v:.10g}", f"{g:.8g}")
                for t, h, i, v, g in zip(range(T), HB[c, :T].tolist(), inside[c, :T].tolist(),
                                         Vv[c, :T].tolist(), f0[c, :T].tolist())
            )


def write_curvature_csv(rd: Path, dst: Path) -> None:
    with np.load(rd / "curvature.npz") as z:
        cz = {k: z[k] for k in z.files}
    idx = cz["indices"]
    keys = ("H", "inside", "ell", "u", "d_minus", "d_plus", "max_eigen_residual")
    with open(dst / "curvature_states.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chain", "j", "t", *keys])
        for c in range(cz["d_plus"].shape[0]):
            for j in range(cz["d_plus"].shape[1]):
                w.writerow([c, j, int(idx[j]), *[f"{float(cz[k][c, j]):.10g}" for k in keys]])


def summary_md(out: Path, manifest: dict) -> None:
    L = [
        "# Larger-sample extension results", "",
        f"Generated {manifest['generated_utc']} on `{manifest['hostname']}`"
        + (f" (SLURM job {manifest['slurm_job_id']})" if manifest.get("slurm_job_id") else "")
        + f", commit `{(manifest.get('git_commit') or '?')[:10]}`, config `{manifest['config']}`.",
        f"Targets done: {manifest['n_done']}/{manifest['n_expected']}.", "",
    ]
    val = out / "validation.json"
    if val.exists():
        v = json.loads(val.read_text())
        L += ["## Validation", "", "| check | result |", "|---|---|"]
        L += [f"| {k} | {'PASS' if r['ok'] else 'FAIL'} |" for k, r in v.items() if k != "all_ok"]
        L.append("")
    ss = out / "setting_summary.csv"
    if ss.exists():
        rows = list(csv.DictReader(open(ss, encoding="utf-8")))
        L += ["## Settings (medians over replicates)", "",
              "| data | n | m | n/√m | B | D_th | coverage (min) | exits | q99 H/B | median d₊ | q95 [d₋, d₊] | q95 d₊/D_th | between-rep SD | diag pass |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(
                f"| {r['kind']} | {r['n']} | {r['m']} | {_fmt(r['n_over_sqrt_m'], 3)} | {_fmt(r['B'])} | {_fmt(r['D_th'])} "
                f"| {_fmt(r['coverage_median'])} ({_fmt(r['coverage_min'])}) | {r['total_exits']} "
                f"| {_fmt(r['H_over_B_q99_median'], 3)} | {_fmt(r['median_d_plus_median'], 3)} "
                f"| [{_fmt(r['q95_d_minus_median'], 3)}, {_fmt(r['q95_d_plus_median'], 3)}] "
                f"| {_fmt(r['q95_d_plus_over_D_th'], 3)} | {_fmt(r['q95_d_plus_between_rep_sd'], 2)} "
                f"| {r['all_reps_pass_diagnostics']} |"
            )
        L.append("")
    L += ["## Targets", "",
          "T beyond the addendum's retained-update cap (`sampling.protocol_max_retained`) is marked †. Limiting = observable with the lowest bulk ESS.", "",
          "| target | status | T | diag pass | R̂ max | bulk ESS min | limiting | coverage | budget hit |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in manifest["targets"]:
        T = f"{t['T']}{' †' if t.get('beyond_protocol') else ''}" if "T" in t else "—"
        L.append(f"| {t['name']} | {t['status']} | {T} | {t.get('pass', '—')} | "
                 f"{_fmt(t.get('rhat_max'))} | {_fmt(t.get('ess_bulk_min'))} | {t.get('limiting') or '—'} | "
                 f"{_fmt(t.get('coverage'))} | "
                 f"{t.get('hit_budget', '—')} |")
    figs = sorted((out / "figures").glob("*.png")) if (out / "figures").exists() else []
    if figs:
        L += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs]
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-traces", action="store_true")
    args = ap.parse_args()
    cfg, cfg_path, art, res = load(args)
    out = Path(args.out).resolve() if args.out else res
    out.mkdir(parents=True, exist_ok=True)

    for name in SIDECARS:
        if (art / name).exists():
            shutil.copy2(art / name, out / name)
    if (art / "figures").exists():
        (out / "figures").mkdir(exist_ok=True)
        for p in (art / "figures").iterdir():
            if p.suffix in (".png", ".pdf"):
                shutil.copy2(p, out / "figures" / p.name)
    shutil.copy2(cfg_path, out / "config_used.yaml")

    targets = []
    for t in all_targets(cfg):
        rd = run_dir(art, t)
        rec = {"name": t.name, "status": "missing"}
        if (rd / "metadata.json").exists():
            dst = out / "runs" / t.name
            dst.mkdir(parents=True, exist_ok=True)
            for fn in ("metadata.json", "diagnostics.json"):
                if (rd / fn).exists():
                    shutil.copy2(rd / fn, dst / fn)
            write_curvature_csv(rd, dst)
            if not args.no_traces:
                write_traces(rd, dst)
            meta = json.loads((rd / "metadata.json").read_text())
            diag = json.loads((rd / "diagnostics.json").read_text()) if (rd / "diagnostics.json").exists() else {}
            rec.update(status="done", T=diag.get("T", meta["n_retained"]), **{"pass": diag.get("pass")},
                       rhat_max=diag.get("diagnostics", {}).get("rhat_max"),
                       ess_bulk_min=diag.get("diagnostics", {}).get("ess_bulk_min"),
                       limiting=diag.get("diagnostics", {}).get("ess_bulk_argmin"),
                       beyond_protocol=diag.get("beyond_protocol", False),
                       coverage=diag.get("coverage", {}).get("p_hat"),
                       hit_budget=meta["hit_likelihood_budget"], device=meta["effective_device"])
        targets.append(rec)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "config": str(cfg_path.relative_to(ROOT)) if cfg_path.is_relative_to(ROOT) else str(cfg_path),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "n_expected": len(targets),
        "n_done": sum(t["status"] == "done" for t in targets),
        "targets": targets,
    }
    atomic_save_json(out / "manifest.json", manifest)
    summary_md(out, manifest)
    print(f"Exported {manifest['n_done']}/{manifest['n_expected']} targets to {out}")
    missing = [t["name"] for t in targets if t["status"] != "done"]
    if missing:
        print("Missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
