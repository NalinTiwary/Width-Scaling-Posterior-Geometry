#!/usr/bin/env python3
"""Assess diagnostics and optionally extend retained draws (§4.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cylinder.config import artifacts_root, load_config  # noqa: E402
from cylinder.diagnostics import (  # noqa: E402
    arviz_diagnostics,
    coverage_summary,
    diagnostics_pass,
    half_chain_stability,
)
from cylinder.io import atomic_save_json, run_dir_name  # noqa: E402
from cylinder.run import run_target  # noqa: E402


def evaluate_run(run_dir: Path, cfg: dict) -> dict:
    obs = np.load(run_dir / "observables.npz")
    meta = json.loads((run_dir / "metadata.json").read_text())
    arrays = {
        "H": obs["H"],
        "V": obs["V"],
        "z_sq_over_p": obs["z_sq_over_p"],
        "mean_sq_head": obs["mean_sq_head"],
        "p_probe": obs["p_probe"],
        "f_train": obs["f_train"],
        "projections": obs["projections"],
    }
    # Drop draws that are still NaN (incomplete chain shards)
    valid = np.isfinite(obs["H"]).all(axis=0)
    T = int(np.sum(valid))
    if T == 0:
        return {"status": "empty", "pass": False}
    arrays = {k: v[:, :T] if v.ndim == 2 else v[:, :T, ...] for k, v in arrays.items()}
    diag = arviz_diagnostics(arrays, head_quantiles=cfg["diagnostics"]["head_quantiles"])
    # summary_table is a DataFrame — drop before JSON
    diag_json = {k: v for k, v in diag.items() if k != "summary_table"}
    checks = diagnostics_pass(
        diag,
        rhat_max=float(cfg["diagnostics"]["rhat_max"]),
        bulk_ess_min=float(cfg["diagnostics"]["bulk_ess_min"]),
        tail_ess_min=float(cfg["diagnostics"]["tail_ess_min"]),
        quantile_ess_min=float(cfg["diagnostics"]["quantile_ess_min"]),
    )
    cov = coverage_summary(obs["inside"][:, :T])
    stab_V = half_chain_stability(obs["V"][:, :T])
    stab_H = half_chain_stability(obs["H"][:, :T])
    return {
        "status": "ok",
        "T": T,
        "diagnostics": diag_json,
        "checks": checks,
        "coverage": {k: v for k, v in cov.items() if k != "batch_means"},
        "stability_V": stab_V,
        "stability_H": stab_H,
        "pass": bool(checks["all_ok"]),
        "meta_likelihood_evals": meta.get("likelihood_evals"),
        "hit_budget": meta.get("hit_likelihood_budget", False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--extend", action="store_true", help="Extend failing targets")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    artifacts = artifacts_root(cfg, ROOT)
    runs_root = artifacts / "runs"

    extensions = list(cfg["sampling"]["retained_extensions"])
    results = []

    for m in cfg["widths"]:
        for seed in cfg["prior"]["center_seeds"]:
            for kernel in ("ess",):
                run_dir = runs_root / run_dir_name(m, seed, kernel)
                if not run_dir.exists():
                    print(f"MISSING {run_dir.name}")
                    continue
                report = evaluate_run(run_dir, cfg)
                atomic_save_json(run_dir / "diagnostics.json", report)
                print(
                    f"{run_dir.name}: pass={report.get('pass')} "
                    f"T={report.get('T')} rhat={report.get('diagnostics', {}).get('rhat_max')}"
                )
                if args.extend and not report.get("pass") and not report.get("hit_budget"):
                    meta = json.loads((run_dir / "metadata.json").read_text())
                    current_T = int(meta.get("n_retained", 0))
                    next_T = None
                    for cand in extensions:
                        if cand > current_T:
                            next_T = int(cand)
                            break
                    if next_T is None:
                        print(f"  no further extension for {run_dir.name}")
                        report["unresolved"] = True
                        atomic_save_json(run_dir / "diagnostics.json", report)
                    else:
                        print(f"  extending to T={next_T}")
                        run_target(
                            cfg=cfg,
                            m=m,
                            center_seed=seed,
                            kernel=kernel,
                            artifacts=artifacts,
                            device_request=args.device,
                            n_retained=next_T,
                            overwrite=True,
                        )
                        report = evaluate_run(run_dir, cfg)
                        atomic_save_json(run_dir / "diagnostics.json", report)
                results.append({"run": run_dir.name, **{k: report.get(k) for k in ("pass", "T")}})

    # pCN cross-check presence
    pcn_dir = runs_root / run_dir_name(
        int(cfg["pcn_crosscheck"]["width"]),
        int(cfg["pcn_crosscheck"]["center_seed"]),
        "pcn",
    )
    if pcn_dir.exists():
        report = evaluate_run(pcn_dir, cfg)
        atomic_save_json(pcn_dir / "diagnostics.json", report)
        print(f"{pcn_dir.name}: pass={report.get('pass')}")

    atomic_save_json(artifacts / "extension_summary.json", {"results": results})


if __name__ == "__main__":
    main()
