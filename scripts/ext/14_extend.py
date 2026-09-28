#!/usr/bin/env python3
"""
Diagnostics for every finished target; extend unresolved ones 4000 → 8000 → 16000 retained
updates (rerun from the same seeds) unless the likelihood-evaluation cap was reached.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.diagnostics import (  # noqa: E402
    arviz_diagnostics,
    coverage_summary,
    diagnostics_pass,
    half_chain_stability,
)
from cylinder.ext.cli import base_parser, load  # noqa: E402
from cylinder.ext.run import run_dir, run_ext_target  # noqa: E402
from cylinder.ext.targets import all_targets  # noqa: E402
from cylinder.io import atomic_save_json  # noqa: E402


def evaluate(rd: Path, cfg: dict) -> dict:
    obs = np.load(rd / "observables.npz")
    meta = json.loads((rd / "metadata.json").read_text())
    T = int(np.isfinite(obs["H"]).all(axis=0).sum())
    if T < 4:
        return {"status": "too_few_draws", "T": T, "pass": False,
                "hit_budget": meta.get("hit_likelihood_budget", False)}
    arrays = {k: obs[k][:, :T] for k in ("H", "V", "z_sq_over_p", "mean_sq_head", "f_sub", "projections")}
    diag = arviz_diagnostics(arrays, head_quantiles=cfg["diagnostics"]["head_quantiles"])
    dc = cfg["diagnostics"]
    checks = diagnostics_pass(diag, rhat_max=float(dc["rhat_max"]), bulk_ess_min=float(dc["bulk_ess_min"]),
                              tail_ess_min=float(dc["tail_ess_min"]), quantile_ess_min=float(dc["quantile_ess_min"]))
    cov = coverage_summary(obs["inside"][:, :T])
    return {
        "status": "ok",
        "T": T,
        "T_planned": int(meta["n_retained"]),
        "diagnostics": {k: v for k, v in diag.items() if k != "summary_table"},
        "checks": checks,
        "coverage": {k: v for k, v in cov.items() if k != "batch_means"},
        "stability_V": half_chain_stability(obs["V"][:, :T]),
        "stability_H": half_chain_stability(obs["H"][:, :T]),
        "pass": bool(checks["all_ok"]),
        "likelihood_evals": meta.get("likelihood_evals"),
        "hit_budget": bool(meta.get("hit_likelihood_budget", False)),
    }


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--filter", default=None, help="Regex on target names")
    args = ap.parse_args()
    cfg, _, artifacts, _ = load(args)
    exts = [int(x) for x in cfg["sampling"]["retained_extensions"]]

    results = []
    for t in all_targets(cfg, pattern=args.filter):
        rd = run_dir(artifacts, t)
        if not (rd / "metadata.json").exists():
            print(f"MISSING {t.name}")
            results.append({"target": t.name, "status": "missing"})
            continue
        rep = evaluate(rd, cfg)
        atomic_save_json(rd / "diagnostics.json", rep)
        print(f"{t.name}: pass={rep['pass']} T={rep['T']} rhat={rep.get('diagnostics', {}).get('rhat_max')} "
              f"bulk={rep.get('diagnostics', {}).get('ess_bulk_min')}")
        while args.extend and not rep["pass"] and not rep["hit_budget"]:
            cur = int(json.loads((rd / "metadata.json").read_text())["n_retained"])
            nxt = next((x for x in exts if x > cur), None)
            if nxt is None:
                rep["unresolved"] = True
                atomic_save_json(rd / "diagnostics.json", rep)
                print(f"  {t.name}: unresolved after T={cur}")
                break
            print(f"  extending {t.name} to T={nxt}")
            run_ext_target(cfg=cfg, target=t, artifacts=artifacts, device_request=args.device,
                           n_retained=nxt, overwrite=True)
            rep = evaluate(rd, cfg)
            atomic_save_json(rd / "diagnostics.json", rep)
            print(f"  {t.name}: pass={rep['pass']} T={rep['T']}")
        if rep.get("hit_budget") and not rep["pass"]:
            rep["unresolved"] = True
            atomic_save_json(rd / "diagnostics.json", rep)
        results.append({"target": t.name, "pass": rep["pass"], "T": rep["T"],
                        "unresolved": rep.get("unresolved", False), "hit_budget": rep.get("hit_budget")})
    atomic_save_json(artifacts / "extension_summary.json", {"results": results})


if __name__ == "__main__":
    main()
