#!/usr/bin/env python3
"""Build target_summary.csv, figure1/S1/S2 sidecars, pCN cross-check deltas."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cylinder.config import artifacts_root, load_config  # noqa: E402
from cylinder.diagnostics import coverage_summary, order_stat_quantile  # noqa: E402
from cylinder.io import atomic_save_json, run_dir_name  # noqa: E402
from cylinder.theorem import prior_cdf_H, theorem_bundle  # noqa: E402


def _load_diag(run_dir: Path) -> dict:
    p = run_dir / "diagnostics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    artifacts = artifacts_root(cfg, ROOT)
    runs_root = artifacts / "runs"
    sigma = float(cfg["prior"]["sigma"])
    b0 = float(cfg["prior"]["b0"])

    target_rows = []
    fig1_rows = []
    s1_rows = []
    s2_rows = []

    # Prior reference loss: 2000 independent prior draws per target
    data = np.load(artifacts / "data.npz")
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    from cylinder.centers import assemble_theta0, load_centers
    from cylinder.model import binary_potential

    for m in cfg["widths"]:
        bun = theorem_bundle(
            m=m,
            n=int(cfg["data"]["n"]),
            d=int(cfg["data"]["d"]),
            C=cfg["architecture"]["classes"],
            b0=b0,
            sigma=sigma,
            M2=float(cfg["theorem"]["M2"]),
            M4=float(cfg["theorem"]["M4"]),
            c2=float(cfg["theorem"]["c2"]),
        )
        for seed in cfg["prior"]["center_seeds"]:
            run_dir = runs_root / run_dir_name(m, seed, "ess")
            if not run_dir.exists():
                continue
            obs = np.load(run_dir / "observables.npz")
            meta = json.loads((run_dir / "metadata.json").read_text())
            diag = _load_diag(run_dir)
            T = int(np.isfinite(obs["H"]).all(axis=0).sum())
            H = obs["H"][:, :T]
            inside = obs["inside"][:, :T]
            cov = coverage_summary(inside)
            H_over_B = H / bun["B_m"]
            flat_HB = H_over_B.ravel()

            # Prior reference
            bank = load_centers(artifacts / f"centers_seed{seed}.npz")
            theta0 = assemble_theta0(bank["U"], m=m, b0=b0)
            rng = np.random.Generator(np.random.PCG64(9000 + seed + m))
            prior_V = []
            for _ in range(2000):
                z = rng.standard_normal(theta0.size)
                theta = theta0 + sigma * z
                prior_V.append(binary_potential(theta, X, y, m))
            prior_V = np.asarray(prior_V)
            post_V = obs["V"][:, :T]

            # Curvature summary if present
            curv = {}
            csum = artifacts / "curvature_summary.json"
            if csum.exists():
                for row in json.loads(csum.read_text())["targets"]:
                    if row["m"] == m and row["seed"] == seed:
                        curv = row
                        break

            row = {
                "width": m,
                "p": bun["p"],
                "sigma": sigma,
                "B_m": bun["B_m"],
                "D_th": bun["D_th"],
                "center_seed": seed,
                "data_hash_X": meta.get("data_hash_X"),
                "n_chains": meta.get("n_chains"),
                "burnin": meta.get("burnin"),
                "n_retained": T,
                "likelihood_evals": meta.get("likelihood_evals"),
                "device": meta.get("effective_device"),
                "wall_time_retained_s": meta.get("wall_time_s"),
                "rhat_max": diag.get("diagnostics", {}).get("rhat_max"),
                "ess_bulk_min": diag.get("diagnostics", {}).get("ess_bulk_min"),
                "ess_tail_min": diag.get("diagnostics", {}).get("ess_tail_min"),
                "diagnostics_pass": diag.get("pass"),
                "coverage": cov["p_hat"],
                "exit_count": cov["exit_count"],
                "exit_episodes": cov["exit_episodes"],
                "coverage_mcse_estimable": cov["mcse_estimable"],
                "coverage_mcse": cov["mcse"],
                "H_over_B_q95": order_stat_quantile(flat_HB, 0.95),
                "H_over_B_q99": order_stat_quantile(flat_HB, 0.99),
                "mean_posterior_V": float(np.nanmean(post_V)),
                "mean_prior_V": float(prior_V.mean()),
                "se_posterior_V": float(np.nanstd(post_V) / np.sqrt(post_V.size)),
                "se_prior_V": float(prior_V.std(ddof=1) / np.sqrt(prior_V.size)),
                "curvature_n_inside": curv.get("n_inside"),
                "q95_d_minus": curv.get("q95_d_minus"),
                "q95_d_plus": curv.get("q95_d_plus"),
                "typical_bracket_width": curv.get("typical_bracket_width"),
                "max_eigen_residual": curv.get("max_eigen_residual"),
                "unresolved": bool(diag.get("unresolved", False) or meta.get("hit_likelihood_budget")),
            }
            target_rows.append(row)

            fig1_rows.append(
                {
                    "m": m,
                    "p": bun["p"],
                    "seed": seed,
                    "coverage": cov["p_hat"],
                    "exit_count": cov["exit_count"],
                    "exit_episodes": cov["exit_episodes"],
                    "mcse": cov["mcse"] if cov["mcse_estimable"] else "",
                    "mcse_estimable": cov["mcse_estimable"],
                    "all_inside": cov["all_inside"],
                    "theorem_floor": bun["coverage_floor"],
                }
            )

            # S1 ECDF grid
            zs = np.linspace(0.0, 1.05, 211)
            emp = np.array([(flat_HB <= z).mean() for z in zs])
            prior = prior_cdf_H(zs * bun["B_m"], m=m, b0=b0, sigma=sigma)
            for z, e, pr in zip(zs, emp, prior):
                s1_rows.append(
                    {
                        "m": m,
                        "seed": seed,
                        "z": float(z),
                        "ecdf_posterior": float(e),
                        "cdf_prior": float(pr),
                    }
                )

            # S2 efficiency
            if diag.get("diagnostics"):
                # Approximate: use ess_bulk from p_probe via recompute light ESS proxy
                # Store likelihood-normalized placeholder using ArviZ if diagnostics present
                lik_ret = 0
                for cs in meta.get("chain_stats", []):
                    lik_ret += int(cs.get("likelihood_evals_retained", 0))
                t_ret = float(meta.get("wall_time_s") or 0.0)
                # Per-probe bulk ESS from az if available — fall back to nan
                try:
                    import arviz as az

                    idata = az.from_dict(
                        {
                            "posterior": {
                                f"p{k}": obs["p_probe"][:, :T, k]
                                for k in range(obs["p_probe"].shape[-1])
                            }
                        }
                    )
                    ess = az.ess(idata, method="bulk")
                    post = ess.posterior if hasattr(ess, "posterior") else ess
                    ess_vals = [
                        float(np.asarray(post[f"p{k}"]).item())
                        for k in range(obs["p_probe"].shape[-1])
                    ]
                    med_ess = float(np.median(ess_vals))
                    min_ess = float(np.min(ess_vals))
                except Exception:  # noqa: BLE001
                    med_ess = float("nan")
                    min_ess = float("nan")
                s2_rows.append(
                    {
                        "m": m,
                        "seed": seed,
                        "median_probe_ess": med_ess,
                        "min_probe_ess": min_ess,
                        "likelihood_evals_retained": lik_ret,
                        "wall_time_retained_s": t_ret,
                        "ess_per_1k_evals": med_ess / (lik_ret / 1000.0) if lik_ret else np.nan,
                        "ess_per_sec": med_ess / t_ret if t_ret > 0 else np.nan,
                    }
                )

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            print(f"No rows for {path}")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path} ({len(rows)} rows)")

    write_csv(artifacts / "target_summary.csv", target_rows)
    write_csv(artifacts / "figure1_coverage.csv", fig1_rows)
    write_csv(artifacts / "figureS1_cutoff.csv", s1_rows)
    write_csv(artifacts / "figureS2_efficiency.csv", s2_rows)

    # pCN cross-check comparison
    ess_dir = runs_root / run_dir_name(1024, 0, "ess")
    pcn_dir = runs_root / run_dir_name(1024, 0, "pcn")
    cross = {"status": "missing"}
    if ess_dir.exists() and pcn_dir.exists():
        e = np.load(ess_dir / "observables.npz")
        p = np.load(pcn_dir / "observables.npz")
        Te = int(np.isfinite(e["H"]).all(axis=0).sum())
        Tp = int(np.isfinite(p["H"]).all(axis=0).sum())

        def mean_se(arr):
            arr = arr.ravel()
            return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))

        comps = {}
        for name, ae, ap in [
            ("V", e["V"][:, :Te], p["V"][:, :Tp]),
            ("H", e["H"][:, :Te], p["H"][:, :Tp]),
        ]:
            me, se_e = mean_se(ae)
            mp, se_p = mean_se(ap)
            comps[name] = {
                "ess_mean": me,
                "pcn_mean": mp,
                "diff": mp - me,
                "combined_se": float(np.sqrt(se_e**2 + se_p**2)),
                "flag_gt_3se": abs(mp - me) > 3 * np.sqrt(se_e**2 + se_p**2),
            }
        q95_e = order_stat_quantile(e["H"][:, :Te], 0.95)
        q95_p = order_stat_quantile(p["H"][:, :Tp], 0.95)
        comps["H_q95"] = {"ess": q95_e, "pcn": q95_p, "diff": q95_p - q95_e}
        for k in range(e["p_probe"].shape[-1]):
            me, se_e = mean_se(e["p_probe"][:, :Te, k])
            mp, se_p = mean_se(p["p_probe"][:, :Tp, k])
            comps[f"probe_{k}"] = {
                "ess_mean": me,
                "pcn_mean": mp,
                "diff": mp - me,
                "combined_se": float(np.sqrt(se_e**2 + se_p**2)),
                "flag_gt_3se": abs(mp - me) > 3 * np.sqrt(se_e**2 + se_p**2),
            }
        cross = {"status": "ok", "comparisons": comps}
    atomic_save_json(artifacts / "pcn_crosscheck.json", cross)
    print(f"pCN cross-check: {cross['status']}")


if __name__ == "__main__":
    main()
