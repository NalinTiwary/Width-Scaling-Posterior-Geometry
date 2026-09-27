#!/usr/bin/env python3
"""Evaluate curvature brackets at 128 prespecified states per ESS target (§6)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cylinder.centers import assemble_theta0, load_centers  # noqa: E402
from cylinder.config import artifacts_root, load_config  # noqa: E402
from cylinder.curvature import deficit_bracket  # noqa: E402
from cylinder.data import load_data_bundle  # noqa: E402
from cylinder.diagnostics import order_stat_quantile  # noqa: E402
from cylinder.io import atomic_save_json, atomic_save_npz, run_dir_name  # noqa: E402
from cylinder.observables import curvature_state_indices  # noqa: E402
from cylinder.theorem import theorem_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    artifacts = artifacts_root(cfg, ROOT)
    data = load_data_bundle(artifacts / "data.npz")
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    n, d = X.shape
    sigma = float(cfg["prior"]["sigma"])
    K = int(cfg["curvature"]["K"])
    n_per_chain = int(cfg["curvature"]["states_per_chain"])
    d_th_tol = float(cfg["curvature"]["d_th_violation_tol"])

    rows = []
    summary_rows = []

    for m in cfg["widths"]:
        bun = theorem_bundle(
            m=m,
            n=n,
            d=d,
            C=cfg["architecture"]["classes"],
            b0=float(cfg["prior"]["b0"]),
            sigma=sigma,
            M2=float(cfg["theorem"]["M2"]),
            M4=float(cfg["theorem"]["M4"]),
            c2=float(cfg["theorem"]["c2"]),
        )
        for seed in cfg["prior"]["center_seeds"]:
            run_dir = artifacts / "runs" / run_dir_name(m, seed, "ess")
            if not run_dir.exists():
                print(f"skip missing {run_dir.name}")
                continue
            obs = np.load(run_dir / "observables.npz")
            if "z_curv" in obs.files:
                Zc = obs["z_curv"]  # (C, n_per_chain, p)
                idxs = np.asarray(obs["curv_indices"], dtype=np.int64)
            else:
                # Older runs stored every draw; select the same prespecified states.
                Z = obs["z"]
                valid_T = int(np.isfinite(Z[:, :, 0]).all(axis=0).sum())
                idxs = curvature_state_indices(valid_T, n_per_chain)
                Zc = Z[:, idxs]
            C = Zc.shape[0]
            bank = load_centers(artifacts / f"centers_seed{seed}.npz")
            theta0 = assemble_theta0(bank["U"], m=m, b0=float(cfg["prior"]["b0"]))

            d_minus_in = []
            d_plus_in = []
            n_inside = 0
            state_records = []

            for c in range(C):
                for j, t in enumerate(idxs):
                    z = Zc[c, j]
                    if not np.all(np.isfinite(z)):
                        # Chain stopped early (likelihood budget) before this slot.
                        continue
                    theta = theta0 + sigma * z
                    H = float(np.max(np.abs(theta[0 :: (1 + d)])))
                    inside = int(H <= bun["B_m"])
                    br = deficit_bracket(theta, X, y, m, sigma=sigma, K=K)
                    rec = {
                        "m": m,
                        "seed": seed,
                        "chain": c,
                        "draw_index": int(t),
                        "state_slot": j,
                        "inside": inside,
                        "H": H,
                        "ell": br["ell"],
                        "u": br["u"],
                        "d_minus": br["d_minus"],
                        "d_plus": br["d_plus"],
                        "max_eigen_residual": br["max_eigen_residual"],
                        "D_th": bun["D_th"],
                    }
                    state_records.append(rec)
                    rows.append(rec)
                    if inside:
                        n_inside += 1
                        d_minus_in.append(br["d_minus"])
                        d_plus_in.append(br["d_plus"])
                        if br["d_plus"] > bun["D_th"] + d_th_tol:
                            print(
                                f"WARNING d_+ > D_th at m={m} seed={seed} "
                                f"chain={c} t={t}: {br['d_plus']} > {bun['D_th']}"
                            )

            q95_lo = order_stat_quantile(np.asarray(d_minus_in), 0.95) if d_minus_in else np.nan
            q95_hi = order_stat_quantile(np.asarray(d_plus_in), 0.95) if d_plus_in else np.nan
            q50_lo = order_stat_quantile(np.asarray(d_minus_in), 0.50) if d_minus_in else np.nan
            withhold = n_inside < 100
            summary_rows.append(
                {
                    "m": m,
                    "seed": seed,
                    "n_states": len(state_records),
                    "n_inside": n_inside,
                    "q95_d_minus": q95_lo,
                    "q95_d_plus": q95_hi,
                    "median_d_minus": q50_lo,
                    "max_d_plus": float(np.max(d_plus_in)) if d_plus_in else np.nan,
                    "D_th": bun["D_th"],
                    "withhold_q95_conclusion": withhold,
                    "typical_bracket_width": (
                        float(np.median(np.asarray(d_plus_in) - np.asarray(d_minus_in)))
                        if d_minus_in
                        else np.nan
                    ),
                    "max_eigen_residual": (
                        max(r["max_eigen_residual"] for r in state_records)
                        if state_records
                        else np.nan
                    ),
                }
            )
            atomic_save_npz(
                run_dir / "curvature_states.npz",
                indices=idxs,
                **{k: np.asarray([r[k] for r in state_records]) for k in (
                    "inside", "H", "ell", "u", "d_minus", "d_plus", "max_eigen_residual"
                )},
            )
            print(
                f"m={m} seed={seed}: inside={n_inside}/{len(state_records)} "
                f"q95=[{q95_lo:.4g},{q95_hi:.4g}] D_th={bun['D_th']:.4g}"
            )

    # Write global curvature.csv
    curv_path = artifacts / "curvature.csv"
    if rows:
        with open(curv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    atomic_save_json(artifacts / "curvature_summary.json", {"targets": summary_rows})
    # Also CSV for figure 2
    fig2 = artifacts / "figure2_curvature.csv"
    if summary_rows:
        with open(fig2, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"Wrote {curv_path} and {fig2}")


if __name__ == "__main__":
    main()
