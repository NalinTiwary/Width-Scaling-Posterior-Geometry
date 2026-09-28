#!/usr/bin/env python3
"""
Tables and figure sidecars for the extension (addendum §4, §6).

Writes to artifacts_dir:
  target_summary.csv     one row per target (diagnostics, coverage, H/B, curvature, efficiency)
  setting_summary.csv    one row per (kind, n, m): replicate medians and between-replicate spread
  matched_diagonal.csv   settings with n/sqrt(m) = 2
  figure1_coverage.csv, figure2_curvature.csv, figureA_realdata.csv, figureB_cutoff.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.diagnostics import order_stat_quantile, posterior_idata  # noqa: E402
from cylinder.ext.cli import base_parser, load  # noqa: E402
from cylinder.ext.run import run_dir  # noqa: E402
from cylinder.ext.targets import all_targets  # noqa: E402
from cylinder.theorem import prior_cdf_H  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"no rows for {path.name}")
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path.name} ({len(rows)} rows)")


def bulk_ess(x: np.ndarray) -> float:
    import arviz as az

    try:
        e = az.ess(posterior_idata({"x": x}), method="bulk")
        e = e.posterior if hasattr(e, "posterior") else e
        return float(np.asarray(e["x"]))
    except Exception:  # noqa: BLE001
        return float("nan")


def curvature_stats(cz: dict[str, np.ndarray]) -> dict:
    inside = cz["inside"] == 1
    finite = np.isfinite(cz["d_plus"])
    ok = inside & finite
    dm, dp = cz["d_minus"][ok], cz["d_plus"][ok]
    out = {
        "curv_n_states": int(finite.sum()),
        "curv_n_inside": int(ok.sum()),
        "median_d_minus": order_stat_quantile(dm, 0.5),
        "median_d_plus": order_stat_quantile(dp, 0.5),
        "q95_d_minus": order_stat_quantile(dm, 0.95),
        "q95_d_plus": order_stat_quantile(dp, 0.95),
        "max_d_plus": float(dp.max()) if dp.size else float("nan"),
        "median_bracket_width": float(np.median(dp - dm)) if dp.size else float("nan"),
        "max_eigen_residual": float(np.nanmax(cz["max_eigen_residual"])) if finite.any() else float("nan"),
    }
    # Monte Carlo spread from chain boundaries: per-chain statistics.
    per_med, per_q95 = [], []
    for c in range(cz["d_plus"].shape[0]):
        row = cz["d_plus"][c][ok[c]]
        if row.size:
            per_med.append(order_stat_quantile(row, 0.5))
            per_q95.append(order_stat_quantile(row, 0.95))
    C = len(per_med)
    out["chain_se_median_d_plus"] = float(np.std(per_med, ddof=1) / math.sqrt(C)) if C > 1 else float("nan")
    out["chain_se_q95_d_plus"] = float(np.std(per_q95, ddof=1) / math.sqrt(C)) if C > 1 else float("nan")
    return out


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg, _, artifacts, _ = load(args)

    rows, fig1, fig2, figA, figB = [], [], [], [], []
    zgrid = np.linspace(0.0, 1.05, 211)
    post = cfg.get("postprocess", {})
    cutoff_width = int(post.get("cutoff_width", 4096))
    diag_ratio = float(post.get("matched_n_over_sqrt_m", 2.0))
    protocol_max = int(cfg["sampling"].get("protocol_max_retained", max(cfg["sampling"]["retained_extensions"])))
    for t in all_targets(cfg):
        rd = run_dir(artifacts, t)
        if not (rd / "metadata.json").exists():
            continue
        meta = json.loads((rd / "metadata.json").read_text())
        diag = json.loads((rd / "diagnostics.json").read_text()) if (rd / "diagnostics.json").exists() else {}
        obs = np.load(rd / "observables.npz")
        cz = dict(np.load(rd / "curvature.npz"))
        T = int(np.isfinite(obs["H"]).all(axis=0).sum())
        th = meta["theory"]
        HB = obs["H_over_B"][:, :T]
        inside = obs["inside"][:, :T]
        cov = diag.get("coverage", {})
        dd = diag.get("diagnostics", {})
        cs = meta["chain_stats"]
        evals_ret = sum(c["likelihood_evals_retained"] for c in cs)
        wall = float(meta["wall_time_sampling_s"])
        ess_sub = [bulk_ess(obs["f_sub"][:, :T, i]) for i in range(obs["f_sub"].shape[-1])]
        med_ess = float(np.nanmedian(ess_sub)) if T > 3 else float("nan")
        row = {
            "target": t.name, "kind": t.kind, "n": t.n, "m": t.m, "rep": t.rep,
            "p_sampled": meta["p"], "p_full": meta["p_full"], "coordinates": meta["coordinates"],
            "B": th["B"], "D_th": th["D_th"], "curvature_margin": th["curvature_margin"],
            "n_over_sqrt_m": th["n_over_sqrt_m"], "M2": th["M2"], "M4": th["M4"],
            "T": T, "beyond_protocol": T > protocol_max,
            "diagnostics_pass": diag.get("pass"), "unresolved": diag.get("unresolved", False),
            "limiting_observable": dd.get("ess_bulk_argmin"),
            "hit_budget": meta["hit_likelihood_budget"], "rhat_max": dd.get("rhat_max"),
            "ess_bulk_min": dd.get("ess_bulk_min"), "ess_tail_min": dd.get("ess_tail_min"),
            "coverage": float(inside.mean()) if T else float("nan"),
            "exit_count": int((inside == 0).sum()),
            "coverage_mcse": cov.get("mcse"), "all_inside": bool((inside == 1).all()) if T else False,
            "H_over_B_q95": order_stat_quantile(HB, 0.95), "H_over_B_q99": order_stat_quantile(HB, 0.99),
            "H_over_B_max": float(np.nanmax(HB)) if T else float("nan"),
            "mean_V": float(np.nanmean(obs["V"][:, :T])) if T else float("nan"),
            **curvature_stats(cz),
            "median_f_sub_bulk_ess": med_ess,
            "likelihood_evals": meta["likelihood_evals"], "likelihood_evals_retained": evals_ret,
            "ess_per_1k_evals": med_ess / (evals_ret / 1000.0) if evals_ret else float("nan"),
            "ess_per_sec": med_ess / wall if wall > 0 else float("nan"),
            "wall_time_sampling_s": wall, "wall_time_curvature_s": meta["wall_time_curvature_s"],
            "device": meta["effective_device"],
        }
        rows.append(row)
        base = {k: row[k] for k in ("target", "kind", "n", "m", "rep", "B", "D_th", "n_over_sqrt_m")}
        (figA if t.kind == "fmnist" else fig1).append(
            {**base, **{k: row[k] for k in ("coverage", "exit_count", "all_inside", "coverage_mcse",
                                             "H_over_B_q99")}, "coverage_floor": 1 - 1 / t.m}
        )
        c2 = {**base, **{k: row[k] for k in ("curv_n_inside", "median_d_minus", "median_d_plus", "q95_d_minus",
                                              "q95_d_plus", "chain_se_median_d_plus", "chain_se_q95_d_plus",
                                              "max_eigen_residual")}}
        if t.kind == "fmnist":
            figA[-1].update({k: v for k, v in c2.items() if k not in base})
        else:
            fig2.append(c2)
        if t.kind == "orth" and t.m == cutoff_width and T:
            flat = np.sort(HB.ravel())
            ecdf = np.searchsorted(flat, zgrid, side="right") / flat.size
            prior = prior_cdf_H(zgrid * th["B"], m=t.m, b0=float(cfg["prior"]["b0"]),
                                sigma=float(cfg["prior"]["sigma"]))
            for z, e, p in zip(zgrid, ecdf, prior):
                figB.append({"n": t.n, "m": t.m, "rep": t.rep, "z": float(z), "ecdf_posterior": float(e),
                             "cdf_prior": float(p)})

    settings = []
    keyset = sorted({(r["kind"], r["n"], r["m"]) for r in rows})
    for kind, n, m in keyset:
        g = [r for r in rows if (r["kind"], r["n"], r["m"]) == (kind, n, m)]

        def med(key: str) -> float:
            v = np.asarray([r[key] for r in g], dtype=float)
            return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")

        def sd(key: str) -> float:
            v = np.asarray([r[key] for r in g], dtype=float)
            v = v[np.isfinite(v)]
            return float(np.std(v, ddof=1)) if v.size > 1 else float("nan")

        settings.append({
            "kind": kind, "n": n, "m": m, "n_reps": len(g), "B": g[0]["B"], "D_th": g[0]["D_th"],
            "curvature_margin": g[0]["curvature_margin"], "n_over_sqrt_m": g[0]["n_over_sqrt_m"],
            "all_reps_pass_diagnostics": all(bool(r["diagnostics_pass"]) for r in g),
            "coverage_median": med("coverage"), "coverage_min": min(r["coverage"] for r in g),
            "total_exits": sum(r["exit_count"] for r in g),
            "H_over_B_q99_median": med("H_over_B_q99"), "H_over_B_q99_max": max(r["H_over_B_q99"] for r in g),
            "median_d_plus_median": med("median_d_plus"), "median_d_minus_median": med("median_d_minus"),
            "q95_d_plus_median": med("q95_d_plus"), "q95_d_minus_median": med("q95_d_minus"),
            "q95_d_plus_between_rep_sd": sd("q95_d_plus"),
            "q95_d_plus_chain_se_mean": med("chain_se_q95_d_plus"),
            "q95_d_plus_over_D_th": med("q95_d_plus") / g[0]["D_th"],
            "ess_per_1k_evals_median": med("ess_per_1k_evals"), "ess_per_sec_median": med("ess_per_sec"),
            "rhat_max": max(float(r["rhat_max"] or np.nan) for r in g),
            "likelihood_evals_total": sum(r["likelihood_evals"] for r in g),
        })
    matched = [s for s in settings if s["kind"] == "orth" and abs(s["n_over_sqrt_m"] - diag_ratio) < 1e-9]

    write_csv(artifacts / "target_summary.csv", rows)
    write_csv(artifacts / "setting_summary.csv", settings)
    write_csv(artifacts / "matched_diagonal.csv", matched)
    write_csv(artifacts / "figure1_coverage.csv", fig1)
    write_csv(artifacts / "figure2_curvature.csv", fig2)
    write_csv(artifacts / "figureA_realdata.csv", figA)
    write_csv(artifacts / "figureB_cutoff.csv", figB)


if __name__ == "__main__":
    main()
