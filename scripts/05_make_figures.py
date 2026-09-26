#!/usr/bin/env python3
"""Publication figures 1, 2, S1, S2 from CSV sidecars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cylinder.config import artifacts_root, load_config  # noqa: E402


def fig1_coverage(artifacts: Path, out: Path) -> None:
    path = artifacts / "figure1_coverage.csv"
    if not path.exists():
        print("Skip Fig1: missing figure1_coverage.csv")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    widths = sorted(df["m"].unique())
    for seed, g in df.groupby("seed"):
        g = g.sort_values("m")
        y = g["coverage"].to_numpy()
        x = g["m"].to_numpy()
        all_inside = g["all_inside"].astype(bool).to_numpy()
        ax.plot(x, y, "-", lw=1.0, label=f"seed {seed}")
        for xi, yi, inside, mcse_ok, mcse in zip(
            x, y, all_inside, g["mcse_estimable"], g["mcse"]
        ):
            if inside:
                ax.plot(xi, yi, "o", mfc="none", mec="C0" if seed == 0 else None, ms=8)
            else:
                ax.plot(xi, yi, "o", ms=6)
            if bool(mcse_ok) and pd.notna(mcse) and mcse != "":
                ax.errorbar(xi, yi, yerr=float(mcse), fmt="none", capsize=3, elinewidth=0.8)
    # theorem floor
    m_grid = np.array(widths, dtype=float)
    ax.plot(m_grid, 1.0 - 1.0 / m_grid, "k--", lw=1.2, label="theorem lower bound")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("width $m$")
    ax.set_ylabel(r"posterior fraction inside $G_{B_m}$")
    ymin = min(0.99, float(df["coverage"].min()) - 0.002)
    ax.set_ylim(ymin, 1.0005)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cylinder coverage vs width")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure1_coverage.pdf")
    fig.savefig(out / "figure1_coverage.png", dpi=150)
    plt.close(fig)
    print("Wrote figure1_coverage")


def fig2_curvature(artifacts: Path, out: Path) -> None:
    path = artifacts / "figure2_curvature.csv"
    if not path.exists():
        print("Skip Fig2: missing figure2_curvature.csv")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    widths = sorted(df["m"].unique())
    # theorem envelope
    from cylinder.theorem import theorem_bundle

    ms = np.array(widths, dtype=float)
    Dth = [theorem_bundle(m=int(m))["D_th"] for m in ms]
    ax.plot(ms, Dth, "k--", lw=2.0, label=r"$D_{\mathrm{th}}(m)$")
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0, label="$d=1$")

    offsets = {-0: -0.05, 0: 0.0, 1: 0.05, 2: 0.1}
    for seed, g in df.groupby("seed"):
        g = g.sort_values("m")
        x = g["m"].to_numpy(dtype=float) * (2 ** (0.04 * (seed - 1)))
        lo = g["q95_d_minus"].to_numpy(dtype=float)
        hi = g["q95_d_plus"].to_numpy(dtype=float)
        for xi, a, b in zip(x, lo, hi):
            if not np.isfinite(b):
                continue
            if np.isfinite(a) and a > 0 and b > 0:
                ax.vlines(xi, a, b, color=f"C{seed}", lw=2)
                ax.plot(xi, b, "o", color=f"C{seed}", ms=5)
            elif (not np.isfinite(a) or a <= 0) and b > 0:
                ax.annotate(
                    "",
                    xy=(xi, b),
                    xytext=(xi, b * 0.3 if b * 0.3 > 0 else b * 0.5),
                    arrowprops=dict(arrowstyle="->", color=f"C{seed}"),
                )
                ax.plot(xi, b, "o", color=f"C{seed}", ms=5)
            elif a == 0 and b == 0:
                ax.plot(xi, 1e-4, "x", color=f"C{seed}", ms=8)
        # connect upper endpoints
        mask = np.isfinite(hi) & (hi > 0)
        ax.plot(x[mask], hi[mask], "-", color=f"C{seed}", lw=1.0, label=f"seed {seed} upper")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("width $m$")
    ax.set_ylabel(r"normalized negative likelihood curvature $d_H$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Curvature deficit vs width")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure2_curvature.pdf")
    fig.savefig(out / "figure2_curvature.png", dpi=150)
    plt.close(fig)
    print("Wrote figure2_curvature")


def figS1_cutoff(artifacts: Path, out: Path) -> None:
    path = artifacts / "figureS1_cutoff.csv"
    if not path.exists():
        print("Skip S1: missing figureS1_cutoff.csv")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    color = {256: "C0", 1024: "C1", 4096: "C2"}
    for (m, seed), g in df.groupby(["m", "seed"]):
        g = g.sort_values("z")
        ax.plot(g["z"], g["ecdf_posterior"], "-", color=color.get(m, "k"), lw=1.0, alpha=0.9)
        ax.plot(
            g["z"],
            g["cdf_prior"],
            "--",
            color=color.get(m, "k"),
            lw=1.0,
            alpha=0.7,
        )
    ax.axvline(1.0, color="k", lw=1.2)
    ax.set_xlabel(r"$z = H / B_m$")
    ax.set_ylabel("empirical CDF")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.02)
    # legend proxies
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="C0", label="m=256"),
        Line2D([0], [0], color="C1", label="m=1024"),
        Line2D([0], [0], color="C2", label="m=4096"),
        Line2D([0], [0], color="k", ls="-", label="posterior"),
        Line2D([0], [0], color="k", ls="--", label="prior"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    ax.set_title("Cylinder cutoff calibration")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figureS1_cutoff.pdf")
    fig.savefig(out / "figureS1_cutoff.png", dpi=150)
    plt.close(fig)
    print("Wrote figureS1_cutoff")


def figS2_efficiency(artifacts: Path, out: Path) -> None:
    path = artifacts / "figureS2_efficiency.csv"
    if not path.exists():
        print("Skip S2: missing figureS2_efficiency.csv")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("Skip S2: empty")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for seed, g in df.groupby("seed"):
        g = g.sort_values("m")
        axes[0].plot(g["m"], g["ess_per_1k_evals"], "o-", label=f"seed {seed}")
        axes[1].plot(g["m"], g["ess_per_sec"], "o-", label=f"seed {seed}")
    for ax, ylab, title in zip(
        axes,
        ["median probe ESS / 1k lik. evals", "median probe ESS / second"],
        ["ESS per evaluation", "ESS per second"],
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["m"].unique()))
        ax.set_xticklabels([str(w) for w in sorted(df["m"].unique())])
        ax.set_xlabel("width $m$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figureS2_efficiency.pdf")
    fig.savefig(out / "figureS2_efficiency.png", dpi=150)
    plt.close(fig)
    print("Wrote figureS2_efficiency")


def write_captions(artifacts: Path) -> None:
    text = """# Figure captions (fill observed results after runs)

## Figure 1
Posterior coverage of the theorem-defined cylinder as network width grows. Each curve represents one paired prior-center construction, using four unrestricted elliptical slice chains per target. The horizontal axis is width on a logarithmic scale; the vertical axis is the retained fraction satisfying $\\max_j|a_j|\\le B_m$, with $B_m$ fixed from Lemma 4.1 using $s=1$. Hidden weights remain unrestricted. The dashed curve is the analytic lower bound $1-1/m$. [Describe observed results here.] Open markers denote no observed exits and do not imply zero Monte Carlo uncertainty. High occupancy supports the domain’s relevance despite increasing parameter dimension; it does not resolve the rate of rare exits.

## Figure 2
Negative likelihood curvature versus width on the high-mass cylinder. For each paired prior-center seed, vertical intervals bracket the empirical 95th percentile of $d_H=\\sigma^2[-\\lambda_{\\min}(\\nabla^2 V)]_+$ over 128 prespecified posterior states, restricted to the cylinder. Endpoints use the block-diagonal residual Hessian and a Rayleigh–Ritz projection of the full Hessian; they are numerical bounds, not confidence intervals. Axes are logarithmic for positive values. The dashed curve is the finite-width uniform envelope from Theorem 2. [Describe observed results here.] A decrease indicates less adverse likelihood curvature relative to Gaussian prior precision. Together with coverage, this tests the proposed mechanism on a region containing substantial posterior probability.

## Figure S1
How much of the permitted head radius is used by posterior draws? The horizontal axis is the maximum absolute head coefficient divided by the theorem cutoff; the vertical axis is its empirical CDF under unrestricted posterior sampling. The vertical line at one is the cylinder boundary. Solid curves show posterior draws; dashed curves show the exact Gaussian-prior maximum distribution. Concentration to the left of one explains near-unit coverage and quantifies cutoff conservatism. The prior reference is not substituted for posterior mass.

## Figure S2
Practical sampling efficiency of elliptical slice sampling across widths. Left: median bulk ESS across eight fixed predictive probabilities per 1,000 likelihood evaluations. Right: the same ESS per second on fixed hardware. Each point represents one prior-center seed with four chains. Unsuccessful slice-angle proposals count toward cost. These measurements concern this algorithm, implementation, and observable set; they do not estimate the PI/LSI constants or follow automatically from the continuous-time theorem.
"""
    (artifacts / "captions.md").write_text(text, encoding="utf-8")
    print("Wrote captions.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    artifacts = artifacts_root(cfg, ROOT)
    out = artifacts / "figures"
    fig1_coverage(artifacts, out)
    fig2_curvature(artifacts, out)
    figS1_cutoff(artifacts, out)
    figS2_efficiency(artifacts, out)
    write_captions(artifacts)


if __name__ == "__main__":
    main()
