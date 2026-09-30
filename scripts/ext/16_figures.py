#!/usr/bin/env python3
"""Two main figures and two appendix figures for the extension (addendum §6), plus captions."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.ext.cli import base_parser, load  # noqa: E402

N_COLOR = {64: "C0", 128: "C1", 256: "C2"}
DODGE = 0.05  # in log2(m) units per replicate


def _color(n: int) -> str:
    return N_COLOR.get(int(n), "C3")


def _xaxis(ax, widths) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([f"{w:,}" for w in widths])
    ax.set_xlabel("width $m$")
    ax.minorticks_off()


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    print(f"Wrote {name}")


def _dodge(m: np.ndarray, rep: np.ndarray, n_reps: int, shift: float = 0.0) -> np.ndarray:
    return m * 2.0 ** (DODGE * (rep - (n_reps - 1) / 2) + shift)


def fig1(art: Path, out: Path) -> None:
    p = art / "figure1_coverage.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    widths = sorted(df["m"].unique())
    reps = sorted(df["rep"].unique())
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, (n, g) in enumerate(df.groupby("n")):
        col = _color(n)
        shift = 0.18 * (i - 1) * DODGE * len(reps)
        x = _dodge(g["m"].to_numpy(float), g["rep"].to_numpy(), len(reps), shift)
        filled = ~g["all_inside"].astype(bool).to_numpy()
        ax.scatter(x[~filled], g["coverage"].to_numpy()[~filled], s=18, facecolors="none", edgecolors=col, lw=0.9)
        ax.scatter(x[filled], g["coverage"].to_numpy()[filled], s=18, color=col)
        med = g.groupby("m")["coverage"].median()
        ax.plot(med.index * 2.0**shift, med.values, "-", color=col, lw=1.2)
        ax.scatter(med.index * 2.0**shift, med.values, s=70, marker="D", facecolors="none", edgecolors=col,
                   lw=1.5, label=f"$n={n}$ (median)")
    mg = np.array(widths, dtype=float)
    ax.plot(mg, 1 - 1 / mg, "k--", lw=1.1, label="theorem floor $1-1/m$")
    ax.scatter([], [], s=18, facecolors="none", edgecolors="0.4", label="replicate, no observed exits")
    _xaxis(ax, widths)
    ax.set_ylabel(r"posterior fraction inside $G_{B_{m,n}}$")
    lo = min(0.99, float(df["coverage"].min()) - 0.002)
    ax.set_ylim(lo, 1.0006)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Cylinder coverage vs width, by sample size")
    fig.tight_layout()
    _save(fig, out, "ext_figure1_coverage")


def fig2(art: Path, out: Path) -> None:
    p = art / "figure2_curvature.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    widths = sorted(df["m"].unique())
    reps = sorted(df["rep"].unique())
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for i, (n, g) in enumerate(df.groupby("n")):
        col = _color(n)
        shift = 0.18 * (i - 1) * DODGE * len(reps)
        x = _dodge(g["m"].to_numpy(float), g["rep"].to_numpy(), len(reps), shift)
        pos = g["q95_d_plus"].to_numpy() > 0
        ax.vlines(x[pos], g["q95_d_minus"].to_numpy()[pos].clip(min=1e-12), g["q95_d_plus"].to_numpy()[pos],
                  color=col, lw=1.6)
        ax.scatter(x[pos], g["q95_d_plus"].to_numpy()[pos], marker="^", s=22, color=col)
        ax.vlines(x[pos], g["median_d_minus"].to_numpy()[pos].clip(min=1e-12),
                  g["median_d_plus"].to_numpy()[pos], color=col, lw=1.0, alpha=0.7)
        ax.scatter(x[pos], g["median_d_plus"].to_numpy()[pos], marker="o", s=16, facecolors="none", edgecolors=col)
        mq = g.groupby("m")["q95_d_plus"].median()
        ax.plot(mq.index * 2.0**shift, mq.values, "-", color=col, lw=1.1, label=f"$n={n}$")
        dth = g.groupby("m")["D_th"].first()
        ax.plot(dth.index, dth.values, "--", color=col, lw=1.0, alpha=0.8)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax.text(widths[0], 1.04, "$d=1$", color="0.4", fontsize=8, va="bottom")
    ax.scatter([], [], marker="^", color="0.3", s=22, label="95th pct. bracket $[d_-,d_+]$")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="0.3", s=16, label="median bracket")
    ax.plot([], [], "--", color="0.3", lw=1.0, label=r"$D_{\rm th}(m,n)$")
    ax.set_yscale("log")
    _xaxis(ax, widths)
    ax.set_ylabel(r"$d_H=\sigma^2[-\lambda_{\min}(\nabla^2 V)]_+$ (inside cylinder)")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax.set_title("Negative likelihood curvature vs width, by sample size")
    fig.tight_layout()
    _save(fig, out, "ext_figure2_curvature")


def figA(art: Path, out: Path) -> None:
    p = art / "figureA_realdata.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    widths = sorted(df["m"].unique())
    reps = sorted(df["rep"].unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    x = _dodge(df["m"].to_numpy(float), df["rep"].to_numpy(), len(reps))
    allin = df["all_inside"].astype(bool).to_numpy()
    a1.scatter(x[allin], df["coverage"][allin], facecolors="none", edgecolors="C2", s=30, label="no observed exits")
    a1.scatter(x[~allin], df["coverage"][~allin], color="C2", s=30, label="replicate")
    mg = np.array(widths, dtype=float)
    a1.plot(mg, 1 - 1 / mg, "k--", lw=1.0, label="$1-1/m$")
    _xaxis(a1, widths)
    a1.set_ylim(min(0.99, float(df["coverage"].min()) - 0.002), 1.0006)
    a1.set_ylabel(r"fraction inside $G_{B_{m,n}}$")
    a1.set_title("Coverage (Fashion-MNIST 0 vs 6, $n=256$)")
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    pos = df["q95_d_plus"].to_numpy() > 0
    a2.vlines(x[pos], df["q95_d_minus"][pos].clip(lower=1e-12), df["q95_d_plus"][pos], color="C2", lw=1.6)
    a2.scatter(x[pos], df["q95_d_plus"][pos], marker="^", color="C2", s=22, label="95th pct. bracket")
    a2.vlines(x[pos], df["median_d_minus"][pos].clip(lower=1e-12), df["median_d_plus"][pos], color="C2",
              lw=1.0, alpha=0.7)
    a2.scatter(x[pos], df["median_d_plus"][pos], marker="o", facecolors="none", edgecolors="C2", s=16,
               label="median bracket")
    a2.axhline(1.0, color="0.5", ls=":", lw=1.0)
    dth = df.groupby("m")["D_th"].median()
    a2.plot(dth.index, dth.values, "k--", lw=1.0, label=r"safe envelope $D_{\rm th}$ (data $M_2$)")
    a2.set_yscale("log")
    _xaxis(a2, widths)
    a2.set_ylabel(r"$d_H$ (inside cylinder)")
    a2.set_title("Curvature brackets")
    a2.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    _save(fig, out, "ext_figureA_realdata")


def figB(art: Path, out: Path) -> None:
    p = art / "figureB_cutoff.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for n, g in df.groupby("n"):
        col = _color(n)
        for j, (rep, gr) in enumerate(g.groupby("rep")):
            ax.plot(gr["z"], gr["ecdf_posterior"], "-", color=col, lw=1.0, alpha=0.85,
                    label=f"$n={n}$ posterior" if j == 0 else None)
        pr = g[g["rep"] == g["rep"].min()]
        ax.plot(pr["z"], pr["cdf_prior"], "--", color=col, lw=1.0, alpha=0.8, label=f"$n={n}$ prior")
    ax.axvline(1.0, color="k", lw=1.2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$z=H/B_{m,n}$,  $H=\max_j|a_j|$")
    ax.set_ylabel("CDF")
    m = int(df["m"].iloc[0])
    ax.set_title(f"Cutoff calibration at $m={m:,}$")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    _save(fig, out, "ext_figureB_cutoff")


CAPTIONS = """# Extension figure captions (draft)

**Figure 1 (main).** Posterior coverage of the theorem cylinder G_{B_{m,n}} versus width for
n = 64, 128, 256 on orthonormal inputs in d = 256. Small markers are the three joint
data/prior-center replicates (open = no observed exits, which does not imply zero Monte Carlo
uncertainty); diamonds are replicate medians. The dashed curve is the analytic floor 1 − 1/m,
common to all n because s = 1. B_{m,n} is fixed from the addendum's formula before sampling and
grows roughly like sqrt(n); see Appendix Figure B and the H/B table for how much of the cutoff
the posterior uses. [Describe observed results here.]

**Figure 2 (main).** Negative likelihood curvature d_H = σ²[−λ_min(∇²V)]₊ over 256 prespecified
posterior states per target (64 evenly spaced retained states per chain), restricted to the
cylinder. For each replicate, the thick bar with a triangle brackets the empirical 95th
percentile and the thin bar with a circle brackets the median; endpoints are numerical bounds
(block residual spectrum below; Rayleigh–Ritz on the 32 worst neuron blocks above), not
confidence intervals. Solid lines join replicate medians of the upper 95th-percentile bound;
dashed lines show the exact envelope D_th(m, n) for each n; the dotted line marks d = 1.
[Describe observed results here, including the matched n/sqrt(m) = 2 diagonal from
matched_diagonal.csv.] No PI/LSI constants, width thresholds or exponents are estimated.

**Appendix Figure A.** Fashion-MNIST T-shirt/top vs shirt, n = 256 (128 per class), inputs
centered and projected to 32 principal components fit on the official two-class training pool,
then unit-normalized; widths 4,096 and 16,384; three paired subset/prior-center replicates.
Left: coverage. Right: median and 95th-percentile d_H brackets (full-coordinate dense block
code). The dashed "safe envelope" uses the dataset's M₂ and the bound M₄² ≤ max_i‖x_i‖ M₂; it may
be far above one and is not informative at these settings. [Describe observed results here.]

**Appendix Figure B.** CDF of H/B_{m,n} at m = 4,096 (available for every n), pooled over all
retained draws of each replicate (solid, one line per replicate) with the exact Gaussian-prior
maximum distribution (dashed). The vertical line is the cylinder boundary. The horizontal
separation between colors shows how much of the higher-n coverage relies on the larger cutoff.
"""


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    _, _, art, _ = load(args)
    out = art / "figures"
    fig1(art, out)
    fig2(art, out)
    figA(art, out)
    figB(art, out)
    (art / "captions.md").write_text(CAPTIONS, encoding="utf-8")
    print("Wrote captions.md")


if __name__ == "__main__":
    main()
