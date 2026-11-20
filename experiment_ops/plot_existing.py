"""Plot audited archived comparisons and preliminary probe and geometry diagnostics."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("verified_evidence"))
    base = parser.parse_args().dir
    archived = pd.read_csv(base / "two_kernel_archived_loss.csv")
    slow = pd.read_csv(base / "short_worst_observed.csv")
    geometry = pd.read_csv(base / "dataset0_geometry_screen.csv")
    colors = ["#234f71", "#b74d26", "#426f54"]
    plt.rcParams.update({"axes.spines.right": False, "axes.spines.top": False,
                         "font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, grid = plt.subplots(2, 2, figsize=(10, 8.2))
    fig.subplots_adjust(left=.09, right=.975, top=.85, bottom=.115, hspace=.85, wspace=.33)
    fig.text(.09, .957, "What changes when the network gets wider?", size=18, weight="bold")
    fig.text(.09, .917, "Archived training-loss efficiency and two short-run diagnostic checks", size=11)
    offsets = [-.13, 0, .13]
    formats = ["o", "s", "^"]
    ax, clock = grid[0]
    for j, (_, rows) in enumerate(archived.groupby("architecture", sort=True)):
        for row in rows.itertuples():
            r = row.dataset
            x = np.array([j*2.45,j*2.45+.95]) + offsets[r]
            ax.plot(x, [row.pcnl_wide_narrow_bulk_per_transition, row.slice_wide_narrow_bulk_per_transition],
                    color=colors[r], marker=formats[r], lw=.8, ms=5)
            clock.scatter(j+offsets[r], row.pcnl_wide_narrow_bulk_per_A10_s,
                          color=colors[r], marker=formats[r], s=35)
    ax.set(title="A. Effective draws per transition", ylabel="Wide / narrow loss bulk ESS",
           xticks=[0,.95,2.45,3.4], xticklabels=["Deep\nPCNL","Deep\nslice","Shallow\nPCNL","Shallow\nslice"],
           ylim=(.85,2.22), xlim=(-.43,3.84))
    clock.set(title="B. Effective draws per second on one A10", ylabel="Wide / narrow loss bulk ESS per second",
              xticks=[0,1], xticklabels=["Deep\n32 → 256","Shallow\n64 → 4,096"], ylim=(.5,2.05), xlim=(-.5,1.5))
    legend = [Line2D([0],[0],color=colors[i],marker=formats[i],linestyle="",label=f"Dataset {i}") for i in range(3)]
    ax.legend(handles=legend, loc="upper center",bbox_to_anchor=(.52,-.23),ncol=3,frameon=False,fontsize=8)
    ax, chi = grid[1]
    for row in slow.itertuples():
        depth,width,rep = row.cell.split("_")
        group = 0 if depth=="deep" else 1
        ix = ([32,64,128,256] if group==0 else [64,256,1024,4096]).index(int(width[1:]))
        r = int(rep[1:])
        ax.scatter(group + (ix-1.5)*.083 + offsets[r]*.18, row.iat_over_loss,
                   c=colors[r], marker=formats[r], s=24, alpha=.88)
    ax.set(title="C. The slowest monitored scalar in 24 runs", ylabel="Slowest scalar IAT / loss IAT",
           xticks=[0,1],xticklabels=["Deep","Shallow"],xlim=(-.42,1.42),ylim=(.5,9))
    chi.plot(range(len(geometry)),geometry.wide_narrow_bulk_per_transition, color=colors[0],marker="o")
    chi.set(title="D. Width effect at three input geometries",ylabel="Wide / narrow loss bulk ESS per transition",
            xticks=range(len(geometry)),xticklabels=[f"χ = {x:g}" for x in geometry.chi],ylim=(.68,1.33), xlim=(-.3,2.3))
    for a in grid.flat:
        a.axhline(1,color="#797979",lw=.85,ls="--",zorder=0)
        a.set_axisbelow(True)
        a.grid(axis="y",lw=.4,alpha=.25)
        a.title.set_fontweight("bold")
    fig.text(.09,.040,"C. The two parameter projections fail the rank diagnostic in all 24 short runs.\n"
             "D. Dataset 0 only. Each of the six loss bulk ESS estimates is below 300.",fontsize=9,color="#433f3b")
    fig.savefig(base/"width_and_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
