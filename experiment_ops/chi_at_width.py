"""Estimate changes in sampling efficiency across input geometries at each fixed width."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WIDTHS = [64, 256, 1024, 4096]
PROBES = ["V", *[f"test_prob_{i}" for i in range(0,1024,128)], "q_A", "q_W"]
KEYS = ["rep", "m", "name"]


def fixed_width_contrasts(data):
    """Return a row for every planned comparison, including absent arms."""
    d = data.copy()
    ids = d.cell.str.extract(r"^geometry_r(0|0\.2|0\.8)_m(\d+)_r(\d+)$")
    if ids.isna().any().any():
        raise ValueError("Unrecognized geometry target")
    if not (ids[1].astype(int).eq(d.m) & ids[2].astype(int).eq(d.rep)).all():
        raise ValueError("Cell and tabulated width or dataset disagree")
    if (not d.m.isin(WIDTHS).all() or not d.rep.isin(range(3)).all()
            or not d.name.isin(PROBES).all() or not np.allclose(d.h_over_sigma2,.01)):
        raise ValueError("Unexpected width, replicate, observable, or step")
    if "chains" in d and not d.chains.eq(4).all():
        raise ValueError("Expected four synchronized chains")
    d["chi"] = ids[0].map({"0":1, "0.2":4, "0.8":13})
    if d.duplicated([*KEYS, "chi"]).any() or not (d.steps > 0).all():
        raise ValueError("Duplicate geometry cell/probe or empty window")
    d["raw_per_transition"] = d.ess_mean / (4*d.steps)
    d["bulk_per_transition"] = d.ess_bulk / (4*d.steps)
    d["source_gate"] = d.passed.astype("string").str.lower().eq("true").fillna(False)
    values = ["steps", "source_gate", "raw_per_transition", "bulk_per_transition", "raw_ess_s", "bulk_ess_s", "joint_step_s"]
    design = pd.MultiIndex.from_product([range(3), WIDTHS, PROBES, [4,13]], names=[*KEYS, "chi"]).to_frame(index=False)
    result = design.merge(d[[*KEYS, "chi", *values]], on=[*KEYS, "chi"], how="left", validate="one_to_one")
    base = d.loc[d.chi.eq(1), [*KEYS, *values]].rename(columns={k:k+"_chi1" for k in values})
    result = result.merge(base, on=KEYS, how="left", validate="many_to_one")
    complete = result.steps.notna() & result.steps_chi1.notna()
    result["both_source_gates"] = (result.source_gate.eq(True) & result.source_gate_chi1.eq(True)).fillna(False).astype(bool)
    result["status"] = np.select([~complete, ~result.both_source_gates], ["missing_arm", "diagnostics_incomplete"], default="source_gates_pass")
    result["equal_steps"] = complete & result.steps.eq(result.steps_chi1)
    for measure in values[2:]:
        a, b = result[measure], result[measure + "_chi1"]
        result[measure + "_ratio"] = a.div(b).where(a.gt(0) & b.gt(0) & np.isfinite(a) & np.isfinite(b))
    return result


def plot_loss(contrasts, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = contrasts[contrasts.name.eq("V")]
    fig, axes = plt.subplots(2, 4, figsize=(11,5.8), sharex=True, sharey="row", layout="constrained")
    colors = ["#184e70", "#933b21", "#336449"]
    for col, width in enumerate(WIDTHS):
        for row, (metric, label) in enumerate([("bulk_per_transition_ratio","Bulk ESS / transition"),("bulk_ess_s_ratio","Bulk ESS / A10 second")]):
            ax = axes[row,col]
            ax.axhline(1, color="#888", lw=.9, ls="--")
            for rep, color in enumerate(colors):
                part = data[(data.rep==rep)&(data.m==width)].sort_values("chi")
                if part[metric].notna().any():
                    ax.plot([1,4,13], [1,*part[metric]], c=color, lw=.9)
                for _,p in part.dropna(subset=[metric]).iterrows():
                    ax.scatter(p.chi, p[metric], edgecolor=color, s=30, facecolor=color if p.both_source_gates else "white",zorder=3)
            ax.set_xticks([1,4,13])
            if col==0: ax.set_ylabel(label+"\nrelative to χ = 1")
            if row==0: ax.set_title(f"Width {width:,}")
            else: ax.set_xlabel("Input χ")
    fig.suptitle("Change in loss sampling efficiency at fixed width",fontsize=14)
    from matplotlib.lines import Line2D
    handles = [Line2D([],[],color=c,label=f"Dataset {i}") for i,c in enumerate(colors)]
    handles.append(Line2D([],[],marker="o",color="black",markerfacecolor="white",lw=0,label="Open: source diagnostics incomplete"))
    fig.legend(handles=handles,loc="outside lower center",ncol=4,frameon=False,fontsize=8)
    fig.savefig(dest)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/revision_next"))
    out = p.parse_args().out
    for src, suffix in [("gates_geometry.csv",""),("diagnostics_geometry_6192.csv","_6192")]:
        if not (out/src).exists():
            if not suffix: raise SystemExit(f"Missing {out/src}. Run geometry gates first.")
            continue
        results = fixed_width_contrasts(pd.read_csv(out/src))
        results.to_csv(out/f"chi_at_fixed_width{suffix}.csv", index=False)
        plot_loss(results, out/f"chi_at_fixed_width{suffix}.pdf")


if __name__ == "__main__":
    main()
