"""Rebuild the current figures from the published A10 attachment."""
from pathlib import Path
from zipfile import ZipFile
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STYLE = ["#086969", "#c2672d", "#655098"]
MARK = ["o", "s", "^"]

def panels(rows, cols, title, subtitle, size):
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, layout="constrained")
    fig.suptitle(title + "\n" + subtitle, x=.06, ha="left", fontsize=13, color="#163e48")
    return fig, axes

def lines(ax, frame, x, y, log=True):
    vals = sorted(frame[x].unique())
    for r, g in frame.groupby("replicate"):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker=MARK[r], color=STYLE[r], label=f"Dataset {r}", lw=1.6, ms=4)
    if log: ax.set_xscale("log", base=2)
    ax.set_xticks(vals, [f"{i:,}" if x=="m" else f"{i:g}" for i in vals])
    ax.minorticks_off()
    ax.grid(axis="y", color="#e0e8e8")
    ax.set_axisbelow(True)
    ax.spines[["right","top"]].set_visible(False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, default=HERE/"a10_revision_attachment.zip")
    p.add_argument("--out", type=Path, default=HERE/"figures")
    a=p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with ZipFile(a.zip) as z:
        def table(n): return pd.read_csv(z.open(n + ".csv"))
        s=table("replay_s1_with_a10_cost")
        t=table("tuning_candidates").rename(columns={"rep":"replicate"})
        f=table("fixed_step_per_cell")
        b=table("pilot_timing_blocks")
    assert not s.duplicated(["target_id","observable"]).any() and len(s)==216
    assert (s.steps_per_chain==102400).all() and (s.chain_count==4).all()
    np.testing.assert_allclose(s.ess_bulk/(s.steps_per_chain*s.seconds_per_joint_step), s.bulk_ess_per_timed_batch_second)
    np.testing.assert_allclose(4*f.stored_ess_per_second, f.ess_mean/f.timed_batch_seconds)
    assert len(b)==10 and (b.block_seconds>=1).all()
    loss=s[s.observable=="V"].copy()
    loss["ess1k"]=1000*loss.ess_bulk/loss.pooled_transitions
    loss["ms"]=1000*loss.seconds_per_joint_step
    pred=s[s.observable!="V"].groupby(["architecture","m","replicate"],as_index=False).bulk_ess_per_timed_batch_second.median()
    def width_fig(stem, items, title, subtitle):
        fig,ax=panels(len(items),2,title,subtitle,(8.7,2.25*len(items)+1.15))
        for j,arch in enumerate(("deep","shallow")):
            for i,(data,col,label) in enumerate(items):
                aa=ax[i,j]; lines(aa,data[data.architecture==arch],"m",col)
                if i==0: aa.set_title("Two hidden layers" if arch=="deep" else "One hidden layer",loc="left",fontsize=11)
                if j==0: aa.set_ylabel(label)
                if i==len(items)-1: aa.set_xlabel("Hidden width")
                aa.set_ylim(bottom=0)
        for i in range(len(items)):
            hi=max(aa.get_ylim()[1] for aa in ax[i]); [aa.set_ylim(0,hi) for aa in ax[i]]
        ax[0,1].legend(fontsize=8,ncol=3,loc="upper left")
        fig.savefig(a.out/(stem+".pdf")); plt.close(fig)
    width_fig("cost_decomposition",[(loss,"ess1k","Loss bulk ESS / 1,000\nchain-transitions"),(loss,"ms","ms / four-chain step"),(loss,"bulk_ess_per_timed_batch_second","Loss bulk ESS / s")],
              "Where width gains and loses wall time","Fixed step 0.01  ·  archived ESS and matched A10 kernel timing")
    width_fig("fixed_efficiency",[(loss,"bulk_ess_per_timed_batch_second","Loss bulk ESS / s"),(pred,"bulk_ess_per_timed_batch_second","Median prediction\nbulk ESS / s")],
              "Sampling efficiency across hidden widths","Four batched chains  ·  eight fixed prediction probes per target")
    fig,ax=panels(2,2,"Step-size pilot at the four endpoint widths","2,048 retained steps per chain  ·  open circles mark the stored selection",(9,6.2))
    for aa,((arch,m),g) in zip(ax.flat,t.groupby(["arch","m"],sort=True)):
        lines(aa,g,"h_over_sigma2","ess_per_transition")
        w=g[g.h_over_sigma2==g.winner]
        aa.scatter(w.h_over_sigma2,w.ess_per_transition,facecolors="none",edgecolors="#293c45",s=95,lw=1.1)
        aa.set_title(f"{arch.capitalize()}  ·  width {m:,}",loc="left",fontsize=11)
        aa.set_xlabel("h / σ²"); aa.set_ylabel("Loss raw-mean ESS /\nchain-transition")
        aa.set_ylim(bottom=0)
    ax[0,1].legend(fontsize=8,loc="upper left")
    fig.savefig(a.out/"step_sensitivity.pdf"); plt.close(fig)
if __name__=="__main__": main()
