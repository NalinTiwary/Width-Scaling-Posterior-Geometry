"""Draw a formal prior-start loss figure or the matching prior/posterior loss CDFs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("mode",choices=["cold-band","tv"])
    p.add_argument("--out",type=Path,default=Path("results/revision_next")); a=p.parse_args()
    cold=a.mode=="cold-band"
    data=pd.read_csv(a.out/("cold-band.csv" if cold else "tv_cdf.csv"))
    fig,axs=plt.subplots(2,3,figsize=(11.5,6.2),layout="constrained")
    for ax,(arch,r) in zip(axs.flat,[(aa,rr) for aa in ("deep","shallow") for rr in range(3)]):
        ax.set_title(f"{arch.capitalize()}  ·  dataset {r}")
        if cold:
            ax.axhspan(-.25,.25,color="#dae8e6"); ax.axhline(0,color="#526970",lw=.8)
            ax.set_xscale("symlog",base=2,linthresh=1)
        for m,col in zip(((32,256) if arch=="deep" else (64,4096)),("#176a6b","#ab5c35")):
            g=data[data.cell==f"{arch}_m{m:04d}_r{r}"]
            if g.empty: continue
            if cold:
                g=g.sort_values("t"); x=g.t.to_numpy(); y=g.error.to_numpy(); b=g.simultaneous_halfwidth.to_numpy()
                ax.plot(x,y,".-",color=col,label=f"m = {m:,}"); ax.fill_between(x,y-b,y+b,color=col,alpha=.13)
                t=g.first_agreement_or_last.iloc[0]
                ax.plot(t,g.loc[g.t==t,"error"].iloc[0],marker=">" if g.censored.iloc[0] else "D",mfc="white",mec=col)
            else:
                g=g.sort_values("V")
                for k,ls in (("prior","--"),("posterior","-")): ax.plot(g.V,g[k],ls=ls,c=col,label=f"{k}, {m:,}")
        ax.set_xlabel("Transitions" if cold else "Training loss")
        ax.set_ylabel("Loss error / reference SD" if cold else "Cumulative probability")
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    fig.suptitle("Mean loss from the prior  ·  simultaneous bands; diamond = agreement, triangle = censored"
                 if cold else "Loss distributions under the matching prior and posterior")
    fig.savefig(a.out/(a.mode+".pdf"))
if __name__=="__main__": main()
