"""Figures from gated follow-up results. Open symbols identify unresolved cells."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("scope")
    p.add_argument("--out",type=Path,default=Path("results/revision_next")); a=p.parse_args()
    d=pd.read_csv(a.out/f"gates_{a.scope}.csv")
    def parse(g):
        u=g[g.name.str.startswith("test_prob_")]
        if len(u)!=8 or sum(g.name=="V")!=1: raise ValueError("incomplete prediction or loss bank")
        loss=g[g.name=="V"].iloc[0]; return [(loss.raw_ess_s,loss.passed),(u.raw_ess_s.median(),u.passed.all())]
    pts=[]
    for (cell,h),g in d.groupby(["cell","h_over_sigma2"]):
        x=g.iloc[0]
        if a.scope=="endpoint":
            panel=x.arch; curve="fixed 0.01" if h==.01 else "selected step"
        elif a.scope=="geometry":
            panel="geometry"; r=float(cell.split("_")[1][1:]); curve=f"χ = {round(1+15*r)}"
        else: panel=a.scope; curve=a.scope
        for i,(y,ok) in enumerate(parse(g)):
            pts.append(dict(panel=panel,curve=curve,rep=int(x.rep),m=int(x.m),y=y,ok=ok,metric=i))
    frames=pd.DataFrame(pts)
    fig,ax=plt.subplots(2,frames.panel.nunique(),figsize=(max(6,4.6*frames.panel.nunique()),6.8),squeeze=False,layout="constrained")
    for j,(pnl,sub) in enumerate(frames.groupby("panel")):
        colors=dict(zip(sorted(sub.curve.unique()),["#16696b","#a3562b","#6b5195"]))
        for i in (0,1):
            aa=ax[i,j]
            for (curve,r),g in sub[sub.metric==i].groupby(["curve","rep"]):
                g=g.sort_values("m"); col=colors[curve]
                aa.plot(g.m,g.y,color=col,lw=.9,alpha=.55,ls=["-","--",":"][r],label=curve if r==0 else None)
                for v in g.itertuples(): aa.plot(v.m,v.y,marker=["o","s","^"][r],ms=5,mfc=col if v.ok else "white",mec=col)
            widths=sorted(sub.m.unique())
            aa.set_xscale("log",base=2); aa.set_xticks(widths,labels=[f"{m:,}" for m in widths]); aa.minorticks_off()
            aa.set_title(pnl.capitalize()); aa.set_xlabel("Width")
            aa.set_ylabel("Loss raw-mean ESS/s" if i==0 else "Median prediction raw-mean ESS/s"); aa.set_ylim(bottom=0)
            aa.grid(axis="y",alpha=.2)
            if i==0: aa.legend(fontsize=8)
    fig.suptitle("Four-chain A10 kernel timing  ·  open points have unresolved diagnostics")
    fig.savefig(a.out/f"efficiency_{a.scope}.pdf")
if __name__=="__main__": main()
