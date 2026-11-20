"""Descriptive prior-start trajectories using the independent reference mean and SD."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=Path("results/revision_next/recovery"))
    p.add_argument("--ref",type=Path,default=HERE/"Prior_Start_Reference_Values.csv")
    a=p.parse_args(); starts=pd.read_csv(a.folder/"prior_checkpoints.csv")
    if starts.empty: raise SystemExit("No complete cold target. Inspect inventory.csv.")
    ref=pd.read_csv(a.ref).set_index(["target_id","observable"],verify_integrity=True)
    rows=[]
    for (cell,t),g in starts.groupby(["cell","t"]):
        assert g.chain.is_unique and len(g)>=2
        for probe in starts.columns.difference(["cell","chain","t"]):
            r=ref.loc[cell,probe]; s=r.posterior_reference_sd
            err=(g[probe].mean()-r.posterior_reference_mean)/s
            se=np.hypot(g[probe].std()/np.sqrt(len(g)),r.posterior_reference_mean_mcse)/s
            rows.append(dict(cell=cell,t=t,probe=probe,std_error_mean=err,
                pointwise_halfwidth=student.ppf(.975,len(g)-1)*se,starts=len(g)))
    d=pd.DataFrame(rows); d.to_csv(a.folder/"cold_errors_pointwise.csv",index=False)
    fig,axs=plt.subplots(2,3,figsize=(11,6),layout="constrained",sharex=True)
    for ax,(arch,r) in zip(axs.flat,[(aa,rr) for aa in ("deep","shallow") for rr in range(3)]):
        ax.axhspan(-.25,.25,color="#e4eded"); ax.axhline(0,color="#607473",lw=.7)
        for m,col in zip(((32,256) if arch=="deep" else (64,4096)),("#156d71","#ac562f")):
            g=d[(d.probe=="V")&(d.cell==f"{arch}_m{m:04d}_r{r}")].sort_values("t")
            if g.empty: continue
            x=g.t.to_numpy(); y=g.std_error_mean.to_numpy(); b=g.pointwise_halfwidth.to_numpy()
            ax.plot(x,y,".-",c=col,label=f"m = {m:,}"); ax.fill_between(x,y-b,y+b,color=col,alpha=.14)
        ax.set_title(f"{arch.capitalize()}  ·  dataset {r}"); ax.set_xscale("log",base=2); ax.set_xlabel("Transitions")
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    axs[0,0].set_ylabel("Loss error / reference SD"); axs[1,0].set_ylabel("Loss error / reference SD")
    fig.suptitle("Prior-start mean loss  ·  approximate pointwise 95% intervals")
    fig.savefig(a.folder/"prior_start_loss.pdf")
if __name__=="__main__": main()
