"""Endpoint equilibrium figures from original, validated 102,400-step trajectories."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bnn_geometry import config as core
from bnn_geometry.revision import config as rc
from bnn_geometry.paper_cleanup.supplement_export import _Ctx,read_trajectory
from bnn_geometry.paper_figures.figure_metrics import acf_replicate
from bnn_geometry.revision.efficiency import scalar_rate
from bnn_geometry.revision.slow import symmetrized_lag_cov,both_folds
from bnn_geometry.storage import clean_json
from traces import four

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch",choices=["deep","shallow"],required=True)
    p.add_argument("--rep",type=int,choices=[0,1,2],default=0)
    p.add_argument("--config",default="configs/revision.yaml");p.add_argument("--out",type=Path,default=Path("results/revision_next/equilibrium"))
    a=p.parse_args(); cfg=rc.load(a.config); base=rc.baseline_root(cfg); campaign=core.load(cfg["baseline_config"])
    ctx=_Ctx(campaign,base,"cpu"); a.out.mkdir(parents=True,exist_ok=True)
    names=["V","test_prob_0","q_A","q_W"]
    with ZipFile(HERE/"a10_revision_attachment.zip") as z:
        timing=pd.read_csv(z.open("fixed_step_per_cell.csv")).set_index("task_id")
    fig,axs=plt.subplots(2,2,figsize=(9,6),layout="constrained")
    covfig,covaxes=plt.subplots(1,2,figsize=(8.8,4),layout="constrained")
    rows=[]; mats=[]; cdata=[]; details=[]
    cells=(32,256) if a.arch=="deep" else (64,4096)
    for j,m in enumerate(cells):
        tgt=core.Target(a.arch,m,a.rep); cell=tgt.target_id
        traj=read_trajectory(campaign,base,tgt,.01,ctx,names)
        if traj["n_retained"]<102400: raise ValueError(f"short window {cell}")
        x=traj["X"][:,-102400:,:]; clock=timing.loc[cell,"seconds_per_joint_step"]
        spec=json.loads((base/"targets"/cell/"spec.json").read_text())
        ca,_=four(base/"targets"/cell/"reference","calibration",names,target_hash=spec["target_hash"])
        ca=ca.reshape(-1,len(names)); sd=ca.std(axis=0,ddof=1)
        if np.any(sd<=1e-12): raise ValueError(f"constant calibration component: {cell}")
        k=50; norm=(x-ca.mean(axis=0))/sd
        lag,zero=symmetrized_lag_cov(list(norm),k)
        mats.append((cell,zero))
        for u,p1 in enumerate(names):
            for v,p2 in enumerate(names):
                cdata.append(dict(cell=cell,row=p1,column=p2,calibrated_cov=zero[u,v],lag=k,lagged_cov=lag[u,v]))
        slow=both_folds(ca,list(x),lag=k,names=names); details.append(dict(cell=cell,**slow))
        for i,pname in enumerate(names):
            s=x[:,:,i]; raw=scalar_rate(s)
            rows.append(dict(cell=cell,probe=pname,iat=raw["tau"],raw_ess=raw["ess_mean"],bulk_ess=raw["ess_bulk"],
                             raw_ess_s=raw["ess_mean"]/(102400*clock),n=102400,chains=4))
            if i<2:
                y=acf_replicate(s,400,102400)["replicate_acf"]; ls=np.arange(len(y))
                for l,t in enumerate((ls,ls*clock)):
                    axs[i,l].plot(t,y,label=f"m = {m:,}")
                    axs[i,l].set_ylabel(pname+" ACF"); axs[i,l].set_xlabel("Lag (steps)" if l==0 else "Lag (estimated A10 seconds)")
    for ax in axs.flat: ax.axhline(0,color="#999",lw=.7); ax.legend(fontsize=8)
    bound=max(abs(mat).max() for _,mat in mats)
    for ax,(cell,mat) in zip(covaxes,mats):
        im=ax.imshow(mat,cmap="RdBu_r",vmin=-bound,vmax=bound)
        ax.set_xticks(range(4),names,rotation=35,ha="right"); ax.set_yticks(range(4),names)
        ax.set_title(cell)
    covfig.colorbar(im,ax=list(covaxes),shrink=.65,label="Covariance / calibration SD product")
    stem=f"{a.arch}_r{a.rep}"
    pd.DataFrame(rows).to_csv(a.out/f"{stem}_iat.csv",index=False)
    pd.DataFrame(cdata).to_csv(a.out/f"{stem}_covariance.csv",index=False)
    (a.out/f"{stem}_slow.json").write_text(json.dumps(clean_json(details),indent=2))
    fig.savefig(a.out/f"{stem}_acf.pdf"); covfig.savefig(a.out/f"{stem}_covariance.pdf")
if __name__=="__main__": main()
