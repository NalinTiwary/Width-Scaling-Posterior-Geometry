"""Search a frozen 15-probe family using fit and evaluation chains that are disjoint."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arch",choices=["deep","shallow"],required=True)
    p.add_argument("--rep",type=int,choices=[0,1,2],required=True);p.add_argument("--source",choices=["archive","confirm"],default="archive")
    p.add_argument("--steps",type=int,default=8192);p.add_argument("--out",type=Path,default=Path("results/revision_next"));a=p.parse_args()
    sys.path.insert(0,str(Path("a10_next").resolve()))
    from traces import four,PROBES
    from bnn_geometry import config as core
    from bnn_geometry.revision import config as rc
    from bnn_geometry.diagnostics import scalar_diagnostics
    from bnn_geometry.revision.slow import both_folds
    from bnn_geometry.paper_cleanup.supplement_export import _Ctx,read_trajectory
    from bnn_geometry.storage import clean_json
    cfg=rc.load("configs/revision.yaml");base=rc.baseline_root(cfg);campaign=core.load(cfg["baseline_config"])
    ctx=_Ctx(campaign,base,"cpu") if a.source=="archive" else None
    names=list(PROBES)+[f"interaction_{j}" for j in range(4)]
    widths=(32,256) if a.arch=="deep" else (64,4096)
    clocks=a.out/"slow_family";clocks.mkdir(exist_ok=True,parents=True)
    metrics=[];directions=[]
    for m in widths:
        target=core.Target(a.arch,m,a.rep);cell=target.target_id;legacy=base/"targets"/cell
        if a.source=="archive":
            needed = legacy / "dynamics" / "h_0p01" / "chain_0" / "stage_1.h5"
            if not needed.is_file():
                raise SystemExit(
                    f"STOP: the original trajectory is not in this checkout:\n  {needed}\n"
                    "This step needs the archived HDF5 stages. It is not an import failure, and there is nothing to resubmit."
                )
            ident=json.loads((legacy/"spec.json").read_text())
            calib=four(legacy/"reference","calibration",names,target_hash=ident["target_hash"])[0]
            tr=read_trajectory(campaign,base,target,.01,ctx,names)
            if tr["n_retained"]<102400:raise ValueError("short archive")
            arms=[("fixed",tr["X"][:,-102400:,:])]
        else:
            # New independent slice calibration is available even if the original raw files were lost.
            sr=a.out/"reference"/"endpoint"/cell/"slice"
            ident=json.loads((sr/"identity.json").read_text())
            calib=four(sr,"calibration",names,2048,ident["target_hash"])[0]
            winners=pd.read_csv(a.out/"winners.csv")
            win=winners[(winners.arch==a.arch)&(winners.m==m)].h_over_sigma2.item()
            arms=[]
            for label,h in (("fixed",.01),("selected",win)):
                folder=a.out/"confirm"/"endpoint"/cell/f"h_{h:g}"
                run=json.loads((folder/f"run_{a.steps}.json").read_text())
                if any(run["counters"]["numerical_failures"]):raise ValueError(folder)
                data=[four(folder,seg,names,n,ident["target_hash"])[0] for seg,n in run["segments"]]
                x=np.concatenate(data,axis=1)
                if x.shape!=(4,a.steps,len(names)):raise ValueError(folder)
                arms.append((label,x))
        for arm,x in arms:
            for j,pname in enumerate(names):
                s=scalar_diagnostics(pname,x[:,:,j]);n=x.shape[1]
                metrics.append(dict(cell=cell,arm=arm,steps=n,name=pname,raw_iat=n*4/s.ess_mean if s.ess_mean>0 else np.nan,
                     rhat=s.rhat,ess_mean=s.ess_mean,ess_bulk=s.ess_bulk,ess_tail=s.ess_tail,status=s.status,drift_pass=s.drift_pass))
            fits=both_folds(calib.reshape(-1,len(names)),list(x),lag=50,names=names)
            directions.append(dict(cell=cell,arm=arm,n=x.shape[1],names=names,lag=50,folds=fits))
    stem=f"{a.source}_{a.arch}_r{a.rep}"+(f"_{a.steps}" if a.source=="confirm" else "")
    pd.DataFrame(metrics).to_csv(clocks/f"{stem}_scalars.csv",index=False)
    (clocks/f"{stem}_heldout.json").write_text(json.dumps(clean_json(directions),indent=2))
if __name__=="__main__":main()
