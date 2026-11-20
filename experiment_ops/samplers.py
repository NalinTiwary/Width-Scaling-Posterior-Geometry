"""Production-clock comparison for endpoint PCNL and independent elliptical-slice runs."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--steps",type=int,default=8192)
    p.add_argument("--out",type=Path,default=Path("results/revision_next"));a=p.parse_args()
    import h5py
    src=pd.read_csv("a10_next/Prior_Start_Reference_Values.csv").set_index(["target_id","observable"])
    out=[]
    for folder in sorted((a.out/"reference"/"endpoint").glob("*/slice")):
        cell=folder.parent.name
        for d in [folder,*sorted((a.out/"confirm"/"endpoint"/cell).glob("h_*"))]:
            file=d/f"metrics_{a.steps}.csv";audit=d/f"run_{a.steps}.json"
            if not file.is_file() or not audit.is_file():continue
            meta=json.loads(audit.read_text()); metrics=pd.read_csv(file)
            if any(meta["counters"]["numerical_failures"]):raise ValueError(audit)
            evals=0
            for seg,n in meta["segments"]:
                for c in range(4):
                    with h5py.File(d/f"chain_{c}"/(seg+".h5"),"r") as f:
                        if len(f["evals"])!=n:raise ValueError(f.filename)
                        evals+=int(f["evals"][:].sum())
            for _,x in metrics.iterrows():
                q=src.loc[(cell,x["name"]),:];clock=x.measured_production_s
                aok=x.status=="ok" and x.rhat<1.01 and x.ess_bulk>=1000 and x.ess_tail>=400 and x.drift_pass
                agree=abs(x["mean"]-q.posterior_reference_mean)<=3*np.hypot(x.mcse,q.posterior_reference_mean_mcse)+.05*q.posterior_reference_sd
                out.append(dict(cell=cell,arm=d.name,name=x["name"],steps=a.steps,ess_mean=x.ess_mean,
                    raw_ess_per_wall_s=x.raw_ess_per_production_s,clock_present=bool(np.isfinite(clock) and clock>0),
                    posterior_check=bool(aok and agree),evals_per_transition=evals/(4*a.steps)))
    if not out:raise SystemExit("No matched-stage files found.")
    pd.DataFrame(out).to_csv(a.out/f"sampler_comparison_{a.steps}.csv",index=False)
if __name__=="__main__":main()
