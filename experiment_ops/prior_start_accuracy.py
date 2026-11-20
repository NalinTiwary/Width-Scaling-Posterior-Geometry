"""Supplementary accuracy of a finite prior-start run; use the saved scalar paths."""
import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np
import pandas as pd

def accuracy(values,reference,reference_sd,reference_mcse,seed=252):
    """One vector entry per independent start, with the same discard/window policy."""
    values=np.asarray(values)
    rng=np.random.default_rng(seed)
    resamples=rng.choice(values,(2000,len(values)),replace=True)
    ref_est=rng.normal(reference,reference_mcse,size=(2000,1))
    err=(values-reference)/reference_sd
    boot_err=(resamples-ref_est)/reference_sd
    bias=boot_err.mean(axis=1); rmse=np.sqrt(np.mean(boot_err**2,axis=1))
    return dict(standardized_bias=err.mean(),standardized_rmse=np.sqrt(np.mean(err**2)),
                bias_lo=np.quantile(bias,.025),bias_hi=np.quantile(bias,.975),
                rmse_lo=np.quantile(rmse,.025),rmse_hi=np.quantile(rmse,.975))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=Path("results/revision_next"))
    a=p.parse_args()
    sys.path.insert(0,str(Path("a10_next").resolve()))
    from traces import one
    from bnn_geometry.revision import config as rc
    cfg=rc.load("configs/revision.yaml"); root=rc.output_root(cfg)/"runs"
    ref=pd.read_csv("a10_next/Prior_Start_Reference_Values.csv").set_index(["target_id","observable"])
    with ZipFile("a10_next/a10_revision_attachment.zip") as z:
        timing=pd.read_csv(z.open("fixed_step_per_cell.csv")).set_index("task_id")
    probes=["V","test_prob_0"]; rows=[]
    for ci,cell in enumerate(sorted(ref.index.get_level_values(0).unique())):
        folder=root/cell/"prior-starts"; ident=json.loads((folder/"identity.json").read_text())
        if ident["cell"]!=cell: raise ValueError(cell)
        first=folder/"batch_0"/"chain_0"/"from_prior.h5"
        if not first.is_file():
            raise SystemExit(
                f"STOP: prior-start trace is not in this checkout:\n  {first}\n"
                "Run this on the cluster checkout that already passed trace recovery."
            )
        draws=[]
        for c in range(32):
            path=folder/f"batch_{c//4*4}"/f"chain_{c}"/"from_prior.h5"
            draws.append(one(path,c,probes,8192,ident["target_hash"])[0])
        x=np.stack(draws); cost=timing.loc[cell,"seconds_per_joint_step"]
        windows=[("prefix_steps",t,0,t) for t in (128,256,1024,4096,8192)]
        for seconds in (.25,1,4,16):
            n=int(np.floor(seconds/cost))
            if 1<=n<=8192:windows.append(("estimated_seconds",seconds,0,n))
        windows.append(("late_half",8192,4096,8192))
        for label,budget,first,last in windows:
            means=x[:,first:last,:].mean(axis=1)
            for j,name in enumerate(probes):
                q=ref.loc[(cell,name),:]
                result=accuracy(means[:,j],q.posterior_reference_mean,q.posterior_reference_sd,
                                q.posterior_reference_mean_mcse,seed=574+2*ci+j)
                rows.append(dict(cell=cell,probe=name,window=label,budget=budget,first_transition=first+1,
                      last_transition=last,starts=32,estimated_elapsed_s=last*cost,**result))
    pd.DataFrame(rows).to_csv(a.out/"prior_start_accuracy.csv",index=False)
if __name__=="__main__":main()
