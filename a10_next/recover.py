"""Inventory and export the existing revision HDF5 before submitting fresh jobs."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from bnn_geometry.diagnostics import scalar_diagnostics
from bnn_geometry.revision import config as rc
from bnn_geometry.revision.runner import tasks_for
from bnn_geometry.revision.efficiency import warmup_steps
from traces import PROBES, one, four

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/revision.yaml")
    p.add_argument("--out",type=Path,default=Path("results/revision_next/recovery"))
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    cfg=rc.load(a.config); root=rc.output_root(cfg)/"runs"
    frozen=pd.read_csv(Path(__file__).parent/"baseline_identity.csv").set_index("target_id")
    inventory=[]; cold=[]; metrics=[]
    stages=("prior-starts","tune","fixed-step","geometry-pilot","relu","sin","depth4","concentrated","nonlinear","prior-only")
    for stage in stages:
        for task in tasks_for(cfg,stage):
            cell=task["id"]; base=root/cell/stage
            identity=base/"identity.json"
            groups=["pilot_"+str(h) for h in cfg["tuning"]["h_over_sigma2"]] if stage=="tune" else ["all"]
            for group in groups:
                meta=dict(stage=stage,cell=cell,group=group)
                try:
                    ident=json.loads(identity.read_text()); assert ident["cell"]==cell
                    fp=ident["target_hash"]; ss=float(ident["sigma"])
                    if task["kind"]=="baseline":
                        key=frozen.loc[cell]
                        assert all(ident[k]==key[k] for k in ("hash_X","hash_y","n","d","L","m","sigma","arch")), cell
                    if stage=="prior-starts":
                        times=np.array([2**k for k in range(14)])
                        pending=[]
                        for c in range(32):
                            path=base/f"batch_{c//4*4}"/f"chain_{c}"/"from_prior.h5"
                            data,_=one(path,c,PROBES,8192,fp,rows=times-1)
                            for i,t in enumerate(times):
                                pending.append(dict(cell=cell,chain=c,t=int(t),**dict(zip(PROBES,data[i]))))
                        cold.extend(pending)
                    else:
                        n=int(cfg["fixed_step"]["transitions"]); burn=0; hdim=float(cfg["fixed_step"]["h_over_sigma2"])
                        name="production"; sub="prior_only" if stage=="prior-only" else "fixed"
                        if stage=="tune":
                            hdim=float(group.removeprefix("pilot_")); sub=group; name="pilot"
                            burn=warmup_steps(hdim*ss**2,ss,time_over_sigma2=float(cfg["tuning"]["warmup_time_over_sigma2"]))
                            n=burn+int(cfg["tuning"]["pilot_transitions"])
                        data,acc=four(base/sub,name,PROBES,n,fp)
                        windows=[("after_warmup",burn)] if stage=="tune" else [("original",0)]
                        if stage not in ("prior-only","tune"):
                            windows.append(("after_nominal_warmup",warmup_steps(hdim*ss**2,ss)))
                        for window,skip in windows:
                            for j,probe in enumerate(PROBES):
                                d=scalar_diagnostics(probe,data[:,skip:,j]).row()
                                metrics.append(dict(**meta,window=window,discard=skip,h_over_sigma2=hdim,
                                    steps_per_chain=n-skip,chains=4,acceptance=acc[:,skip:].mean(),**d))
                    inventory.append(dict(**meta,status="ok",reason=""))
                except (OSError, KeyError, ValueError, AssertionError) as exc:
                    inventory.append(dict(**meta,status="unavailable",reason=f"{type(exc).__name__}: {exc}"))
    pd.DataFrame(inventory).to_csv(a.out/"inventory.csv",index=False)
    pd.DataFrame(cold,columns=["cell","chain","t",*PROBES]).to_csv(a.out/"prior_checkpoints.csv",index=False)
    pd.DataFrame(metrics).to_csv(a.out/"recovered_diagnostics.csv",index=False)
    print(pd.DataFrame(inventory).groupby(["stage","status"]).size().to_string())
if __name__=="__main__": main()
