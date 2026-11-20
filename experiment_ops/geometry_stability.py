"""Matched full- and half-window sensitivity of the raw-mean geometry interactions."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

def expressions(rate,baseline,cost,cost_baseline):
    raw=rate[1]/rate[0]
    control=baseline[1]/baseline[0]
    wall=(raw/(cost[1]/cost[0]))/(control/(cost_baseline[1]/cost_baseline[0]))
    return raw/control,wall

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("results/revision_next"));a=p.parse_args()
    sys.path.insert(0,str(Path("a10_next").resolve()))
    from bnn_geometry.revision import config as rc
    from bnn_geometry.diagnostics import ess_mean_raw
    from traces import four
    cfg=rc.load("configs/revision.yaml");legacy=rc.output_root(cfg)/"runs"
    names=["V",*[f"test_prob_{j}" for j in range(0,1024,128)]]
    tab=pd.read_csv(a.out/"gates_geometry.csv")
    result=[]
    for rep,targets in tab[tab.m.isin([64,4096])].groupby("rep"):
        data={};cost={};gate={}
        for cell,rows in targets.groupby("cell"):
            meta=rows.iloc[0]; m=int(meta.m); chi=round(1+15*float(cell.split("_")[1][1:]));key=(chi,m)
            gate[key]=rows.set_index("name").passed.to_dict()
            cost[key]=meta.joint_step_s
            if meta.origin=="compare":
                path=a.out/"compare"/"geometry"/cell/"h_0.01"
                run=json.loads((path/f"run_{int(meta.steps)}.json").read_text())
                hh=json.loads((path/"identity.json").read_text())["target_hash"]
                xs=[four(path,seg,names,n,hh)[0] for seg,n in run["segments"]]
                data[key]=np.concatenate(xs,axis=1)
            else:
                base=next(legacy/cell/st for st in ("geometry-pilot","geometry") if (legacy/cell/st/"fixed/chain_0/production.h5").is_file())
                hh=json.loads((base/"identity.json").read_text())["target_hash"]
                data[key]=four(base/"fixed","production",names,8192,hh,slice(2000,None))[0]
        for chi in (4,13):
            keys=[(1,64),(1,4096),(chi,64),(chi,4096)]
            if not all(k in data for k in keys):continue
            n=min(data[k].shape[1] for k in keys)
            for label,sl in (("full",slice(0,n)),("first_half",slice(0,n//2)),("last_half",slice(n-n//2,n))):
                for j,name in enumerate(names):
                    rate=[ess_mean_raw(data[k][:,sl,j])[0] for k in keys]
                    if not np.isfinite(rate).all() or min(rate)<=0:tr=wall=np.nan
                    else:tr,wall=expressions(rate[2:],rate[:2],[cost[k] for k in keys[2:]],[cost[k] for k in keys[:2]])
                    result.append(dict(rep=rep,chi=chi,reference_chi=1,probe=name,portion=label,
                        common_retained_n=n,steps=n if label=="full" else n//2,
                        qualified_parent_windows=all(gate[k][name] for k in keys),
                        width_interaction_per_transition=tr,width_interaction_per_second=wall))
    if not result:raise SystemExit("No complete endpoint comparison with chi=1 yet")
    pd.DataFrame(result).to_csv(a.out/"chi_window_stability.csv",index=False)
if __name__=="__main__":main()
