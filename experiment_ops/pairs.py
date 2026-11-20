"""Paired ESS/s ratios within dataset and observable, retaining the diagnostic status."""
import argparse
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np
import pandas as pd

def matched(g,scope,out):
    # A common initial post-discard window for the fixed-step geometry or activation comparison.
    sys.path.insert(0,str(Path("a10_next").resolve()))
    from bnn_geometry.revision import config as rc
    from bnn_geometry.diagnostics import scalar_diagnostics
    from traces import PROBES,four
    cfg=rc.load("configs/revision.yaml")
    legacy=rc.output_root(cfg)/"runs"
    with ZipFile("a10_next/a10_revision_attachment.zip") as z:
        timing=pd.read_csv(z.open("fixed_step_per_cell.csv")).set_index("task_id")
    frozen=pd.read_csv("a10_next/baseline_identity.csv").set_index("target_id")
    import json
    result=[]
    def append(meta,root,seg,n,sl,ident,cost,flag):
        x,_=four(root,seg,PROBES,n,ident["target_hash"],sl)
        if x.shape!=(4,6192,len(PROBES)): raise ValueError(root)
        for j,name in enumerate(PROBES):
            diag=scalar_diagnostics(name,x[:,:,j])
            result.append(dict(**meta,name=name,steps=6192,passed=flag.get(name,np.nan),joint_step_s=cost,
                 raw_ess_s=diag.ess_mean/(6192*cost),bulk_ess_s=diag.ess_bulk/(6192*cost),
                 ess_mean=diag.ess_mean,ess_bulk=diag.ess_bulk,ess_tail=diag.ess_tail,rhat=diag.rhat))
    for cell,rows in g.groupby("cell"):
        r=rows.iloc[0]
        meta=dict(cell=cell,arch=r.arch,m=int(r.m),rep=int(r.rep),h_over_sigma2=.01)
        if r.origin=="compare":
            root=out/"compare"/scope/cell/"h_0.01"
            ident=json.loads((root/"identity.json").read_text()); rootdir=root; seg="production_1"; n=8192; sl=slice(0,6192)
        else:
            stages=["geometry-pilot","geometry"] if scope=="geometry" else [scope]
            folder=next(legacy/cell/s for s in stages if (legacy/cell/s/"fixed"/"chain_0"/"production.h5").is_file())
            ident=json.loads((folder/"identity.json").read_text());rootdir=folder/"fixed";seg="production";n=8192;sl=slice(2000,None)
        append(meta,rootdir,seg,n,sl,ident,r.joint_step_s,rows.set_index("name").passed.to_dict())
        if scope!="geometry":
            tanh=f'deep_m{int(r.m):04d}_r{int(r.rep)}';folder=legacy/tanh/"fixed-step"
            ident=json.loads((folder/"identity.json").read_text())
            baseline=frozen.loc[tanh]
            if any(ident[k]!=baseline[k] for k in ("hash_X","hash_y","L","n","m","sigma")):raise ValueError(tanh)
            meta["cell"]=tanh;append(meta,folder/"fixed","production",8192,slice(2000,None),ident,
                                       timing.loc[tanh,"seconds_per_joint_step"],{})
    return pd.DataFrame(result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scope");p.add_argument("--out",type=Path,default=Path("results/revision_next"))
    p.add_argument("--matched",action="store_true")
    a=p.parse_args();out=a.out
    d=pd.read_csv(out/f"gates_{a.scope}.csv"); suffix="_6192" if a.matched else ""
    if a.matched:
        if a.scope=="endpoint":p.error("use --matched for fixed-step geometry/variants")
        d=matched(d,a.scope,out)
        d.to_csv(out/f"diagnostics_{a.scope}{suffix}.csv",index=False)
    if a.scope=="endpoint": d["condition"]=np.where(np.isclose(d.h_over_sigma2,.01),"fixed","selected")
    elif a.scope=="geometry": d["condition"]=d.cell.str.extract(r"^geometry_r([^_]+)_")[0].map(lambda r:f"chi{1+15*float(r):g}")
    else: d["condition"]=np.where(d.cell.str.startswith("deep_m"),"tanh",a.scope)
    keys=["arch","rep","condition","name"]; frames=[]
    for arch,sub in d.groupby("arch"):
        m0,m1=(32,256) if arch=="deep" else (64,4096)
        q=sub[sub.m==m0].merge(sub[sub.m==m1],on=keys,suffixes=("_narrow","_wide"),validate="one_to_one")
        if q.empty: continue
        for k in ("raw_ess_s","bulk_ess_s","joint_step_s","raw_ess_per_production_s","bulk_ess_per_production_s"):
            if k+"_wide" in q:
                q[k+"_ratio"]=q[k+"_wide"]/q[k+"_narrow"]
        q["raw_ess_per_transition_ratio"]=(q.ess_mean_wide/q.steps_wide)/(q.ess_mean_narrow/q.steps_narrow)
        q["equal_steps"]=q.steps_wide==q.steps_narrow
        frames.append(q)
    if not frames: raise SystemExit("No completed endpoint pairs yet; inspect gate table.")
    pairs=pd.concat(frames,ignore_index=True);pairs.to_csv(out/f"pairs_{a.scope}{suffix}.csv",index=False)
    pred=pairs[pairs.name.str.startswith("test_prob_")].groupby(keys[:-1],as_index=False).agg(
        predictions=("name","nunique"),median_ratio=("raw_ess_s_ratio","median"),
        minimum_ratio=("raw_ess_s_ratio","min"),maximum_ratio=("raw_ess_s_ratio","max"))
    if not pred.predictions.eq(8).all():raise ValueError("incomplete held-out bank")
    pred.to_csv(out/f"prediction_ratios_{a.scope}{suffix}.csv",index=False)
if __name__=="__main__": main()
