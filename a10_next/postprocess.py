"""Cross-target summaries for the next campaign. Run from the project root."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bnn_geometry.diagnostics import ess_mean_raw, batch_means_mcse
from bnn_geometry.revision import config as rc
from bnn_geometry.revision.geometry import paired_designs
from bnn_geometry.revision.bounds import bound_row
from bnn_geometry.revision.initcheck import time_to_agreement
from traces import four

HERE=Path(__file__).resolve().parent

from stats_tools import gap_fit, band

def legacy_ref(cfg,cell,ref_hash):
    path=rc.baseline_root(cfg)/"targets"/cell/"reference"
    post=np.concatenate([four(path,f"production_s{k}",["V"],target_hash=ref_hash)[0][:,:,0] for k in range(1,5)],axis=1)
    if post.shape!=(4,65536): raise ValueError(f"{cell}: wrong legacy reference window {post.shape}")
    return post

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["geometry","cold-band","tv","gates"])
    p.add_argument("--out",type=Path,default=Path("results/revision_next")); p.add_argument("--config",default="configs/revision.yaml")
    p.add_argument("--scope",default="endpoint"); p.add_argument("--cell"); a=p.parse_args()
    cfg=rc.load(a.config); out=a.out; out.mkdir(parents=True,exist_ok=True)
    ref=pd.read_csv(HERE/"Prior_Start_Reference_Values.csv").set_index(["target_id","observable"],verify_integrity=True)
    if a.action=="geometry":
        res=[]
        for r in range(3):
            _,_,checks=paired_designs(cfg["master_seed"],r,[0,.2,.8],n=16,d=32,n_test=1024)
            for check in checks:
                for m in (64,256,1024,4096):
                    row=bound_row(label=f"r{r}",L=2,m=m,n=16,chi=check["chi"])
                    res.append(dict(row,replicate=r,r=check["r"]))
        pd.DataFrame(res).to_csv(out/"geometry_certificates.csv",index=False)
        return
    if a.action=="gates":
        base=out/("confirm" if a.scope=="endpoint" else "compare")/a.scope
        rows=[]
        hdirs={(x.parent.name,x.name): x for x in (out/"reuse"/a.scope).glob("*/h_*")} if a.scope!="endpoint" else {}
        hdirs.update({(x.parent.name,x.name):x for x in base.glob("*/h_*")})
        for hdir in sorted(hdirs.values()):
            def latest(path):
                paths=list(path.glob("metrics_*.csv"))
                return max(paths,key=lambda f:int(f.stem.split("_")[-1])) if paths else None
            mf=latest(hdir)
            if mf is None: continue
            md=pd.read_csv(mf); cell=hdir.parent.name
            ref_file=latest(out/"reference"/a.scope/cell/"slice")
            if a.scope!="endpoint":
                rtab=pd.read_csv(ref_file).set_index("name") if ref_file else None
            for i,x in md.iterrows():
                if a.scope=="endpoint":
                    q=ref.loc[cell,x["name"]]; mean,sd,se=q.posterior_reference_mean,q.posterior_reference_sd,q.posterior_reference_mean_mcse
                    rg=(q.rhat<1.01 and q.ess_bulk>=1000 and q.ess_tail>=400)
                elif ref_file:
                    q=rtab.loc[x["name"]]; mean,sd,se=q["mean"],q.sd,q.mcse
                    rg=(q.status=="ok" and q.rhat<1.01 and q.ess_bulk>=1000 and q.ess_tail>=400 and bool(q.drift_pass))
                else: mean=sd=se=np.nan; rg=False
                ok=(x.status=="ok" and x.rhat<1.01 and x.ess_bulk>=1000 and x.ess_tail>=400 and bool(x.drift_pass))
                gap=abs(x["mean"]-mean); limit=3*np.hypot(x.mcse,se)+.05*sd
                rows.append(dict(x.to_dict(),origin=hdir.parents[2].name,reference_available=bool(np.isfinite(mean)),reference_gate=rg,chain_gate=ok,
                       mean_gap=gap,mean_threshold=limit,passed=bool(rg and ok and gap<=limit)))
        if not rows: raise SystemExit("No new production metrics found")
        pd.DataFrame(rows).to_csv(out/f"gates_{a.scope}.csv",index=False)
        return
    starts=pd.read_csv(out/"recovery"/"prior_checkpoints.csv") if a.action=="cold-band" else None
    skipped=[]; rows=[]; cdfs=[]
    for cell in ([a.cell] if a.cell else ref.index.get_level_values(0).unique()):
        try:
            q=ref.loc[cell,"V"]; post=legacy_ref(cfg,cell,q.target_hash)
            np.testing.assert_allclose([post.mean(),post.std(ddof=1)],[q.posterior_reference_mean,q.posterior_reference_sd],rtol=1e-9)
            if a.action=="cold-band":
                g=starts[starts.cell==cell].pivot(index="chain",columns="t",values="V")
                if len(g)<2 or g.isna().any().any(): raise ValueError("missing start histories")
                zero=out/"cold-zero"/"endpoint"/cell/"verified_zero.csv"
                if zero.is_file():
                    z=pd.read_csv(zero).set_index("chain",verify_integrity=True)
                    if set(z.index)!=set(g.index) or not (z.cell==cell).all(): raise ValueError("unmatched t=0 chains")
                    g[0]=z.V.reindex(g.index); g=g.sort_index(axis=1)
                times=g.columns.to_numpy(); observed,half,b=band(g.to_numpy(),post,13371)
                hit=time_to_agreement(times,observed,np.full(len(times),half))
                rows.extend(dict(cell=cell,t=t,error=y,simultaneous_halfwidth=half,block=b,starts=len(g),
                                 first_agreement_or_last=hit["time"],censored=hit["censored"]) for t,y in zip(times,observed))
            else:
                prior=np.load(out/"iid"/"endpoint"/cell/"prior_loss.npy",allow_pickle=False)
                assert prior.shape==(4096,) and np.isfinite(prior).all()
                _,sup=gap_fit(post,prior)
                vp=np.sort(post.ravel()); vg=np.sort(prior)
                grid=np.unique(np.quantile(np.r_[vp,vg],np.linspace(0,1,401)))
                cdfs.extend(dict(cell=cell,V=v,posterior=fp,prior=fg) for v,fp,fg in zip(grid,
                            np.searchsorted(vp,grid,"right")/len(vp),np.searchsorted(vg,grid,"right")/len(vg)))
                for fold in (0,1):
                    p0,p1=np.split(post,2); q0,q1=np.split(prior,2)
                    if fold: p0,p1=p1,p0; q0,q1=q1,q0
                    c,_=gap_fit(p0,q0)
                    p_event=(p1<=c).astype(float); q_event=(q1<=c).astype(float)
                    se=np.hypot(batch_means_mcse(p_event),q_event.std(ddof=1)/np.sqrt(len(q_event)))
                    rows.append(dict(cell=cell,fold=fold,threshold=c,heldout_gap=p_event.mean()-q_event.mean(),
                                conditional_mcse=se,plugin_sup=sup))
        except (OSError,ValueError,KeyError,AssertionError) as err:
            skipped.append(dict(cell=cell,reason=str(err)))
    suffix=("_"+a.cell) if a.cell else ""
    pd.DataFrame(rows).to_csv(out/(a.action+suffix+".csv"),index=False)
    pd.DataFrame(skipped,columns=["cell","reason"]).to_csv(out/(a.action+suffix+"_unavailable.csv"),index=False)
    if cdfs: pd.DataFrame(cdfs).to_csv(out/("tv_cdf"+suffix+".csv"),index=False)
    print(f"{len(rows)} rows; {len(skipped)} unavailable cells")
if __name__=="__main__": main()
