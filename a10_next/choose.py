"""Freeze one finalist choice per architecture and width across all three datasets."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("results/revision_next")); a=p.parse_args()
    files=sorted((a.out/"finalists"/"endpoint").glob("*/h_*/metrics_8192.csv"))
    if len(files)!=36: raise SystemExit(f"Expected 36 candidate tables; found {len(files)}")
    rows=[]
    for f in files:
        audit=json.loads((f.parent/"run_8192.json").read_text())
        if any(audit["counters"]["numerical_failures"]): raise ValueError(f"failed candidate {f}")
        d=pd.read_csv(f); use=d[d.name.str.startswith("test_prob_") | d.name.eq("V")]
        if len(use)!=9 or use.name.nunique()!=9 or not use.status.eq("ok").all() or not np.isfinite(use.raw_ess_s).all():
            raise ValueError(f"invalid candidates in {f}")
        if not (use.steps.eq(8192)&use.chains.eq(4)&use.acceptance.gt(0)&use.acceptance.le(1)).all():
            raise ValueError(f"invalid window in {f}")
        loss=use.loc[use.name.eq("V"),"raw_ess_s"].item(); pred=use.loc[use.name.ne("V"),"raw_ess_s"].median()
        r=use.iloc[0]; rows.append(dict(arch=r.arch,m=r.m,rep=r.rep,h_over_sigma2=r.h_over_sigma2,
                                        score=min(loss,pred),loss=loss,pred=pred))
    d=pd.DataFrame(rows)
    expected={(aa,m,r,h) for aa,ms in (("deep",(32,256)),("shallow",(64,4096))) for m in ms for r in (0,1,2) for h in (.08,.16,.32)}
    if set(d[["arch","m","rep","h_over_sigma2"]].itertuples(index=False,name=None))!=expected: raise ValueError("incomplete grid")
    d.to_csv(a.out/"candidate_scores.csv",index=False)
    med=d.groupby(["arch","m","h_over_sigma2"],as_index=False).score.median(); picks=[]
    for _,g in med.groupby(["arch","m"]):
        pick=g[g.score>=.95*g.score.max()].sort_values("h_over_sigma2").iloc[0]
        picks.append(pick.to_dict())
    w=pd.DataFrame(picks); f=a.out/"winners.csv"
    data=w.to_csv(index=False)
    if f.exists() and f.read_text()!=data: raise RuntimeError("selection already frozen; use a new output root")
    f.write_text(data); print(w.to_string(index=False))
if __name__=="__main__": main()
