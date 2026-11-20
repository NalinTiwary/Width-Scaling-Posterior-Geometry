"""Planning sensitivity for expanding prior starts; holds the reference fixed."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recovery",type=Path,default=Path("results/revision_next/recovery"))
    p.add_argument("--ref",type=Path,default=Path("a10_next/Prior_Start_Reference_Values.csv")); a=p.parse_args()
    df=pd.read_csv(a.recovery/"prior_checkpoints.csv")
    rr=pd.read_csv(a.ref).query("observable=='V'").set_index("target_id")
    rows=[]
    for cell,g in df.groupby("cell"):
        v=g.pivot(index="chain",columns="t",values="V").to_numpy()
        if v.shape!=(32,14) or not np.isfinite(v).all():raise ValueError(cell)
        for n in (32,128,256):
            rng=np.random.default_rng(13371)
            w=rng.multinomial(n,[1/32]*32,2000)/n
            err=(w@v-v.mean(axis=0))/rr.loc[cell,"posterior_reference_sd"]
            half=np.quantile(np.abs(err).max(axis=1),.95)
            rows.append(dict(cell=cell,starts=n,start_only_projected_halfwidth=half))
    pd.DataFrame(rows).to_csv(a.recovery/"start_precision_forecast.csv",index=False)
if __name__=="__main__":main()
