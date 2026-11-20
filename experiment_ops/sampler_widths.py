"""Match the selected PCNL step at each width and calculate within-kernel width ratios."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def ratios(data,winners):
    d=data.copy();labels=d.cell.str.extract(r"^(deep|shallow)_m(\d+)_r(\d+)$")
    d["arch"]=labels[0];d["m"]=labels[1].astype(int);d["rep"]=labels[2].astype(int)
    d=d.merge(winners[["arch","m","h_over_sigma2"]],on=["arch","m"],validate="many_to_one")
    d["kernel"]=np.where(d.arm.eq("slice"),"slice",np.where(d.arm.eq("h_0.01"),"pcnl_fixed","pcnl_selected"))
    selected=d.kernel.eq("pcnl_selected")
    if not np.allclose(d.loc[selected,"arm"].str.removeprefix("h_").astype(float),d.loc[selected,"h_over_sigma2"]):raise ValueError("unexpected selected arm")
    blocks=[]
    for arch,sub in d.groupby("arch"):
        m0,m1=(32,256) if arch=="deep" else (64,4096)
        q=sub[sub.m==m0].merge(sub[sub.m==m1],on=["kernel","arch","rep","name","steps"],validate="one_to_one",suffixes=("_narrow","_wide"))
        q["wide_narrow_raw_per_transition"]=q.ess_mean_wide/q.ess_mean_narrow
        q["wide_narrow_raw_per_wall_s"]=q.raw_ess_per_wall_s_wide/q.raw_ess_per_wall_s_narrow
        q["qualified"]=q.posterior_check_wide & q.posterior_check_narrow & q.clock_present_wide & q.clock_present_narrow
        blocks.append(q)
    paired=pd.concat(blocks,ignore_index=True)
    group=paired.groupby(["arch","rep","name","steps"])
    paired["all_three_arms_present"]=group.kernel.transform("nunique").eq(3)
    paired["all_three_arms_qualified"]=paired.all_three_arms_present & group.qualified.transform("all")
    return paired
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=Path("results/revision_next"))
    p.add_argument("--steps",type=int,default=8192);a=p.parse_args()
    d=pd.read_csv(a.out/f"sampler_comparison_{a.steps}.csv")
    ratios(d,pd.read_csv(a.out/"winners.csv")).to_csv(a.out/f"sampler_width_ratios_{a.steps}.csv",index=False)
if __name__=="__main__":main()
