"""Contrast the width effects of chi=4 and chi=13 to chi=1 within each dataset."""
import argparse
from pathlib import Path
import pandas as pd

def contrasts(d):
    keys=["arch","rep","name"]
    measures=["raw_ess_s_ratio","bulk_ess_s_ratio","raw_ess_per_transition_ratio","joint_step_s_ratio"]
    cols=["condition",*keys,"passed_narrow","passed_wide",*measures]
    sub=d[cols].copy()
    if sub.duplicated(["condition",*keys]).any():raise ValueError("duplicate matched probe")
    for c in ("passed_wide","passed_narrow"):
        sub[c]=sub[c].astype("string").str.lower().eq("true").fillna(False)
    result=sub[sub.condition.ne("chi1")].merge(sub[sub.condition.eq("chi1")],on=keys,suffixes=("","_base"),validate="many_to_one")
    result["passed_all_four"]=result.passed_narrow & result.passed_wide & result.passed_narrow_base & result.passed_wide_base
    for m in measures:result["contrast_"+m]=result[m]/result[m+"_base"]
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=Path("results/revision_next"));a=p.parse_args()
    for suffix in ("","_6192"):
        src=a.out/f"pairs_geometry{suffix}.csv"
        if src.exists():contrasts(pd.read_csv(src)).to_csv(a.out/f"chi_interactions{suffix}.csv",index=False)
if __name__=="__main__":main()
