"""Reproduce the four-day plan's numerical baselines from saved summaries."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,default=Path("a10_next/Prior_Start_Reference_Values.csv"))
    p.add_argument("--archive",type=Path,default=Path("a10_next/a10_revision_attachment.zip"))
    p.add_argument("--recovered",type=Path,default=Path("results/revision_next/recovery/recovered_diagnostics.csv"))
    p.add_argument("--out",type=Path,default=Path("results/revision_next/existing"))
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    r=pd.read_csv(a.reference).query("observable=='V'").copy()
    assert len(r)==12 and not r.target_id.duplicated().any()
    assert (r.first_draw.eq(0)&r.last_draw.eq(65535)&r.saved_stride.eq(1)&r.law.eq("full_posterior")&r.stage.eq("reference_production_4")).all()
    with ZipFile(a.archive) as z:
        d=pd.read_csv(z.open("replay_endpoint_ratios.csv")).query("observable=='V'").copy()
    rs=[]
    for x in d.itertuples():
        sub=r[(r.architecture==x.architecture)&(r.replicate==x.replicate)].set_index("m")
        es=sub.loc[x.wide_m,"ess_bulk"]/sub.loc[x.narrow_m,"ess_bulk"]
        rs.append(dict(architecture=x.architecture,dataset=x.replicate,probe="V",
            pcnl_wide_narrow_bulk_per_transition=x.ess_ratio_wide_over_narrow,
            slice_wide_narrow_bulk_per_transition=es,pcnl_wide_narrow_bulk_per_A10_s=x.bulk_ess_per_second_ratio,
            pcnl_steps_per_chain=102400,slice_steps_per_chain=65536,
            pcnl_wide_extra_time_percent_for_equal_ess=100*(1/x.bulk_ess_per_second_ratio-1)))
    pd.DataFrame(rs).to_csv(a.out/"two_kernel_archived_loss.csv",index=False)
    old=pd.read_csv(a.recovered).query("window=='after_nominal_warmup'").copy()
    fixed=old[old.stage.eq("fixed-step")].copy()
    assert (fixed.groupby('cell').name.nunique()==11).all() and fixed.cell.nunique()==24
    fixed["iat_steps"]=fixed.chains*fixed.steps_per_chain/fixed.ess_mean
    worst=fixed.loc[fixed.groupby("cell").iat_steps.idxmax(),["cell","name","iat_steps","rhat","ess_bulk"]].set_index("cell")
    worst["loss_iat"]=fixed.query("name=='V'").set_index("cell").iat_steps
    worst["iat_over_loss"]=worst.iat_steps/worst.loss_iat
    worst.to_csv(a.out/"short_worst_observed.csv")
    geo=old.query("stage=='geometry-pilot' and name=='V'").copy()
    geo["chi"]=geo.cell.str.extract(r"^geometry_r([^_]+)_")[0].astype(float)*15+1
    geo["width"]=geo.cell.str.extract(r"_m(\d+)_")[0].astype(int)
    values=geo.pivot(index="chi",columns="width",values="ess_bulk")
    values.columns=["narrow_bulk","wide_bulk"];values["wide_narrow_bulk_per_transition"]=values.wide_bulk/values.narrow_bulk
    values.to_csv(a.out/"dataset0_geometry_screen.csv")
if __name__=="__main__":main()
