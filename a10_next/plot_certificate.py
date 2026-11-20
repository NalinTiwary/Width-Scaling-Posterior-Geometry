"""Make the deterministic geometry figure from the controlled-design constants."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bnn_geometry.revision.bounds import bound_row

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--out",type=Path,default=Path(__file__).parent/"figures")
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    d=pd.DataFrame([bound_row(label=f"chi_{chi}",L=2,m=m,n=16,chi=chi) for chi in (1,4,13) for m in (64,256,1024,4096)])
    d.to_csv(a.out/"certificate_grid.csv",index=False)
    fig,ax=plt.subplots(1,2,figsize=(9,3.9),layout="constrained")
    for i,(chi,g) in enumerate(d.groupby("chi")):
        col=["#16696b","#b75b2a","#635399"][i]
        ax[0].plot(g.m,g.rho,"o-",c=col,label=f"χ = {chi}")
        valid=g[g.rho_positive]
        ax[1].plot(valid.m,valid.lambda_max_M_over_sigma2,"o-",c=col,label=f"χ = {chi}")
    ax[0].axhline(0,color="#555",ls="--",lw=1); ax[0].set_ylabel("Curvature bound ρ")
    ax[1].set_yscale("log"); ax[1].axhline(1,color="#555",ls="--",lw=1); ax[1].set_ylabel("LSI upper bound / prior constant")
    for k in ax:
        k.set_xscale("log",base=2); k.set_xticks([64,256,1024,4096],["64","256","1,024","4,096"])
        k.minorticks_off(); k.set_xlabel("Hidden width"); k.grid(axis="y",alpha=.18)
    ax[0].legend(fontsize=8); ax[0].set_title("Positive curvature condition"); ax[1].set_title("Bound where ρ > 0")
    fig.suptitle("Shallow input geometry  ·  deterministic bounds, n = 16")
    fig.savefig(a.out/"geometry_certificate.pdf")
if __name__=="__main__": main()
