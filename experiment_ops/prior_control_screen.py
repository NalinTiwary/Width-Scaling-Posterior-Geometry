"""Compare observed width effects for the same Langevin kernel with and without the likelihood."""
import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def screen(prior, posterior):
    data = prior.query("stage == 'prior-only' and window == 'original'").copy()
    names = ["V", *[f"test_prob_{i}" for i in range(0,1024,128)]]
    data = data[data.name.isin(names)]
    by_width = []
    for width in [32,256]:
        part = data[data.cell.eq(f"deep_m{width:04d}_r0")].set_index("name",verify_integrity=True)
        if (set(part.index)!=set(names) or not part.steps_per_chain.eq(8192).all()
                or not part.chains.eq(4).all() or not np.allclose(part.h_over_sigma2,.01) or not part.acceptance.eq(1).all()):
            raise ValueError("Prior-only target, window or transition mismatch")
        by_width.append(part)
    q = posterior.query("architecture == 'deep' and replicate == 0 and narrow_m == 32 and wide_m == 256").set_index("observable",verify_integrity=True)
    if set(q.index)!=set(names): raise ValueError("Missing matched posterior probe")
    out = pd.DataFrame(index=names)
    for label, data in zip(["narrow","wide"],by_width):
        out[f"prior_bulk_{label}"] = data.ess_bulk
        out[f"prior_rank_rhat_{label}"] = data.rhat
    out["prior_width_ratio"] = out.prior_bulk_wide/out.prior_bulk_narrow
    out["posterior_width_ratio"] = q.ess_ratio_wide_over_narrow
    out["posterior_ratio_over_prior_ratio"] = out.posterior_width_ratio/out.prior_width_ratio
    out["prior_steps_per_chain"] = 8192
    out["posterior_steps_per_chain"] = 102400
    return out.rename_axis("probe").reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recovered",type=Path,default=Path("results/revision_next/recovery/recovered_diagnostics.csv"))
    p.add_argument("--archive",type=Path,default=Path("a10_next/a10_revision_attachment.zip"))
    p.add_argument("--out",type=Path,default=Path("results/revision_next/existing"))
    a = p.parse_args()
    with ZipFile(a.archive) as z:
        result = screen(pd.read_csv(a.recovered), pd.read_csv(z.open("replay_endpoint_ratios.csv")))
    a.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.out/"prior_posterior_control.csv", index=False)


if __name__=="__main__":
    main()
