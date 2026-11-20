"""Tabulate fixed, previously monitored directions on the long elliptical-slice reference."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("verified_evidence"))
    args = parser.parse_args()
    names = ["V", "q_A", "q_W"] + [f"test_prob_{i}" for i in range(0, 1024, 128)]
    data = pd.read_csv(args.input)
    data = data[data.name.isin(names)].copy()
    keys = ["target_id", "name"]
    if len(data) != 264 or data.duplicated(keys).any() or not data.groupby("target_id").size().eq(11).all():
        raise ValueError("Expected 24 targets and 11 unique archived observables per target")
    protocol = (data.first_draw.eq(0) & data.last_draw.eq(65_535) & data.saved_stride.eq(1)
                & data.stage.eq("reference_production_4") & data.law.eq("full_posterior"))
    if not protocol.all():
        raise ValueError("Reference windows are inconsistent")
    data["raw_iat_steps"] = 4 * 65_536 / data.ess_mean
    loss = data[data.name.eq("V")].set_index("target_id").raw_iat_steps
    data["iat_over_loss"] = data.raw_iat_steps / data.target_id.map(loss)
    args.out.mkdir(parents=True, exist_ok=True)
    fields = [*keys,"architecture","m","replicate","raw_iat_steps","iat_over_loss",
              "rhat","ess_mean","ess_bulk","ess_tail","gate_pass"]
    data[fields].to_csv(args.out / "slice_reference_probes.csv", index=False)
    rows = []
    for arch, sub in data.groupby("architecture"):
        lo, hi = (32,256) if arch == "deep" else (64,4096)
        endpoints = sub[sub.m.eq(lo)].merge(sub[sub.m.eq(hi)],on=["name","replicate"],
                                            suffixes=("_narrow","_wide"),validate="one_to_one")
        for r in endpoints.itertuples():
            rows.append(dict(architecture=arch,dataset=r.replicate,probe=r.name,
                             wide_narrow_raw_ess_per_transition=r.ess_mean_wide/r.ess_mean_narrow,
                             both_passed=r.gate_pass_narrow and r.gate_pass_wide))
    pd.DataFrame(rows).to_csv(args.out / "slice_reference_widths.csv", index=False)


if __name__ == "__main__":
    main()
