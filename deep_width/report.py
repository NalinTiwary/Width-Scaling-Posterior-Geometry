"""Assemble measured throughput, qualification, paired ratios and figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import PROBES, WIDTHS, diagnostic_pass, reference_agreement

HERE = Path(__file__).resolve().parent
COLORS = ("#196088", "#9C4A0F", "#535377")


def one_device(names):
    found = sorted({str(name) for name in names if str(name)})
    if len(found) != 1:
        raise ValueError(f"Metrics must come from one GPU, found {found}")
    return found[0]


def latest(path, steps=None):
    files = list(path.glob("metrics_*.csv"))
    if steps is not None:
        files = [f for f in files if int(f.stem.split("_")[-1]) == steps]
    return max(files, key=lambda f: int(f.stem.split("_")[-1])) if files else None


def collect(out, step, steps=None):
    archived = pd.read_csv(HERE / "archived_reference.csv")
    old_identity = pd.read_csv(HERE / "baseline_identity.csv").set_index("target_id")
    rows = []
    for path in sorted((out / "run" / f"pcnl_h_{step:g}").glob("deep_*")):
        data = latest(path, steps)
        if data is None:
            continue
        ref_path = out / "run" / "slice" / path.name
        ref_data = latest(ref_path)
        ref = pd.read_csv(ref_data) if ref_data else archived[archived.target_id.eq(path.name)]
        ref = ref.set_index("probe" if ref_data else "name", verify_integrity=True)
        ref_type = "fresh_slice" if ref_data else ("archived_slice" if len(ref) else "missing")
        exec_info = json.loads((path / "execution.json").read_text())
        ident = exec_info["target"]
        if ref_data:
            ref_info = json.loads((ref_path / "execution.json").read_text())
            if ident != ref_info["target"] or exec_info["input_geometry"] != ref_info["input_geometry"]:
                raise ValueError(f"Fresh reference target differs for {path.name}")
        elif len(ref):
            old = old_identity.loc[path.name]
            if any(ident[k] != old[k] for k in ("n", "d", "L", "sigma", "m", "arch")):
                raise ValueError(f"Archived model identity differs for {path.name}")
            geo = exec_info["input_geometry"]
            if any(geo[k] != old[k] for k in ("hash_X", "hash_y", "hash_X_test", "hash_center_bank")):
                raise ValueError(f"Archived data identity differs for {path.name}")
            if not (ref.target_hash.eq(old.target_hash) & ref.law.eq("full_posterior")
                    & ref.saved_stride.eq(1) & ref.last_draw.eq(65535) & ref.first_draw.eq(0)
                    & ref.stage.eq("reference_production_4")).all():
                raise ValueError(f"Invalid archive reference in {path.name}")
        frame = pd.read_csv(data)
        one_device(frame.device)
        if (set(frame.probe) != set(PROBES) or not frame.chains.eq(4).all()
                or not frame.target_hash.eq(ident["target_hash"]).all()):
            raise ValueError(f"Unexpected metrics identity in {data}")
        for row in frame.to_dict("records"):
            q = ref.loc[row["probe"]].to_dict() if row["probe"] in ref.index else None
            passed, gap, limit = reference_agreement(row, q)
            measured = np.isfinite(row["ess_mean_per_second"]) and row["ess_mean_per_second"] > 0
            if measured:
                expected = row["ess_mean"] / row["elapsed_seconds"]
                np.testing.assert_allclose(row["ess_mean_per_second"], expected)
            rows.append(dict(row, reference=ref_type, chain_pass=diagnostic_pass(row),
                             reference_pass=diagnostic_pass(q), mean_gap=gap,
                             agreement_threshold=limit, passed=passed, timed_pass=passed and measured))
    return pd.DataFrame(rows)


def pairs(metrics):
    rows = []
    for (rep, probe), group in metrics.groupby(["rep", "probe"]):
        base = group[group.width.eq(256)]
        if base.empty:
            continue
        b = base.iloc[0]
        for r in group[group.width.gt(256)].itertuples():
            def ratio(value, baseline):
                return value / baseline if np.isfinite(baseline) and baseline > 0 else np.nan
            gain = ratio(r.ess_mean_per_1000_transitions, b.ess_mean_per_1000_transitions)
            cost = ratio(r.seconds_per_joint_step, b.seconds_per_joint_step)
            rate = ratio(r.ess_mean_per_second, b.ess_mean_per_second)
            if np.isfinite(rate):
                np.testing.assert_allclose(rate, gain / cost)
            rows.append(dict(rep=rep, probe=probe, chi=r.chi, width=r.width, baseline_width=256,
                             steps=r.steps, baseline_steps=b.steps, matched_steps=r.steps == b.steps,
                             ess_gain_per_transition=gain, cost_ratio=cost, rate_ratio=rate,
                             timed_pass=bool(r.timed_pass and b.timed_pass)))
    return pd.DataFrame(rows)


def figure(out, series, panels, name, caption):
    fig, axes = plt.subplots(1, len(panels), figsize=(9.4, 4.1))
    for ax, (key, ylabel, gate) in zip(np.atleast_1d(axes), panels):
        for rep, color in enumerate(COLORS):
            data = series[series.rep.eq(rep)].set_index("width").reindex(WIDTHS)
            qualified = data[gate].eq(True)
            y = data[key].astype(float)
            good = y.where(qualified)
            bad = y.where(~qualified)
            ax.plot(WIDTHS, good, "o-", color=color, label=f"Dataset {rep}", lw=1.5, ms=5)
            ax.plot(WIDTHS, bad, "o", mfc="white", mec=color, ms=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(WIDTHS, [f"{w:,}" for w in WIDTHS], rotation=35)
        ax.set_xlabel("Hidden-layer width")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.26)
    axes = np.atleast_1d(axes)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(out / f"{name}.{fmt}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (out / f"{name}_caption.txt").write_text(caption + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/deep_width"))
    parser.add_argument("--step", type=float, default=0.16)
    parser.add_argument("--steps", type=int, help="Use this exact retained length per chain")
    args = parser.parse_args()
    inputs = pd.read_csv(args.out / "inputs" / "input_geometry.csv").sort_values("rep")
    if inputs.rep.tolist() != [0, 1, 2] or not (inputs.n.eq(128) & inputs.d.eq(32)).all():
        raise ValueError("Input manifest missing the three frozen datasets")
    work = collect(args.out, args.step, args.steps)
    if work.empty:
        raise SystemExit("No matching production metrics")
    for rep, group in work.groupby("rep"):
        np.testing.assert_allclose(group.chi, inputs.set_index("rep").at[rep, "chi"])
    window = args.steps if args.steps is not None else "latest"
    chi = ", ".join(f"{v:.3f}" for v in inputs.chi)
    groups = list(work.groupby("device", sort=True))
    for device_name, part in groups:
        slug = ""
        if len(groups) > 1:
            slug = "_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in str(device_name)).strip("_")
        out = args.out / f"report_h_{args.step:g}_{window}{slug}"
        out.mkdir(parents=True, exist_ok=True)
        part.to_csv(out / "probe_metrics.csv", index=False)
        pairs(part).to_csv(out / "ratios_vs_256.csv", index=False)
        desc = ("Two-hidden-layer tanh BNNs on three frozen synthetic binary-classification datasets. "
                "Rows of each training input matrix X are unit length, with X of shape 128 by 32. "
                f"The largest eigenvalues of X X^T are {chi} in dataset order 0, 1, 2. "
                "Each dataset uses the same X across widths. See inputs/ for the actual matrices and hashes. "
                f"Four float64 adjusted Langevin chains ran on one {device_name} per cell at h/sigma^2={args.step:g}. "
                "The production clock includes scalar recording and checkpoint writes. "
                "Open markers indicate incomplete diagnostic or reference qualification.")
        loss = part[part.probe.eq("V")]
        figure(out, loss, [("ess_mean_per_second", "Loss mean ESS per second", "timed_pass"),
                           ("ess_mean_per_1000_transitions", "Loss mean ESS per 1,000 transitions", "passed")],
               "loss_throughput", desc)
        pred = (part[part.probe.str.startswith("test_prob_")].groupby(["rep", "width"], as_index=False)
                .agg(ess_mean_per_second=("ess_mean_per_second", "median"),
                     seconds_per_joint_step=("seconds_per_joint_step", "first"),
                     plot_pass=("timed_pass", "all"), probes=("probe", "size")))
        if not pred.probes.eq(8).all():
            raise ValueError("Each prediction summary needs eight predeclared predictions")
        pred["clock_present"] = np.isfinite(pred["seconds_per_joint_step"])
        figure(out, pred, [("ess_mean_per_second", "Median prediction mean ESS per second", "plot_pass"),
                           ("seconds_per_joint_step", "Seconds per four-chain joint step", "clock_present")],
               "prediction_throughput_and_cost",
               desc + " The prediction panel shows the median rate across the eight fixed held-out probabilities. The cost panel uses all available production clocks.")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
