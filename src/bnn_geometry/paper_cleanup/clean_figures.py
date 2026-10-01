"""Render the two paper figures from audited, unrounded CSV adapter outputs.

See figure_cleanup_guide.md for schemas. This module never samples or estimates
ACF. The spectral mode computes empirical quantiles and coverage from saved norms. It contains no campaign values. NumPy and Matplotlib are required.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

WIDTH_STYLE = {
    32: ("#0072B2", "o"), 64: ("#56B4E9", "s"),
    128: ("#009E73", "^"), 256: ("#E69F00", "D"),
    1024: ("#D55E00", "v"), 4096: ("#CC79A7", "P"),
}
QUANTILE_STYLE = {
    0.5: ("#56B4E9", "o", "Median"),
    0.95: ("#0072B2", "s", "95th percentile"),
    0.99: ("#253494", "^", "99th percentile"),
}
WIDTHS = {"shallow": [64, 256, 1024, 4096], "deep": [32, 64, 128, 256]}
REPS = (0, 1, 2)
RC = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.65, "axes.edgecolor": "#444444",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.transparent": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def read_csv(path, columns):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != set(columns):
            raise ValueError(f"Expected columns exactly {columns}; got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty input table")
    return rows


def base_figure(width_in, spectral=False):
    if not 5.0 <= width_in <= 8.0:
        raise ValueError("width-in must be in [5,8]; redesign smaller panels instead of shrinking text")
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(width_in, 2.65))
    if spectral:
        left = fig.add_axes([0.11, 0.29, 0.36, 0.57])
        right = fig.add_axes([0.62, 0.29, 0.36, 0.57])
    else:
        left = fig.add_axes([0.105, 0.29, 0.405, 0.57])
        right = fig.add_axes([0.570, 0.29, 0.405, 0.57], sharey=left)
        right.tick_params(labelleft=False)
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
    return fig, (left, right)


def legend_handle(color, marker, size=4):
    return Line2D([], [], color=color, marker=marker, linewidth=1.6,
                  markersize=size, markeredgewidth=0.5)


def check_legends(fig, expected):
    if len(fig.legends) != expected:
        raise AssertionError("Wrong figure-legend count")
    child_ids = [id(a) for a in fig.get_children()]
    for legend in fig.legends:
        if child_ids.count(id(legend)) != 1:
            raise AssertionError("Duplicate legend artist: do not re-add a figure legend")


def empirical_cdf(values, cutoffs):
    x = np.asarray(values, dtype=np.float64)
    a = np.asarray(cutoffs, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Invalid statewise spectral values")
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("Invalid cutoff array")
    return np.searchsorted(np.sort(x), a, side="right") / x.size


def threshold_counts(values, threshold, guard=1e-10):
    x = np.asarray(values, dtype=np.float64)
    empirical_cdf(x, [threshold])  # validates values and threshold
    if not np.isfinite(guard) or guard < 0:
        raise ValueError("Invalid numerical guard")
    return {"n": int(x.size), "n_above": int(np.sum(x > threshold)),
            "n_at_or_below": int(np.sum(x <= threshold)),
            "fraction_above": float(np.mean(x > threshold)),
            "n_near_threshold": int(np.sum(np.abs(x-threshold) <= guard)),
            "n_definitely_above": int(np.sum(x > threshold+guard)),
            "n_possibly_above": int(np.sum(x > threshold-guard)),
            "maximum": float(x.max())}


def read_theory_context(path):
    context = json.loads(Path(path).read_text())
    required = {"weight_layers": 3, "n": 128, "sigma": 1.0, "r0": 0.0,
                "hidden_center_max_normalized_opnorm": 0.0,
                "statistic": "opnorm_W2_over_sqrt_m"}
    for key, value in required.items():
        if context.get(key) != value:
            raise ValueError(f"This campaign requires {key}={value}; verify original target metadata")
    digest = context.get("target_manifest_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("A SHA-256 of the audited target manifest is required")
    return context, float(context["r0"] + 2*context["sigma"])


def spectral_figure(path, width_in=5.5, context_path=None, expected_per_chain=4096):
    if context_path is None:
        raise ValueError("Provide the audited theory-context JSON, not a chosen display margin")
    context, threshold = read_theory_context(context_path)
    if expected_per_chain < 2:
        raise ValueError("Expected archive count must be at least 2")
    rows = read_csv(path, ["width", "replicate", "chain", "draw", "T"])
    groups, seen = {}, set()
    for row in rows:
        key = (int(row["width"]), int(row["replicate"]), int(row["chain"]))
        draw, value = int(row["draw"]), float(row["T"])
        if key+(draw,) in seen:
            raise ValueError(f"Duplicate state ID {key+(draw,)}")
        if draw < 0 or not np.isfinite(value) or value < 0:
            raise ValueError("Invalid draw ID or norm")
        seen.add(key+(draw,)); groups.setdefault(key, []).append((draw, value))
    expected = {(m, r, c) for m in WIDTHS["deep"] for r in REPS for c in range(4)}
    if set(groups) != expected:
        raise ValueError("Missing or unexpected width/replicate/chain combination")
    for key, values in groups.items():
        values.sort()
        if len(values) != expected_per_chain:
            raise ValueError(f"Wrong archive count for {key}: expected {expected_per_chain}")
        if not np.all(np.diff([x[0] for x in values]) == 16):
            raise ValueError("Expected original regularly archived transition IDs at stride 16")
    samples = {(m, r): np.array([v for c in range(4) for _, v in groups[m, r, c]])
               for m in WIDTHS["deep"] for r in REPS}
    quantiles, count_rows, quantile_rows = {}, [], []
    for (m, r), values in samples.items():
        quantiles[m, r] = np.quantile(values, list(QUANTILE_STYLE), method="linear")
        count_rows.append({"width": m, "replicate": r, "threshold": threshold,
                           **threshold_counts(values, threshold)})
        for q, val in zip(QUANTILE_STYLE, quantiles[m, r]):
            quantile_rows.append({"width": m, "replicate": r, "quantile": q, "T": float(val)})
    all_q = np.concatenate(list(quantiles.values()))
    ylow = min(1.85, float(np.floor((all_q.min()-0.01)*100)/100))
    yhigh = max(2.16, float(np.ceil((all_q.max()+0.01)*100)/100))
    minimum = min(float(x.min()) for x in samples.values())
    maximum = max(float(x.max()) for x in samples.values())
    xlow = min(1.75, float(np.floor((minimum-0.01)*100)/100))
    xhigh = max(2.30, float(np.ceil((maximum+0.01)*100)/100))
    fig, (left, right) = base_figure(width_in, spectral=True)
    left.set_title("(a) Spectral quantiles", loc="left", pad=6)
    right.set_title("(b) Coverage vs. cutoff", loc="left", pad=6)
    left.set_xscale("log", base=2)
    left.set_xlim(2**(5-0.12), 2**(8+0.12))
    left.xaxis.set_major_locator(FixedLocator(WIDTHS["deep"]))
    left.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    left.xaxis.set_minor_locator(NullLocator())
    left.set_xlabel(r"Width $m$")
    left.set_ylabel(r"$T=\|W_2\|_{\mathrm{op}}/\sqrt{m}$")
    left.set_ylim(ylow, yhigh); left.set_yticks([1.9, 2.0, 2.1])
    left.axhline(threshold, color="#555555", linewidth=.9, linestyle=(0,(4,3)), zorder=1)
    left.annotate(r"$a_*=2$", xy=(.025, threshold), xycoords=left.get_yaxis_transform(),
                  xytext=(0,-3), textcoords="offset points", ha="left", va="top", color="#555555")
    for qi, (q, (color, marker, _)) in enumerate(QUANTILE_STYLE.items()):
        array = np.array([[quantiles[m, r][qi] for m in WIDTHS["deep"]] for r in REPS])
        middle = np.median(array, axis=0)
        left.errorbar(WIDTHS["deep"], middle,
                      yerr=np.vstack([middle-array.min(0),array.max(0)-middle]),
                      fmt="none", color=color, elinewidth=.75, capsize=2, capthick=.75, zorder=2)
        left.plot(WIDTHS["deep"], middle, color=color, marker=marker,
                  linewidth=1.6, markersize=4, markeredgewidth=.5, zorder=3)
    coverage_arrays = {}
    for m in WIDTHS["deep"]:
        color, marker = WIDTH_STYLE[m]
        knots = np.unique(np.concatenate([samples[m, r] for r in REPS] + [[xlow, threshold, xhigh]]))
        cdfs = np.stack([empirical_cdf(samples[m, r], knots) for r in REPS])
        middle = np.median(cdfs, axis=0)
        right.fill_between(knots, cdfs.min(0), cdfs.max(0), step="post",
                           color=color, alpha=.10, linewidth=0, zorder=1)
        right.step(knots, middle, where="post", color=color, linewidth=1.6, zorder=3)
        at_threshold = np.median([empirical_cdf(samples[m, r], [threshold])[0] for r in REPS])
        right.plot([threshold], [at_threshold], marker=marker, color=color,
                   markersize=4, markeredgewidth=.5, linestyle="none", zorder=4)
        coverage_arrays[f"width_{m}_a"] = knots
        coverage_arrays[f"width_{m}_cdf_replicates"] = cdfs
    right.axvline(threshold, color="#555555", linewidth=.9, linestyle=(0,(4,3)), zorder=2)
    right.annotate(r"$a_*=2$", xy=(threshold,.08), xycoords=right.get_xaxis_transform(),
                   xytext=(4,0), textcoords="offset points", ha="left", va="bottom", color="#555555")
    right.set_xlim(xlow,xhigh); right.set_xticks([1.8,2.0,2.2])
    right.set_ylim(0,1.02); right.set_yticks([0,.25,.5,.75,1])
    right.set_xlabel(r"Cutoff $a$"); right.set_ylabel("Empirical coverage")
    fig.legend([legend_handle(c,m) for c,m,_ in QUANTILE_STYLE.values()],
               ["Median","95th","99th"], ncols=3, loc="center", bbox_to_anchor=(.29,.085),
               frameon=False, columnspacing=.6, handletextpad=.35, handlelength=1.4)
    order = [32,128,64,256]
    fig.legend([legend_handle(*WIDTH_STYLE[m]) for m in order], [f"$m={m}$" for m in order],
               ncols=2, loc="center", bbox_to_anchor=(.80,.075), frameon=False,
               columnspacing=1.0, handletextpad=.4)
    check_legends(fig,2)
    return fig, {"display_units":"opnorm_W2_over_sqrt_m", "admissibility_threshold":threshold,
                 "strict_cutoff_condition":"a > r0 + 2*sigma", "theory_context":context,
                 "theory_context_sha256":hashlib.sha256(Path(context_path).read_bytes()).hexdigest(),
                 "spectral_rows":len(rows), "quantile_ylim":[ylow,yhigh], "coverage_xlim":[xlow,xhigh],
                 "numerical_guard":1e-10, "threshold_counts":count_rows,
                 "replicate_quantiles":quantile_rows, "_coverage_arrays":coverage_arrays}


def acf_figure(path, width_in=5.5, lag_limit=200):
    if lag_limit not in (200, 400):
        raise ValueError("Main loss display supports 200 or 400 lags")
    rows = read_csv(path, ["architecture", "width", "replicate", "lag", "acf"])
    groups = {}
    for row in rows:
        key = (row["architecture"], int(row["width"]), int(row["replicate"]))
        k, val = int(row["lag"]), float(row["acf"])
        if k in groups.setdefault(key, {}):
            raise ValueError(f"Duplicate ACF key {key + (k,)}")
        groups[key][k] = val
    expected = {(a, m, r) for a in WIDTHS for m in WIDTHS[a] for r in REPS}
    if set(groups) != expected:
        raise ValueError("Missing or unexpected architecture/width/replicate curves")
    lengths = {len(g) for g in groups.values()}
    if len(lengths) != 1 or min(lengths) < 401:
        raise ValueError("All curves must have equal length and at least lags 0..400")
    size = lengths.pop()
    arrays = {}
    for key, g in groups.items():
        if set(g) != set(range(size)):
            raise ValueError("Lag grid is not consecutive from zero")
        v = np.array([g[k] for k in range(size)])
        if not np.isfinite(v).all() or np.max(np.abs(v)) > 1+1e-10 or abs(v[0]-1) > 1e-12:
            raise ValueError("Invalid ACF values or lag-zero normalization")
        arrays[key] = v
    shown_min = min(float(v[:lag_limit+1].min()) for v in arrays.values())
    lower = -0.08 if shown_min >= -0.08 else np.floor((shown_min-0.02)/0.05)*0.05
    fig, axes = base_figure(width_in)
    x = np.arange(lag_limit+1)
    for ax, a, title, center in zip(axes, WIDTHS, ("(a) Shallow", "(b) Deep"), (0.3075, 0.7725)):
        ax.set_title(title, loc="left", pad=6)
        for m in WIDTHS[a]:
            color, marker = WIDTH_STYLE[m]
            arr = np.stack([arrays[a, m, r][:lag_limit+1] for r in REPS])
            ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=0.10,
                            linewidth=0, zorder=1)
            ax.plot(x, np.median(arr, axis=0), color=color, linewidth=1.6,
                    marker=marker, markersize=3.2, markeredgewidth=0.5,
                    markevery=list(range(lag_limit//8, lag_limit+1, lag_limit//8)), zorder=3)
        ax.axhline(0, color="#777777", linewidth=0.65, zorder=2)
        ax.set_xlim(0, lag_limit)
        ax.set_xticks(np.linspace(0, lag_limit, 5))
        ax.set_ylim(lower, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Lag (sampler iterations)")
        # Matplotlib fills columns; reorder so rows read in increasing width.
        ordered = [WIDTHS[a][i] for i in (0, 2, 1, 3)]
        fig.legend([legend_handle(*WIDTH_STYLE[m], size=3.2) for m in ordered],
                   [f"$m={m}$" for m in ordered], ncols=2, loc="center",
                   bbox_to_anchor=(center, 0.075), frameon=False,
                   columnspacing=1.3, handletextpad=0.5)
    axes[0].set_ylabel("Loss autocorrelation")
    check_legends(fig, 2)
    return fig, {"display_lags": [0, lag_limit], "source_lags": [0, size-1],
                 "ylim": [float(lower), 1.02], "acf_rows": len(rows),
                 "aggregation": "pointwise median and min/max of 3 four-chain replicate curves"}


def write_outputs(fig, out, source, metadata):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(str(out)+suffix, dpi=300, facecolor="white", transparent=False)
    if "_coverage_arrays" in metadata:
        arrays = metadata.pop("_coverage_arrays")
        np.savez_compressed(str(out)+"_coverage.npz", **arrays)
        for key, suffix in [("threshold_counts", "_threshold_counts.csv"),
                            ("replicate_quantiles", "_quantiles.csv")]:
            rows = metadata[key]
            with Path(str(out)+suffix).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    metadata.update({
        "source_file": str(Path(source).resolve()),
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "figure_inches": fig.get_size_inches().tolist(),
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "rc": RC,
        "width_style": WIDTH_STYLE, "quantile_style": QUANTILE_STYLE,
    })
    Path(str(out)+".json").write_text(json.dumps(metadata, indent=2)+"\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["spectral", "acf"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True, help="Output filename prefix, without extension")
    parser.add_argument("--width-in", type=float, default=5.5)
    parser.add_argument("--lag-limit", type=int, default=200)
    parser.add_argument("--context", help="Required spectral theory-context JSON")
    parser.add_argument("--expected-per-chain", type=int, default=4096)
    args = parser.parse_args()
    if args.kind == "spectral":
        fig, meta = spectral_figure(args.input, args.width_in, args.context, args.expected_per_chain)
    else:
        fig, meta = acf_figure(args.input, args.width_in, args.lag_limit)
    write_outputs(fig, args.out, args.input, meta)


if __name__ == "__main__":
    main()
