"""Rendering: computed tables -> figure_1_spectral_domain and figure_2_loss_acf (guide §4, §7, §8).

Loads only the finished tables; it runs no sampler, chooses no observable or replicate and makes no validity
decision. Regenerate both figures with
    python -m bnn_geometry.paper_figures.render --tables <paper_figures>/tables --figures <paper_figures>/figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from .common import MPL_STYLE, style  # noqa: E402

FIXTURE_PALETTE = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]


def _save(fig, figures: Path, stem: str) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    out = [figures / f"{stem}.pdf", figures / f"{stem}.png"]
    fig.savefig(out[0])
    fig.savefig(out[1], dpi=300)
    plt.close(fig)
    return out


def _fmt_int(n: int) -> str:
    return f"{int(n):,}"


# ---- Figure 1 ---------------------------------------------------------------------------------------
def figure1(tables: Path, figures: Path) -> list[Path]:
    S = style()
    c, sp = S["common"], S["spectral"]
    plot = pd.read_csv(tables / "spectral_plot.csv")
    counts = json.loads((tables / "spectral_counts.json").read_text())
    widths = sorted(int(m) for m in plot.m.unique())
    with plt.rc_context(MPL_STYLE):
        fig = plt.figure(figsize=tuple(sp["figure_inches"]))
        ax = fig.add_axes(sp["axes_rect"])
        ax.set_xscale("log", base=2)
        pad = float(sp["x_padding_log2"])
        ax.set_xlim(2 ** (math.log2(widths[0]) - pad), 2 ** (math.log2(widths[-1]) + pad))
        ax.xaxis.set_major_locator(FixedLocator(widths))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(m) for m in widths])
        ax.set_xlabel(r"Width $m$")
        ax.set_ylabel(r"$\|W_2\|_{\mathrm{op}}/(2.5\sqrt{m})$")

        lo_data = float(np.nanmin(plot[["replicate_min", "posterior_median", "prior"]].to_numpy()))
        hi_data = float(np.nanmax(plot[["replicate_max", "posterior_median", "prior"]].to_numpy()))
        y0, y1 = sp["y_limits_default"]
        if lo_data < y0:
            y0 = math.floor(lo_data / 0.05) * 0.05 - 0.02
        if hi_data > 1.02:
            y1 = hi_data + 0.02
        ax.set_ylim(y0, y1)
        ticks = [t for t in sp["y_ticks"]]
        if y0 < ticks[0]:
            ticks = sorted(set([round(x, 2) for x in np.arange(math.ceil(y0 * 10) / 10, 1.0001, 0.1)]))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", which="major", color=c["grid_color"], linewidth=c["grid_linewidth"])
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

        ax.axhline(sp["boundary"], color=sp["boundary_color"], linewidth=sp["boundary_linewidth"],
                   dashes=tuple(sp["boundary_dash"]), zorder=1.5)
        ax.text(sp["boundary_label_axes_x"], sp["boundary_label_data_y"], "Domain boundary",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, zorder=6)

        qstyle = sp["quantiles"]
        for qlab, qs in qstyle.items():
            d = plot[plot["quantile"].map(lambda v: f"{float(v):.2f}") == qlab].sort_values("m")
            x = d.m.to_numpy(dtype=float)
            med = d.posterior_median.to_numpy()
            err = np.vstack([med - d.replicate_min.to_numpy(), d.replicate_max.to_numpy() - med])
            # 1) posterior line + replicate-range whiskers
            ax.plot(x, med, "-", color=qs["color"], linewidth=sp["posterior_linewidth"], zorder=2)
            ax.errorbar(x, med, yerr=err, fmt="none", ecolor=qs["color"], elinewidth=sp["whisker_linewidth"],
                        capsize=sp["whisker_cap_points"], capthick=sp["whisker_linewidth"], zorder=2)
            # 2) prior dashed curve
            ax.plot(x, d.prior.to_numpy(), color=qs["color"], linewidth=sp["prior_linewidth"],
                    dashes=tuple(sp["prior_dash"]), alpha=sp["prior_alpha"], zorder=3)
        for qlab, qs in qstyle.items():
            d = plot[plot["quantile"].map(lambda v: f"{float(v):.2f}") == qlab].sort_values("m")
            # 3) posterior filled points
            ax.plot(d.m, d.posterior_median, linestyle="none", marker=qs["marker"], markersize=sp["posterior_marker_size"],
                    color=qs["color"], markeredgewidth=0, zorder=4)
        for qlab, qs in qstyle.items():
            d = plot[plot["quantile"].map(lambda v: f"{float(v):.2f}") == qlab].sort_values("m")
            # 4) hollow prior outlines last
            ax.plot(d.m, d.prior, linestyle="none", marker=qs["marker"], markersize=sp["prior_marker_size"],
                    markerfacecolor=sp["prior_marker_face"], markeredgecolor=qs["color"],
                    markeredgewidth=sp["prior_marker_edgewidth"], zorder=5)

        tx, ty = sp["title_figure_xy"]
        fig.text(tx, ty, "Deep posterior spectral domain", ha="center", va="top", fontsize=9)
        n_out, n_amb, n_ins = counts["n_outside"], counts["n_ambiguous"], counts["n_inspected"]
        cnt = f"{_fmt_int(n_out)} exits / {_fmt_int(n_ins)} inspected states"
        if n_amb:
            cnt += f"; {_fmt_int(n_amb)} numerically unresolved"
        cx, cy = sp["count_figure_xy"]
        fig.text(cx, cy, cnt, ha="center", va="top", fontsize=7)
        if counts.get("fixture"):
            fig.text(0.02, 0.5, "FIXTURE - NOT A RESULT", rotation=90, ha="left", va="center", fontsize=6,
                     color="#C00000")

        qh = [Line2D([], [], color=qs["color"], marker=qs["marker"], markersize=sp["posterior_marker_size"],
                     linewidth=sp["posterior_linewidth"], markeredgewidth=0, label=qs["label"])
              for qs in qstyle.values()]
        lq = fig.legend(handles=qh, loc="center", bbox_to_anchor=tuple(sp["quantile_legend_figure_xy"]), ncol=3,
                        frameon=False, fontsize=7, handlelength=2.2, columnspacing=1.2)
        fig.add_artist(lq)
        rh = [Line2D([], [], color="black", linewidth=sp["posterior_linewidth"], marker="o",
                     markersize=sp["posterior_marker_size"], markeredgewidth=0, label="Posterior (solid/filled)"),
              Line2D([], [], color="black", linewidth=sp["prior_linewidth"], dashes=tuple(sp["prior_dash"]),
                     marker="o", markersize=sp["prior_marker_size"], markerfacecolor="none",
                     markeredgewidth=sp["prior_marker_edgewidth"], label="Prior (dashed/hollow)")]
        fig.legend(handles=rh, loc="center", bbox_to_anchor=tuple(sp["role_legend_figure_xy"]), ncol=2,
                   frameon=False, fontsize=7, handlelength=2.4, columnspacing=0.9, handletextpad=0.5)
        return _save(fig, figures, "figure_1_spectral_domain")


# ---- Figure 2 ---------------------------------------------------------------------------------------
def _width_style(acf: dict, m: int, fixture: bool, idx: int) -> dict:
    s = acf["width_styles"].get(str(m))
    if s is not None:
        return s
    if not fixture:
        raise KeyError(f"no plot style for width {m}")
    return {"color": FIXTURE_PALETTE[idx % len(FIXTURE_PALETTE)], "marker": "osD^vP"[idx % 6]}


def figure2(tables: Path, figures: Path) -> list[Path]:
    S = style()
    c, acf = S["common"], S["acf"]
    plot = pd.read_csv(tables / "acf_plot.csv")
    st = json.loads((tables / "acf_settings.json").read_text())
    K, fixture = int(st["K"]), bool(st.get("fixture"))
    y0, y1 = acf["y_limits_default"]
    lo = float(plot.replicate_min.min())
    if lo < y0:
        y0 = math.floor(lo / 0.05) * 0.05 - 0.02
    all_widths = sorted({int(m) for m in plot.m.unique()})
    with plt.rc_context(MPL_STYLE):
        fig = plt.figure(figsize=tuple(acf["figure_inches"]))
        axes = [fig.add_axes(r) for r in acf["axes_rects"]]
        axes[1].sharex(axes[0])
        axes[1].sharey(axes[0])
        titles = [r"(a) Shallow, $L=2$", r"(b) Deep, $L=3$"]
        for i, (ax, arch) in enumerate(zip(axes, ["shallow", "deep"])):
            d = plot[plot.architecture == arch]
            widths = sorted(int(m) for m in d.m.unique())
            handles = []
            for m in widths:
                ws = _width_style(acf, m, fixture, all_widths.index(m))
                g = d[d.m == m].sort_values("lag")
                ax.fill_between(g.lag, g.replicate_min, g.replicate_max, color=ws["color"], alpha=acf["range_alpha"],
                                linewidth=0, zorder=2)
            for m in widths:
                ws = _width_style(acf, m, fixture, all_widths.index(m))
                g = d[d.m == m].sort_values("lag")
                ax.plot(g.lag, g["median"], "-", color=ws["color"], linewidth=acf["median_linewidth"], zorder=3)
                mk = [K * j // 8 for j in range(1, 9)]
                gm = g.set_index("lag").loc[mk]
                ax.plot(mk, gm["median"], linestyle="none", marker=ws["marker"], markersize=acf["marker_size"],
                        color=ws["color"], markeredgewidth=0, zorder=4)
                handles.append(Line2D([], [], color=ws["color"], linewidth=acf["median_linewidth"],
                                      marker=ws["marker"], markersize=acf["marker_size"], markeredgewidth=0,
                                      label=rf"$m={m}$"))
            ax.axhline(0.0, color=acf["zero_color"], linewidth=acf["zero_linewidth"], zorder=1.5)
            ax.set_xlim(0, K)
            ax.set_ylim(y0, y1)
            ax.xaxis.set_major_locator(FixedLocator([0, K // 4, K // 2, 3 * K // 4, K]))
            ax.xaxis.set_minor_locator(FixedLocator([K // 8, 3 * K // 8, 5 * K // 8, 7 * K // 8]))
            ax.yaxis.set_major_locator(FixedLocator(acf["y_ticks"]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(axis="y", which="major", color=c["grid_color"], linewidth=c["grid_linewidth"])
            ax.grid(axis="x", visible=False)
            ax.grid(which="minor", visible=False)
            ax.set_axisbelow(True)
            ax.set_xlabel("Lag (sampler iterations)")
            if i == 0:
                ax.set_ylabel(r"Autocorrelation of training loss $V$")
            else:
                ax.tick_params(labelleft=True)
            tx, ty = acf["title_figure_xy"][i]
            fig.text(tx, ty, titles[i], ha="center", va="top", fontsize=9)
            lx, ly = acf["legend_figure_xy"][i]
            leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(lx, ly), ncol=2, frameon=False,
                             fontsize=7, handlelength=2.2, columnspacing=1.4)
            fig.add_artist(leg)
        nx, ny = acf["shared_note_figure_xy"]
        h = st["h"]
        fig.text(nx, ny, rf"Same adjusted sampler; $h = {h:g}$; $n = {st['n']}$", ha="center", va="top", fontsize=8)
        if fixture:
            fig.text(0.005, 0.5, "FIXTURE - NOT A RESULT", rotation=90, ha="left", va="center", fontsize=6,
                     color="#C00000")
        return _save(fig, figures, "figure_2_loss_acf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate both paper figures from the computed tables only")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--only", choices=["1", "2"], default=None)
    a = ap.parse_args(argv)
    t, f = Path(a.tables), Path(a.figures)
    if a.only in (None, "1"):
        print("\n".join(map(str, figure1(t, f))))
    if a.only in (None, "2"):
        print("\n".join(map(str, figure2(t, f))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
