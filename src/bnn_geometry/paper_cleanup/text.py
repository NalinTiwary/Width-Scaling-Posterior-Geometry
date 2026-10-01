"""captions.tex and figure_results.md from the cleanup tables (template wording from docs/paper_cleanup)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .common import FIGURE_WIDTH_IN, S1_LAGS, S2_TIMES, Layout

COVERAGE_CUTOFF = 2.1

SPECTRAL = r"""\textbf{Width dependence of the posterior spectral statistic and domain coverage.}
Here $T=\|W_2\|_{\mathrm{op}}/\sqrt m$, the hidden-layer prior center is zero,
and $\sigma=1$, so the theorem permits any fixed cutoff
$a>a_*=r_0+2\sigma=2$. (a) Median, 95th, and 99th posterior percentiles of $T$
versus width. Points summarize three dataset/prior-center replicates by their
median, with whiskers showing their range. (b) Empirical coverage
$\widehat\pi(G_a)$ versus cutoff $a$, with one curve per width; curves and
shading show the pointwise median and range across replicates. Both panels
use unrestricted posterior samples and mark the threshold $a_*=2$.
For every fixed $a>2$, the theorem predicts coverage tending to one as width
increases; it makes no such assertion at $a=2$. Exact sample counts above 2
are reported in the accompanying table. Here $n=128$ and each target uses
four chains."""

LOSS = r"""\textbf{Width improves loss decorrelation, with diminishing returns in shallow networks.}
Autocorrelation of the summed training cross-entropy $V$ is shown over the
first 200 sampler lags for shallow ($L=2$) and deep ($L=3$) networks.
All runs use the same Metropolis-adjusted Gaussian-preserving Langevin
kernel with step size $h=0.01$ and $n=128$. Each of four chains contributes
its last 102,400 retained transitions, including repeats after rejection.
Curves are the pointwise median across three dataset/prior-center replicates
after averaging the four chain autocorrelations within each replicate;
shading shows the replicate range. The deep curves show a pronounced width
benefit, while the shallow improvement largely levels off after $m=256$.
The comparison concerns decorrelation per sampler iteration."""

PRED = r"""\textbf{Prediction autocorrelation across widths.}
We compute autocorrelation separately for class-1 probabilities at eight
fixed held-out inputs, average over four chains and the eight inputs within
each replicate, and show the median and range across three
dataset/prior-center replicates. Each chain contributes the same final
{window} retained transitions used in the loss analysis, with $h=0.01$.
Colors identify width. This tests whether the width pattern observed for
training loss also appears in predictive observables."""

STEP = r"""\textbf{Step-size sensitivity of the deep width comparison.}
Training-loss autocorrelation is compared at widths {lo} and {hi} for two
step sizes of the same adjusted Langevin kernel. The horizontal axis is
the discrete algorithmic lag $kh$. Each chain contributes the final {dur}
units of retained algorithmic duration; curves show the median and range
of four-chain averages across three dataset/prior-center replicates.
The comparison tests whether the endpoint width ordering persists when
the step size is halved."""


def _rng(x) -> str:
    x = np.asarray(x, float)
    return f"{np.median(x):+.3f} [{x.min():+.3f}, {x.max():+.3f}]"


def spectral_facts(L: Layout) -> dict[str, Any]:
    q = pd.read_csv(L.figures / "figure_1_spectral_domain_quantiles.csv")
    tc = pd.read_csv(L.figures / "figure_1_spectral_domain_threshold_counts.csv")
    s = pd.read_csv(L.exports / "spectral_states.csv.gz")
    widths = sorted(int(m) for m in q.width.unique())
    q99 = {m: float(q[(q.width == m) & np.isclose(q["quantile"], 0.99)]["T"].median()) for m in widths}
    cov = {}
    for m in widths:
        f = [float((g["T"] <= COVERAGE_CUTOFF).mean()) for _, g in s[s.width == m].groupby("replicate")]
        cov[m] = {"median": float(np.median(f)), "min": float(min(f)), "max": float(max(f))}
    pct = {m: (100 * tc[tc.width == m].fraction_above).tolist() for m in widths}
    return {"widths": widths, "q99_median": q99, "coverage_cutoff": COVERAGE_CUTOFF, "coverage": cov,
            "pct_above_2": pct, "n_per_target": int(tc.n.iloc[0]), "max_T": {m: float(tc[tc.width == m].maximum.max())
                                                                             for m in widths},
            "n_near_threshold": int(tc.n_near_threshold.sum())}


def spectral_sentence(f: dict[str, Any]) -> str:
    lo, hi = f["widths"][0], f["widths"][-1]
    allp = [p for v in f["pct_above_2"].values() for p in v]
    c = f["coverage"]
    n = f"{f['n_per_target']:,}".replace(",", "{,}")
    q_dir = "decreases" if f["q99_median"][hi] < f["q99_median"][lo] else "does not decrease"
    tail = ", illustrating a narrowing upper tail above the theorem's admissibility threshold" \
        if q_dir == "decreases" else ""
    c_dir = "rises" if c[hi]["median"] > c[lo]["median"] else "does not rise"
    return (f"The 99th-percentile curve {q_dir} from {f['q99_median'][lo]:.3f} at width {lo} to "
            f"{f['q99_median'][hi]:.3f} at width {hi}{tail}; at $a={f['coverage_cutoff']:g}$ the median coverage "
            f"{c_dir} from "
            f"{100 * c[lo]['median']:.1f}\\% to {100 * c[hi]['median']:.1f}\\%. "
            f"Between {min(allp):.1f}\\% and {max(allp):.1f}\\% of the {n} inspected states per "
            f"target lie above 2 at every width, which the theorem does not exclude.")


def s1_facts(L: Layout) -> Optional[dict[str, Any]]:
    p = L.tables / "prediction_width_differences.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    st = json.loads((L.tables / "prediction_acf_settings.json").read_text())
    out = {"K": st["K"], "all_series_valid": st["all_series_valid"], "n_invalid": st["n_invalid"], "by_arch": {}}
    for arch, g in d.groupby("architecture"):
        m = g[g.test_index == "mean8"]
        out["by_arch"][arch] = {
            "width_narrow": int(m.width_narrow.iloc[0]), "width_wide": int(m.width_wide.iloc[0]),
            "lags": {int(k): {"D_mean8": m[m.lag == k].difference.tolist(),
                              "n_reps_positive": int((m[m.lag == k].difference > 0).sum()),
                              "points_positive": int(m[m.lag == k].n_points_positive.sum()),
                              "points_total": int(m[m.lag == k].n_points.sum())} for k in sorted(m.lag.unique())}}
    return out


def s1_sentence(f: dict[str, Any]) -> str:
    parts = []
    for arch in ("deep", "shallow"):
        a = f["by_arch"][arch]
        k = 100 if 100 in a["lags"] else max(a["lags"])
        x = a["lags"][k]
        parts.append(f"{arch} {_rng(x['D_mean8'])} ({x['points_positive']}/{x['points_total']} point-level "
                     f"differences positive)")
    return (f"At lag {k}, the narrowest-minus-widest difference of the mean curve (positive: lower autocorrelation "
            f"at the widest width), as median [range] over replicates, is " + "; ".join(parts) + ".")


def s2_facts(L: Layout) -> Optional[dict[str, Any]]:
    p = L.tables / "step_width_differences.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    st = json.loads((L.tables / "step_acf_settings.json").read_text())
    return {"widths": st["widths"], "steps": st["steps"], "duration": st["duration"], "t_max": st["t_max"],
            "all_series_valid": st["all_series_valid"], "n_invalid": st["n_invalid"],
            "D": {f"{h:g}": {f"{t:g}": d[np.isclose(d.h, h) & np.isclose(d.algorithmic_lag, t)].difference.tolist()
                             for t in S2_TIMES if t <= st["t_max"]} for h in st["steps"]}}


def s2_sentence(f: dict[str, Any]) -> str:
    lo, hi = f["widths"]
    t = "0.5" if "0.5" in next(iter(f["D"].values())) else next(iter(next(iter(f["D"].values()))))
    seg = [f"{_rng(f['D'][h][t])} at $h={h}$ ({sum(v > 0 for v in f['D'][h][t])}/3 replicates positive)"
           for h in f["D"]]
    return (f"At $kh={t}$ the width-{lo}-minus-width-{hi} autocorrelation difference, as median [range] over "
            f"replicates, is " + " and ".join(seg) + ".")


def write_text(L: Layout) -> dict[str, Any]:
    sf = spectral_facts(L)
    s1, s2 = s1_facts(L), s2_facts(L)
    s1st = json.loads((L.tables / "prediction_acf_settings.json").read_text()) if s1 else None
    s2st = json.loads((L.tables / "step_acf_settings.json").read_text()) if s2 else None
    tex = ["% Generated by `bnn_geometry paper-cleanup --steps text`; template wording from docs/paper_cleanup.",
           "% Main captions: template text plus the computed result sentence.",
           "\\newcommand{\\SpectralFigureCaption}{%", SPECTRAL, spectral_sentence(sf) + "}", "",
           "\\newcommand{\\LossAcfFigureCaption}{%", LOSS + "}", ""]
    tex.append("% Supplementary captions: neutral template plus a factual outcome sentence from the tables.")
    tex.append("% Review the outcome sentence against the figure before submission.")
    window = f"{int(s1st['window']):,}".replace(",", "{,}") if s1st else "102{,}400"
    tex += ["\\newcommand{\\PredictionAcfFigureCaption}{%", PRED.replace("{window}", window)
            + ("\n" + s1_sentence(s1) if s1 else "\n% outcome sentence pending: run the S1 supplement") + "}", ""]
    lo, hi = (s2["widths"] if s2 else (32, 256))
    dur = f"{s2['duration']:g}" if s2 else "512"
    tex += ["\\newcommand{\\StepSizeFigureCaption}{%",
            STEP.replace("{lo}", str(lo)).replace("{hi}", str(hi)).replace("{dur}", dur)
            + ("\n" + s2_sentence(s2) if s2 else "\n% outcome sentence pending: run the S2 supplement") + "}", ""]
    (L.out / "captions.tex").write_text("\n".join(tex))
    write_json_safe = {"spectral": sf, "s1": s1, "s2": s2}
    (L.tables / "text_facts.json").write_text(json.dumps(write_json_safe, indent=2, default=str) + "\n")
    _results_md(L, sf, s1, s2, s1st, s2st)
    return write_json_safe


def _results_md(L: Layout, sf, s1, s2, s1st, s2st) -> None:
    pr = pd.read_csv(L.tables / "prior_reference_T.csv")
    tc = pd.read_csv(L.tables / "threshold_counts_paper.csv")
    w = [f"# Paper figures after cleanup (figure width {FIGURE_WIDTH_IN} in)", "",
         "## Figure 1: spectral statistic $T=\\|W_2\\|_{op}/\\sqrt m$ and coverage", "",
         spectral_sentence(sf).replace("\\%", "%").replace("{,}", ","), "",
         "| width | above 2 (r0/r1/r2 of %d) | %% above 2 median [range] | max T | posterior q99 | prior q99 | "
         "coverage at a=%g median [range] |" % (sf["n_per_target"], sf["coverage_cutoff"]),
         "|---|---|---|---|---|---|---|"]
    for r in tc.itertuples():
        p = pr[pr.width == r.width].iloc[0]
        c = sf["coverage"][r.width]
        w.append(f"| {r.width} | {r.n_above_r0}/{r.n_above_r1}/{r.n_above_r2} | {r.pct_above_median:.2f} "
                 f"[{r.pct_above_min:.2f}, {r.pct_above_max:.2f}] | {r.max_T:.3f} | {p.posterior_q99:.3f} | "
                 f"{p.prior_q99:.3f} | {100 * c['median']:.1f}% [{100 * c['min']:.1f}, {100 * c['max']:.1f}] |")
    w += ["", f"States within the numerical guard of 2: {sf['n_near_threshold']}.", "",
          "## Figure 2: loss autocorrelation", "",
          "Unchanged accepted curves (lags 0-400 exported, 0-200 displayed); see "
          "`paper_figures/figure_results.md` for the per-lag numbers.", ""]
    w += ["## Supplement S1: prediction autocorrelation", ""]
    if s1:
        w += [s1_sentence(s1), "", f"Lag horizon K = {s1['K']} (history in `tables/prediction_acf_settings.json`). "
              f"Diagnostics: {s1st['n_series'] - s1st['n_invalid']}/{s1st['n_series']} series pass "
              f"R-hat<1.01, bulk ESS>=1000, tail ESS>=400; drift flags: {s1st['n_drift_flags']}.", "",
              "| arch | lag | D (mean of 8 points) median [range] | replicates positive | points positive |",
              "|---|---|---|---|---|"]
        for arch in ("shallow", "deep"):
            for k, x in s1["by_arch"][arch]["lags"].items():
                w.append(f"| {arch} | {k} | {_rng(x['D_mean8'])} | {x['n_reps_positive']}/3 | "
                         f"{x['points_positive']}/{x['points_total']} |")
    else:
        w.append("Not yet computed (requires the cluster export).")
    w += ["", "## Supplement S2: step-size sensitivity (deep endpoints)", ""]
    if s2:
        w += [s2_sentence(s2), "", f"Matched duration {s2['duration']:g}, display range kh in [0, {s2['t_max']:g}]. "
              f"Diagnostics: {s2st['n_invalid']} of the series fail a protocol gate; drift flags: "
              f"{s2st['n_drift_flags']}.", "", "| h | kh | D median [range] | replicates positive |", "|---|---|---|---|"]
        for h, byt in s2["D"].items():
            for t, v in byt.items():
                w.append(f"| {h} | {t} | {_rng(v)} | {sum(x > 0 for x in v)}/3 |")
    else:
        w.append("Not yet computed (requires the cluster export).")
    (L.out / "figure_results.md").write_text("\n".join(w) + "\n")
