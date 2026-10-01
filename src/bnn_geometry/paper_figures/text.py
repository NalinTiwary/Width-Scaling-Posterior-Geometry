"""Captions, results text and recovery log generated from the computed tables (guide §4.4, §7.3, §11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .common import Layout

REGEN = ("python -m bnn_geometry.paper_figures.render --tables {t} --figures {f}")


def _n(x: int) -> str:
    return f"{int(x):,}"


def _tex_n(x: int) -> str:
    return f"{int(x):,}".replace(",", "{,}")


# ---- spectral --------------------------------------------------------------------------------------
def spectral_facts(L: Layout) -> dict[str, Any]:
    plot = pd.read_csv(L.tables / "spectral_plot.csv")
    counts = json.loads((L.tables / "spectral_counts.json").read_text())
    q = lambda lab: plot[np.isclose(plot["quantile"].astype(float), lab)].sort_values("m")  # noqa: E731
    q99 = q(0.99)
    diff = (plot.posterior_median - plot.prior).abs()
    return {**counts, "q99_min": float(q99.posterior_median.min()), "q99_max": float(q99.posterior_median.max()),
            "q99_rep_max": float(q99.replicate_max.max()), "q50_min": float(q(0.5).posterior_median.min()),
            "q50_max": float(q(0.5).posterior_median.max()), "max_prior_diff": float(diff.max()),
            "q99_by_m": dict(zip(q99.m.astype(int), q99.posterior_median.round(4)))}


def spectral_sentence(f: dict[str, Any]) -> str:
    w = f["widths"]
    if f["n_outside"] == 0 and f["n_ambiguous"] == 0:
        s = (f"Across widths m = {w[0]}-{w[-1]} (n = {f['n']}, two hidden layers), all {_n(f['n_inspected'])} "
             f"inspected unrestricted posterior states ({_n(f['per_target_inspected'][0])} per target, "
             f"{f['n_targets']} targets) lie inside the spectral restriction G_2.5 (0 exits).")
    else:
        inside = f["n_inspected"] - f["n_outside"] - f["n_ambiguous"]
        s = (f"Across widths m = {w[0]}-{w[-1]}, {_n(inside)} of {_n(f['n_inspected'])} inspected unrestricted "
             f"posterior states lie inside G_2.5 ({_n(f['n_outside'])} exits, {_n(f['n_ambiguous'])} numerically "
             f"unresolved).")
    s += (f" The replicate-median 99th percentile of S = ||W2||op/(2.5 sqrt(m)) ranges from {f['q99_min']:.3f} to "
          f"{f['q99_max']:.3f} across widths (largest single-replicate value {f['q99_rep_max']:.3f}), "
          f"{'below' if f['q99_rep_max'] < 1 else 'reaching'} the boundary S = 1; medians lie between "
          f"{f['q50_min']:.3f} and {f['q50_max']:.3f}. The posterior median, 95th and 99th percentiles differ from "
          f"the iid Gaussian-prior quantiles by at most {f['max_prior_diff']:.4f} in S, so these norm statistics are "
          f"close to their prior values.")
    return s


def caption1(f: dict[str, Any]) -> str:
    base = (r"Posterior coverage of the deep spectral domain. For two-hidden-layer networks with $n="
            f"{f['n']}" r"$, we plot the median, 95th, and 99th percentiles of "
            r"$S=\|W_2\|_{\mathrm{op}}/(2.5\sqrt{m})$ from unrestricted posterior samples. The dashed horizontal "
            r"line $S=1$ is the boundary of the domain used by the local deep LSI result. Solid curves show the "
            r"median of the three replicate quantile estimates, with whiskers spanning the replicate range; dashed "
            r"curves and hollow markers show the iid Gaussian-prior reference. ")
    if f["n_outside"] == 0 and f["n_ambiguous"] == 0:
        base += (f"All ${_tex_n(f['n_inspected'])}$ inspected posterior states satisfied the restriction. "
                 "The domain therefore contains all observed samples across the tested widths.")
    else:
        inside = f["n_inspected"] - f["n_outside"] - f["n_ambiguous"]
        base += (f"${_tex_n(inside)}$ of ${_tex_n(f['n_inspected'])}$ inspected posterior states satisfied the "
                 f"restriction (${_tex_n(f['n_outside'])}$ exits, ${_tex_n(f['n_ambiguous'])}$ numerically "
                 "unresolved). The domain contains this fraction of the observed samples across the tested widths.")
    return base


# ---- ACF --------------------------------------------------------------------------------------------
def acf_facts(L: Layout) -> Optional[dict[str, Any]]:
    if not (L.tables / "acf_plot.csv").exists():
        return None
    plot = pd.read_csv(L.tables / "acf_plot.csv")
    con = pd.read_csv(L.tables / "acf_width_contrasts.csv")
    st = json.loads((L.tables / "acf_settings.json").read_text())
    facts = {"settings": st, "arch": {}}
    for arch in ("deep", "shallow"):
        p = plot[plot.architecture == arch]
        ws = sorted(int(m) for m in p.m.unique())
        lags = sorted(int(k) for k in con[con.architecture == arch].lag.unique())
        med = {k: {m: float(p[(p.m == m) & (p.lag == k)]["median"].iloc[0]) for m in ws} for k in lags}
        c = con[con.architecture == arch]
        neg = {k: int((c[c.lag == k].delta < 0).sum()) for k in lags}
        dmed = {k: med[k][ws[-1]] - med[k][ws[0]] for k in lags}
        nrep = int(c.replicate.nunique())
        mono = {k: all(np.diff([med[k][m] for m in ws]) < 0) for k in lags}
        facts["arch"][arch] = {"widths": ws, "lags": lags, "median": med, "n_negative": neg, "delta_median": dmed,
                               "n_rep": nrep, "monotone_decreasing": mono}
    return facts


def _lags(ks: list[int]) -> str:
    ks = [str(k) for k in ks]
    return ks[0] if len(ks) == 1 else ", ".join(ks[:-1]) + " and " + ks[-1]


def _describe(arch: str, a: dict[str, Any]) -> str:
    ws, lags, med, neg, n, dmed = (a["widths"], a["lags"], a["median"], a["n_negative"], a["n_rep"],
                                   a["delta_median"])
    lo, hi = ws[0], ws[-1]
    name = "deep" if arch == "deep" else "shallow"
    agree = [k for k in lags if neg[k] == n]
    against = [k for k in lags if neg[k] == 0]
    mixed = [k for k in lags if 0 < neg[k] < n]
    parts = []
    if agree:
        diffs = ", ".join(f"{dmed[k]:+.2f} at lag {k}" for k in agree)
        parts.append(f"the widest network (m={hi}) has lower loss autocorrelation than the narrowest (m={lo}) in all "
                     f"{n} replicates at lags {_lags(agree)} (pointwise-median differences {diffs})")
    if against:
        parts.append(f"the widest network has higher autocorrelation in all {n} replicates at lags {_lags(against)}")
    if mixed:
        near0 = max(abs(med[k][m]) for k in mixed for m in ws)
        parts.append(f"at lag{'s' if len(mixed) > 1 else ''} {_lags(mixed)} the sign differs across replicates "
                     f"({', '.join(f'{neg[k]}/{n}' for k in mixed)} lower at m={hi}), where all medians are within "
                     f"{near0:.3f} of zero")
    s = f"In the {name} model, " + "; ".join(parts) + "."
    sh = _shape(a)
    if sh and sh[0] == "plateau":
        s += (f" Most of the reduction occurs between m={ws[0]} and m={ws[1]}; the curves for m={ws[1]} to m={hi} "
              f"nearly coincide (within {sh[1]:.3f} at lags {_lags(sh[2])}).")
    elif sh and sh[0] == "monotone":
        s += f" The pointwise-median autocorrelation decreases monotonically with width at lags {_lags(sh[1])}."
    return s


def _shape(a: dict[str, Any]):
    """Pattern across intermediate widths over the lags where all replicates agree on the narrow-wide ordering."""
    ws, med, neg, n = a["widths"], a["median"], a["n_negative"], a["n_rep"]
    agree = [k for k in a["lags"] if neg[k] == n]
    mono = [k for k in agree if a["monotone_decreasing"].get(k)]
    shape_lags = [k for k in agree if k <= 100] or agree
    if not shape_lags or len(ws) < 3:
        return None
    first = np.mean([med[k][ws[0]] - med[k][ws[1]] for k in shape_lags])
    total = np.mean([med[k][ws[0]] - med[k][ws[-1]] for k in shape_lags])
    spread = max(max(med[k][m] for m in ws[1:]) - min(med[k][m] for m in ws[1:]) for k in shape_lags)
    if total > 0 and first >= 0.75 * total and not set(shape_lags) <= set(mono):
        return ("plateau", spread, shape_lags)
    if mono:
        return ("monotone", mono)
    return None


def _short(arch: str, a: dict[str, Any]) -> str:
    """One caption clause per architecture; the numbers live in figure_results.md."""
    ws, neg, n = a["widths"], a["n_negative"], a["n_rep"]
    agree = [k for k in a["lags"] if neg[k] == n]
    mixed = [k for k in a["lags"] if 0 < neg[k] < n]
    if not agree:
        return f"in the {arch} model the width ordering is not consistent across replicates"
    s = (f"in the {arch} model, autocorrelation is lower at m={ws[-1]} than at m={ws[0]} in all {n} replicates "
         f"through lag {max(agree)}")
    sh = _shape(a)
    if sh and sh[0] == "plateau":
        s += f", with most of the change between m={ws[0]} and m={ws[1]} and nearly coincident curves beyond"
    elif sh and sh[0] == "monotone":
        s += f" and decreases monotonically with width over that range"
    if mixed:
        s += f", and all curves are near zero by lag {min(mixed)}"
    return s


def acf_caption_effect(fa: dict[str, Any]) -> str:
    c = "; ".join(_short(a, fa["arch"][a]) for a in ("deep", "shallow")) + "."
    return c[0].upper() + c[1:]


def acf_sentence(fa: dict[str, Any]) -> str:
    return " ".join(_describe(a, fa["arch"][a]) for a in ("deep", "shallow"))


def _texify_m(s: str) -> str:
    import re
    return re.sub(r"m=(\d+)", r"$m=\1$", s)


# ---- writers ----------------------------------------------------------------------------------------
def write_all(L: Layout, *, export_log: Optional[dict] = None) -> dict[str, Path]:
    fs = spectral_facts(L)
    fa = acf_facts(L)
    out = {}
    cap = ["% Generated by bnn_geometry.paper_figures.text from the computed tables; do not edit numbers by hand.",
           "% Figure 1: figures/figure_1_spectral_domain.pdf", r"\newcommand{\FigOneCaption}{" + caption1(fs) + "}"]
    if fa is not None:
        c2 = (r"Width and loss autocorrelation under a fixed sampling algorithm. We compare the autocorrelation of "
              r"the summed training cross-entropy $V$ for shallow and deep networks at $n=" f"{fa['settings']['n']}"
              r"$, using the same Metropolis-adjusted sampler and step size $h=" f"{fa['settings']['h']:g}"
              r"$ at every width. Each chain contributes its final $" + _tex_n(fa["settings"]["window"]) +
              r"$ retained transitions, including rejected moves. Curves show pointwise medians across three "
              r"data/center replicates after averaging four chains within each replicate; shaded regions show the "
              r"replicate range. " + _texify_m(acf_caption_effect(fa)) + r" Lag is measured in sampler iterations, so "
              r"the figure compares the implemented discrete kernels.")
        cap += ["% Figure 2: figures/figure_2_loss_acf.pdf", r"\newcommand{\FigTwoCaption}{" + c2 + "}"]
    else:
        cap += ["% Figure 2: pending (loss-trace NPZs not yet exported)"]
    out["captions"] = L.out / "captions.tex"
    out["captions"].write_text("\n".join(cap) + "\n")

    md = ["# Figure results (two-figure package)", "",
          "Generated from the computed tables in `tables/`. Scope: empirical spectral-domain coverage and fixed-step "
          "loss autocorrelation. No PI/LSI constant, integrated autocorrelation time, continuous-time rate or fitted "
          "width exponent is estimated.", "", "## Figure 1: posterior coverage of the deep spectral domain", "",
          spectral_sentence(fs), "", "| m | q50 | q95 | q99 | replicate range q99 | prior q99 |", "|---|---|---|---|---|---|"]
    plot = pd.read_csv(L.tables / "spectral_plot.csv")
    for m, d in plot.groupby("m"):
        g = {round(float(r["quantile"]), 2): r for _, r in d.iterrows()}
        md.append(f"| {m} | {g[0.5].posterior_median:.4f} | {g[0.95].posterior_median:.4f} | "
                  f"{g[0.99].posterior_median:.4f} | {g[0.99].replicate_min:.4f}-{g[0.99].replicate_max:.4f} | "
                  f"{g[0.99].prior:.4f} |")
    md += ["", "Per-target counts and quantiles: `tables/spectral_replicate_summary.csv`; statewise values: "
           "`tables/spectral_states.csv.gz`; cross-check against the campaign summary: `tables/spectral_crosscheck.csv`.", ""]
    md += ["## Figure 2: width and loss autocorrelation", ""]
    if fa is None:
        md += ["Pending: the h=0.01 loss-trace NPZs have not been exported yet (`export-loss-traces`).", ""]
    else:
        st = fa["settings"]
        md += [f"Status `fixed_step_loss_acf_status = {st['fixed_step_loss_acf_status']}`, "
               f"`analysis_scope = {st['analysis_scope']}`; displayed lag range K = {st['K']} "
               f"(extension history: {[h['K'] for h in st['K_history']]}); window {st['window']:,} transitions per "
               f"chain at h = {st['h']:g}.", "", acf_sentence(fa), "",
               "_The paragraph above and the caption clause are generated from `tables/acf_width_contrasts.csv` and "
               "`tables/acf_plot.csv` (agreement across replicates per lag, monotonicity, plateau share); re-inspect "
               "`figures/figure_2_loss_acf.png` whenever the tables change._", ""]
        for arch in ("deep", "shallow"):
            a = fa["arch"][arch]
            md += [f"### {arch}: pointwise-median ACF at fixed lags", "",
                   "| lag | " + " | ".join(f"m={m}" for m in a["widths"]) + " | widest - narrowest | replicates negative |",
                   "|---|" + "---|" * (len(a["widths"]) + 2)]
            for k in a["lags"]:
                md.append(f"| {k} | " + " | ".join(f"{a['median'][k][m]:.3f}" for m in a["widths"]) +
                          f" | {a['delta_median'][k]:+.3f} | {a['n_negative'][k]}/{a['n_rep']} |")
            md.append("")
        dg = pd.read_csv(L.tables / "loss_window_diagnostics.csv")
        md += ["### Selected-window sampling checks (V)", "",
               f"R-hat max {dg.V_rhat.max():.4f} (threshold < 1.01); bulk ESS min {dg.V_ess_bulk.min():.0f} "
               f"(threshold >= 1000); window acceptance {dg.acceptance_window_mean.min():.3f}-"
               f"{dg.acceptance_window_mean.max():.3f}; drift flags {int(dg.drift_flag.sum())}; failed targets: "
               f"{', '.join(dg[dg.check_status != 'pass'].target_id) or 'none'}.", ""]
    md += ["## Regeneration", "", "```bash", REGEN.format(t="results/final_geometry/paper_figures/tables",
                                                           f="results/final_geometry/paper_figures/figures"), "```", ""]
    out["results"] = L.out / "figure_results.md"
    out["results"].write_text("\n".join(md))

    rl = ["# Recovery log", "", "Inputs were recovered from existing campaign outputs before computing anything new "
          "(guide §10.1). No posterior sampling was rerun for this package.", "",
          "- Spectral states: existing statewise table `tables/spectral_states.csv` (re-exported as `paper_figures/tables/spectral_states.csv.gz`) of the unrestricted reference "
          "archive (every archived state, stride 16, 4 chains x 4,096 per deep target), cross-checked against the "
          "campaign's pooled per-target summary (quantiles identical, counts identical) and its float64 SVD backend "
          "check.", "- Prior reference: existing 4,096 iid Gaussian W2 matrices per width "
          "(`controls/prior_spectral_values.npz`, seeds in `tables/spectral_prior_summary.csv`)."]
    if export_log:
        n_ok, n_fail = len(export_log.get("exported", [])), len(export_log.get("failures", []))
        rl.append(f"- Loss traces: {n_ok} targets exported from the existing h={export_log['h']:g} dynamics HDF5 "
                  f"traces (full per-transition V incl. rejected repeats; priority 1 of §10.3), {n_fail} failures. "
                  "Each export verified target/execution hashes, kernel and step, contiguous unit-stride transitions, "
                  "unchanged V at rejected transitions, agreement of the mean V with the campaign analysis and the "
                  "final-checkpoint state's recomputed V.")
        for e in export_log.get("exported", []):
            rl.append(f"  - {e['target_id']}: {e['trajectory']} segments {','.join(e['segments'])}; retained "
                      f"{e['retained_per_chain']:,}/chain; window transitions {e['iteration_start'][0]:,}-"
                      f"{e['iteration_end'][0]:,}; status {e['source_status']}")
        for f in export_log.get("failures", []):
            rl.append(f"  - FAILED {f['target_id']}: {f['error']}")
    else:
        rl.append("- Loss traces: pending cluster export.")
    if export_log and not export_log.get("failures"):
        rl += ["", "No targeted new run (§10.4) was needed or made."]
    elif export_log:
        rl += ["", "Targets marked FAILED need the §10.3/§10.4 recovery procedure before Figure 2 is final."]
    out["recovery"] = L.out / "recovery_log.md"
    out["recovery"].write_text("\n".join(rl) + "\n")
    return out
