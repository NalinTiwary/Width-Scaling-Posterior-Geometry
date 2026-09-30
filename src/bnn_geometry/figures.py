"""Main and appendix figures, sidecars and captions, generated exclusively from saved tables (runbook §14, §16.2)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .storage import atomic_write_json, clean_json, read_json  # noqa: E402

OKABE = {"loss": "#000000", "train_logit": "#0072B2", "test_probability": "#E69F00", "interaction": "#009E73"}
MARK = {"loss": "o", "train_logit": "s", "test_probability": "^", "interaction": "D"}
LABEL = {"loss": "V (loss)", "train_logit": "training logits", "test_probability": "held-out probabilities",
         "interaction": "interactions"}
FAIL_TEXT = "Numerical validity criteria not met within the prescribed budget"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "pdf.fonttype": 42})


class CaptionError(RuntimeError):
    pass


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "missing"


def load(root: Path, name: str) -> pd.DataFrame:
    p = root / "tables" / f"{name}.csv"
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).lower() in ("true", "1", "1.0"))


def check_caption(text: str) -> str:
    if re.search(r"\[[A-Za-z][A-Za-z ,/\-]*\]", text):
        raise CaptionError(f"placeholder left in caption: {text}")
    return text


def save(fig, root: Path, name: str, plotted: pd.DataFrame, meta: dict[str, Any], dpi: int) -> None:
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", dpi=dpi, metadata={"Software": None})
    plt.close(fig)
    plotted.to_csv(out / f"{name}.csv", index=False)
    atomic_write_json(out / f"{name}.json", clean_json({**meta, "plotted_csv": f"{name}.csv",
                                                        "plotting_code_revision": C.code_revision()}))


def _xoff(m: np.ndarray, rep: int, reps: list[int]) -> np.ndarray:
    k = reps.index(rep) - (len(reps) - 1) / 2.0
    return m * (1.0 + 0.045 * k)


def _panel(ax, df: pd.DataFrame, fams: list[str], widths: list[int], reps: list[int], ycol: str, lo: str, hi: str):
    """Replicate points (hollow if invalid), thin replicate lines between valid neighbours, thick medians."""
    any_valid = False
    for fam in fams:
        d = df[df["family"] == fam]
        for rep in reps:
            dr = d[d["replicate"] == rep].set_index("m").reindex(widths)
            x = _xoff(np.array(widths, dtype=float), rep, reps)
            y = dr[ycol].to_numpy(dtype=float)
            v = np.array([isinstance(z, (bool, np.bool_)) and bool(z) for z in dr["validity_b"]], dtype=bool)
            yl = dr[lo].to_numpy(dtype=float) if lo in dr else np.full(len(widths), np.nan)
            yh = dr[hi].to_numpy(dtype=float) if hi in dr else np.full(len(widths), np.nan)
            for i in range(len(widths)):
                if not np.isfinite(y[i]):
                    continue
                err = None
                if np.isfinite(yl[i]) and np.isfinite(yh[i]):
                    err = [[max(y[i] - yl[i], 0)], [max(yh[i] - y[i], 0)]]
                ax.errorbar([x[i]], [y[i]], yerr=err, fmt=MARK[fam], ms=3.5, lw=0.6, capsize=1.5, alpha=0.55,
                            color=OKABE[fam], mfc=OKABE[fam] if v[i] else "white", mec=OKABE[fam])
                any_valid |= bool(v[i])
            for i in range(len(widths) - 1):
                if v[i] and v[i + 1] and np.isfinite(y[i]) and np.isfinite(y[i + 1]):
                    ax.plot(x[i:i + 2], y[i:i + 2], color=OKABE[fam], lw=0.6, alpha=0.35)
        med = []
        for m in widths:
            dm = d[d["m"] == m]
            ok = dm[dm["validity_b"]]
            med.append(float(np.median(ok[ycol])) if len(ok) == len(reps) and len(reps) else np.nan)
        med = np.array(med)
        for i in range(len(widths) - 1):
            if np.isfinite(med[i]) and np.isfinite(med[i + 1]):
                ax.plot(widths[i:i + 2], med[i:i + 2], color=OKABE[fam], lw=2.2)
        ax.plot([], [], color=OKABE[fam], marker=MARK[fam], lw=2.0, label=LABEL[fam])
    ax.set_xscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.minorticks_off()
    return any_valid


def _yscale(axes, values: np.ndarray) -> str:
    v = values[np.isfinite(values) & (values > 0)]
    if v.size and v.max() / v.min() > 10:
        for ax in axes:
            ax.set_yscale("log")
        return "log"
    top = (np.nanmax(values) if np.isfinite(values).any() else 1.0) * 1.15
    for ax in axes:
        ax.set_ylim(0, max(top, 1.2))
    return "linear"


def _trend(ratios: pd.DataFrame, exp: str, arch: str) -> str:
    r = ratios[(ratios["experiment"] == exp) & (ratios["architecture"] == arch)] if len(ratios) else ratios
    if not len(r):
        return f"no widest/narrowest ratios are available for the {arch} panel"
    parts = []
    for fam, g in r.groupby("family"):
        g = g.copy()
        val = _bool(g["validity"])
        if not val.all():
            parts.append(f"{LABEL.get(fam, fam)}: unresolved ({int((~val).sum())} of {len(g)} replicate ratios invalid)")
            continue
        dec = ((g["mc_high"] < 1).sum()) if "mc_high" in g else 0
        inc = ((g["mc_low"] > 1).sum()) if "mc_low" in g else 0
        rng = f"{g['estimate'].min():.2f}-{g['estimate'].max():.2f}"
        if dec == len(g):
            word = "decrease supported in every replicate"
        elif inc == len(g):
            word = "increase supported in every replicate"
        else:
            word = "no consistent direction across replicates"
        parts.append(f"{LABEL.get(fam, fam)}: widest/narrowest ratio {rng}, {word}")
    return f"{arch}: " + "; ".join(parts)


# ---- main figure 1 ----------------------------------------------------------------------------------------
def figure_relaxation(cfg, root: Path, dpi: int) -> dict[str, Any]:
    df = load(root, "relaxation_families")
    ratios = load(root, "width_ratios")
    meta_steps = read_json(root / "tables" / "analysis_metadata.json")["final_steps"]
    reps = [int(r) for r in cfg["replicates"]]
    fams = ["loss", "train_logit", "test_probability", "interaction"]
    if len(df):
        df = df[df["role"] == "production_final"].copy()
        df["validity_b"] = _bool(df["validity"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    plotted = []
    for ax, arch, title in zip(axes, ("shallow", "deep"), ("Shallow (L=2), full posterior", "Deep (L=3), full posterior")):
        widths = [int(w) for w in cfg["model"][arch]["widths"]]
        d = df[df["architecture"] == arch] if len(df) else df
        ok = _panel(ax, d, fams, widths, reps, "estimate_over_sigma2", "mc_low_over_sigma2", "mc_high_over_sigma2") \
            if len(d) else False
        ax.axhline(1.0, ls="--", lw=0.8, color="0.3")
        ax.set_title(title)
        ax.set_xlabel("Width m (log scale)")
        st = meta_steps[arch]
        Ts = d["T_per_chain"].dropna() if len(d) else pd.Series(dtype=float)
        note = f"h={st['h']:.4g}" if st["h"] else "h: calibration failed"
        if len(Ts):
            note += f", T/chain {Ts.min():.0f}-{Ts.max():.0f}"
        ax.text(0.02, 0.97, note + f"\nstep check: {st['status']}", transform=ax.transAxes, va="top", fontsize=6.5)
        if not ok:
            ax.text(0.5, 0.5, FAIL_TEXT, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, wrap=True,
                    color="0.25")
        if len(d):
            plotted.append(d)
    axes[0].set_ylabel("Held-out family relaxation diagnostic / sigma^2")
    allv = pd.concat(plotted)["estimate_over_sigma2"].to_numpy(dtype=float) if plotted else np.array([1.0])
    scale = _yscale(axes, allv)
    axes[1].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    steps_txt = ", ".join(f"{a} h={meta_steps[a]['h']:.4g}" if meta_steps[a]["h"] else f"{a} unresolved"
                          for a in ("shallow", "deep"))
    checks = ", ".join(f"{a}: {meta_steps[a]['status']}" + (" after one refinement" if meta_steps[a]["refined"] else "")
                       for a in ("shallow", "deep"))
    trend = "; ".join(_trend(ratios, "relaxation", a) for a in ("shallow", "deep"))
    n = int(cfg["data"]["train_size"])
    caption = check_caption(
        "Width dependence of observable relaxation under unrestricted posterior dynamics. The horizontal axes give "
        f"network width for the one-hidden-layer and two-hidden-layer models, at fixed n={n}. Vertical values are "
        "cross-evaluated family autocorrelation integrals in the physical-time convention of the adjusted "
        "Gaussian-preserving Langevin proposal, normalized by the prior variance. Families contain the likelihood "
        "potential, eight training logits, eight held-out probabilities, and four fixed head--hidden interactions. "
        "Thin points and lines show three independent data/center replicates; the thick line is their median, and "
        "error bars show within-target Monte Carlo uncertainty. Hollow points did not meet the validity criteria. "
        "The horizontal line at one is the continuous Gaussian linear-observable timescale. "
        f"The production steps were {steps_txt}; endpoint step-halving checks: {checks}. Over the tested width "
        f"range, {trend}. These finite-observable diagnostics do not estimate the optimal PI coefficient, and no "
        "global head or hidden-layer restriction was imposed on the chains.")
    pl = pd.concat(plotted) if plotted else pd.DataFrame()
    save(fig, root, "main_1_relaxation", pl, {
        "sources": {"tables/relaxation_families.csv": sha(root / "tables/relaxation_families.csv"),
                    "tables/width_ratios.csv": sha(root / "tables/width_ratios.csv")},
        "filters": "role == production_final (architecture's final common step)", "y_normalization": "estimate / sigma^2",
        "y_scale": scale, "uncertainty": "percentile 2.5-97.5% within-chain moving-block bootstrap (400 resamples)",
        "target_law": {"shallow": "full_posterior", "deep": "full_posterior"},
        "selected_probe_ids": _selected(pl), "flagged_rows": _flagged(pl), "caption": caption}, dpi)
    return {"caption": caption}


def _selected(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [c for c in ("target_id", "family", "selected_fold_A_to_B", "selected_fold_B_to_A") if c in df]
    return df[cols].to_dict("records") if len(df) and cols else []


def _flagged(df: pd.DataFrame) -> list[dict[str, Any]]:
    if not len(df) or "validity_b" not in df:
        return []
    bad = df[~df["validity_b"]]
    cols = [c for c in ("target_id", "family", "failure_reason") if c in bad]
    return bad[cols].to_dict("records")


# ---- main figure 2 ----------------------------------------------------------------------------------------
def figure_entropy(cfg, root: Path, dpi: int) -> dict[str, Any]:
    df = load(root, "entropy_families")
    ratios = load(root, "width_ratios")
    reps = [int(r) for r in cfg["replicates"]]
    fams = ["loss", "train_logit", "interaction"]
    sigma2 = float(cfg["model"]["sigma"]) ** 2
    if len(df):
        df = df.copy()
        df["validity_b"] = _bool(df["validity"])
        df["estimate_over_sigma2"] = df["estimate"] / sigma2
        df["mc_low_over_sigma2"] = df["mc_low"] / sigma2
        df["mc_high_over_sigma2"] = df["mc_high"] / sigma2
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    plotted = []
    labels = {"shallow": ("Shallow (L=2)", "law: full posterior pi"),
              "deep": ("Deep (L=3)", "law: pi( . | G_2.5), ||W2||op/sqrt(m) <= 2.5")}
    for ax, arch in zip(axes, ("shallow", "deep")):
        widths = [int(w) for w in cfg["model"][arch]["widths"]]
        d = df[df["architecture"] == arch] if len(df) else df
        ok = _panel(ax, d, fams, widths, reps, "estimate_over_sigma2", "mc_low_over_sigma2", "mc_high_over_sigma2") \
            if len(d) else False
        ax.axhline(1.0, ls="--", lw=0.8, color="0.3")
        ax.set_title(labels[arch][0])
        ax.text(0.02, 0.97, labels[arch][1], transform=ax.transAxes, va="top", fontsize=6.5)
        ax.set_xlabel("Width m (log scale)")
        if not ok:
            ax.text(0.5, 0.5, FAIL_TEXT, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color="0.25")
        if len(d):
            plotted.append(d)
    axes[0].set_ylabel("Held-out entropy-Fisher diagnostic / sigma^2")
    allv = pd.concat(plotted)["estimate_over_sigma2"].to_numpy(dtype=float) if plotted else np.array([1.0])
    scale = _yscale(axes, allv)
    axes[1].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    pl = pd.concat(plotted) if plotted else pd.DataFrame()
    inc = pl[~pl["validity_b"]][["target_id", "family"]].astype(str).agg(":".join, axis=1).tolist() if len(pl) else []
    trend = "; ".join(_trend(ratios, "entropy", a) for a in ("shallow", "deep"))
    inc_txt = (f"Incomplete or unresolved family points (hollow): {len(inc)} of {len(pl)}." if inc
               else "All family points met their validity criteria.")
    caption = check_caption(
        "Finite-probe entropy--Fisher diagnostics for the final LSI domains. For each standardized smooth probe f and "
        "fixed tilt t in {-1,-0.5,0.5,1}, we form the normalized density r proportional to exp(tf) relative to the "
        "shallow posterior or the deep posterior conditioned on the later-hidden-layer spectral event. The vertical "
        "axis reports held-out family estimates of 2 Ent(r)/E[r||grad log r||^2], divided by the prior variance; the "
        "horizontal axis is width. The head and first hidden layer remain unrestricted in both panels. Probe "
        "selection and evaluation use separate chain folds. The value-one line is attained by Gaussian linear tilts "
        f"and is a reference, not a required lower limit for these probes. Measured trend: {trend}. {inc_txt} "
        "LSI bounds these population ratios above by its coefficient, so the plotted tests are finite-probe lower "
        "diagnostics of that coefficient; they neither determine its optimum nor establish global deep LSI.")
    save(fig, root, "main_2_entropy", pl, {
        "sources": {"tables/entropy_families.csv": sha(root / "tables/entropy_families.csv"),
                    "tables/width_ratios.csv": sha(root / "tables/width_ratios.csv")},
        "filters": "reference stopping-stage static estimates", "y_normalization": "estimate / sigma^2",
        "y_scale": scale, "uncertainty": "percentile 2.5-97.5% within-chain moving-block bootstrap",
        "target_law": {"shallow": "full_posterior", "deep": "conditional_G_2.5"},
        "selected_probe_ids": _selected(pl), "flagged_rows": _flagged(pl), "caption": caption}, dpi)
    return {"caption": caption}


# ---- main figure 3 ----------------------------------------------------------------------------------------
def figure_spectral(cfg, root: Path, dpi: int) -> dict[str, Any]:
    df = load(root, "spectral_summary")
    prior = pd.read_csv(root / "controls" / "prior_spectral.csv") if (root / "controls" / "prior_spectral.csv").exists() \
        else pd.DataFrame()
    reps = [int(r) for r in cfg["replicates"]]
    widths = [int(w) for w in cfg["model"]["deep"]["widths"]]
    qs = [float(q) for q in cfg["spectral"]["posterior_quantiles"]]
    styles = dict(zip(qs, ["-", "--", ":"]))
    if len(df):
        df = df[df["chain_id"].astype(str) == "all"].copy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for rep in reps:
        d = df[df["replicate"] == rep].set_index("m").reindex(widths) if len(df) else pd.DataFrame(index=widths)
        x = _xoff(np.array(widths, float), rep, reps)
        if "occupancy" in d:
            a1.plot(x, d["occupancy"], "o", ms=4, color="#0072B2", alpha=0.7)
            for xi, (_, r) in zip(x, d.iterrows()):
                if np.isfinite(r.get("inspected_states", np.nan)):
                    a1.annotate(f"{int(r['exits'])}/{int(r['inspected_states'])}", (xi, r["occupancy"]),
                                textcoords="offset points", xytext=(0, -9 - 6 * reps.index(rep)), fontsize=5,
                                ha="center")
        for q in qs:
            col = f"S_q{q:g}"
            if col in d:
                a2.plot(x, d[col], MARK["train_logit"], ms=3, color="#D55E00", alpha=0.6)
    for q in qs:
        col = f"S_q{q:g}"
        if len(df) and col in df:
            med = [df[(df["m"] == m)][col].median() if (df["m"] == m).sum() == len(reps) else np.nan for m in widths]
            a2.plot(widths, med, styles[q], color="#D55E00", lw=1.8, label=f"posterior q{q:g}")
        if len(prior) and col in prior:
            pr = prior.set_index("m").reindex(widths)
            a2.plot(widths, pr[col], styles[q], color="0.55", lw=0.9, label=f"prior q{q:g}")
    a1.set_ylim(0, 1.02)
    a1.set_ylabel("Observed posterior fraction in G_2.5")
    a2.axhline(1.0, ls="--", color="k", lw=0.8)
    a2.axhline(0.8, ls=":", color="0.6", lw=0.8)
    a2.text(widths[0], 0.81, "prior large-width reference", fontsize=5.5, color="0.4")
    a2.set_ylabel("||W2||op / (2.5 sqrt(m))")
    vals = [v for c in [f"S_q{q:g}" for q in qs] for v in (df[c].tolist() if len(df) and c in df else [])]
    a2.set_ylim(min([0.0] + vals) * 0.9, max([1.05] + vals) * 1.05)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths])
        ax.minorticks_off()
        ax.set_xlabel("Width m (log scale)")
    a2.legend(fontsize=5.5, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    n_ins = int(df["inspected_states"].sum()) if len(df) else 0
    exits = int(df["exits"].sum()) if len(df) else 0
    per = ", ".join(f"m={int(r.m)} r{int(r.replicate)}: {int(r.exits)}" for r in df.itertuples()) if len(df) else "none"
    if len(df) and len(prior):
        gaps = []
        for m in widths:
            dm = df[df["m"] == m]
            pm = prior[prior["m"] == m]
            if len(dm) and len(pm):
                gaps.append(f"m={m}: posterior median {dm['S_q0.5'].median():.3f} vs prior {float(pm['S_q0.5'].iloc[0]):.3f},"
                            f" posterior q99 {dm['S_q0.99'].max():.3f} (max over replicates)")
        agree = "; ".join(gaps)
    else:
        agree = "prior comparator unavailable"
    allin = " All inspected states were inside the event; no zero-width interval is reported." if exits == 0 and n_ins else ""
    n = int(cfg["data"]["train_size"])
    caption = check_caption(
        "Posterior relevance of the final deep spectral domain. Left: empirical fractions of archived draws from the "
        f"unrestricted two-hidden-layer posterior satisfying ||W2||op/sqrt(m) <= 2.5, at fixed n={n}. {n_ins} states "
        f"were inspected, with {exits} observed exits in total (per target: {per}); counts retain chain identity and are "
        f"not treated as iid rare-event trials.{allin} Right: the median, 95th, and 99th percentiles of the normalized "
        "spectral norm, with matching iid Gaussian-prior reference quantiles. The boundary is one. Posterior/prior "
        f"comparison: {agree}. This assesses whether the conditional LSI domain contains a substantial part of the "
        "sampled posterior. It does not estimate the theorem's exponential exit-rate constants, and high observed mass "
        "does not turn conditional deep LSI into a global statement.")
    pl = df.copy() if len(df) else pd.DataFrame()
    save(fig, root, "main_3_spectral", pl, {
        "sources": {"tables/spectral_summary.csv": sha(root / "tables/spectral_summary.csv"),
                    "controls/prior_spectral.csv": sha(root / "controls/prior_spectral.csv")},
        "filters": "deep targets, chain_id == all (target rows)", "y_normalization": "none (fraction; S)",
        "uncertainty": "exit counts shown; quantile intervals in table (chain-aware block bootstrap)",
        "target_law": {"deep": "full_posterior archive; event G_2.5"}, "selected_probe_ids": [],
        "flagged_rows": [], "caption": caption}, dpi)
    return {"caption": caption}


# ---- appendix S1 ------------------------------------------------------------------------------------------
def figure_s1(cfg, root: Path, dpi: int) -> dict[str, Any]:
    ep = load(root, "endpoint_comparisons")
    calp = root / "tests" / "calibration_results.csv"
    cal = pd.read_csv(calp) if calp.exists() else pd.DataFrame()
    sigma2 = float(cfg["model"]["sigma"]) ** 2
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.4))
    a, b, c, d = ax.ravel()
    fam = ep[ep["kind"] == "family"] if len(ep) else ep
    if len(fam):
        final_round = fam["round"].max()
        for arch, mk in (("shallow", "o"), ("deep", "s")):
            for f in OKABE:
                g = fam[(fam["architecture"] == arch) & (fam["family"] == f) & (fam["round"] == final_round)]
                if len(g):
                    a.plot(g["estimate_h"] / sigma2, g["estimate_h_half"] / sigma2, mk, color=OKABE[f], ms=3.5,
                           mfc="white" if arch == "deep" else OKABE[f], label=f"{arch} {LABEL[f]}")
                    b.plot(g["m"] * (1.05 if arch == "deep" else 1.0), g["relative_difference"], mk, color=OKABE[f],
                           ms=3.5, mfc="white" if arch == "deep" else OKABE[f])
        lim = [0, max(1.0, float(np.nanmax(fam[["estimate_h", "estimate_h_half"]].to_numpy() / sigma2)) * 1.1)]
        a.plot(lim, lim, "k-", lw=0.6)
        a.set_xlim(lim)
        a.set_ylim(lim)
        b.axhline(float(cfg["dynamics"]["step_validation_relative_difference_max"]), ls="--", color="k", lw=0.8)
        b.set_xscale("log")
        a.legend(fontsize=4.5, frameon=False)
    else:
        a.text(0.5, 0.5, "no endpoint comparisons", transform=a.transAxes, ha="center")
    a.set_xlabel("step h: family diagnostic / sigma^2")
    a.set_ylabel("step h/2: family diagnostic / sigma^2")
    a.set_title("(a) endpoint step-halving")
    b.set_xlabel("Width m (log scale)")
    b.set_ylabel("relative difference |h - h/2| / mean")
    b.set_title("(b) relative endpoint differences")
    if len(cal):
        ou = cal[cal["test"] == "ou_calibration"]
        for i, (_, r) in enumerate(ou.iterrows()):
            x = r["sigma"] + (0.02 if r["probe"] == "unit_sum" else -0.02)
            c.errorbar([x], [r["tau_hat"] / r["sigma"] ** 2],
                       yerr=[[r["tau_hat"] / r["sigma"] ** 2 - r["tau_low"] / r["sigma"] ** 2],
                             [r["tau_high"] / r["sigma"] ** 2 - r["tau_hat"] / r["sigma"] ** 2]],
                       fmt="o" if r["probe"] == "unit_sum" else "s", color="#0072B2", ms=3.5, capsize=2)
            c.plot([x - 0.015, x + 0.015], [r["tau_exact_discrete"] / r["sigma"] ** 2] * 2, "k-", lw=1.2)
        ge = cal[cal["test"] == "gaussian_entropy"]
        for s, col in ((0.7, "#E69F00"), (1.0, "#009E73")):
            g = ge[np.isclose(ge["sigma"], s)]
            d.errorbar(g["t"] + (0.03 if s == 1.0 else -0.03), g["R_over_sigma2"], yerr=g["R_mcse"] / s**2 * 1.96,
                       fmt="o", color=col, ms=3.5, capsize=2, label=f"sigma={s}")
        d.legend(fontsize=6, frameon=False)
    c.set_xlabel("sigma")
    c.set_ylabel("linear-probe IAT / sigma^2 (black: exact discrete)")
    c.set_title("(c) Gaussian OU calibration")
    d.axhline(1.0, ls="--", color="k", lw=0.8)
    d.set_xlabel("tilt t")
    d.set_ylabel("entropy-Fisher ratio / sigma^2")
    d.set_title("(d) Gaussian linear tilts")
    fig.tight_layout()
    n_fail = int((~_bool(fam["pass"])).sum()) if len(fam) else 0
    caption = check_caption(
        f"Numerical support. (a) Endpoint family relaxation diagnostics at the production step versus half step, "
        f"normalized by sigma^2, with the identity line (final comparison round shown; {n_fail} family comparisons "
        "failed and are listed in tables/endpoint_comparisons.csv). (b) Relative endpoint differences with the 20% "
        "protocol threshold for all replicates and both architectures. (c) Gaussian OU linear-probe integrated "
        "autocorrelation times against the exact discrete value, with bootstrap intervals. (d) Gaussian linear "
        "entropy--Fisher ratios at t=-1,-0.5,0.5,1 divided by sigma^2, with the value-one reference.")
    pl = pd.concat([fam.assign(panel="ab") if len(fam) else pd.DataFrame(),
                    cal.assign(panel="cd") if len(cal) else pd.DataFrame()], ignore_index=True)
    save(fig, root, "appendix_s1_validation", pl, {
        "sources": {"tables/endpoint_comparisons.csv": sha(root / "tables/endpoint_comparisons.csv"),
                    "tests/calibration_results.csv": sha(calp)},
        "filters": "family endpoint comparisons; OU and Gaussian-entropy calibration rows",
        "y_normalization": "divided by sigma^2", "uncertainty": "bootstrap MCSE / intervals",
        "target_law": {"endpoints": "full_posterior", "calibration": "Gaussian"}, "selected_probe_ids": [],
        "flagged_rows": fam[~_bool(fam["pass"])].to_dict("records") if len(fam) else [], "caption": caption}, dpi)
    return {"caption": caption}


# ---- appendix S2 ------------------------------------------------------------------------------------------
def figure_s2(cfg, root: Path, dpi: int) -> dict[str, Any]:
    pr = load(root, "predictive_scores")
    pi = load(root, "static_pi")
    reps = [int(r) for r in cfg["replicates"]]
    sigma2 = float(cfg["model"]["sigma"]) ** 2
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for arch, col, mk in (("shallow", "#0072B2", "o"), ("deep", "#D55E00", "s")):
        widths = [int(w) for w in cfg["model"][arch]["widths"]]
        d = pr[pr["architecture"] == arch] if len(pr) else pr
        for rep in reps:
            g = d[d["replicate"] == rep].set_index("m").reindex(widths) if len(d) else pd.DataFrame(index=widths)
            if "estimate" not in g:
                continue
            x = _xoff(np.array(widths, float), rep, reps)
            v = _bool(g["validity"]) if "validity" in g else pd.Series(False, index=g.index)
            for xi, (_, r), vi in zip(x, g.iterrows(), v):
                if np.isfinite(r.get("estimate", np.nan)):
                    err = [[r["estimate"] - r["mc_low"]], [r["mc_high"] - r["estimate"]]] \
                        if np.isfinite(r.get("mc_low", np.nan)) else None
                    ax[0].errorbar([xi], [r["estimate"]], yerr=err, fmt=mk, color=col, ms=3.5, capsize=1.5,
                                   mfc=col if vi else "white")
        ax[0].plot([], [], mk, color=col, label=arch)
    ax[0].axhline(0.0, ls="--", color="k", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Width m (log scale)")
    ax[0].set_ylabel("NLS(prior) - NLS(posterior)")
    ax[0].set_title("(a) predictive improvement")
    ax[0].legend(fontsize=6, frameon=False)
    plotted = [pr.assign(panel="a")] if len(pr) else []
    for k, arch in enumerate(("shallow", "deep"), 1):
        widths = [int(w) for w in cfg["model"][arch]["widths"]]
        d = pi[(pi["architecture"] == arch) & (pi["kind"] == "family")].copy() if len(pi) else pd.DataFrame()
        if len(d):
            d["validity_b"] = _bool(d["validity"])
            d["mc_low_over_sigma2"] = d["mc_low"] / sigma2
            d["mc_high_over_sigma2"] = d["mc_high"] / sigma2
            ok = _panel(ax[k], d, ["loss", "train_logit", "interaction"], widths, reps, "estimate_over_sigma2",
                        "mc_low_over_sigma2", "mc_high_over_sigma2")
            if not ok:
                ax[k].text(0.5, 0.5, FAIL_TEXT, transform=ax[k].transAxes, ha="center", fontsize=5, wrap=True)
            plotted.append(d.assign(panel=f"b_{arch}"))
        ax[k].axhline(1.0, ls="--", color="0.3", lw=0.8)
        ax[k].set_title(f"(b) static PI, {arch}, full posterior")
        ax[k].set_xlabel("Width m (log scale)")
    ax[1].set_ylabel("Held-out family Var/E||grad||^2 / sigma^2")
    ax[2].legend(fontsize=5, frameon=False)
    fig.tight_layout()
    imp = pr["estimate"].dropna() if len(pr) else pd.Series(dtype=float)
    caption = check_caption(
        "Appendix checks at no additional sampling cost. (a) Bayesian predictive negative log score of the prior mixture "
        "minus that of the posterior on all held-out points (positive: the posterior predicts better); "
        f"observed range {imp.min():.3f} to {imp.max():.3f} nats per point. " if len(imp) else
        "Appendix checks at no additional sampling cost. (a) Predictive scores unavailable. ") + check_caption(
        "(b) Held-out family static Poincare ratios Var/E||grad g||^2 on the unrestricted posterior for both "
        "architectures, divided by sigma^2; hollow points did not meet the 15% relative MCSE or reference-validity "
        "gate. Small ratios are not upper bounds on the Poincare constant.")
    save(fig, root, "appendix_s2_static_checks", pd.concat(plotted, ignore_index=True) if plotted else pd.DataFrame(), {
        "sources": {"tables/predictive_scores.csv": sha(root / "tables/predictive_scores.csv"),
                    "tables/static_pi.csv": sha(root / "tables/static_pi.csv")},
        "filters": "static_pi kind == family", "y_normalization": "(a) nats per point; (b) / sigma^2",
        "uncertainty": "block bootstrap (posterior), iid bootstrap (prior)",
        "target_law": {"static_pi": "full_posterior", "predictive": "posterior vs prior"},
        "selected_probe_ids": _selected(pi[pi["kind"] == "family"]) if len(pi) else [], "flagged_rows": [],
        "caption": caption}, dpi)
    return {"caption": caption}


def make_figures(cfg, root: Path) -> dict[str, str]:
    dpi = int(cfg["outputs"]["png_dpi"])
    caps = {"main_1_relaxation": figure_relaxation(cfg, root, dpi)["caption"],
            "main_2_entropy": figure_entropy(cfg, root, dpi)["caption"],
            "main_3_spectral": figure_spectral(cfg, root, dpi)["caption"],
            "appendix_s1_validation": figure_s1(cfg, root, dpi)["caption"],
            "appendix_s2_static_checks": figure_s2(cfg, root, dpi)["caption"]}
    tex = []
    for name, cap in caps.items():
        esc = cap.replace("%", r"\%").replace("_", r"\_").replace("^", r"\^{}").replace("&", r"\&")
        tex.append(f"% {name}\n\\begin{{figure}}[t]\n\\centering\n\\includegraphics[width=\\linewidth]"
                   f"{{figures/{name}.pdf}}\n\\caption{{{esc}}}\n\\label{{fig:{name}}}\n\\end{{figure}}\n")
    (root / "captions.tex").write_text("\n".join(tex))
    return caps
