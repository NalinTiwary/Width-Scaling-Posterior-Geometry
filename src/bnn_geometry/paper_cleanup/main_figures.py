"""Revised main figures (figure_cleanup_guide.md §4-§6) from existing campaign outputs; no sampling.

Figure 1: the stored statewise S = sigma_max(W2)/(2.5 sqrt m) is converted once to T = 2.5 S; the theorem threshold
a_* = r0 + 2 sigma is taken from an audited theory context (config + the centers actually used by the model).
Figure 2: the accepted replicate-level loss ACF curves (paper_figures/tables/acf_replicate.csv), cropped to 200 lags.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..paper_export import sha256_file
from .common import FIGURE_WIDTH_IN, HISTORICAL_A, Layout, clean_figures, write_json


# ---- theory context -------------------------------------------------------------------------------
def theory_context(cfg: dict[str, Any], root: Path, L: Layout, device: str = "cpu") -> dict[str, Any]:
    """Verify r0 and sigma from the configuration and from the W2 block of every deep target's prior center."""
    import torch
    from ..context import TargetContext
    from ..model import unpack

    m = cfg["model"]
    if int(m["deep"]["weight_layers"]) != 3:
        raise ValueError("deep architecture must have 3 weight layers")
    if cfg["prior_centers"]["later_hidden"] != "zero":
        raise ValueError(f"later-hidden prior center is {cfg['prior_centers']['later_hidden']!r}, not zero")
    rows = []
    for t in C.targets(cfg):
        if t.arch != "deep":
            continue
        ctx = TargetContext(cfg, t, root, device=torch.device(device))
        W2 = unpack(ctx.model.theta0.unsqueeze(0), ctx.lay)["W2"][0].detach().cpu().numpy()
        op = float(np.linalg.svd(W2, compute_uv=False)[0]) if np.any(W2) else 0.0
        rows.append({"target_id": t.target_id, "m": t.m, "replicate": t.rep, "target_hash": ctx.thash,
                     "W2_center_opnorm_over_sqrt_m": op / math.sqrt(t.m), "W2_center_max_abs": float(np.abs(W2).max())})
    r0 = max(r["W2_center_opnorm_over_sqrt_m"] for r in rows)
    manifest = root / "manifest.json"
    if not manifest.exists():
        manifest = root / "checksums.sha256"
    ctx_json = {
        "weight_layers": int(m["deep"]["weight_layers"]), "n": int(cfg["data"]["train_size"]),
        "sigma": float(m["sigma"]), "r0": float(r0), "hidden_center_max_normalized_opnorm": float(r0),
        "statistic": "opnorm_W2_over_sqrt_m", "target_manifest_sha256": sha256_file(manifest),
        "target_manifest": str(manifest.relative_to(root.parent) if root.parent in manifest.parents else manifest),
        "config_sha256": json.loads(manifest.read_text()).get("config_sha256") if manifest.suffix == ".json" else None,
        "center_files_sha256": {p.name: sha256_file(p) for p in sorted((root / "centers").glob("rep_*.npz"))},
        "prior_center_later_hidden": cfg["prior_centers"]["later_hidden"],
        "admissibility_threshold": float(r0 + 2 * float(m["sigma"])),
        "historical_normalizer_a": HISTORICAL_A,
        "verification": "W2 block of each deep target's model prior center (theta0) computed from the saved "
                        "center/data files; sigma from the frozen configuration",
        "per_target_center_check": rows}
    write_json(L.exports / "theory_context.json", ctx_json)
    return ctx_json


# ---- spectral export -------------------------------------------------------------------------------
def spectral_export(root: Path, L: Layout) -> dict[str, Any]:
    src = root / "paper_figures" / "tables" / "spectral_states.csv.gz"
    s = pd.read_csv(src)
    if set(s.source_normalization) != {"campaign_S_sigma_max_over_a_sqrt_m"} or set(s.a) != {HISTORICAL_A}:
        raise ValueError("unexpected stored spectral normalization; refusing to convert")
    if set(s.reference_status) != {"reference_pass"}:
        raise ValueError("spectral states include targets whose reference run did not pass")
    s = s.sort_values(["m", "replicate_id", "chain_id", "original_draw_index"])
    T = s.S.to_numpy(np.float64) * HISTORICAL_A
    out = pd.DataFrame({"width": s.m.to_numpy(), "replicate": s.replicate_id.to_numpy(),
                        "chain": s.chain_id.to_numpy(), "draw": s.original_draw_index.to_numpy(), "T": T})
    fn = L.exports / "spectral_states.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(out.columns))
        for r in out.itertuples(index=False):
            w.writerow([int(r.width), int(r.replicate), int(r.chain), int(r.draw), repr(float(r.T))])
    with open(fn, "rb") as fi, open(L.exports / "spectral_states.csv.gz", "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as fo:
        fo.write(fi.read())
    # unit check: the campaign recomputed sigma_max for 20 archived W2 matrices per target with an independent
    # LAPACK SVD and compared S; T = 2.5 S inherits that check. A fresh direct check written on the cluster by
    # `export-supplements` (spectral_unit_check_direct.csv) is merged when present.
    summ = pd.read_csv(root / "tables" / "spectral_summary.csv")
    summ = summ.dropna(subset=["backend_check_ok"])[["target_id", "backend_check_ok", "backend_max_rel_diff"]]
    if sorted(summ.target_id) != sorted(f"deep_m{m:04d}_r{r}" for m in (32, 64, 128, 256) for r in (0, 1, 2)):
        raise ValueError("campaign SVD backend check missing for some deep targets")
    rows = [{"target_id": r.target_id, "check": "campaign_independent_lapack_svd_20_states",
             "pass": bool(r.backend_check_ok), "max_rel_diff": float(r.backend_max_rel_diff)} for r in summ.itertuples()]
    direct = L.exports / "spectral_unit_check_direct.csv"
    if direct.exists():
        d = pd.read_csv(direct)
        for tid, g in d.groupby("target_id"):
            rows.append({"target_id": tid, "check": "fresh_direct_svd_T_from_archived_W2",
                         "pass": bool(g["pass"].all()), "max_rel_diff": float(g.rel_diff.max())})
    uc = pd.DataFrame(rows).sort_values(["target_id", "check"])
    uc.to_csv(L.tables / "spectral_unit_check.csv", index=False)
    if not uc["pass"].all():
        raise ValueError("spectral unit check failed")
    # cross-check: T quantiles equal 2.5 x the accepted unrounded S summaries
    acc = pd.read_csv(root / "paper_figures" / "tables" / "spectral_replicate_summary.csv")
    worst = 0.0
    for r in acc.itertuples():
        x = out[(out.width == r.m) & (out.replicate == r.replicate_id)]["T"].to_numpy()
        q = np.quantile(x, [0.5, 0.95, 0.99], method="linear")
        worst = max(worst, float(np.max(np.abs(q - HISTORICAL_A * np.array([r.q50, r.q95, r.q99])))))
        if len(x) != r.n_inspected:
            raise ValueError(f"{r.target_id}: {len(x)} rows != accepted {r.n_inspected}")
    if worst > 1e-12:
        raise ValueError(f"T quantiles differ from 2.5 x accepted S quantiles by {worst}")
    return {"rows": int(len(out)), "source": str(src.relative_to(root)), "source_sha256": sha256_file(src),
            "conversion": "T = 2.5 * S (historical normalizer inverted once)", "quantile_crosscheck_max_abs": worst,
            "unit_checks": rows, "csv": str(fn)}


# ---- tables ------------------------------------------------------------------------------------------
def threshold_tables(L: Layout, threshold: float) -> pd.DataFrame:
    c = pd.read_csv(L.figures / "figure_1_spectral_domain_threshold_counts.csv")
    if not (c.n_above + c.n_at_or_below == c.n).all():
        raise ValueError("threshold bookkeeping failed")
    if (c.n_near_threshold > 0).any():
        raise ValueError("states within the 1e-10 guard of the threshold need an independent SVD check")
    rows = []
    for m, g in c.groupby("width"):
        g = g.sort_values("replicate")
        pct = 100 * g.fraction_above.to_numpy()
        rows.append({"width": int(m), **{f"n_above_r{int(r.replicate)}": int(r.n_above) for r in g.itertuples()},
                     "n_per_target": int(g.n.iloc[0]), "pct_above_median": float(np.median(pct)),
                     "pct_above_min": float(pct.min()), "pct_above_max": float(pct.max()),
                     "max_T": float(g.maximum.max()), "n_near_threshold": int(g.n_near_threshold.sum())})
    paper = pd.DataFrame(rows)
    paper.to_csv(L.tables / "threshold_counts_paper.csv", index=False)
    return paper


def prior_table(root: Path, L: Layout) -> pd.DataFrame:
    q = pd.read_csv(L.figures / "figure_1_spectral_domain_quantiles.csv")
    pri = np.load(root / "controls" / "prior_spectral_values.npz", allow_pickle=False)
    rows = []
    for m in sorted(q.width.unique()):
        row = {"width": int(m)}
        for qq, lab in ((0.5, "q50"), (0.95, "q95"), (0.99, "q99")):
            sel = q[(q["width"] == m) & np.isclose(q["quantile"], qq)]["T"]
            if len(sel) != 3:
                raise ValueError(f"expected 3 replicate quantiles at width {m}, q={qq}")
            row[f"posterior_{lab}"] = float(np.median(sel))
        x = HISTORICAL_A * pri[f"m{int(m)}"].astype(np.float64)
        pq = np.quantile(x, [0.5, 0.95, 0.99], method="linear")
        row.update({"prior_q50": float(pq[0]), "prior_q95": float(pq[1]), "prior_q99": float(pq[2]),
                    "n_iid_prior": int(x.size)})
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(L.tables / "prior_reference_T.csv", index=False)
    return t


# ---- Figure 2 input ----------------------------------------------------------------------------------
def acf_export(root: Path, L: Layout) -> dict[str, Any]:
    src = root / "paper_figures" / "tables" / "acf_replicate.csv"
    a = pd.read_csv(src)
    out = a.rename(columns={"m": "width", "rho": "acf"})[["architecture", "width", "replicate", "lag", "acf"]]
    if out.lag.max() < 400:
        raise ValueError("accepted ACF curves must extend to lag 400")
    fn = L.exports / "loss_acf_replicates.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(out.columns))
        for r in out.itertuples(index=False):
            w.writerow([r.architecture, int(r.width), int(r.replicate), int(r.lag), repr(float(r.acf))])
    back = pd.read_csv(fn)
    if not np.array_equal(back.acf.to_numpy(), a.rho.to_numpy()):
        raise ValueError("Figure 2 input differs from the accepted replicate curves")
    return {"rows": int(len(out)), "source": str(src.relative_to(root)), "source_sha256": sha256_file(src)}


# ---- build -------------------------------------------------------------------------------------------
def build_main(cfg: dict[str, Any], root: Path, L: Layout, *, device: str = "cpu",
               width_in: float = FIGURE_WIDTH_IN) -> dict[str, Any]:
    cf = clean_figures()
    L.mkdirs()
    ctx = theory_context(cfg, root, L, device=device)
    sp = spectral_export(root, L)
    fig, meta = cf.spectral_figure(str(L.exports / "spectral_states.csv"), width_in,
                                   str(L.exports / "theory_context.json"))
    cf.write_outputs(fig, L.figures / "figure_1_spectral_domain", str(L.exports / "spectral_states.csv"), meta)
    thr = threshold_tables(L, ctx["admissibility_threshold"])
    pri = prior_table(root, L)
    ac = acf_export(root, L)
    fig, meta = cf.acf_figure(str(L.exports / "loss_acf_replicates.csv"), width_in, 200)
    cf.write_outputs(fig, L.figures / "figure_2_loss_acf", str(L.exports / "loss_acf_replicates.csv"), meta)
    return {"theory_context": {k: v for k, v in ctx.items() if k != "per_target_center_check"},
            "spectral_export": sp, "acf_export": ac, "threshold_rows": int(len(thr)), "prior_rows": int(len(pri))}
