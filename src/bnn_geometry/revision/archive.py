"""Read-only recovery of the published campaign summaries.

Scalar ESS values come from the supplement table in the geometry package. Spectral counts
are recomputed from spectral_states.csv and checked against the published threshold table.
Original HDF5 trajectories are not in the package; missing traces are labeled, not interpolated.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..data import load_replicate
from . import config as RC
from .bounds import bound_row, deep_mass_example, n16_widths, n4_fixture
from .efficiency import TRANSITIONS_S1, break_even_ratio, ess_per_transition
from .geometry import paired_designs, probe_train_indices
from .gram import gram_spectrum

PUBLISHED_MEDIAN_ESS = {
    ("shallow", 64): 4199, ("shallow", 256): 5125, ("shallow", 1024): 5549, ("shallow", 4096): 5375,
    ("deep", 32): 3309, ("deep", 64): 4388, ("deep", 128): 5624, ("deep", 256): 6119,
}
PUBLISHED_RATIOS = {
    "deep": {0: 2.068, 1: 1.815, 2: 1.849},
    "shallow": {0: 1.360, 1: 1.241, 2: 1.239},
}


def _read_zip_text(path: Path, name: str) -> str:
    with zipfile.ZipFile(path) as zf:
        return zf.read(name).decode()


def supplement_rows(zip_path: Path) -> list[dict[str, str]]:
    text = _read_zip_text(zip_path, "final_geometry_package/6_supplement_tables/supplement_diagnostics.csv")
    return list(csv.DictReader(io.StringIO(text)))


def section2_loss_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r["analysis"] != "S1_prediction_acf" or r["observable"] != "V":
            continue
        ess = float(r["ess_bulk"])
        out.append({
            "target_id": r["target_id"], "architecture": r["architecture"], "m": int(r["m"]),
            "replicate": int(r["replicate"]), "h": float(r["h"]), "ess_bulk": ess,
            "acceptance": float(r["acceptance"]), "transitions": TRANSITIONS_S1,
            "ess_per_transition": ess_per_transition(ess, TRANSITIONS_S1),
            "ess_per_1000_transitions": 1000.0 * ess_per_transition(ess, TRANSITIONS_S1),
            "window": r.get("window"), "iteration_start": r.get("iteration_start"),
            "iteration_end": r.get("iteration_end"), "source": "archive_S1_prediction_acf",
        })
    if len(out) != 24:
        raise RuntimeError(f"expected 24 S1 loss rows, found {len(out)}")
    return out


def median_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["architecture"], r["m"])].append(r)
    out = []
    for key in sorted(groups, key=lambda k: (0 if k[0] == "shallow" else 1, k[1])):
        grp = groups[key]
        ess = float(np.median([g["ess_bulk"] for g in grp]))
        acc = float(np.median([g["acceptance"] for g in grp]))
        out.append({
            "architecture": key[0], "m": key[1], "n_replicates": len(grp),
            "acceptance_median": acc, "ess_bulk_median": ess,
            "ess_per_1000_transitions": 1000.0 * ess / TRANSITIONS_S1,
            "published_ess_bulk": PUBLISHED_MEDIAN_ESS[key],
        })
    return out


def replicate_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {(r["architecture"], r["m"], r["replicate"]): r["ess_bulk"] for r in rows}
    out = []
    for arch, narrow, wide in (("shallow", 64, 4096), ("deep", 32, 256)):
        for rep in (0, 1, 2):
            ratio = break_even_ratio(by[(arch, wide, rep)], by[(arch, narrow, rep)])
            out.append({"architecture": arch, "replicate": rep, "narrow": narrow, "wide": wide,
                        "ess_ratio": ratio, "published_ratio": PUBLISHED_RATIOS[arch][rep],
                        "note": "wider endpoint wins on ESS/second on this replicate iff measured cost ratio < ess_ratio"})
    return out


def spectral_counts(zip_path: Path) -> list[dict[str, Any]]:
    text = _read_zip_text(zip_path, "final_geometry_package/1_main_figure_data/spectral_states.csv")
    df = pd.read_csv(io.StringIO(text))
    published = pd.read_csv(io.StringIO(_read_zip_text(
        zip_path, "final_geometry_package/1_main_figure_data/figure_1_spectral_domain_threshold_counts.csv")))
    published = published[np.isclose(published["threshold"].astype(float), 2.0)]
    rows = []
    for (width, rep), g in df.groupby(["width", "replicate"]):
        T = g["T"].to_numpy(dtype=np.float64)
        pub = published[(published["width"] == width) & (published["replicate"] == rep)]
        if len(pub) != 1:
            raise RuntimeError(f"missing published count for width {width} replicate {rep}")
        n_above_2 = int(np.sum(T > 2.0))
        rows.append({
            "width": int(width), "replicate": int(rep), "n": int(T.size),
            "n_above_2": n_above_2, "published_n_above_2": int(pub["n_above"].iloc[0]),
            "match_published": n_above_2 == int(pub["n_above"].iloc[0]),
            "n_above_2_05": int(np.sum(T > 2.05)), "n_above_2_10": int(np.sum(T > 2.10)),
            "max_T": float(T.max()), "stride_note": "archived draws; autocorrelation is not removed",
        })
    rows.sort(key=lambda r: (r["width"], r["replicate"]))
    return rows


def original_chi(baseline: Path) -> list[dict[str, Any]]:
    rows = []
    for rep in (0, 1, 2):
        D, _bank, hashes = load_replicate(baseline, rep)
        spec = gram_spectrum(D["X"])
        rows.append({"replicate": rep, **spec, "class_fraction_y1": float(np.mean(D["y"] == 1.0)),
                     "hash_X": hashes["hash_X"], "unit_row_floor": 4.0,
                     "chi_at_least_n_over_d": bool(spec["chi"] + 1e-9 >= spec["n"] / spec["d"])})
    return rows


def theory_tables(chi_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [n4_fixture(), deep_mass_example()]
    for item in n16_widths():
        rows.append(bound_row(label=f"geometry_chi_{item['chi']:g}_width_at_sufficient", L=2, m=int(round(item["sufficient_width_72"])),
                              n=16, chi=item["chi"]))
        for m in (64, 256, 1024, 4096):
            rows.append(bound_row(label=f"geometry_chi_{item['chi']:g}_m{m}", L=2, m=m, n=16, chi=item["chi"]))
    for chi in chi_rows:
        for arch, widths, L in (("shallow", (64, 256, 1024, 4096), 2), ("deep", (32, 64, 128, 256), 3)):
            for m in widths:
                kwargs = {"label": f"original_{arch}_m{m}_r{chi['replicate']}", "L": L, "m": m,
                          "n": int(chi["n"]), "chi": chi["chi"] if L == 2 else None}
                if L > 2:
                    kwargs["a"] = 2.10
                rows.append(bound_row(**kwargs))
    return rows


def planned_status(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every planned cell, with archive rows already measured and sampling rows not started."""
    rows = []

    def add(cell: str, stage: str, status: str, reason: str, **extra: Any) -> None:
        rows.append({"cell": cell, "stage": stage, "status": status, "reason": reason, **extra})

    for arch, widths in (("shallow", (64, 256, 1024, 4096)), ("deep", (32, 64, 128, 256))):
        for m in widths:
            for rep in (0, 1, 2):
                cell = f"{arch}_m{m:04d}_r{rep}"
                add(cell, "archive_ess", "archive_reported", "S1 bulk ESS recovered; transition cost not in the package",
                    architecture=arch, m=m, replicate=rep)
                add(cell, "fixed_step_timing", "not_started", "needs a same-hardware replay or a fresh timed run",
                    architecture=arch, m=m, replicate=rep)
                endpoint = m in (widths[0], widths[-1])
                if endpoint:
                    add(cell, "tuned_step", "not_started", "endpoint tuning is the first tuned block",
                        architecture=arch, m=m, replicate=rep)
                    add(cell, "prior_starts", "not_started", "32 prior starts; expand only if the margin is unresolved",
                        architecture=arch, m=m, replicate=rep)
                else:
                    add(cell, "tuned_step", "not_started", "intermediate width; do not label as tuned until measured",
                        architecture=arch, m=m, replicate=rep)
        for m in (widths[0], widths[-1]):
            if arch == "deep":
                add(f"{arch}_m{m:04d}_r0", "prior_only_control", "not_started",
                    "replicate 0 first; cite all three replicates only after they exist", architecture=arch, m=m, replicate=0)
    for r in (0.0, 0.2, 0.8):
        for m in (64, 256, 1024, 4096):
            for rep in (0, 1, 2):
                first = rep == 0 and m in (64, 4096)
                add(f"geometry_r{r:g}_m{m:04d}_r{rep}", "geometry",
                    "not_started", "endpoint pilot first" if first else "after the six-cell endpoint pilot",
                    r=r, m=m, replicate=rep)
    for name in ("relu", "concentrated_likelihood", "depth4", "prior_scale_0.5", "nonlinear_teacher", "sin"):
        for m in (32, 256):
            add(f"{name}_deep_m{m:04d}_r0", name, "not_started", "one-factor contrast, replicate 0 first",
                architecture="deep", m=m, replicate=0)
    add("relu_shallow_m0064_r0", "relu", "not_started", "shallow endpoint if budget remains", m=64)
    add("relu_shallow_m4096_r0", "relu", "not_started", "shallow endpoint if budget remains", m=4096)
    add("uci_wisconsin", "uci", "blocked_missing_data",
        "load_wisconsin is ready; set uci_csv in the revision config to the UCI wdbc.data file",
        dataset="breast_cancer_wisconsin_diagnostic")
    add("slice_endpoint_timing", "second_sampler", "not_started",
        "reuse archived elliptical-slice scalars; time the kernel only if those timings are absent", )
    rows.append({"cell": "campaign", "stage": "config", "status": "configured", "reason": cfg["campaign_id"]})
    return rows


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def run_archive(cfg: dict[str, Any], *, zip_path: Optional[Path] = None) -> dict[str, Any]:
    zip_path = zip_path or RC.archive_zip(cfg)
    if zip_path is None or not zip_path.exists():
        raise FileNotFoundError(f"archive zip not found: {zip_path}")
    root = RC.output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    loss = section2_loss_rows(supplement_rows(zip_path))
    med = median_table(loss)
    ratios = replicate_ratios(loss)
    counts = spectral_counts(zip_path)
    chi = original_chi(RC.baseline_root(cfg))
    bounds = theory_tables(chi)
    status = planned_status(cfg)
    write_table(root / "efficiency.csv", [
        {**r, "probe": "V", "estimator": "rank_bulk_ESS", "numerator": r["ess_bulk"],
         "denominator": r["transitions"], "unit": "ESS per transition", "mcse": None,
         "status": "archive_ess_cost_unmeasured",
         "reason": "numerator from S1; seconds per transition were not in the package"}
        for r in loss
    ])
    write_table(root / "bounds.csv", bounds)
    write_table(root / "status.csv", status)
    write_table(root / "section2_medians.csv", med)
    write_table(root / "break_even_ratios.csv", ratios)
    write_table(root / "spectral_counts.csv", counts)
    write_table(root / "input_spectra.csv", chi)
    # Geometry is deterministic and has no chains. Record the certificate inputs beside the bounds.
    _latents, _designs, checks = paired_designs(int(cfg["master_seed"]), 0, [0.0, 0.2, 0.8])
    (root / "geometry_replicate0_checks.json").write_text(json.dumps({
        "checks": checks, "probe_indices_n16": probe_train_indices(16),
        "n16_widths": n16_widths(),
    }, indent=2))
    manifest = {
        "archive_zip": str(zip_path),
        "baseline_root": str(RC.baseline_root(cfg)),
        "baseline_config_sha256": C.config_sha256(cfg["baseline_config"]),
        "code_revision": C.code_revision(),
        "s1_loss_rows": len(loss),
        "spectral_matches": all(r["match_published"] for r in counts),
        "deep_m256_exceedances_at_2_10": [
            r["n_above_2_10"] for r in counts if r["width"] == 256
        ],
        "traces": "HDF5 stage files and per-window device times are not in the zip; no ESS was interpolated",
    }
    (root / "archive_manifest.json").write_text(json.dumps(manifest, indent=2))
    # initialization.csv is created by the prior-start stage. Leave an empty header so the handoff name exists.
    write_table(root / "initialization.csv", [{
        "cell": None, "checkpoint": None, "observable": "V", "standardized_error": None,
        "status": "not_started", "reason": "prior-start runs have not been launched",
    }])
    return manifest
