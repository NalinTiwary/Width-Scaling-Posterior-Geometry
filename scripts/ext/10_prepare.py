#!/usr/bin/env python3
"""Theorem table check + frozen datasets for the extension (addendum §3–4, Appendix A)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.centers import generate_center_bank  # noqa: E402
from cylinder.data import sha256_array  # noqa: E402
from cylinder.ext.cli import ROOT, base_parser, load  # noqa: E402
from cylinder.ext.data import (  # noqa: E402
    class_counts,
    fmnist_pca_map,
    fmnist_replicate,
    load_npz,
    orth_replicate,
    save_npz,
)
from cylinder.ext.theory import assert_ext_preflight, ext_bundle  # noqa: E402
from cylinder.io import atomic_save_json  # noqa: E402


def prepare_fmnist(cfg: dict, data_dir: Path) -> dict:
    rd = cfg["realdata"]
    proc = ROOT / rd["processed_dir"]
    reps = [int(r) for r in rd["replicates"]]
    need = [r for r in reps if not (proc / f"subset_r{r}.npz").exists()] or (
        [] if (proc / "pca_map.npz").exists() else reps
    )
    if need:
        print(f"Fitting Fashion-MNIST PCA map and subsets for replicates {need}")
        pca = fmnist_pca_map(ROOT / rd["raw_dir"], classes=tuple(rd["classes"]), dim=int(rd["pca_dim"]))
        save_npz(proc / "pca_map.npz", {k: v for k, v in pca.items() if not k.startswith("pool_")}
                 | {"pool_index": pca["pool_index"]})
        for r in reps:
            rep = fmnist_replicate(r, pca, n_per_class=int(rd["n_per_class"]),
                                   subset_seed_base=int(rd["subset_seed_base"]),
                                   center_seed_base=int(rd["center_seed_base"]),
                                   center_bank_rows=int(rd["center_bank_rows"]))
            save_npz(proc / f"subset_r{r}.npz", {k: v for k, v in rep.items() if k not in ("U", "hash_U")})
    summary = {}
    for r in reps:
        sub = load_npz(proc / f"subset_r{r}.npz")
        if sha256_array(sub["X"]) != sub["hash_X"]:
            raise RuntimeError(f"fmnist subset r{r}: X hash mismatch")
        bank = generate_center_bank(n_pairs=int(rd["center_bank_rows"]), d=sub["X"].shape[1],
                                    seed=int(rd["center_seed_base"]) + r)
        save_npz(data_dir / f"fmnist_r{r}.npz", {**sub, "U": bank["U"], "hash_U": bank["hash_U"]})
        n = sub["X"].shape[0]
        summary[f"r{r}"] = {
            "n": n, "class_counts": {"y0": int((sub["y"] == 0).sum()), "y1": int((sub["y"] == 1).sum())},
            "M2": float(sub["M2"]), "M4_sq_upper_bound": float(sub["M4_sq_bound"]),
            "hash_X": sub["hash_X"], "hash_y": sub["hash_y"], "hash_U": bank["hash_U"],
            "D_th_safe_envelope": {
                str(m): ext_bundle(n, int(m), M2=float(sub["M2"]), M4=float(np.sqrt(sub["M4_sq_bound"])))["D_th"]
                for m in rd["widths"]
            },
        }
    pca = load_npz(proc / "pca_map.npz")
    summary["pca_explained_variance"] = float(np.sum(pca["explained_variance_ratio"]))
    summary["pca_hash_components"] = pca["hash_components"]
    return summary


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg, _, artifacts, _ = load(args)
    data_dir = artifacts / "data"

    table = assert_ext_preflight()
    print("Theorem table matches the addendum:")
    for key, b in table.items():
        print(f"  {key}: B={b['B']:.4f} D_th={b['D_th']:.4f} 1-D_th={b['curvature_margin']:+.4f}")

    ctl = cfg["controlled"]
    orth = {}
    for r in ctl["replicates"]:
        rep = orth_replicate(int(r), d=int(ctl["d"]), input_seed_base=int(ctl["input_seed_base"]),
                             label_seed_base=int(ctl["label_seed_base"]),
                             center_seed_base=int(ctl["center_seed_base"]),
                             center_bank_rows=int(ctl["center_bank_rows"]),
                             teacher_scale=float(ctl["teacher_scale"]))
        X = rep["X_bank"]
        err = float(np.abs(X @ X.T - np.eye(X.shape[0])).max())
        if err > 1e-12:
            raise RuntimeError(f"orth r{r}: input bank not orthonormal (err {err})")
        save_npz(data_dir / f"orth_r{r}.npz", rep)
        ns = sorted({int(n) for n, _ in ctl["settings"]})
        orth[f"r{r}"] = {"class_counts": class_counts(rep["y_all"], ns), "orthonormality_err": err,
                         "hash_X": rep["hash_X"], "hash_y": rep["hash_y"], "hash_U": rep["hash_U"]}
        print(f"orth r{r}: class counts {orth[f'r{r}']['class_counts']}")

    summary = {"theorem_table": table, "orth": orth}
    if cfg.get("realdata", {}).get("enabled", False):
        summary["fmnist"] = prepare_fmnist(cfg, data_dir)
        print("fmnist:", {k: v for k, v in summary["fmnist"].items() if k.startswith("r")})
    atomic_save_json(artifacts / "data_summary.json", summary)
    print(f"Wrote {data_dir} and data_summary.json")


if __name__ == "__main__":
    main()
