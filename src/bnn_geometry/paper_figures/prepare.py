"""Preparation: resolve campaign inputs, check provenance, write validated source tables (guide §2, §3, §5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config as C
from ..paper_export import selection_rule
from ..randomness import stream_seed
from .common import OBSERVABLE, SPECTRAL_NORMALIZATION, Layout, sha256


class InputError(RuntimeError):
    pass


def _rel(p: Path, root: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(p)


def _map_row(logical: str, path: Path, root: Path, dataset: str, shape: Any, dtype: str, mapping: str) -> dict:
    return {"logical_input": logical, "path": _rel(path, root), "dataset": dataset, "shape": str(shape),
            "dtype": dtype, "sha256": sha256(path), "mapping": mapping}


def deep_targets(cfg: dict[str, Any]) -> list[C.Target]:
    return [t for t in C.targets(cfg) if t.arch == "deep"]


def prepare_spectral(cfg: dict[str, Any], root: Path, L: Layout) -> list[dict]:
    """Per-state S table from the unrestricted reference archive and the iid prior values (guide §3, §9.2)."""
    rows = []
    src = root / "tables" / "spectral_states.csv"
    summ = root / "tables" / "spectral_summary.csv"
    if not src.exists():
        raise InputError(f"{src} missing: recover the statewise table or recompute S from the W2 archive (§10.2)")
    st = pd.read_csv(src)
    rows.append(_map_row("spectral_state_table", src, root, "csv rows", st.shape, "float64 S",
                         "one row per archived reference state of each deep target and chain"))
    if set(st.campaign_id) != {cfg["campaign_id"]}:
        raise InputError(f"spectral states belong to {set(st.campaign_id)}, not {cfg['campaign_id']}")
    a = float(cfg["domain"]["spectral_cutoff_a"])
    stride = int(cfg["reference"]["parameter_archive_stride"])
    out = []
    for t in deep_targets(cfg):
        tdir = root / "targets" / t.target_id
        spec = json.loads((tdir / "spec.json").read_text())
        status = json.loads((tdir / "status.json").read_text())
        rows.append(_map_row("target_specification", tdir / "spec.json", root, "json", "-", "json",
                             f"{t.target_id}: architecture, m, n, sigma, data/center hashes, sampler identity"))
        d = st[st.target_id == t.target_id]
        ref = status.get("reference") or {}
        if ref.get("status") != "reference_pass":
            print(f"WARNING {t.target_id}: reference status {ref.get('status')} (recorded in the summary table)")
        per_chain = int(ref["production_per_chain"]) // stride
        if set(d.execution_hash) != {spec["reference_execution_hash"]}:
            raise InputError(f"{t.target_id}: states do not come from the unrestricted reference sampler run")
        if set(d.target_hash) != {spec["target_hash"]}:
            raise InputError(f"{t.target_id}: target hash mismatch")
        if float(spec["a"]) != a or set(d.m) != {t.m} or len(d) == 0:
            raise InputError(f"{t.target_id}: a or m mismatch")
        for c in cfg["chain_ids"]:
            dc = d[d.chain_id == c]
            exp = np.arange(per_chain, dtype=np.int64) * stride
            if not np.array_equal(np.sort(dc.draw.to_numpy()), exp):
                raise InputError(f"{t.target_id}/chain {c}: archived draws are not the stride-{stride} schedule "
                                 f"of {per_chain} states")
        S = d.S.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(S)) or np.any(S < 0):
            raise InputError(f"{t.target_id}: invalid S values")
        out.append(pd.DataFrame({
            "architecture": "deep", "L": t.L, "m": t.m, "n": int(cfg["data"]["train_size"]),
            "sigma": float(cfg["model"]["sigma"]), "a": a, "replicate_id": t.rep, "chain_id": d.chain_id.astype(int),
            "original_draw_index": d.draw.astype(np.int64), "S": S, "source_normalization": SPECTRAL_NORMALIZATION,
            "target_hash": spec["target_hash"], "source_run_id": spec["reference_execution_hash"],
            "target_id": t.target_id, "reference_status": ref.get("status")}))
    states = pd.concat(out, ignore_index=True)
    L.tables.mkdir(parents=True, exist_ok=True)
    states.to_csv(L.tables / "spectral_states.csv.gz", index=False)

    # campaign per-target pooled summary, used as an independent cross-check of the recomputed quantiles
    if summ.exists():
        rows.append(_map_row("spectral_summary", summ, root, "csv rows (chain_id='all' pooled)",
                             pd.read_csv(summ).shape, "float64", "per-target quantiles, inspected/exit counts, "
                             "SVD backend cross-check"))

    pv = root / "controls" / "prior_spectral_values.npz"
    pc = root / "controls" / "prior_spectral.csv"
    if not pv.exists():
        raise InputError(f"{pv} missing: regenerate 4,096 iid prior matrices per width (§3.4)")
    z = np.load(pv, allow_pickle=False)
    prior = []
    for m in sorted({t.m for t in deep_targets(cfg)}):
        v = np.asarray(z[f"m{m}"], dtype=np.float64)
        if v.shape != (int(cfg["spectral"]["prior_iid_matrices_per_width"]),):
            raise InputError(f"prior m={m}: {v.shape} values")
        seed = stream_seed(int(cfg["master_seed"]), rep="shared", arch="deep", width=m, chain="shared",
                           stage="prior_spectral", stream="W2_prior")
        prior.append(pd.DataFrame({"m": m, "prior_index": np.arange(v.size), "S": v, "seed": seed,
                                   "normalization": SPECTRAL_NORMALIZATION}))
        rows.append(_map_row("prior_spectral_reference", pv, root, f"m{m}", v.shape, str(v.dtype),
                             f"iid N(0,1) {m}x{m} W2 matrices, S per matrix, shared across replicates"))
    pd.concat(prior, ignore_index=True).to_csv(L.tables / "spectral_prior_states.csv", index=False)
    if pc.exists():
        rows.append(_map_row("prior_spectral_summary", pc, root, "csv rows", pd.read_csv(pc).shape, "float64",
                             "campaign prior quantiles (cross-check)"))
    return rows


def load_loss_npz(path: Path, *, t: C.Target, h: float, window: int, n: int, spec: dict | None) -> dict[str, Any]:
    """Adapter-side metadata checks before the array-only helper sees a trace (guide §5, §9.1)."""
    z = np.load(path, allow_pickle=False)
    V = np.asarray(z["V"], dtype=np.float64)
    chk = []
    if V.shape != (4, window):
        chk.append(f"V shape {V.shape} != (4, {window})")
    if not np.array_equal(z["chain_ids"], np.arange(4)):
        chk.append("chain ids")
    if int(z["save_stride"]) != 1:
        chk.append("save_stride != 1")
    if float(z["h"]) != float(h):
        chk.append(f"h={float(z['h'])}")
    if str(z["architecture"]) != t.arch or int(z["m"]) != t.m or int(z["replicate_id"]) != t.rep or int(z["L"]) != t.L:
        chk.append("architecture/m/replicate/L do not match the file name")
    if int(z["n"]) != n:
        chk.append(f"n={int(z['n'])}")
    if str(z["observable"]) != OBSERVABLE:
        chk.append(f"observable {z['observable']}")
    if not bool(z["contains_rejected_transitions"]):
        chk.append("rejected transitions missing")
    if str(z["selection_rule"]) != selection_rule(window):
        chk.append("selection rule")
    start, end = np.asarray(z["iteration_start"]), np.asarray(z["iteration_end"])
    if np.any(end - start + 1 != window):
        chk.append("iteration range length != window")
    if spec is not None and str(z["target_hash"]) != spec["target_hash"]:
        chk.append("target hash differs from the campaign spec")
    if not np.all(np.isfinite(V)):
        chk.append("non-finite V")
    if chk:
        raise InputError(f"{path.name}: {'; '.join(chk)}")
    return {"V": V, "accepted": np.asarray(z["accepted"]) if "accepted" in z.files else None,
            "iteration_start": start, "iteration_end": end, "target_hash": str(z["target_hash"]),
            "sampler_code_hash": str(z["sampler_code_hash"]), "execution_hash": str(z["execution_hash"]),
            "source_run_ids": [str(s) for s in z["source_run_ids"]],
            "retained_per_chain": int(z["retained_transitions_per_chain"]) if "retained_transitions_per_chain" in z.files else None,
            "sha256": sha256(path)}
