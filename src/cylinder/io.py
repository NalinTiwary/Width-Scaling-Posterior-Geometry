"""Atomic checkpoints, metadata, and resume validation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

METADATA_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Atomically write a compressed npz. Temp path must end in .npz (NumPy quirk)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        # Pass path without relying on NumPy's auto-.npz append on open handles.
        np.savez_compressed(str(tmp_path), **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists() and tmp_path != path:
            tmp_path.unlink(missing_ok=True)


def atomic_save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".json", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists() and tmp_path != path:
            tmp_path.unlink(missing_ok=True)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_dir_name(m: int, seed: int, kernel: str = "ess") -> str:
    return f"m{m}_seed{seed}_{kernel}"


def validate_resume(
    meta: dict[str, Any],
    expected: dict[str, Any],
    keys: list[str] | None = None,
) -> None:
    """
    Fail if a resumed run's identity keys disagree with the live target.
    """
    if keys is None:
        keys = [
            "schema_version",
            "m",
            "center_seed",
            "sigma",
            "data_hash_X",
            "data_hash_y",
            "centers_hash_U",
            "loss_normalization",
            "kernel",
            "B_m",
        ]
    mismatches = []
    for k in keys:
        if k not in expected:
            continue
        if meta.get(k) != expected.get(k):
            mismatches.append((k, meta.get(k), expected.get(k)))
    if mismatches:
        msg = "; ".join(f"{k}: stored={a!r} live={b!r}" for k, a, b in mismatches)
        raise RuntimeError(f"Resume validation failed: {msg}")


def build_run_metadata(
    *,
    m: int,
    center_seed: int,
    sigma: float,
    B_m: float,
    D_th: float,
    kernel: str,
    data_hashes: dict[str, str],
    centers_hash: str,
    device_info: dict[str, Any],
    n_chains: int,
    burnin: int,
    n_retained: int,
    likelihood_evals: int,
    wall_time_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "m": m,
        "p": m * 33,  # d=32 fixed in protocol
        "center_seed": center_seed,
        "sigma": sigma,
        "B_m": B_m,
        "D_th": D_th,
        "kernel": kernel,
        "loss_normalization": "sum",
        "activation": "tanh",
        "architecture": "one_hidden_layer",
        "data_hash_X": data_hashes.get("hash_X"),
        "data_hash_y": data_hashes.get("hash_y"),
        "data_hash_X_probe": data_hashes.get("hash_X_probe"),
        "centers_hash_U": centers_hash,
        "n_chains": n_chains,
        "burnin": burnin,
        "n_retained": n_retained,
        "likelihood_evals": likelihood_evals,
        "wall_time_s": wall_time_s,
        **device_info,
    }
    if extra:
        meta.update(extra)
    return meta
