"""Chunked HDF5 traces, atomic checkpoints and resume validation (runbook §4).

A trajectory directory holds one subdirectory per chain. Each segment (burn-in, calibration, production stage,
...) is written as atomically-renamed chunk files while running and consolidated into a single file per chain
and segment when complete. One process writes every file of a trajectory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch


class StorageError(RuntimeError):
    pass


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def clean_json(obj: Any) -> Any:
    """Replace non-finite floats by None recursively (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    return obj


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def atomic_torch_save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ---- trace files ------------------------------------------------------------------------------------
def write_trace_file(path: Path, *, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in arrays.items():
            v = np.asarray(v)
            if v.size == 0:
                f.create_dataset(k, data=v)
            else:
                chunks = (min(v.shape[0], 4096),) + v.shape[1:] if v.ndim >= 1 else None
                if v.ndim == 2 and v.shape[1] > 4096:
                    chunks = (1, v.shape[1])
                f.create_dataset(k, data=v, chunks=chunks)
        for k, v in attrs.items():
            f.attrs[k] = json.dumps(v, default=_json_default) if isinstance(v, (dict, list)) else v
        f.flush()
    os.replace(tmp, path)


def read_trace_file(path: Path, keys: Optional[list[str]] = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with h5py.File(path, "r") as f:
            ks = keys if keys is not None else list(_walk(f))
            arrays = {k: f[k][()] for k in ks if k in f}
            attrs = {k: _attr(v) for k, v in f.attrs.items()}
    except OSError as exc:
        raise StorageError(f"unreadable or truncated trace file {path}: {exc}") from exc
    return arrays, attrs


def _walk(f: h5py.Group, prefix: str = "") -> list[str]:
    out = []
    for k, v in f.items():
        name = f"{prefix}{k}"
        if isinstance(v, h5py.Group):
            out += _walk(v, name + "/")
        else:
            out.append(name)
    return out


def _attr(v: Any) -> Any:
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return v
    if isinstance(v, np.generic):
        return v.item()
    return v


CHUNK_RE = re.compile(r"^(?P<seg>.+)\.chunk(?P<start>\d{9})\.h5$")


def chunk_path(chain_dir: Path, segment: str, start: int) -> Path:
    return chain_dir / f"{segment}.chunk{start:09d}.h5"


def segment_path(chain_dir: Path, segment: str) -> Path:
    return chain_dir / f"{segment}.h5"


def list_chunks(chain_dir: Path, segment: str) -> list[tuple[int, Path]]:
    out = []
    if chain_dir.exists():
        for p in chain_dir.iterdir():
            mt = CHUNK_RE.match(p.name)
            if mt and mt.group("seg") == segment:
                out.append((int(mt.group("start")), p))
    return sorted(out)


def validate_trace(arrays: dict[str, np.ndarray], attrs: dict[str, Any], *, expect_first: int, expect_n: int,
                   hashes: dict[str, str], chain: int, where: str) -> None:
    for k, v in hashes.items():
        if attrs.get(k) != v:
            raise StorageError(f"{where}: {k} mismatch (incompatible trajectory)")
    if int(attrs.get("chain", -1)) != chain:
        raise StorageError(f"{where}: chain id {attrs.get('chain')} != {chain} (incorrect chain concatenation)")
    draw = arrays["draw"]
    if draw.shape[0] != expect_n:
        raise StorageError(f"{where}: {draw.shape[0]} transitions, expected {expect_n}")
    if expect_n and (draw[0] != expect_first or np.any(np.diff(draw) != 1)):
        if len(np.unique(draw)) != len(draw):
            raise StorageError(f"{where}: duplicate draw ids")
        raise StorageError(f"{where}: draw ids not contiguous from {expect_first} (missing or shuffled)")
    for k, v in arrays.items():
        if k.startswith("archive/") or k == "draw":
            continue
        if v.shape[0] != expect_n:
            raise StorageError(f"{where}: {k} has {v.shape[0]} rows, expected {expect_n}")
    if "archive/draw" in arrays:
        ad = arrays["archive/draw"]
        stride = int(attrs.get("archive_stride", 0))
        if stride:
            exp = np.arange(expect_first + (-expect_first) % stride, expect_first + expect_n, stride)
            if not np.array_equal(ad, exp):
                raise StorageError(f"{where}: archive draw ids are not the stride-{stride} schedule")
        for k, v in arrays.items():
            if k.startswith("archive/") and v.shape[0] != ad.shape[0]:
                raise StorageError(f"{where}: {k} count mismatch with archive draw ids")


def concat_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
