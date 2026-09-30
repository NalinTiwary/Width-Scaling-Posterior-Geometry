"""Teacher datasets and first-layer prior-center banks, generated once per replicate (runbook §2.2, §2.4)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from .randomness import numpy_rng, stream_seed


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _seed(cfg: dict[str, Any], rep: int, stage: str, stream: str) -> int:
    return stream_seed(int(cfg["master_seed"]), rep=rep, arch="shared", width="shared", chain="shared",
                       stage=stage, stream=stream, schema=int(cfg["schema_version"]))


def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    out = np.empty((n, d))
    for i in range(n):
        while True:
            z = rng.standard_normal(d)
            nz = np.linalg.norm(z)
            if nz != 0.0:
                out[i] = z / nz
                break
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def make_replicate_data(cfg: dict[str, Any], rep: int) -> dict[str, np.ndarray]:
    dc = cfg["data"]
    n, nt, d = int(dc["train_size"]), int(dc["test_size"]), int(dc["input_dimension"])
    k = float(dc["teacher_log_odds_multiplier"])
    v = numpy_rng(_seed(cfg, rep, "data", "teacher")).standard_normal(d)
    X = _unit_rows(numpy_rng(_seed(cfg, rep, "data", "train_inputs")), n, d)
    p = _sigmoid(k * X @ v)
    y = (numpy_rng(_seed(cfg, rep, "data", "train_labels")).random(n) < p).astype(np.float64)
    Xt = _unit_rows(numpy_rng(_seed(cfg, rep, "data", "test_inputs")), nt, d)
    pt = _sigmoid(k * Xt @ v)
    yt = (numpy_rng(_seed(cfg, rep, "data", "test_labels")).random(nt) < pt).astype(np.float64)
    return {"X": X, "y": y, "p_teacher": p, "X_test": Xt, "y_test": yt, "p_teacher_test": pt, "teacher": v}


def make_center_bank(cfg: dict[str, Any], rep: int) -> np.ndarray:
    rows, d = int(cfg["prior_centers"]["row_bank_size"]), int(cfg["data"]["input_dimension"])
    return numpy_rng(_seed(cfg, rep, "centers", "center_bank")).standard_normal((rows, d))


def data_summary(D: dict[str, np.ndarray]) -> dict[str, Any]:
    X = D["X"]
    norms = np.linalg.norm(X, axis=1)
    ev = np.linalg.eigvalsh(X @ X.T)[::-1]
    tol = ev[0] * X.shape[0] * np.finfo(float).eps
    return {
        "n": int(X.shape[0]), "d": int(X.shape[1]), "n_test": int(D["X_test"].shape[0]),
        "class_counts": {"y0": int((D["y"] == 0).sum()), "y1": int((D["y"] == 1).sum())},
        "test_class_counts": {"y0": int((D["y_test"] == 0).sum()), "y1": int((D["y_test"] == 1).sum())},
        "X_op_norm": float(np.linalg.norm(X, 2)),
        "input_norm_min": float(norms.min()), "input_norm_max": float(norms.max()),
        "XXT_nonzero_eigenvalues": [float(e) for e in ev if e > tol],
        "rank": int((ev > tol).sum()),
        "teacher_norm": float(np.linalg.norm(D["teacher"])),
    }


def data_hashes(D: dict[str, np.ndarray], bank: np.ndarray) -> dict[str, str]:
    out = {f"hash_{k}": sha256_array(v) for k, v in sorted(D.items())}
    out["hash_center_bank"] = sha256_array(bank)
    return out


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def freeze_replicate(cfg: dict[str, Any], root: Path, rep: int) -> dict[str, Any]:
    """Write data/rep_r.npz and centers/rep_r.npz once; verify byte-identical regeneration afterwards."""
    D = make_replicate_data(cfg, rep)
    bank = make_center_bank(cfg, rep)
    dp, cp = root / "data" / f"rep_{rep}.npz", root / "centers" / f"rep_{rep}.npz"
    hashes = data_hashes(D, bank)
    for path, arrays in ((dp, D), (cp, {"W1_bank": bank})):
        if path.exists():
            old = dict(np.load(path))
            if set(old) != set(arrays) or any(sha256_array(old[k]) != sha256_array(arrays[k]) for k in arrays):
                raise RuntimeError(f"{path} exists but differs from its deterministic regeneration")
        else:
            _atomic_savez(path, **arrays)
    return {"rep": rep, "summary": data_summary(D), "hashes": hashes}


def load_replicate(root: Path, rep: int) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, str]]:
    D = dict(np.load(root / "data" / f"rep_{rep}.npz"))
    bank = np.load(root / "centers" / f"rep_{rep}.npz")["W1_bank"]
    return D, bank, data_hashes(D, bank)
