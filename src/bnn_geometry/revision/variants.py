"""One-factor changes of a frozen target. Each variant keeps the other factors fixed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import sha256_array
from ..randomness import numpy_rng, stream_seed


def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    out = np.empty((n, d))
    for i in range(n):
        z = rng.standard_normal(d)
        nz = np.linalg.norm(z)
        while nz == 0.0:
            z = rng.standard_normal(d)
            nz = np.linalg.norm(z)
        out[i] = z / nz
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _seed(master: int, rep: int, stream: str) -> int:
    return stream_seed(master, rep=rep, arch="variant", width="shared", chain="shared",
                       stage="revision_variant", stream=stream, schema=1)


def add_training_rows(D: dict[str, np.ndarray], n_new: int, master: int, rep: int, *,
                      log_odds: float = 2.0) -> dict[str, np.ndarray]:
    """Append independent rows from the same teacher. The original prefix is unchanged."""
    out = {k: np.array(v, copy=True) for k, v in D.items()}
    d = out["X"].shape[1]
    rng_x = numpy_rng(_seed(master, rep, "extra_inputs"))
    rng_y = numpy_rng(_seed(master, rep, "extra_labels"))
    Xn = _unit_rows(rng_x, n_new, d)
    teacher = np.asarray(out["teacher"], dtype=np.float64)
    p = _sigmoid(log_odds * Xn @ teacher)
    y = (rng_y.random(n_new) < p).astype(np.float64)
    out["X"] = np.concatenate([out["X"], Xn], axis=0)
    out["y"] = np.concatenate([out["y"], y], axis=0)
    out["p_teacher"] = np.concatenate([out["p_teacher"], p], axis=0)
    return out


def odd_score(X: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """directions[0] is v and directions[1:4] are the three product directions."""
    linear = X @ directions[0]
    prod = np.ones(X.shape[0])
    for j in range(1, 4):
        prod *= X @ directions[j]
    return linear + 0.75 * prod


def nonlinear_labels(D: dict[str, np.ndarray], master: int, rep: int, *, n_cal: int = 4096) -> dict[str, Any]:
    """Keep X and replace labels by an odd teacher whose log-odds RMS matches the baseline on fresh inputs."""
    X = np.asarray(D["X"], dtype=np.float64)
    teacher = np.asarray(D["teacher"], dtype=np.float64)
    baseline = 2.0 * (X @ teacher)
    target_rms = float(np.sqrt(np.mean(baseline ** 2)))
    rng = numpy_rng(_seed(master, rep, "nonlinear_directions"))
    directions = rng.standard_normal((4, X.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cal = _unit_rows(numpy_rng(_seed(master, rep, "nonlinear_calibration")), n_cal, X.shape[1])
    raw = odd_score(cal, directions)
    rms = float(np.sqrt(np.mean(raw ** 2)))
    if rms == 0.0:
        raise RuntimeError("nonlinear calibration score has zero RMS")
    multiplier = target_rms / rms
    log_odds = multiplier * odd_score(X, directions)
    uniforms = numpy_rng(_seed(master, rep, "nonlinear_labels")).random(X.shape[0])
    p = _sigmoid(log_odds)
    y = (uniforms < p).astype(np.float64)
    out = {k: np.array(v, copy=True) for k, v in D.items()}
    out["y"] = y
    out["p_teacher"] = p
    out["teacher"] = directions[0]
    return {"data": out, "multiplier": multiplier, "target_rms": target_rms, "calibration_rms": rms,
            "directions_hash": sha256_array(directions), "class_fraction": float(y.mean())}


def load_wisconsin(path: Path, seed: int, *, n_train: int = 128, split: int = 0) -> dict[str, Any]:
    """UCI Wisconsin Diagnostic. Standardize columns on the training split only, then normalize rows.

    The file may be the headerless UCI `wdbc.data` (id, diagnosis, 30 features) or a CSV with a header.
    Three seeded stratified splits are split=0,1,2. Overlapping splits are a sensitivity check, not
    independent datasets.
    """
    frame = pd.read_csv(path, header=None if _headerless(path) else 0)
    if frame.shape[1] < 12:
        raise ValueError(f"{path} does not look like the Wisconsin diagnostic table")
    diagnosis = frame.iloc[:, 1].astype(str).str.upper().str[0]
    features = frame.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any() or not set(diagnosis.unique()).issubset({"M", "B"}):
        raise ValueError(f"{path} has non-numeric features or labels other than M/B")
    y_all = (diagnosis == "M").astype(float).to_numpy()
    X_all = features.to_numpy(dtype=np.float64)
    rng = numpy_rng(stream_seed(seed, rep=split, arch="uci", width="shared", chain="shared",
                                stage="revision_uci", stream="split", schema=1))
    train_idx = _stratified_indices(y_all, n_train, rng)
    test_idx = np.array([i for i in range(len(y_all)) if i not in set(train_idx)])
    X_train, y = X_all[train_idx], y_all[train_idx]
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    def transform(X):
        Z = (X - mu) / sd
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return Z / norms
    return {"X": transform(X_train), "y": y, "X_test": transform(X_all[test_idx]),
            "y_test": y_all[test_idx], "train_index": train_idx, "class_fraction": float(y.mean()),
            "split": split}


def _headerless(path: Path) -> bool:
    with path.open() as f:
        first = f.readline()
    try:
        float(first.split(",")[0])
        return True
    except ValueError:
        return False


def _stratified_indices(y: np.ndarray, n_train: int, rng: np.random.Generator) -> np.ndarray:
    if n_train >= len(y):
        raise ValueError("training split must leave held-out rows")
    chosen = []
    for label in (0.0, 1.0):
        pool = np.flatnonzero(y == label)
        take = int(round(n_train * (len(pool) / len(y))))
        take = min(max(take, 1), len(pool) - 1)
        chosen.append(rng.choice(pool, size=take, replace=False))
    idx = np.concatenate(chosen)
    if len(idx) > n_train:
        idx = rng.choice(idx, size=n_train, replace=False)
    elif len(idx) < n_train:
        rest = np.array([i for i in range(len(y)) if i not in set(idx)])
        idx = np.concatenate([idx, rng.choice(rest, size=n_train - len(idx), replace=False)])
    return np.sort(idx)


def admissible_cutoff(sigma: float, *, slack: float = 0.10, r0: float = 0.0) -> float:
    """Fixed absolute slack above the Gaussian edge r0 + 2 sigma."""
    return r0 + 2.0 * sigma + slack
