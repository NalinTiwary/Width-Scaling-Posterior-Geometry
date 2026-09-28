"""
Extension datasets (addendum §4 and Appendix A).

Controlled (orthogonal), replicate r:
  1. PCG64(1000 + r): G ~ N(0,1)^{256x256}; QR with diag(R) > 0; input bank = rows of Qᵀ.
     The n-point dataset is the first n rows, so X_n X_nᵀ = I_n and datasets are nested.
  2. PCG64(2000 + r): teacher v* ~ N(0, I_256), then all 256 labels once,
     y_i = 1{U_i < sigmoid(2 x_iᵀ v*)} with U_i ~ Uniform(0,1) from the same stream.
  3. PCG64(3000 + r): 8,192 hidden-row centers u_j ~ N(0, I_256) (paired heads ±1).

Real data (Fashion-MNIST classes 0 vs 6, official training split):
  pixels/255 -> center -> top-32 PCA directions (no whitening) -> unit-normalize rows.
  The centering/PCA map is fit once on the full two-class training pool.
  Replicate r: PCG64(4000 + r) draws 128 examples per class and shuffles them;
  PCG64(5000 + r) draws the 8,192 x 32 center bank.
"""

from __future__ import annotations

import gzip
import os
import hashlib
import struct
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ..centers import generate_center_bank
from ..data import _qr_positive_diagonal, sha256_array


def orth_replicate(
    r: int,
    *,
    d: int = 256,
    input_seed_base: int = 1000,
    label_seed_base: int = 2000,
    center_seed_base: int = 3000,
    center_bank_rows: int = 8192,
    teacher_scale: float = 2.0,
) -> dict[str, Any]:
    rng_x = np.random.Generator(np.random.PCG64(input_seed_base + r))
    Q = _qr_positive_diagonal(rng_x.standard_normal(size=(d, d)))
    X_bank = np.ascontiguousarray(Q.T, dtype=np.float64)

    rng_y = np.random.Generator(np.random.PCG64(label_seed_base + r))
    v_star = rng_y.standard_normal(size=d)
    prob = 1.0 / (1.0 + np.exp(-teacher_scale * (X_bank @ v_star)))
    y_all = (rng_y.random(size=d) < prob).astype(np.float64)

    bank = generate_center_bank(n_pairs=center_bank_rows, d=d, seed=center_seed_base + r)
    return {
        "kind": "orth",
        "replicate": r,
        "X_bank": X_bank,
        "y_all": y_all,
        "v_star": v_star,
        "label_prob": prob,
        "U": bank["U"],
        "input_seed": input_seed_base + r,
        "label_seed": label_seed_base + r,
        "center_seed": center_seed_base + r,
        "teacher_scale": teacher_scale,
        "hash_X": sha256_array(X_bank),
        "hash_y": sha256_array(y_all),
        "hash_U": bank["hash_U"],
    }


def class_counts(y: np.ndarray, prefixes: list[int]) -> dict[str, dict[str, int]]:
    return {
        str(n): {"y0": int(np.sum(y[:n] == 0)), "y1": int(np.sum(y[:n] == 1))} for n in prefixes
    }


# ---------------------------------------------------------------- Fashion-MNIST

FMNIST_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
}
FMNIST_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download_fmnist(raw_dir: Path) -> dict[str, Path]:
    """Fetch the official training split (if missing) and verify the published MD5s."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, md5 in FMNIST_FILES.items():
        path = raw_dir / name
        if not path.exists() or _md5(path) != md5:
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(FMNIST_URL + name, tmp)
            tmp.replace(path)
        got = _md5(path)
        if got != md5:
            raise RuntimeError(f"MD5 mismatch for {name}: {got} != {md5}")
        out[name] = path
    return out


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack(">I", data[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def fmnist_pca_map(
    raw_dir: Path,
    classes: tuple[int, int] = (0, 6),
    dim: int = 32,
) -> dict[str, Any]:
    """Fit centering + PCA on the whole official training pool of the two classes."""
    files = download_fmnist(raw_dir)
    images = _read_idx(files["train-images-idx3-ubyte.gz"])
    labels = _read_idx(files["train-labels-idx1-ubyte.gz"])
    keep = np.isin(labels, classes)
    pool_idx = np.flatnonzero(keep)
    Xp = images[keep].reshape(-1, 28 * 28).astype(np.float64) / 255.0
    mean = Xp.mean(axis=0)
    _, svals, Vt = np.linalg.svd(Xp - mean, full_matrices=False)
    comps = Vt[:dim].copy()
    # Deterministic sign: largest-magnitude loading of each component is positive.
    flip = np.sign(comps[np.arange(dim), np.argmax(np.abs(comps), axis=1)])
    comps *= flip[:, None]
    proj = (Xp - mean) @ comps.T
    X_unit = proj / np.linalg.norm(proj, axis=1, keepdims=True)
    var = svals**2
    return {
        "mean": mean,
        "components": comps,
        "explained_variance_ratio": var[:dim] / var.sum(),
        "pool_index": pool_idx,  # indices into the official 60k training split
        "pool_labels": labels[keep].astype(np.int64),
        "pool_X": X_unit,
        "classes": np.asarray(classes),
        "hash_mean": sha256_array(mean),
        "hash_components": sha256_array(comps),
        "raw_md5": dict(FMNIST_FILES),
    }


def fmnist_replicate(
    r: int,
    pca: dict[str, Any],
    *,
    n_per_class: int = 128,
    subset_seed_base: int = 4000,
    center_seed_base: int = 5000,
    center_bank_rows: int = 8192,
) -> dict[str, Any]:
    c0, c1 = (int(c) for c in pca["classes"])
    rng = np.random.Generator(np.random.PCG64(subset_seed_base + r))
    labels = np.asarray(pca["pool_labels"])
    pick0 = rng.choice(np.flatnonzero(labels == c0), size=n_per_class, replace=False)
    pick1 = rng.choice(np.flatnonzero(labels == c1), size=n_per_class, replace=False)
    rows = rng.permutation(np.concatenate([pick0, pick1]))
    X = np.ascontiguousarray(pca["pool_X"][rows], dtype=np.float64)
    y = (labels[rows] == c1).astype(np.float64)
    d = X.shape[1]
    bank = generate_center_bank(n_pairs=center_bank_rows, d=d, seed=center_seed_base + r)
    M2 = float(np.linalg.norm(X, 2))
    max_row = float(np.max(np.linalg.norm(X, axis=1)))
    return {
        "kind": "fmnist",
        "replicate": r,
        "X": X,
        "y": y,
        "official_train_index": np.asarray(pca["pool_index"])[rows],
        "U": bank["U"],
        "subset_seed": subset_seed_base + r,
        "center_seed": center_seed_base + r,
        "M2": M2,
        # ‖X‖²_{2→4} ≤ max_i‖x_i‖ · ‖X‖_{2→2}  (safe, possibly loose)
        "M4_sq_bound": max_row * M2,
        "hash_X": sha256_array(X),
        "hash_y": sha256_array(y),
        "hash_U": bank["hash_U"],
    }


def save_npz(path: Path, obj: dict[str, Any]) -> None:
    arrays = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            continue
        arrays[k] = np.asarray(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_npz(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        v = z[k]
        out[k] = v.item() if v.shape == () else v
    return out
