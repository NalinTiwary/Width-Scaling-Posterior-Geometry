"""Paired shallow designs with chi exactly 1, 4 and 13 at fixed labels.

Per replicate, one orthonormal frame and one latent teacher are shared across r.
Training row i is sqrt(1-r) q_i + sqrt(r) s_i q_0 with s_i = (-1)^i.
The teacher direction v(r) = Q b / sqrt(1-r) keeps v(r)^T x_i(r) = b_i.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..randomness import numpy_rng, stream_seed
from .gram import gram_spectrum


def _qr_positive(G: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(G, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    return Q * signs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def probe_train_indices(n: int, count: int = 8) -> list[int]:
    return [int(math.floor(j * n / count)) for j in range(count)]


def geometry_seed(master: int, rep: int, stream: str) -> int:
    return stream_seed(master, rep=rep, arch="geometry", width="shared", chain="shared",
                       stage="revision_geometry", stream=stream, schema=1)


def draw_latents(master: int, rep: int, n: int, d: int, n_test: int) -> dict[str, np.ndarray]:
    if n + 1 > d:
        raise ValueError(f"need d >= n+1 to place the frame, got n={n} d={d}")
    frame = numpy_rng(geometry_seed(master, rep, "frame")).standard_normal((d, n + 1))
    Qfull = _qr_positive(frame)
    b = numpy_rng(geometry_seed(master, rep, "teacher")).standard_normal(n)
    uniforms = numpy_rng(geometry_seed(master, rep, "labels")).random(n)
    Z = numpy_rng(geometry_seed(master, rep, "test_latent")).standard_normal((n_test, n))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    s_test = np.where(numpy_rng(geometry_seed(master, rep, "test_signs")).random(n_test) < 0.5, -1.0, 1.0)
    return {"q0": Qfull[:, 0], "Q": Qfull[:, 1:], "b": b, "uniforms": uniforms, "Z": Z, "s_test": s_test}


def design_at_r(latents: dict[str, np.ndarray], r: float, *, log_odds: float = 2.0) -> dict[str, np.ndarray]:
    if not 0.0 <= r < 1.0:
        raise ValueError("r must lie in [0, 1)")
    q0, Q, b = latents["q0"], latents["Q"], latents["b"]
    n = b.shape[0]
    s = np.array([(-1.0) ** i for i in range(n)])
    scale_q = math.sqrt(1.0 - r)
    scale_0 = math.sqrt(r)
    X = scale_q * Q.T + scale_0 * s[:, None] * q0[None, :]
    X_test = scale_q * (latents["Z"] @ Q.T) + scale_0 * latents["s_test"][:, None] * q0[None, :]
    p = _sigmoid(log_odds * b)
    y = (latents["uniforms"] < p).astype(np.float64)
    p_test = _sigmoid(log_odds * (latents["Z"] @ b))
    v = (Q @ b) / scale_q
    return {
        "X": X, "y": y, "p_teacher": p, "X_test": X_test, "p_teacher_test": p_test,
        "teacher": v, "b": b, "r": np.array([r]),
    }


def assert_design(latents: dict[str, np.ndarray], r: float, data: dict[str, np.ndarray], *,
                  chi_atol: float = 1e-10, norm_atol: float = 1e-12, label_atol: float = 1e-12) -> dict[str, Any]:
    X, b = data["X"], latents["b"]
    n = b.shape[0]
    spec = gram_spectrum(X)
    chi = 1.0 + (n - 1) * r
    if abs(spec["chi"] - chi) > chi_atol:
        raise AssertionError(f"chi {spec['chi']} != {chi}")
    if abs(spec["row_norm_max"] - 1.0) > norm_atol or abs(spec["row_norm_min"] - 1.0) > norm_atol:
        raise AssertionError(f"row norms not one: {spec['row_norm_min']} {spec['row_norm_max']}")
    scores = X @ data["teacher"]
    if np.max(np.abs(scores - b)) > label_atol:
        raise AssertionError("training teacher scores are not the shared latent b")
    s = np.array([(-1.0) ** i for i in range(n)])
    gram = (1.0 - r) * np.eye(n) + r * np.outer(s, s)
    if np.max(np.abs(X @ X.T - gram)) > 1e-9:
        raise AssertionError("row Gram is not (1-r)I + r ss^T")
    test_scores = data["X_test"] @ data["teacher"]
    expected = latents["Z"] @ b
    if np.max(np.abs(test_scores - expected)) > 1e-9:
        raise AssertionError("held-out teacher scores depend on r")
    return {"chi": spec["chi"], "chi_exact": chi, "row_norm_max": spec["row_norm_max"],
            "class_balance": float(data["y"].mean()), "n": n, "d": int(X.shape[1]), "r": r}


def paired_designs(master: int, rep: int, radii: list[float], *, n: int = 16, d: int = 32,
                   n_test: int = 64) -> tuple[dict[str, np.ndarray], dict[float, dict[str, np.ndarray]], list[dict[str, Any]]]:
    latents = draw_latents(master, rep, n, d, n_test)
    out, checks = {}, []
    labels = None
    for r in radii:
        data = design_at_r(latents, float(r))
        checks.append(assert_design(latents, float(r), data))
        if labels is None:
            labels = data["y"].copy()
        elif not np.array_equal(labels, data["y"]):
            raise AssertionError("labels changed with r")
        probs = _sigmoid(2.0 * latents["b"])
        if np.max(np.abs(data["p_teacher"] - probs)) > 1e-12:
            raise AssertionError("training probabilities changed with r")
        out[float(r)] = data
    return latents, out, checks
