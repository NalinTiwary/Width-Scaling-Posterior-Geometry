"""Per-draw observables recorded at every retained update (§4.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model import (
    SQRT2,
    forward_f_torch,
    head_coefficients,
    predict_probs_torch,
    theta_from_z,
    unpack_torch,
)


def compute_observables_torch_full(
    z: torch.Tensor,
    theta0: torch.Tensor,
    sigma: float,
    X: torch.Tensor,
    y: torch.Tensor,
    X_probe: torch.Tensor,
    U_proj: torch.Tensor,
    m: int,
    B_m: float,
) -> dict[str, Any]:
    d = X.shape[1]
    p = z.numel()
    theta = theta_from_z(z, theta0, sigma)
    a, _W = unpack_torch(theta, m, d)
    H = float(torch.max(torch.abs(a)).item())
    f_train = forward_f_torch(theta, X, m)
    s = SQRT2 * f_train
    V = float(torch.sum(torch.nn.functional.softplus(s) - y * s).item())
    p_probe = predict_probs_torch(theta, X_probe, m)
    z_sq_over_p = float((z * z).sum().item() / p)
    a0 = theta0[0 :: (1 + d)]
    z_head = (a - a0) / sigma
    mean_sq_head = float((z_head * z_head).mean().item())
    projections = (U_proj @ z).detach().cpu().numpy().astype(np.float64)

    return {
        "H": H,
        "H_over_B": H / B_m,
        "inside": int(H <= B_m),
        "V": V,
        "f_train": f_train.detach().cpu().numpy().astype(np.float64),
        "p_probe": p_probe.detach().cpu().numpy().astype(np.float64),
        "z_sq_over_p": z_sq_over_p,
        "mean_sq_head": mean_sq_head,
        "projections": projections,
    }


def empty_observable_buffers(
    n_chains: int,
    n_draws: int,
    n_train: int,
    n_probes: int,
    n_proj: int,
) -> dict[str, np.ndarray]:
    return {
        "H": np.full((n_chains, n_draws), np.nan, dtype=np.float64),
        "H_over_B": np.full((n_chains, n_draws), np.nan, dtype=np.float64),
        "inside": np.full((n_chains, n_draws), -1, dtype=np.int8),
        "V": np.full((n_chains, n_draws), np.nan, dtype=np.float64),
        "f_train": np.full((n_chains, n_draws, n_train), np.nan, dtype=np.float64),
        "p_probe": np.full((n_chains, n_draws, n_probes), np.nan, dtype=np.float64),
        "z_sq_over_p": np.full((n_chains, n_draws), np.nan, dtype=np.float64),
        "mean_sq_head": np.full((n_chains, n_draws), np.nan, dtype=np.float64),
        "projections": np.full((n_chains, n_draws, n_proj), np.nan, dtype=np.float64),
    }


def fill_draw(
    buf: dict[str, np.ndarray],
    chain: int,
    draw: int,
    obs: dict[str, Any],
) -> None:
    buf["H"][chain, draw] = obs["H"]
    buf["H_over_B"][chain, draw] = obs["H_over_B"]
    buf["inside"][chain, draw] = obs["inside"]
    buf["V"][chain, draw] = obs["V"]
    buf["f_train"][chain, draw] = obs["f_train"]
    buf["p_probe"][chain, draw] = obs["p_probe"]
    buf["z_sq_over_p"][chain, draw] = obs["z_sq_over_p"]
    buf["mean_sq_head"][chain, draw] = obs["mean_sq_head"]
    buf["projections"][chain, draw] = obs["projections"]


def head_max_from_theta_np(theta: np.ndarray, m: int, d: int) -> float:
    return float(np.max(np.abs(head_coefficients(theta, m, d))))


def curvature_state_indices(T: int, n_per_chain: int = 32) -> np.ndarray:
    """Evenly spaced indices: t_j = floor((j + 1/2) T / 32), j=0..31."""
    js = np.arange(n_per_chain, dtype=np.float64)
    return np.floor((js + 0.5) * T / n_per_chain).astype(np.int64)
