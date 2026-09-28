"""
Posterior targets for the extension.

Parameters are stored neuron-major as blocks (m, k):  [a_j, w_j]  with k = 1 + width_in.

* ``orth`` (reduced coordinates, addendum §5): because X_n X_nᵀ = I_n, write
  w_j = X_nᵀ z_j + w_{j,⊥}.  The likelihood depends only on (a_j, z_j), the prior on
  z_j is N(X_n w_{j,0}, σ² I_n), and w_⊥ stays an independent Gaussian that integrates
  out exactly.  This is the pilot network with *identity* inputs and hidden weights
  z_j ∈ R^n, so ``X is None`` and pre-activations are the z block itself (O(nm) work).
* ``fmnist``: full coordinates, X is the (n, 32) projected input matrix.

Likelihood (unchanged from the pilot): p_i = sigmoid(√2 f_i),
V = Σ_i softplus(√2 f_i) − √2 y_i f_i,   f_i = m^{-1/2} Σ_j a_j tanh(pre_ij).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..model import SQRT2
from .data import load_npz
from .theory import ext_bundle


@dataclass
class Problem:
    kind: str
    n: int
    m: int
    rep: int
    k: int  # parameters per neuron (1 + n for orth, 1 + d for fmnist)
    X: Optional[torch.Tensor]  # None => identity inputs (reduced coordinates)
    y: torch.Tensor
    theta0: torch.Tensor  # flat, length m*k
    sigma: float
    theory: dict[str, Any]
    hashes: dict[str, str]
    sub_idx: np.ndarray  # fixed training rows used as diagnostic observables
    p_full: int  # parameter count of the original (unreduced) network
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def p(self) -> int:
        return self.m * self.k

    @property
    def B(self) -> float:
        return float(self.theory["B"])

    @property
    def device(self) -> torch.device:
        return self.theta0.device


def preacts(theta: torch.Tensor, prob: Problem) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a (m,) and pre-activations (m, n)."""
    blocks = theta.view(prob.m, prob.k)
    a = blocks[:, 0]
    W = blocks[:, 1:]
    pre = W if prob.X is None else W @ prob.X.T
    return a, pre


def forward_f(theta: torch.Tensor, prob: Problem) -> torch.Tensor:
    a, pre = preacts(theta, prob)
    return (a @ torch.tanh(pre)) / math.sqrt(prob.m)


def potential(theta: torch.Tensor, prob: Problem) -> torch.Tensor:
    s = SQRT2 * forward_f(theta, prob)
    return torch.sum(F.softplus(s) - prob.y * s)


def paired_theta0(a_sign_rows: np.ndarray, m: int, b0: float) -> np.ndarray:
    """(m/2, width) center rows -> flat [(+b0, row_j)]_{j<m/2} ++ [(-b0, row_j)]_{j<m/2}."""
    half = m // 2
    rows = np.asarray(a_sign_rows[:half], dtype=np.float64)
    if rows.shape[0] < half:
        raise ValueError(f"center bank has {rows.shape[0]} rows; need {half} for m={m}")
    heads = np.concatenate([np.full(half, b0), np.full(half, -b0)])[:, None]
    return np.concatenate([heads, np.concatenate([rows, rows], axis=0)], axis=1).ravel()


def data_path(artifacts: Path, kind: str, rep: int) -> Path:
    return artifacts / "data" / f"{kind}_r{rep}.npz"


def build_problem(
    cfg: dict[str, Any],
    kind: str,
    n: int,
    m: int,
    rep: int,
    artifacts: Path,
    device: torch.device,
) -> Problem:
    sigma = float(cfg["prior"]["sigma"])
    b0 = float(cfg["prior"]["b0"])
    C = int(cfg["architecture"]["classes"])
    n_sub = int(cfg["observables"]["n_train_subset"])
    raw = load_npz(data_path(artifacts, kind, rep))
    U = raw["U"]

    if kind == "orth":
        X_n = raw["X_bank"][:n]
        y = raw["y_all"][:n]
        # Projected prior means z_{j,0} = X_n u_j.
        theta0 = paired_theta0(U[: m // 2] @ X_n.T, m, b0)
        X_t = None
        k = 1 + n
        theory = ext_bundle(n, m, sigma=sigma, b0=b0, C=C)
        p_full = m * (1 + raw["X_bank"].shape[1])
        extra = {"d": int(raw["X_bank"].shape[1])}
    elif kind == "fmnist":
        X_np = raw["X"]
        if X_np.shape[0] != n:
            raise ValueError(f"fmnist replicate has n={X_np.shape[0]}, requested {n}")
        y = raw["y"]
        theta0 = paired_theta0(U[: m // 2], m, b0)
        X_t = torch.as_tensor(X_np, dtype=torch.float64, device=device)
        k = 1 + X_np.shape[1]
        M2 = float(raw["M2"])
        M4 = math.sqrt(float(raw["M4_sq_bound"]))
        theory = ext_bundle(n, m, sigma=sigma, b0=b0, C=C, M2=M2, M4=M4)
        theory["D_th_is_safe_upper_envelope"] = True
        p_full = m * k
        extra = {"d": int(X_np.shape[1])}
    else:
        raise ValueError(f"unknown kind {kind!r}")

    return Problem(
        kind=kind,
        n=n,
        m=m,
        rep=rep,
        k=k,
        X=X_t,
        y=torch.as_tensor(y, dtype=torch.float64, device=device),
        theta0=torch.as_tensor(theta0, dtype=torch.float64, device=device),
        sigma=sigma,
        theory=theory,
        hashes={"hash_X": str(raw["hash_X"]), "hash_y": str(raw["hash_y"]), "hash_U": str(raw["hash_U"])},
        sub_idx=np.arange(min(n_sub, n)),
        p_full=p_full,
        extra=extra,
    )
