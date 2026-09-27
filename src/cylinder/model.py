"""Two-layer tanh BNN, binary CE likelihood, and parameter packing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


SQRT2 = float(np.sqrt(2.0))


def param_dim(m: int, d: int) -> int:
    """p = m * (1 + d); each neuron has scalar head a_j and weight w_j in R^d."""
    return m * (1 + d)


def pack_params(a: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Interleave neurons as [a_0, w_0, a_1, w_1, ..., a_{m-1}, w_{m-1}].
    a: (m,), W: (m, d)
    """
    blocks = np.concatenate(
        [np.asarray(a, dtype=np.float64)[:, None], np.asarray(W, dtype=np.float64)],
        axis=1,
    )
    return blocks.ravel()


def unpack_params(theta: np.ndarray, m: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
    blocks = np.asarray(theta, dtype=np.float64).reshape(m, 1 + d)
    return blocks[:, 0].copy(), blocks[:, 1:].copy()


def head_coefficients(theta: np.ndarray, m: int, d: int) -> np.ndarray:
    return theta[0 :: (1 + d)].copy()


def unpack_torch(
    theta: torch.Tensor, m: int, d: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a (m,), W (m, d) views/copies from a flat θ."""
    blocks = theta.view(m, 1 + d)
    a = blocks[:, 0]
    W = blocks[:, 1:]
    return a, W


def forward_f_torch(
    theta: torch.Tensor,
    X: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """
    f_i = m^{-1/2} sum_j a_j tanh(w_j^T x_i), shape (n,).
    X: (n, d)
    """
    d = X.shape[1]
    a, W = unpack_torch(theta, m, d)
    # preacts: (n, m)
    pre = X @ W.T
    act = torch.tanh(pre)
    return (act @ a) / np.sqrt(m)


def binary_potential_torch(
    theta: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """
    Summed binary CE in the Q = (-1,1)^T/√2 coordinate:

      p_i = sigmoid(√2 f_i)
      V = sum_i [ softplus(√2 f_i) - √2 y_i f_i ]
    """
    f = forward_f_torch(theta, X, m)
    s = SQRT2 * f
    return torch.sum(F.softplus(s) - y * s)


def forward_f(theta: np.ndarray, X: np.ndarray, m: int) -> np.ndarray:
    """NumPy forward (CPU) for tests and theorem checks."""
    d = X.shape[1]
    a, W = unpack_params(theta, m, d)
    pre = X @ W.T
    act = np.tanh(pre)
    return (act @ a) / np.sqrt(m)


def binary_potential(theta: np.ndarray, X: np.ndarray, y: np.ndarray, m: int) -> float:
    f = forward_f(theta, X, m)
    s = SQRT2 * f
    # stable softplus
    softplus = np.where(s > 0, s + np.log1p(np.exp(-s)), np.log1p(np.exp(s)))
    return float(np.sum(softplus - y * s))


def predict_probs_torch(
    theta: torch.Tensor,
    X: torch.Tensor,
    m: int,
) -> torch.Tensor:
    f = forward_f_torch(theta, X, m)
    return torch.sigmoid(SQRT2 * f)


def theta_from_z(
    z: torch.Tensor, theta0: torch.Tensor, sigma: float
) -> torch.Tensor:
    return theta0 + sigma * z


def z_from_theta(
    theta: torch.Tensor, theta0: torch.Tensor, sigma: float
) -> torch.Tensor:
    return (theta - theta0) / sigma
