from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bnn_geometry.model import Layout, Model, prior_center  # noqa: E402

torch.set_default_dtype(torch.float64)
CPU = torch.device("cpu")


def tiny_model(arch: str, m: int = 4, d: int = 3, n: int = 5, sigma: float = 1.0, seed: int = 0,
               center_scale: float = 1.0) -> tuple[Model, np.ndarray]:
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = (g.random(n) < 0.5).astype(float)
    lay = Layout(arch, m, d)
    bank = g.standard_normal((max(m, 8), d)) * center_scale
    th0 = prior_center(lay, bank)
    model = Model(lay=lay, X=torch.as_tensor(X), y=torch.as_tensor(y), theta0=torch.as_tensor(th0), sigma=sigma,
                  X_probe_test=torch.as_tensor(X[:2]))
    return model, bank


@pytest.fixture(params=["shallow", "deep"])
def arch(request):
    return request.param
