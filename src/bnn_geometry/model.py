"""Pure forward pass, likelihood potential V and its gradient, parameter packing (runbook §2.3, §3.1).

Parameters are flattened in the immutable order A (1×m), W1 (m×d), W2 (m×m, deep only), row-major within
blocks. All functions are batched over a leading chain dimension C; chains never mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Layout:
    arch: str
    m: int
    d: int

    @property
    def blocks(self) -> list[tuple[str, tuple[int, ...]]]:
        b = [("A", (1, self.m)), ("W1", (self.m, self.d))]
        if self.arch == "deep":
            b.append(("W2", (self.m, self.m)))
        return b

    @property
    def offsets(self) -> dict[str, tuple[int, int]]:
        out, o = {}, 0
        for name, shape in self.blocks:
            size = int(np.prod(shape))
            out[name] = (o, o + size)
            o += size
        return out

    @property
    def p(self) -> int:
        return sum(int(np.prod(s)) for _, s in self.blocks)

    def to_json(self) -> dict[str, Any]:
        return {"arch": self.arch, "m": self.m, "d": self.d, "p": self.p,
                "blocks": [{"name": n, "shape": list(s), "offset": list(self.offsets[n])} for n, s in self.blocks],
                "order": "A,W1,W2 row-major", "schema_version": 1}


def unpack(theta: torch.Tensor, lay: Layout) -> dict[str, torch.Tensor]:
    """theta (C, p) -> views A (C, m), W1 (C, m, d), W2 (C, m, m)."""
    C = theta.shape[0]
    out = {}
    for name, shape in lay.blocks:
        a, b = lay.offsets[name]
        blk = theta[:, a:b].reshape(C, *shape)
        out[name] = blk[:, 0, :] if name == "A" else blk
    return out


def pack(blocks: dict[str, torch.Tensor], lay: Layout) -> torch.Tensor:
    parts = []
    for name, _ in lay.blocks:
        t = blocks[name]
        parts.append(t.reshape(t.shape[0], -1))
    return torch.cat(parts, dim=1)


def prior_center(lay: Layout, bank: np.ndarray) -> np.ndarray:
    m = lay.m
    A0 = ((-1.0) ** np.arange(m)) / math.sqrt(m)
    parts = [A0, np.asarray(bank[:m], dtype=np.float64).reshape(-1)]
    if lay.arch == "deep":
        parts.append(np.zeros(m * m))
    return np.concatenate(parts)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


@dataclass
class Model:
    """A posterior target's likelihood on a device: data, center, sigma and layout."""

    lay: Layout
    X: torch.Tensor          # (n, d)
    y: torch.Tensor          # (n,)
    theta0: torch.Tensor     # (p,)
    sigma: float
    X_probe_test: Optional[torch.Tensor] = None   # (k, d) held-out inputs whose probabilities are recorded
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.X.device

    @property
    def m(self) -> int:
        return self.lay.m

    # ---- forward -------------------------------------------------------------------------------
    def hidden(self, theta: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, ...]:
        blk = unpack(theta, self.lay)
        H1 = torch.tanh(torch.matmul(X, blk["W1"].transpose(1, 2)))          # (C, n, m)
        if self.lay.arch == "shallow":
            return (H1,)
        H2 = torch.tanh(torch.matmul(H1, blk["W2"].transpose(1, 2)) / math.sqrt(self.m))
        return H1, H2

    def logits(self, theta: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        X = self.X if X is None else X
        H = self.hidden(theta, X)[-1]
        A = unpack(theta, self.lay)["A"]
        return torch.matmul(H, A.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.m)   # (C, n)

    def V_from_f(self, f: torch.Tensor) -> torch.Tensor:
        z = SQRT2 * f
        return (softplus(z) - self.y * z).sum(dim=-1)

    def V(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.logits(theta)
        return self.V_from_f(f), f

    def V_and_grad(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analytic backpropagation; returns V (C,), grad V (C, p) in original coordinates, logits (C, n)."""
        m = self.m
        rm = 1.0 / math.sqrt(m)
        blk = unpack(theta, self.lay)
        A, W1 = blk["A"], blk["W1"]
        H1 = torch.tanh(torch.matmul(self.X, W1.transpose(1, 2)))
        if self.lay.arch == "deep":
            W2 = blk["W2"]
            H2 = torch.tanh(torch.matmul(H1, W2.transpose(1, 2)) * rm)
            H = H2
        else:
            H = H1
        f = torch.matmul(H, A.unsqueeze(-1)).squeeze(-1) * rm
        z = SQRT2 * f
        V = (softplus(z) - self.y * z).sum(dim=-1)
        r = SQRT2 * (torch.sigmoid(z) - self.y)                                # dV/df (C, n)
        gA = torch.matmul(r.unsqueeze(1), H).squeeze(1) * rm                  # (C, m)
        dH = r.unsqueeze(-1) * A.unsqueeze(1) * rm                            # (C, n, m)
        grads = {"A": gA}
        if self.lay.arch == "deep":
            dP2 = dH * (1.0 - H2 * H2)
            grads["W2"] = torch.matmul(dP2.transpose(1, 2), H1) * rm          # (C, m, m)
            dH1 = torch.matmul(dP2, W2) * rm
        else:
            dH1 = dH
        dP1 = dH1 * (1.0 - H1 * H1)
        grads["W1"] = torch.matmul(dP1.transpose(1, 2), self.X)              # (C, m, d)
        return V, pack(grads, self.lay), f

    def prior_quadratic(self, theta: torch.Tensor) -> torch.Tensor:
        x = theta - self.theta0
        return (x * x).sum(dim=-1) / (2.0 * self.sigma**2)

    def U(self, theta: torch.Tensor) -> torch.Tensor:
        return self.V(theta)[0] + self.prior_quadratic(theta)


def build_model(lay: Layout, D: dict[str, np.ndarray], bank: np.ndarray, sigma: float, test_idx: list[int],
                device: torch.device) -> Model:
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)  # noqa: E731
    return Model(lay=lay, X=t(D["X"]), y=t(D["y"]), theta0=t(prior_center(lay, bank)), sigma=float(sigma),
                 X_probe_test=t(D["X_test"][list(test_idx)]))


# ---- literal reference implementation for tests (runbook §6.1 A) ------------------------------------
def loop_logits(theta: np.ndarray, lay: Layout, X: np.ndarray) -> np.ndarray:
    m, d = lay.m, lay.d
    o = lay.offsets
    A = theta[o["A"][0]:o["A"][1]]
    W1 = theta[o["W1"][0]:o["W1"][1]].reshape(m, d)
    W2 = theta[o["W2"][0]:o["W2"][1]].reshape(m, m) if lay.arch == "deep" else None
    f = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        h1 = [math.tanh(sum(W1[j, k] * X[i, k] for k in range(d))) for j in range(m)]
        if W2 is not None:
            h = [math.tanh(sum(W2[j, k] * h1[k] for k in range(m)) / math.sqrt(m)) for j in range(m)]
        else:
            h = h1
        f[i] = sum(A[j] * h[j] for j in range(m)) / math.sqrt(m)
    return f


def loop_V(theta: np.ndarray, lay: Layout, X: np.ndarray, y: np.ndarray) -> float:
    f = loop_logits(theta, lay, X)
    out = 0.0
    for fi, yi in zip(f, y):
        z = SQRT2 * fi
        out += math.log1p(math.exp(-abs(z))) + max(z, 0.0) - yi * z
    return out
