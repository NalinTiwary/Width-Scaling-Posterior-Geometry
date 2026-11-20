"""Pure forward pass, likelihood potential V and its gradient, parameter packing (runbook §2.3, §3.1).

Parameters are flattened in the immutable order A (1×m), W1 (m×d), W2 (m×m, deep only), row-major within
blocks. All functions are batched over a leading chain dimension C; chains never mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

ACTIVATIONS = ("tanh", "relu", "sin")

import numpy as np
import torch

SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Layout:
    arch: str
    m: int
    d: int
    weight_layers: Optional[int] = None

    @property
    def n_layers(self) -> int:
        """Number of weight matrices, counting the scalar head. Shallow is 2, the campaign deep net is 3."""
        if self.weight_layers is not None:
            return int(self.weight_layers)
        if self.arch == "deep":
            return 3
        if self.arch == "shallow":
            return 2
        raise ValueError(f"unknown architecture {self.arch}")

    @property
    def blocks(self) -> list[tuple[str, tuple[int, ...]]]:
        b = [("A", (1, self.m)), ("W1", (self.m, self.d))]
        for ell in range(2, self.n_layers):
            b.append((f"W{ell}", (self.m, self.m)))
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
        return {"arch": self.arch, "m": self.m, "d": self.d, "p": self.p, "weight_layers": self.n_layers,
                "blocks": [{"name": n, "shape": list(s), "offset": list(self.offsets[n])} for n, s in self.blocks],
                "order": "A,W1,W2,... row-major", "schema_version": 1}


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
    for _ in range(2, lay.n_layers):
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
    activation: str = "tanh"
    likelihood_scale: float = 1.0

    @property
    def device(self) -> torch.device:
        return self.X.device

    @property
    def m(self) -> int:
        return self.lay.m

    def standard_architecture(self) -> bool:
        """Campaign tanh network: two weight layers if shallow, three if deep."""
        expected = {"shallow": 2, "deep": 3}.get(self.lay.arch)
        return expected is not None and self.lay.n_layers == expected and self.activation == "tanh"

    # ---- forward -------------------------------------------------------------------------------
    def hidden(self, theta: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not self.standard_architecture():
            return tuple(h for h, _ in self._general_layers(theta, X))
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
        v = (softplus(z) - self.y * z).sum(dim=-1)
        if self.likelihood_scale != 1.0:
            v = v * float(self.likelihood_scale)
        return v

    def V(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.logits(theta)
        return self.V_from_f(f), f

    def V_and_grad(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analytic backpropagation; returns V (C,), grad V (C, p) in original coordinates, logits (C, n)."""
        if not self.standard_architecture():
            return self._general_V_and_grad(theta)
        V, g, f = self._legacy_V_and_grad(theta)
        if self.likelihood_scale != 1.0:
            s = float(self.likelihood_scale)
            return s * V, s * g, f
        return V, g, f

    def _legacy_V_and_grad(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _general_layers(self, theta: torch.Tensor, X: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Hidden preactivations and activations, first layer unscaled and later layers scaled by 1/sqrt(m)."""
        blk = unpack(theta, self.lay)
        rm = 1.0 / math.sqrt(self.m)
        z = torch.matmul(X, blk["W1"].transpose(1, 2))
        h = apply_activation(self.activation, z)
        layers = [(h, z)]
        for ell in range(2, self.lay.n_layers):
            z = torch.matmul(h, blk[f"W{ell}"].transpose(1, 2)) * rm
            h = apply_activation(self.activation, z)
            layers.append((h, z))
        return layers

    def _general_V_and_grad(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.m
        rm = 1.0 / math.sqrt(m)
        blk = unpack(theta, self.lay)
        layers = self._general_layers(theta, self.X)
        H = layers[-1][0]
        A = blk["A"]
        f = torch.matmul(H, A.unsqueeze(-1)).squeeze(-1) * rm
        V = self.V_from_f(f)
        z = SQRT2 * f
        # d(scale * V) / df. V_from_f already applied the scale to the value; the derivative scales the same way.
        r = float(self.likelihood_scale) * SQRT2 * (torch.sigmoid(z) - self.y)
        grads = {"A": torch.matmul(r.unsqueeze(1), H).squeeze(1) * rm}
        dH = r.unsqueeze(-1) * A.unsqueeze(1) * rm
        for ell in range(self.lay.n_layers - 1, 1, -1):
            h, pre = layers[ell - 1]
            h_prev = layers[ell - 2][0]
            dP = dH * activation_derivative(self.activation, h, pre)
            W = blk[f"W{ell}"]
            grads[f"W{ell}"] = torch.matmul(dP.transpose(1, 2), h_prev) * rm
            dH = torch.matmul(dP, W) * rm
        h1, pre1 = layers[0]
        dP1 = dH * activation_derivative(self.activation, h1, pre1)
        grads["W1"] = torch.matmul(dP1.transpose(1, 2), self.X)
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
def apply_activation(name: str, z: torch.Tensor) -> torch.Tensor:
    if name == "tanh":
        return torch.tanh(z)
    if name == "relu":
        return torch.relu(z)
    if name == "sin":
        return torch.sin(z)
    raise ValueError(f"unknown activation {name}; expected one of {ACTIVATIONS}")


def activation_derivative(name: str, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """dφ/dz. ReLU uses the right-continuous convention φ'(0) = 0, matching torch.relu."""
    if name == "tanh":
        return 1.0 - h * h
    if name == "relu":
        return (z > 0).to(dtype=z.dtype)
    if name == "sin":
        return torch.cos(z)
    raise ValueError(f"unknown activation {name}; expected one of {ACTIVATIONS}")


def _act_scalar(name: str, z: float) -> float:
    if name == "tanh":
        return math.tanh(z)
    if name == "relu":
        return z if z > 0.0 else 0.0
    if name == "sin":
        return math.sin(z)
    raise ValueError(f"unknown activation {name}")


def loop_logits(theta: np.ndarray, lay: Layout, X: np.ndarray, activation: str = "tanh") -> np.ndarray:
    m, d = lay.m, lay.d
    o = lay.offsets
    A = theta[o["A"][0]:o["A"][1]]
    W1 = theta[o["W1"][0]:o["W1"][1]].reshape(m, d)
    laters = []
    for ell in range(2, lay.n_layers):
        name = f"W{ell}"
        laters.append(theta[o[name][0]:o[name][1]].reshape(m, m))
    f = np.zeros(X.shape[0])
    inv = 1.0 / math.sqrt(m)
    for i in range(X.shape[0]):
        h = [_act_scalar(activation, sum(W1[j, k] * X[i, k] for k in range(d))) for j in range(m)]
        for W in laters:
            h = [_act_scalar(activation, sum(W[j, k] * h[k] for k in range(m)) * inv) for j in range(m)]
        f[i] = sum(A[j] * h[j] for j in range(m)) * inv
    return f


def loop_V(theta: np.ndarray, lay: Layout, X: np.ndarray, y: np.ndarray, activation: str = "tanh",
           likelihood_scale: float = 1.0) -> float:
    f = loop_logits(theta, lay, X, activation=activation)
    out = 0.0
    for fi, yi in zip(f, y):
        z = SQRT2 * fi
        out += math.log1p(math.exp(-abs(z))) + max(z, 0.0) - yi * z
    return float(likelihood_scale) * out
