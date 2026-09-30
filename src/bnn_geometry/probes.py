"""Fixed observables, probe functions and their gradients (runbook §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .model import Model, unpack


@dataclass(frozen=True)
class ProbeSpec:
    train_idx: tuple[int, ...]
    test_idx: tuple[int, ...]
    entropy_train_idx: tuple[int, ...]
    neurons: tuple[int, ...]
    shallow_inputs: tuple[int, ...]

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "ProbeSpec":
        p = cfg["probes"]
        return cls(tuple(p["train_indices"]), tuple(p["test_indices"]), tuple(p["entropy_train_indices"]),
                   tuple(p["interaction_neurons"]), tuple(p["shallow_interaction_input_indices"]))

    # ---- names -------------------------------------------------------------------------------------
    def scalar_names(self, arch: str) -> list[str]:
        names = ["V", "U"]
        names += [f"train_logit_{i}" for i in self.train_idx]
        names += [f"test_prob_{i}" for i in self.test_idx]
        names += [f"interaction_{j}" for j in range(len(self.neurons))]
        names += ["q_A", "q_W", "disp_A", "disp_W1"] + (["disp_W2"] if arch == "deep" else [])
        return names

    def probe_names(self) -> list[str]:
        """Nine substantive probes then the two linear controls (11 gradient norms per state)."""
        return (["V"] + [f"train_logit_{i}" for i in self.entropy_train_idx]
                + [f"interaction_{j}" for j in range(len(self.neurons))] + ["q_A", "q_W"])

    def families(self) -> dict[str, list[str]]:
        return {
            "loss": ["V"],
            "train_logit": [f"train_logit_{i}" for i in self.train_idx],
            "test_probability": [f"test_prob_{i}" for i in self.test_idx],
            "interaction": [f"interaction_{j}" for j in range(len(self.neurons))],
        }

    def static_families(self) -> dict[str, list[str]]:
        return {
            "loss": ["V"],
            "train_logit": [f"train_logit_{i}" for i in self.entropy_train_idx],
            "interaction": [f"interaction_{j}" for j in range(len(self.neurons))],
        }


def alternating_unit(m: int, device: torch.device) -> torch.Tensor:
    return ((-1.0) ** torch.arange(m, device=device, dtype=torch.float64)) / math.sqrt(m)


def qW_direction(m: int, d: int, device: torch.device) -> torch.Tensor:
    j = torch.arange(m, device=device, dtype=torch.float64)[:, None]
    k = torch.arange(d, device=device, dtype=torch.float64)[None, :]
    return ((-1.0) ** (j + k)) / math.sqrt(m * d)


class Observables:
    def __init__(self, model: Model, spec: ProbeSpec):
        self.model, self.spec = model, spec
        lay = model.lay
        dev = model.device
        self.names = spec.scalar_names(lay.arch)
        self.u = alternating_unit(lay.m, dev)
        self.qW = qW_direction(lay.m, lay.d, dev)
        self.blk0 = {k: v[0] for k, v in unpack(model.theta0.unsqueeze(0), lay).items()}
        self.train_idx = torch.as_tensor(spec.train_idx, device=dev)
        self.neurons = torch.as_tensor(spec.neurons, device=dev)
        self.x_int = model.X[list(spec.shallow_inputs)]                     # (4, d)

    def interactions(self, theta: torch.Tensor) -> torch.Tensor:
        blk = unpack(theta, self.model.lay)
        j = self.neurons
        dA = blk["A"][:, j] - self.blk0["A"][j]                             # (C, 4)
        if self.model.lay.arch == "shallow":
            dW = blk["W1"][:, j, :] - self.blk0["W1"][j]                   # (C, 4, d)
            arg = (dW * self.x_int.unsqueeze(0)).sum(-1)
        else:
            dW = blk["W2"][:, j, :] - self.blk0["W2"][j]                   # (C, 4, m)
            arg = torch.matmul(dW, self.u)
        return dA * torch.tanh(arg)

    def linear_controls(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blk = unpack(theta, self.model.lay)
        qA = torch.matmul(blk["A"] - self.blk0["A"], self.u)
        qW = ((blk["W1"] - self.blk0["W1"]) * self.qW).sum(dim=(1, 2))
        return qA, qW

    @torch.no_grad()
    def record(self, theta: torch.Tensor, V: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """All per-transition scalars (C, K) in self.names order; V and f are the cached train values."""
        model = self.model
        U = V + model.prior_quadratic(theta)
        ft = model.logits(theta, model.X_probe_test)
        pt = torch.sigmoid(math.sqrt(2.0) * ft)
        qA, qW = self.linear_controls(theta)
        blk = unpack(theta, model.lay)
        disp = [((blk[k] - self.blk0[k]) ** 2).reshape(theta.shape[0], -1).mean(dim=1)
                for k, _ in model.lay.blocks]
        cols = [V.unsqueeze(1), U.unsqueeze(1), f[:, self.train_idx], pt, self.interactions(theta),
                qA.unsqueeze(1), qW.unsqueeze(1)] + [x.unsqueeze(1) for x in disp]
        return torch.cat(cols, dim=1)

    def probe_values_and_grad_sq(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """For a batch of states (B, p): values (B, 11) and squared Euclidean gradient norms (B, 11)
        over all original coordinates. Loops over probes; no graph survives the call."""
        model = self.model
        th = theta.detach().clone().requires_grad_(True)
        f = model.logits(th)
        V = model.V_from_f(f)
        ints = self.interactions(th)
        qA, qW = self.linear_controls(th)
        probes = [V] + [f[:, i] for i in self.spec.entropy_train_idx] + [ints[:, j] for j in range(ints.shape[1])]
        probes += [qA, qW]
        vals, gsq = [], []
        for k, g in enumerate(probes):
            (grad,) = torch.autograd.grad(g.sum(), th, retain_graph=k < len(probes) - 1)
            vals.append(g.detach())
            gsq.append((grad * grad).sum(dim=1))
        return torch.stack(vals, 1), torch.stack(gsq, 1)
