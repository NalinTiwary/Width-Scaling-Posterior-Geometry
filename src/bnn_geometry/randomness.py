"""Independent seeded streams (runbook §4) and serializable RNG state."""

from __future__ import annotations

import hashlib
from typing import Any, Union

import numpy as np
import torch

Field = Union[int, str]


def stream_seed(master: int, *, rep: Field, arch: Field, width: Field, chain: Field, stage: str,
                stream: str, schema: int = 1) -> int:
    s = f"{master}|schema={schema}|rep={rep}|arch={arch}|width={width}|chain={chain}|stage={stage}|stream={stream}"
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def numpy_rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def torch_gen(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def step_tag(h: float) -> str:
    return float(h).hex()


class ChainRNG:
    """One chain's sampler stream: NumPy PCG64 for scalars, a torch.Generator for Gaussian vectors."""

    def __init__(self, seed: int, device: torch.device):
        self.seed = seed
        self.np = numpy_rng(seed)
        self.torch = torch_gen(seed ^ 0x5DEECE66D, device)

    def state(self) -> dict[str, Any]:
        return {"seed": self.seed, "np": self.np.bit_generator.state,
                "torch": self.torch.get_state().cpu().numpy().copy(),
                "torch_device": str(self.torch.device)}

    def load(self, st: dict[str, Any]) -> None:
        if int(st["seed"]) != self.seed:
            raise ValueError("RNG seed mismatch on resume")
        self.np.bit_generator.state = st["np"]
        self.torch.set_state(torch.as_tensor(st["torch"], dtype=torch.uint8))
