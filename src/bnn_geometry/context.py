"""Per-target execution context: model, probes, identity hashes, seeds and trajectory constructors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from . import config as C
from .chains import Budget, Trajectory, prior_draws
from .data import load_replicate
from .ellipse import EllipticalSlice
from .model import Layout, build_model
from .pcnl import PCNL
from .probes import Observables, ProbeSpec
from .randomness import ChainRNG, step_tag, stream_seed
from .spectral import state_S
from .storage import atomic_write_json, clean_json, read_json, torch_load

torch.set_default_dtype(torch.float64)
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = False


def pick_device(pref: Optional[str] = None) -> torch.device:
    pref = pref or os.environ.get("BNN_DEVICE")
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def h_id(h: float) -> str:
    """Directory-safe step id from the shortest round-trip decimal (exact float hex is kept in metadata)."""
    return "h_" + repr(float(h)).replace(".", "p").replace("-", "m").replace("+", "")


class TargetContext:
    def __init__(self, cfg: dict[str, Any], t: C.Target, root: Path, device: Optional[torch.device] = None):
        self.cfg, self.t, self.root = cfg, t, Path(root)
        self.device = device or pick_device()
        self.D, self.bank, self.data_hashes = load_replicate(self.root, t.rep)
        self.d = int(cfg["data"]["input_dimension"])
        self.lay = Layout(t.arch, t.m, self.d)
        self.sigma = float(cfg["model"]["sigma"])
        self.a = float(cfg["domain"]["spectral_cutoff_a"])
        self.spec = ProbeSpec.from_cfg(cfg)
        self.model = build_model(self.lay, self.D, self.bank, self.sigma, list(self.spec.test_idx), self.device)
        self.obs = Observables(self.model, self.spec)
        self.names = self.obs.names
        self.thash = C.target_hash(cfg, t, self.data_hashes)
        self.dir = self.root / "targets" / t.target_id
        (self.dir / "analysis").mkdir(parents=True, exist_ok=True)
        self.master = int(cfg["master_seed"])
        self.folds = [list(f) for f in cfg["probes"]["selection_chain_folds"]]
        self.chains = [int(c) for c in cfg["chain_ids"]]

    # ---- identity -------------------------------------------------------------------------------
    def seed(self, chain: Any, stage: str, stream: str) -> int:
        return stream_seed(self.master, rep=self.t.rep, arch=self.t.arch, width=self.t.m, chain=chain,
                           stage=stage, stream=stream, schema=int(self.cfg["schema_version"]))

    def ref_exec_hash(self) -> str:
        return C.execution_hash(self.cfg, self.thash, "ess", role="reference")

    def dyn_exec_hash(self, h: float, role: str = "dynamics") -> str:
        return C.execution_hash(self.cfg, self.thash, "pcnl", step=h, role=role)

    @property
    def law_static(self) -> str:
        return "conditional_G_2.5" if self.t.arch == "deep" else "full_posterior"

    def ids(self, *, law: str, stage: str, execution_hash: str) -> dict[str, Any]:
        return {"campaign_id": self.cfg["campaign_id"], "target_id": self.t.target_id, "target_hash": self.thash,
                "execution_hash": execution_hash, "replicate": self.t.rep, "architecture": self.t.arch,
                "L": self.t.L, "n": int(self.cfg["data"]["train_size"]), "d": self.d, "m": self.t.m,
                "p": self.lay.p, "sigma": self.sigma, "law": law, "stage": stage,
                "code_revision": C.code_revision()}

    def write_spec(self) -> None:
        spec = {"target_id": self.t.target_id, "target_hash": self.thash, "layout": self.lay.to_json(),
                "arch": self.t.arch, "L": self.t.L, "m": self.t.m, "rep": self.t.rep, "sigma": self.sigma,
                "n": int(self.cfg["data"]["train_size"]), "d": self.d, "a": self.a if self.t.arch == "deep" else None,
                "data_hashes": self.data_hashes, "head_center_norm": float(np.linalg.norm(
                    self.model.theta0[: self.t.m].cpu().numpy())),
                "reference_execution_hash": self.ref_exec_hash(), "scalar_names": self.names,
                "probe_names": self.spec.probe_names(), "sampler_code_hash": C.sampler_code_hash()}
        atomic_write_json(self.dir / "spec.json", clean_json(spec))

    # ---- status -----------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        p = self.dir / "status.json"
        return read_json(p) if p.exists() else {"target_id": self.t.target_id}

    def update_status(self, **kw: Any) -> dict[str, Any]:
        st = self.status()
        st.update(kw)
        atomic_write_json(self.dir / "status.json", clean_json(st))
        return st

    def save_analysis(self, name: str, obj: Any) -> None:
        atomic_write_json(self.dir / "analysis" / f"{name}.json", clean_json(obj))

    def load_analysis(self, name: str) -> Optional[Any]:
        p = self.dir / "analysis" / f"{name}.json"
        return read_json(p) if p.exists() else None

    # ---- trajectories ---------------------------------------------------------------------------
    def _record(self, st) -> torch.Tensor:
        return self.obs.record(st.theta, st.V, st.cache)

    def reference_traj(self) -> Trajectory:
        r = self.cfg["reference"]
        cap = int(r["max_likelihood_evaluations_per_chain"])
        forced = (self.cfg.get("fixture") or {}).get("force_reference_budget") or {}
        cap = int(forced.get(self.t.target_id, cap))
        rngs = [ChainRNG(self.seed(c, "reference", "sampler"), self.device) for c in self.chains]
        sampler = EllipticalSlice(self.model.V, self.model.theta0, self.sigma, int(r["bracket_guard_evaluations"]))
        archive_fn = (lambda th: {"S": state_S(th, self.lay, self.a)}) if self.t.arch == "deep" else None
        return Trajectory(self.dir / "reference", kind="ess", sampler=sampler, rngs=rngs, chains=self.chains,
                          hashes={"target_hash": self.thash, "execution_hash": self.ref_exec_hash()},
                          record=self._record, names=self.names, checkpoint_stride=int(r["checkpoint_stride"]),
                          archive_fn=archive_fn, per_chain_eval_cap=cap,
                          meta={"target_id": self.t.target_id, "role": "reference"})

    def open_reference(self) -> Trajectory:
        init = [ChainRNG(self.seed(c, "reference", "init"), self.device) for c in self.chains]
        return self.reference_traj().open(lambda: prior_draws(self.model.theta0, self.sigma, init))

    def dynamics_dirs(self) -> list[Path]:
        out = []
        for sub in ("dynamics", "dynamics_calibration"):
            base = self.dir / sub
            if base.exists():
                out += [p for p in sorted(base.iterdir()) if (p / "checkpoint.pt").exists()]
        return out

    def dynamics_grad_evals(self, exclude: Optional[Path] = None) -> int:
        tot = 0
        for p in self.dynamics_dirs():
            if exclude is not None and p.resolve() == Path(exclude).resolve():
                continue
            tot += int(sum(torch_load(p / "checkpoint.pt")["counters"]["grad_evals"]))
        return tot

    def dynamics_traj(self, h: float, *, calibration: bool = False) -> Trajectory:
        sub = "dynamics_calibration" if calibration else "dynamics"
        path = self.dir / sub / h_id(h)
        role = "step_calibration" if calibration else "dynamics"
        chains = [self.chains[0]] if calibration else self.chains
        stage = f"{role}|h={step_tag(h)}"
        rngs = [ChainRNG(self.seed(c, stage, "sampler"), self.device) for c in chains]
        sampler = PCNL(self.model.V_and_grad, self.model.theta0, self.sigma, h)
        limit = int(self.cfg["dynamics"]["max_candidate_grad_evaluations_per_target_all_dynamics"])
        budget = Budget(limit, self.dynamics_grad_evals(exclude=path))
        return Trajectory(path, kind="pcnl", sampler=sampler, rngs=rngs, chains=chains,
                          hashes={"target_hash": self.thash, "execution_hash": self.dyn_exec_hash(h, role)},
                          record=self._record, names=self.names,
                          checkpoint_stride=int(self.cfg["dynamics"]["checkpoint_stride"]), budget=budget,
                          meta={"target_id": self.t.target_id, "role": role, "h": h, "h_hex": step_tag(h)})

    def open_dynamics(self, h: float, ref_final: torch.Tensor, *, calibration: bool = False) -> Trajectory:
        tr = self.dynamics_traj(h, calibration=calibration)
        start = ref_final[: len(tr.chains)].to(self.device).clone()
        return tr.open(lambda: start)
