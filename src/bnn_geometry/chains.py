"""Resumable multi-segment trajectories for a target's chains (runbook §4, §7, §8).

A Trajectory owns one directory: ``chain_<c>/<segment>.h5`` per chain and segment, plus ``checkpoint.pt``.
Segments run in order; each is written in atomically-renamed chunks of ``checkpoint_stride`` transitions and a
restart record follows each chunk, so a resumed run neither repeats nor omits a saved transition.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np
import torch

from .ellipse import EllipticalSlice, ESSState, SamplerFailure
from .pcnl import PCNL, PCNLState
from .randomness import ChainRNG
from .storage import (StorageError, atomic_torch_save, chunk_path, concat_arrays, list_chunks, read_trace_file,
                      segment_path, torch_load, validate_trace, write_trace_file)


class Interrupted(RuntimeError):
    """Raised by the test hook ``stop_after`` to simulate a killed job."""


def prior_draws(theta0: torch.Tensor, sigma: float, rngs: list[ChainRNG]) -> torch.Tensor:
    p = theta0.shape[0]
    z = torch.stack([torch.randn(p, generator=r.torch, device=theta0.device, dtype=theta0.dtype) for r in rngs])
    return theta0 + float(sigma) * z


class Budget:
    """Gradient budget shared by all dynamics trajectories of a target: ``base`` counts the other trajectories."""

    def __init__(self, limit: int, base: int = 0):
        self.limit, self.base = int(limit), int(base)


class Trajectory:
    def __init__(self, path: Path, *, kind: str, sampler: Any, rngs: list[ChainRNG], chains: list[int],
                 hashes: dict[str, str], record: Callable[[Any], torch.Tensor], names: list[str],
                 checkpoint_stride: int, archive_fn: Optional[Callable[[torch.Tensor], dict[str, torch.Tensor]]] = None,
                 per_chain_eval_cap: Optional[int] = None, budget: Optional[Budget] = None,
                 meta: Optional[dict[str, Any]] = None):
        if kind not in ("ess", "pcnl"):
            raise ValueError(kind)
        self.path, self.kind, self.sampler, self.rngs = Path(path), kind, sampler, rngs
        self.chains, self.hashes, self.record, self.names = list(chains), dict(hashes), record, list(names)
        self.stride = int(checkpoint_stride)
        self.archive_fn = archive_fn
        self.cap = per_chain_eval_cap
        self.budget = budget
        self.meta = meta or {}
        self.ck: dict[str, Any] = {}
        self.state: Any = None

    # ---- checkpoint -------------------------------------------------------------------------------
    @property
    def ck_path(self) -> Path:
        return self.path / "checkpoint.pt"

    def chain_dir(self, c: int) -> Path:
        return self.path / f"chain_{c}"

    def _state_dict(self) -> dict[str, Any]:
        st = self.state
        d = {"theta": st.theta.detach().cpu().clone(), "V": st.V.detach().cpu().clone(),
             "cache": None if st.cache is None else st.cache.detach().cpu().clone()}
        if self.kind == "pcnl":
            d["g"] = st.g.detach().cpu().clone()
        return d

    def _save(self) -> None:
        self.ck["state"] = self._state_dict()
        self.ck["rng"] = [r.state() for r in self.rngs]
        atomic_torch_save(self.ck_path, self.ck)

    def grad_evals(self) -> int:
        return int(sum(self.ck["counters"]["grad_evals"])) if self.ck else 0

    def budget_exhausted(self) -> bool:
        return self.budget is not None and self.budget.base + self.grad_evals() >= self.budget.limit

    def open(self, init_theta: Callable[[], torch.Tensor]) -> "Trajectory":
        dev = self.sampler.theta0.device
        if self.ck_path.exists():
            ck = torch_load(self.ck_path)
            for k, v in self.hashes.items():
                if ck["hashes"].get(k) != v:
                    raise StorageError(f"{self.path}: checkpoint {k} differs from the requested run; "
                                       "refusing to resume an incompatible trajectory")
            if ck["chains"] != self.chains or ck["kind"] != self.kind:
                raise StorageError(f"{self.path}: checkpoint chains/kind mismatch")
            for r, s in zip(self.rngs, ck["rng"]):
                r.load(s)
            s = ck["state"]
            th = s["theta"].to(dev)
            cache = None if s["cache"] is None else s["cache"].to(dev)
            if self.kind == "ess":
                self.state = ESSState(th, s["V"].to(dev), cache)
            else:
                self.state = PCNLState(th, s["V"].to(dev), s["g"].to(dev), cache)
            self.ck = ck
        else:
            theta = init_theta()
            C = len(self.chains)
            if self.kind == "ess":
                self.state, ev = self.sampler.init_state(theta)
                grads = np.zeros(C, dtype=np.int64)
            else:
                self.state = self.sampler.init_state(theta)
                ev = np.ones(C, dtype=np.int64)
                grads = np.ones(C, dtype=np.int64)
            self.ck = {"hashes": self.hashes, "kind": self.kind, "chains": self.chains, "meta": self.meta,
                       "done": {}, "order": [], "current": None,
                       "counters": {"evals": ev.tolist(), "grad_evals": grads.tolist(),
                                    "transitions": [0] * C, "accepted": [0] * C, "streak": [0] * C,
                                    "max_streak": [0] * C, "numerical_failures": [0] * C},
                       "elapsed_s": 0.0, "status": "running"}
            self._save()
        return self

    # ---- running ----------------------------------------------------------------------------------
    def done(self, segment: str) -> bool:
        return segment in self.ck["done"]

    def run_segment(self, segment: str, n: int, *, archive_stride: int = 0,
                    stop_after: Optional[int] = None) -> dict[str, Any]:
        """Run ``n`` transitions of ``segment`` (resuming if interrupted). Returns the segment record."""
        if segment in self.ck["done"]:
            return self.ck["done"][segment]
        cur = self.ck["current"]
        if cur is not None and cur["name"] != segment:
            raise StorageError(f"{self.path}: unfinished segment {cur['name']} precedes {segment}")
        if cur is None:
            cur = {"name": segment, "n": int(n), "k": 0, "archive_stride": int(archive_stride),
                   "t_start": time.time()}
            self.ck["current"] = cur
        elif cur["n"] != int(n) or cur["archive_stride"] != int(archive_stride):
            raise StorageError(f"{self.path}: segment {segment} was started with different length/stride")
        for c in self.chains:
            for start, p in list_chunks(self.chain_dir(c), segment):
                if start >= cur["k"]:
                    p.unlink()        # written after the last restart record: redone below
        C = len(self.chains)
        dev = self.state.theta.device
        K = len(self.names)
        k = int(cur["k"])
        t0 = time.time()
        steps_this_call = 0
        stop_reason = "dynamics_budget_exhausted" if self.budget_exhausted() else None
        while k < n and stop_reason is None:
            L = min(self.stride - (k % self.stride), n - k)
            sc = torch.empty((L, C, K), dtype=torch.float64, device=dev)
            ev = np.zeros((L, C), dtype=np.int64)
            acc = np.zeros((L, C), dtype=np.int8)
            lr = np.zeros((L, C))
            arch_draw, arch_states, arch_extra = [], [], {}
            done_L = 0
            for i in range(L):
                if self.kind == "ess":
                    try:
                        e = self.sampler.step(self.state, self.rngs)
                    except SamplerFailure:
                        self.ck["status"] = "sampler_failure"
                        self._flush(segment, k, done_L, sc, ev, acc, lr, arch_draw, arch_states, arch_extra)
                        self._save()
                        raise
                    ev[i] = e
                    for c in range(C):
                        self.ck["counters"]["evals"][c] += int(e[c])
                else:
                    a, lam = self.sampler.step(self.state, self.rngs)
                    acc[i] = a
                    lr[i] = lam
                    ev[i] = 1
                    cn = self.ck["counters"]
                    for c in range(C):
                        cn["evals"][c] += 1
                        cn["grad_evals"][c] += 1
                        cn["accepted"][c] += int(a[c])
                        if not np.isfinite(lam[c]):
                            cn["numerical_failures"][c] += 1
                        cn["streak"][c] = 0 if a[c] else cn["streak"][c] + 1
                        cn["max_streak"][c] = max(cn["max_streak"][c], cn["streak"][c])
                for c in range(C):
                    self.ck["counters"]["transitions"][c] += 1
                sc[i] = self.record(self.state)
                draw = k + i
                if archive_stride and draw % archive_stride == 0:
                    arch_draw.append(draw)
                    arch_states.append(self.state.theta.detach().clone())
                    if self.archive_fn is not None:
                        for key, val in self.archive_fn(self.state.theta).items():
                            arch_extra.setdefault(key, []).append(val.detach().clone())
                done_L = i + 1
                steps_this_call += 1
                if self.cap is not None and max(self.ck["counters"]["evals"]) >= self.cap:
                    stop_reason = "reference_budget_exhausted"
                    break
                if self.budget_exhausted():
                    stop_reason = "dynamics_budget_exhausted"
                    break
                if stop_after is not None and steps_this_call >= stop_after:
                    stop_reason = "interrupted"
                    break
            self._flush(segment, k, done_L, sc, ev, acc, lr, arch_draw, arch_states, arch_extra)
            k += done_L
            cur["k"] = k
            self.ck["elapsed_s"] += time.time() - t0
            t0 = time.time()
            self._save()
            if stop_reason == "interrupted":
                raise Interrupted(f"stopped after {steps_this_call} transitions")
            if stop_reason is not None:
                break
        rec = {"n": k, "planned": int(n), "partial": k < n, "stop_reason": stop_reason,
               "archive_stride": int(archive_stride), "wall_s": time.time() - cur.get("t_start", time.time())}
        self._consolidate(segment, k, archive_stride)
        self.ck["done"][segment] = rec
        self.ck["order"].append(segment)
        self.ck["current"] = None
        if stop_reason is not None:
            self.ck["status"] = stop_reason
        self._save()
        return rec

    def _attrs(self, c: int, segment: str, first: int, archive_stride: int) -> dict[str, Any]:
        return {**self.hashes, "chain": c, "segment": segment, "first_draw": first, "saved_stride": 1,
                "archive_stride": archive_stride, "names": self.names, "kind": self.kind}

    def _flush(self, segment, k, L, sc, ev, acc, lr, arch_draw, arch_states, arch_extra) -> None:
        if L == 0:
            return
        sc_np = sc[:L].detach().cpu().numpy()
        states = torch.stack(arch_states).detach().cpu().numpy() if arch_states else None
        stride = int(self.ck["current"]["archive_stride"])
        if stride and self.archive_fn is not None and not arch_extra:
            probe = self.archive_fn(self.state.theta)
            extra = {key: np.zeros((0,) + tuple(v.shape)) for key, v in probe.items()}
        else:
            extra = {key: torch.stack(v).detach().cpu().numpy() for key, v in arch_extra.items()}
        for ci, c in enumerate(self.chains):
            arrays = {"scalars": sc_np[:, ci, :], "draw": np.arange(k, k + L, dtype=np.int64), "evals": ev[:L, ci]}
            if self.kind == "pcnl":
                arrays["accepted"] = acc[:L, ci]
                arrays["logratio"] = lr[:L, ci]
            if stride:
                arrays["archive/draw"] = np.asarray(arch_draw, dtype=np.int64)
                p = self.state.theta.shape[1]
                arrays["archive/states"] = states[:, ci, :] if states is not None else np.zeros((0, p))
                for key, v in extra.items():
                    arrays[f"archive/{key}"] = v[:, ci]
            write_trace_file(chunk_path(self.chain_dir(c), segment, k), arrays=arrays,
                             attrs=self._attrs(c, segment, k, stride))

    def _consolidate(self, segment: str, n: int, archive_stride: int) -> None:
        for c in self.chains:
            cd = self.chain_dir(c)
            chunks = list_chunks(cd, segment)
            parts, pos = [], 0
            for start, p in chunks:
                arr, attrs = read_trace_file(p)
                ln = arr["draw"].shape[0]
                validate_trace(arr, attrs, expect_first=pos, expect_n=ln, hashes=self.hashes, chain=c,
                               where=str(p))
                if start != pos:
                    raise StorageError(f"{p}: missing chunk before draw {start}")
                parts.append(arr)
                pos += ln
            if pos != n:
                raise StorageError(f"{cd}/{segment}: chunks cover {pos} transitions, expected {n}")
            if parts:
                allarr = concat_arrays(parts)
            else:
                allarr = {"scalars": np.zeros((0, len(self.names))), "draw": np.zeros(0, dtype=np.int64),
                          "evals": np.zeros(0, dtype=np.int64)}
            write_trace_file(segment_path(cd, segment), arrays=allarr,
                             attrs=self._attrs(c, segment, 0, archive_stride) | {"last_draw": n - 1})
            for _, p in chunks:
                p.unlink()

    # ---- reading ----------------------------------------------------------------------------------
    def read(self, segments: list[str], keys: Optional[list[str]] = None) -> list[dict[str, np.ndarray]]:
        """Per chain, the concatenation of complete segments in order, with draw ids renumbered cumulatively."""
        if keys is not None:
            keys = list(dict.fromkeys(list(keys) + ["draw"] + (["archive/draw"] if any(
                k.startswith("archive/") for k in keys) else [])))
        out = []
        for c in self.chains:
            parts, offset = [], 0
            for seg in segments:
                rec = self.ck["done"].get(seg)
                if rec is None:
                    raise StorageError(f"{self.path}: segment {seg} not complete")
                arr, attrs = read_trace_file(segment_path(self.chain_dir(c), seg), keys)
                validate_trace(arr, attrs, expect_first=0, expect_n=rec["n"], hashes=self.hashes, chain=c,
                               where=f"{self.path}/chain_{c}/{seg}")
                arr = dict(arr)
                arr["draw"] = arr["draw"] + offset
                if "archive/draw" in arr:
                    arr["archive/draw"] = arr["archive/draw"] + offset
                parts.append(arr)
                offset += rec["n"]
            out.append(concat_arrays(parts) if parts else {})
        return out

    def archive_rows(self, c: int, segments: list[str], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Archived states of chain c at positions ``idx`` of the concatenated archives of ``segments``
        (returns states (k, p) and cumulative draw ids), reading only the requested rows."""
        idx = np.asarray(idx, dtype=np.int64)
        states, draws = [], []
        base, offset = 0, 0
        for seg in segments:
            rec = self.ck["done"][seg]
            with h5py.File(segment_path(self.chain_dir(c), seg), "r") as f:
                m = f["archive/draw"].shape[0]
                sel = idx[(idx >= base) & (idx < base + m)] - base
                if sel.size:
                    states.append(f["archive/states"][np.sort(sel)])
                    draws.append(f["archive/draw"][np.sort(sel)] + offset)
            base += m
            offset += rec["n"]
        if not states:
            return np.zeros((0, self.state.theta.shape[1])), np.zeros(0, dtype=np.int64)
        return np.concatenate(states), np.concatenate(draws)

    def archive_count(self, segments: list[str]) -> int:
        tot = 0
        for seg in segments:
            with h5py.File(segment_path(self.chain_dir(self.chains[0]), seg), "r") as f:
                tot += f["archive/draw"].shape[0]
        return tot
