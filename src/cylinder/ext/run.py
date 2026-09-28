"""
Run one extension target: 4 unrestricted elliptical-slice chains + online curvature.

Curvature brackets are evaluated during sampling at the prespecified retained indices
t_j = floor((j + ½) T / 64), j = 0..63, per chain (256 states per target), because storing
256 raw states would cost ~8.6 GB per target at m = 16384.  Only per-draw scalar
observables and per-state bracket results are written.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..data import generate_projections
from ..device_utils import configure_dtype, device_metadata, resolve_device, synchronize
from ..io import atomic_save_json, atomic_save_npz, validate_resume
from ..observables import curvature_state_indices
from .curvature import bracket
from .ess import ESSStats, ess_step
from .problem import Problem, build_problem, potential
from .targets import Target

SCHEMA = 1
KIND_ID = {"orth": 1, "fmnist": 2}
IDENTITY_KEYS = ["schema_version", "kind", "n", "m", "rep", "sigma", "p",
                 "data_hash_X", "data_hash_y", "centers_hash_U", "B"]


def run_dir(artifacts: Path, t: Target) -> Path:
    return artifacts / "runs" / t.name


def chain_seeds(cfg: dict[str, Any], t: Target, c: int) -> tuple[np.random.Generator, int]:
    ss = np.random.SeedSequence([int(cfg.get("seed_master", 0)), KIND_ID[t.kind], t.n, t.m, t.rep, c])
    rng = np.random.Generator(np.random.PCG64(ss))
    torch_seed = int(ss.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
    return rng, torch_seed


def _sub_forward(theta: torch.Tensor, prob: Problem) -> torch.Tensor:
    """f at the fixed diagnostic training rows only."""
    blocks = theta.view(prob.m, prob.k)
    a, W = blocks[:, 0], blocks[:, 1:]
    sub = torch.as_tensor(prob.sub_idx, device=theta.device)
    pre = W[:, sub] if prob.X is None else W @ prob.X[sub].T
    return (a @ torch.tanh(pre)) / math.sqrt(prob.m)


def is_complete(artifacts: Path, t: Target, n_retained: int) -> bool:
    d = run_dir(artifacts, t)
    mp = d / "metadata.json"
    if not (mp.exists() and (d / "observables.npz").exists() and (d / "curvature.npz").exists()):
        return False
    meta = json.loads(mp.read_text())
    return int(meta.get("n_retained", 0)) >= n_retained


def run_ext_target(
    *,
    cfg: dict[str, Any],
    target: Target,
    artifacts: Path,
    device_request: Optional[str] = None,
    n_retained: Optional[int] = None,
    overwrite: bool = False,
    skip_existing: bool = False,
    burnin: Optional[int] = None,
    n_chains: Optional[int] = None,
    write: bool = True,
) -> dict[str, Any]:
    configure_dtype()
    req = device_request or cfg.get("device", "auto")
    device = resolve_device(req)
    prob = build_problem(cfg, target.kind, target.n, target.m, target.rep, artifacts, device)
    out = run_dir(artifacts, target)

    identity = {
        "schema_version": SCHEMA, "kind": target.kind, "n": target.n, "m": target.m,
        "rep": target.rep, "sigma": prob.sigma, "p": prob.p,
        "data_hash_X": prob.hashes["hash_X"], "data_hash_y": prob.hashes["hash_y"],
        "centers_hash_U": prob.hashes["hash_U"], "B": prob.B,
    }
    scfg = cfg["sampling"]
    n_chains = int(n_chains if n_chains is not None else scfg["n_chains"])
    burnin = int(burnin if burnin is not None else scfg["burnin"])
    T = int(n_retained if n_retained is not None else scfg["retained_initial"])

    meta_path = out / "metadata.json"
    if write and meta_path.exists() and not overwrite:
        old = json.loads(meta_path.read_text())
        validate_resume(old, identity, keys=IDENTITY_KEYS)
        if skip_existing and is_complete(artifacts, target, T):
            print(f"Skipping completed target {target.name} (T={old['n_retained']})")
            return old

    ocfg = cfg["observables"]
    proj = generate_projections(p=prob.p, n_proj=int(ocfg["n_projections"]), seed=int(ocfg["projection_seed"]))
    U_proj = torch.as_tensor(proj["U"], device=device)
    del proj["U"]

    ccfg = cfg["curvature"]
    spc = int(ccfg["states_per_chain"])
    curv_idx = curvature_state_indices(T, spc)
    slots: dict[int, list[int]] = {}
    for j, t in enumerate(curv_idx):
        slots.setdefault(int(t), []).append(j)

    n_sub, n_proj = len(prob.sub_idx), U_proj.shape[0]
    buf = {
        "H": np.full((n_chains, T), np.nan),
        "H_over_B": np.full((n_chains, T), np.nan),
        "inside": np.full((n_chains, T), -1, dtype=np.int8),
        "V": np.full((n_chains, T), np.nan),
        "f_sub": np.full((n_chains, T, n_sub), np.nan),
        "z_sq_over_p": np.full((n_chains, T), np.nan),
        "mean_sq_head": np.full((n_chains, T), np.nan),
        "projections": np.full((n_chains, T, n_proj), np.nan),
    }
    curv_keys = ("H", "inside", "ell", "u", "d_minus", "d_plus", "max_eigen_residual", "K")
    curv = {k: np.full((n_chains, spc), np.nan) for k in curv_keys}

    theta0, sigma, m, k = prob.theta0, prob.sigma, prob.m, prob.k
    a0 = theta0.view(m, k)[:, 0]
    cap_per_chain = max(1, int(scfg["likelihood_eval_cap_per_target"]) // n_chains)

    def loglik(u: torch.Tensor) -> float:
        return -float(potential(theta0 + sigma * u, prob))

    chain_stats, hit_budget = [], False
    total_evals, t_sample, t_curv = 0, 0.0, 0.0
    for c in range(n_chains):
        rng, tseed = chain_seeds(cfg, target, c)
        gen = torch.Generator(device=device).manual_seed(tseed)
        st = ESSStats()
        u = torch.randn(prob.p, generator=gen, device=device)
        lu = loglik(u)
        st.n_likelihood_evals += 1
        for _ in range(burnin):
            if st.n_likelihood_evals >= cap_per_chain:
                hit_budget = True
                break
            u, lu = ess_step(u, lu, loglik, rng, st, gen=gen)
        ev0 = st.n_likelihood_evals
        synchronize(device)
        t0 = time.perf_counter()
        tc_chain = 0.0
        T_done = 0
        for t in range(T):
            if st.n_likelihood_evals >= cap_per_chain:
                hit_budget = True
                break
            u, lu = ess_step(u, lu, loglik, rng, st, gen=gen)
            th = theta0 + sigma * u
            a = th.view(m, k)[:, 0]
            H = float(a.abs().max())
            buf["H"][c, t] = H
            buf["H_over_B"][c, t] = H / prob.B
            buf["inside"][c, t] = int(H <= prob.B)
            buf["V"][c, t] = -lu
            buf["f_sub"][c, t] = _sub_forward(th, prob).cpu().numpy()
            buf["z_sq_over_p"][c, t] = float(u @ u) / prob.p
            zh = (a - a0) / sigma
            buf["mean_sq_head"][c, t] = float(zh @ zh) / m
            buf["projections"][c, t] = (U_proj @ u).cpu().numpy()
            if t in slots:
                synchronize(device)
                tc = time.perf_counter()
                br = bracket(th, prob, ccfg)
                for j in slots[t]:
                    curv["H"][c, j] = H
                    curv["inside"][c, j] = int(H <= prob.B)
                    for key in ("ell", "u", "d_minus", "d_plus", "max_eigen_residual", "K"):
                        curv[key][c, j] = br[key]
                synchronize(device)
                tc_chain += time.perf_counter() - tc
            T_done = t + 1
        synchronize(device)
        wall = time.perf_counter() - t0 - tc_chain
        t_sample += wall
        t_curv += tc_chain
        total_evals += st.n_likelihood_evals
        chain_stats.append({
            "chain": c, "torch_seed": tseed, "likelihood_evals": st.n_likelihood_evals,
            "likelihood_evals_retained": st.n_likelihood_evals - ev0,
            "bracket_shrinks": st.n_bracket_shrinks, "n_updates": st.n_updates,
            "retained_done": T_done, "wall_time_retained_s": wall, "curvature_time_s": tc_chain,
        })

    meta = {
        **identity,
        "target": target.name,
        "p_full": prob.p_full,
        "k_per_neuron": k,
        "coordinates": "reduced (a, z = X_n w)" if prob.X is None else "full (a, w)",
        "theory": prob.theory,
        "n_chains": n_chains,
        "burnin": burnin,
        "n_retained": T,
        "curvature_indices": curv_idx,
        "curvature_rule": (
            f"orth arrowhead; Rayleigh subspace = {ccfg.get('upper_bracket_blocks', 32)} worst blocks"
            if prob.X is None else "dense (1+d)-blocks; Rayleigh subspace = n+1 lowest residual eigenvectors"
        ),
        "diagnostic_train_rows": prob.sub_idx,
        "projections_hash": proj["hash_U"],
        "likelihood_evals": total_evals,
        "likelihood_eval_cap": int(scfg["likelihood_eval_cap_per_target"]),
        "hit_likelihood_budget": hit_budget,
        "wall_time_sampling_s": t_sample,
        "wall_time_curvature_s": t_curv,
        "chain_stats": chain_stats,
        "extra": prob.extra,
        **device_metadata(req, device),
    }
    if write:
        atomic_save_npz(out / "observables.npz", **buf, curv_indices=curv_idx)
        atomic_save_npz(out / "curvature.npz", **curv, indices=curv_idx)
        atomic_save_json(meta_path, meta)
    meta["_buf"] = buf
    meta["_curv"] = curv
    return meta
