"""Short and production runs on top of the existing PCNL and elliptical-slice kernels.

Production lengths come from the revision config. Smoke lengths are a few dozen transitions
on a tiny network. Checkpoints use the campaign Trajectory, so an interrupted run resumes
without repeating a saved transition.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .. import config as C
from ..chains import Interrupted, Trajectory, prior_draws
from ..context import pick_device
from ..data import load_replicate, sha256_array
from ..ellipse import EllipticalSlice
from ..model import Layout, Model, build_model
from ..pcnl import PCNL, coefficients
from ..probes import Observables, ProbeSpec
from ..randomness import ChainRNG, numpy_rng, stream_seed
from ..storage import StorageError, atomic_write_json, clean_json, read_trace_file, segment_path
from . import config as RC
from .efficiency import scalar_rate, select_step, timed_width_comparison, warmup_steps
from .geometry import design_at_r, draw_latents, probe_train_indices
from .gram import gram_spectrum
from .timing import hardware_record, synchronize, time_blocks
from .variants import add_training_rows, admissible_cutoff, nonlinear_labels

torch.set_default_dtype(torch.float64)


def _campaign(cfg: dict[str, Any]) -> dict[str, Any]:
    return C.load(cfg["baseline_config"])


def tasks_for(cfg: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """Ordered work items for one SLURM array."""
    if stage == "fixed-step":
        return [{"id": f"{arch}_m{m:04d}_r{rep}", "arch": arch, "m": m, "rep": rep, "kind": "baseline"}
                for arch, widths in (("shallow", (64, 256, 1024, 4096)), ("deep", (32, 64, 128, 256)))
                for m in widths for rep in (0, 1, 2)]
    if stage == "tune":
        return [{"id": f"{arch}_m{m:04d}_r{rep}", "arch": arch, "m": m, "rep": rep, "kind": "baseline"}
                for arch, widths in (("shallow", (64, 4096)), ("deep", (32, 256)))
                for m in widths for rep in (0, 1, 2)]
    if stage in ("prior-starts", "prior-only"):
        cells = (("shallow", (64, 4096)), ("deep", (32, 256))) if stage == "prior-starts" else (("deep", (32, 256)),)
        reps = (0, 1, 2) if stage == "prior-starts" else (0,)
        return [{"id": f"{arch}_m{m:04d}_r{rep}", "arch": arch, "m": m, "rep": rep, "kind": "baseline"}
                for arch, widths in cells for m in widths for rep in reps]
    if stage == "geometry-pilot":
        return [{"id": f"geometry_r{r:g}_m{m:04d}_r0", "arch": "shallow", "m": m, "rep": 0, "r": r,
                 "kind": "geometry"} for r in (0.0, 0.2, 0.8) for m in (64, 4096)]
    if stage == "geometry":
        return [{"id": f"geometry_r{r:g}_m{m:04d}_r{rep}", "arch": "shallow", "m": m, "rep": rep, "r": r,
                 "kind": "geometry"} for r in (0.0, 0.2, 0.8) for m in (64, 256, 1024, 4096) for rep in (0, 1, 2)]
    if stage in ("relu", "depth4", "prior-scale", "concentrated", "nonlinear", "sin"):
        kind = {"relu": "relu", "depth4": "depth4", "prior-scale": "prior_scale",
                "concentrated": "concentrated", "nonlinear": "nonlinear", "sin": "sin"}[stage]
        return [{"id": f"{kind}_deep_m{m:04d}_r0", "arch": "deep", "m": m, "rep": 0, "kind": kind} for m in (32, 256)]
    if stage == "local-pilot":
        pilot = cfg.get("local_pilot", {})
        return [{"id": pilot.get("target", "deep_m0032_r0"), "arch": "deep",
                 "m": int(pilot.get("m", 32)), "rep": int(pilot.get("rep", 0)), "kind": "baseline"}]
    if stage == "smoke":
        return [{"id": "smoke_shallow", "arch": "shallow", "m": int(cfg.get("smoke_width", 6)),
                 "rep": 0, "kind": "smoke"}]
    raise ValueError(f"unknown stage {stage}")


def _probe_spec(campaign: dict[str, Any], n: int, n_test: int) -> ProbeSpec:
    base = ProbeSpec.from_cfg(campaign)
    train = base.train_idx if n >= 128 else tuple(probe_train_indices(n))
    test = tuple(i for i in base.test_idx if i < n_test) or (0,)
    neurons = tuple(base.neurons[:4])
    return ProbeSpec(train_idx=train, test_idx=test, entropy_train_idx=tuple(train[:len(neurons)]),
                     neurons=neurons, shallow_inputs=tuple(train[:len(neurons)]))


def _synthetic_smoke(m: int, d: int, n: int, n_test: int, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    teacher = rng.standard_normal(d)
    teacher /= np.linalg.norm(teacher)
    p = 1.0 / (1.0 + np.exp(-2.0 * X @ teacher))
    y = (rng.random(n) < p).astype(np.float64)
    Xt = rng.standard_normal((n_test, d))
    Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
    bank = rng.standard_normal((max(m, 8), d))
    return {"X": X, "y": y, "X_test": Xt, "teacher": teacher, "p_teacher": p}, bank


def build_target(cfg: dict[str, Any], task: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Return model, probes, hashes and the scalar record function for one task."""
    campaign = _campaign(cfg)
    kind = task["kind"]
    master = int(cfg["master_seed"])
    if kind == "smoke":
        n = int(cfg.get("smoke_n", 8))
        d = int(cfg.get("smoke_d", 4))
        D, bank = _synthetic_smoke(task["m"], d, n, n_test=4, seed=master)
        sigma = 1.0
        activation = "tanh"
        layers = 2
        arch = "shallow"
        scale = 1.0
    else:
        D, bank, _hashes = load_replicate(RC.baseline_root(cfg), int(task["rep"]))
        D = {k: np.array(v, copy=True) for k, v in D.items()}
        sigma = float(campaign["model"]["sigma"])
        activation = "tanh"
        arch = task["arch"]
        layers = 3 if arch == "deep" else 2
        scale = 1.0
        if kind == "geometry":
            geo = cfg.get("geometry", {})
            n = int(geo.get("n", 16))
            d = int(geo.get("d", 32))
            n_test = int(geo.get("n_test", 64))
            latents = draw_latents(master, int(task["rep"]), n, d, n_test)
            designed = design_at_r(latents, float(task["r"]))
            D = designed
            bank = numpy_bank(master, int(task["rep"]), d, int(campaign["prior_centers"]["row_bank_size"]))
            arch, layers = "shallow", 2
        elif kind == "concentrated":
            D = add_training_rows(D, 128, master, int(task["rep"]))
        elif kind == "nonlinear":
            packed = nonlinear_labels(D, master, int(task["rep"]))
            D = packed["data"]
        elif kind == "relu":
            activation = "relu"
        elif kind == "sin":
            activation = "sin"
        elif kind == "depth4":
            layers = 4
            arch = "deep"
        elif kind == "prior_scale":
            sigma = 0.5
        elif kind != "baseline":
            raise ValueError(kind)
    lay = Layout(arch, int(task["m"]), int(np.asarray(D["X"]).shape[1]), weight_layers=layers)
    spec = _probe_spec(campaign, int(np.asarray(D["X"]).shape[0]), int(np.asarray(D["X_test"]).shape[0]))
    # Held-out probe columns must exist.
    n_test = int(np.asarray(D["X_test"]).shape[0])
    test_idx = [i for i in spec.test_idx if i < n_test] or [0]
    model = build_model(lay, D, bank, sigma, test_idx, device)
    model.activation = activation
    model.likelihood_scale = scale
    model.X_probe_test = torch.as_tensor(np.asarray(D["X_test"][test_idx], dtype=np.float64), device=device)
    obs = Observables(model, ProbeSpec(spec.train_idx, tuple(test_idx), spec.entropy_train_idx,
                                       spec.neurons, spec.shallow_inputs))
    identity = {
        "campaign_id": cfg["campaign_id"], "cell": task["id"], "kind": kind, "arch": arch,
        "m": int(task["m"]), "rep": int(task.get("rep", 0)), "L": layers, "sigma": sigma,
        "activation": activation, "likelihood_scale": scale, "p": lay.p, "n": int(model.X.shape[0]),
        "d": int(model.X.shape[1]), "hash_X": sha256_array(np.asarray(D["X"])),
        "hash_y": sha256_array(np.asarray(D["y"])),
        "cutoff": admissible_cutoff(sigma) if kind == "prior_scale" else (2.10 if layers > 2 else None),
        "chi": gram_spectrum(np.asarray(D["X"]))["chi"] if layers == 2 else None,
    }
    identity["target_hash"] = C._h(identity)  # noqa: SLF001 — same canonical hash helper the campaign uses

    def record(state) -> torch.Tensor:
        V, f = model.V(state.theta)
        base = obs.record(state.theta, V, f)
        return base

    return {"model": model, "obs": obs, "identity": identity, "record": record, "names": list(obs.names)}


def numpy_bank(master: int, rep: int, d: int, rows: int) -> np.ndarray:
    seed = stream_seed(master, rep=rep, arch="geometry", width="shared", chain="shared",
                       stage="revision_centers", stream="bank", schema=1)
    return numpy_rng(seed).standard_normal((rows, d))


def _seeds(cfg: dict[str, Any], identity: dict[str, Any], chain: int, stage: str, stream: str) -> int:
    return stream_seed(int(cfg["master_seed"]), rep=identity["rep"], arch=identity["arch"],
                       width=identity["m"], chain=chain, stage=stage, stream=stream, schema=1)


def run_pcnl(built: dict[str, Any], cfg: dict[str, Any], *, h: float, n: int, chains: list[int],
             out: Path, segment: str, stage: str, prior_only: bool = False,
             init: str = "center", stop_after: Optional[int] = None,
             checkpoint_stride: int = 32) -> dict[str, Any]:
    model: Model = built["model"]
    dev = model.X.device
    if prior_only:
        def potential(theta):
            zeros = torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)
            return zeros, torch.zeros_like(theta), model.logits(theta)
        pg = potential
    else:
        pg = model.V_and_grad
    sampler = PCNL(pg, model.theta0, model.sigma, h)
    rngs = [ChainRNG(_seeds(cfg, built["identity"], c, stage, "sampler"), dev) for c in chains]
    hashes = {"target_hash": built["identity"]["target_hash"], "segment_family": stage}
    traj = Trajectory(out, kind="pcnl", sampler=sampler, rngs=rngs, chains=chains, hashes=hashes,
                      record=built["record"], names=built["names"], checkpoint_stride=checkpoint_stride,
                      meta={"h": h, "prior_only": prior_only, "init": init})

    def init_theta() -> torch.Tensor:
        init_rngs = [ChainRNG(_seeds(cfg, built["identity"], c, "revision_init", init), dev) for c in chains]
        if init == "prior":
            return prior_draws(model.theta0, model.sigma, init_rngs)
        return model.theta0.unsqueeze(0).expand(len(chains), -1).clone()

    traj.open(init_theta)
    try:
        rec = traj.run_segment(segment, n, stop_after=stop_after)
    except Interrupted as exc:
        rec = {"partial": True, "stop_reason": "interrupted", "message": str(exc), "n": traj.ck["current"]["k"]}
    return {"record": rec, "counters": traj.ck.get("counters"), "elapsed_s": traj.ck.get("elapsed_s"),
            "path": str(out)}


def read_loss(path: Path, chains: list[int], segment: str) -> np.ndarray:
    cols = []
    for c in chains:
        arrays, attrs = read_trace_file(segment_path(path / f"chain_{c}", segment))
        names = list(attrs["names"])
        cols.append(np.asarray(arrays["scalars"])[:, names.index("V")])
    return np.stack(cols, axis=0)


def _arm(built: dict[str, Any], cfg: dict[str, Any], h: float, stage: str) -> dict[str, Any]:
    """One disposable batched kernel. Scalar logging stays inside each timed step."""
    model: Model = built["model"]
    dev = model.theta0.device
    batch = int(cfg.get("fixed_step", {}).get("chains", 4))
    sampler = PCNL(model.V_and_grad, model.theta0, model.sigma, h)
    rngs = [ChainRNG(_seeds(cfg, built["identity"], c, stage, "sampler"), dev) for c in range(batch)]
    state = sampler.init_state(model.theta0.unsqueeze(0).expand(batch, -1).clone())
    total = torch.zeros((), dtype=torch.float64, device=dev)

    def step() -> None:
        nonlocal total
        sampler.step(state, rngs)
        total = total + built["record"](state).sum()

    def warmup(n: int = 4) -> None:
        for _ in range(n):
            step()
        synchronize(dev)

    return {"id": str(built["identity"]["cell"]), "step": step, "warmup": warmup,
            "batch": batch, "device": dev, "checksum": total, "h": h, "sigma": float(model.sigma)}


def kernel_timing(built: dict[str, Any], cfg: dict[str, Any], h: float, *, transitions: int, blocks: int,
                  min_seconds: float) -> dict[str, Any]:
    """Disposable streams. Does not advance the production checkpoint."""
    arm = _arm(built, cfg, h, "timing")
    arm["warmup"](4)
    timed = time_blocks(arm["step"], arm["device"], transitions=transitions, blocks=blocks,
                        min_seconds=min_seconds)
    batch = int(arm["batch"])
    # One call of the kernel advances every chain in the batch. Wall time is not multiplied by batch.
    joint = float(timed["seconds_per_transition"])
    timed["batch_chains"] = batch
    timed["scalar_logging"] = True
    timed["scalar_checksum"] = float(arm["checksum"].detach().cpu())
    timed["seconds_per_joint_step"] = joint
    timed["seconds_per_chain_transition"] = joint / batch
    timed["chain_transitions_per_wall_second"] = batch / joint
    timed["label"] = "replay-timed estimate" if built["identity"]["kind"] != "smoke" else "smoke timing"
    timed["h"] = h
    timed["eta"] = coefficients(h, float(arm["sigma"]))[0]
    timed["convention"] = ("wall seconds cover one joint step of batch_chains chains; "
                           "chain_transitions_per_wall_second counts every chain; "
                           "scalar logging is inside the timed step")
    return timed


def _timed_block(arm: dict[str, Any], transitions: int) -> float:
    synchronize(arm["device"])
    t0 = time.perf_counter()
    for _ in range(int(transitions)):
        arm["step"]()
    synchronize(arm["device"])
    return time.perf_counter() - t0


def interleaved_cost(arms: list[dict[str, Any]], *, transitions: int, blocks: int, min_seconds: float,
                     max_doublings: int = 8) -> dict[str, Any]:
    """Alternate two widths on one device, with a separate clock and length for each.

    The first arm is the narrow cell and the second is the wide cell. Length-finding probes
    alternate as well and are not included in the reported elapsed time. A width whose probe
    is shorter than min_seconds doubles its own block length. The measured blocks then
    alternate, each width at its own length.
    """
    if len(arms) != 2:
        raise ValueError("the cost pilot compares exactly two cells")
    if len({arm["batch"] for arm in arms}) != 1 or len({str(arm["device"]) for arm in arms}) != 1:
        raise ValueError("both cells must use the same device and the same chain batch")
    for arm in arms:
        arm["warmup"](4)
    lengths = {arm["id"]: int(transitions) for arm in arms}
    doublings = {arm["id"]: 0 for arm in arms}
    ready = {arm["id"]: min_seconds <= 0 for arm in arms}
    while not all(ready.values()):
        for arm in arms:
            if ready[arm["id"]]:
                continue
            elapsed = _timed_block(arm, lengths[arm["id"]])
            if elapsed >= min_seconds or doublings[arm["id"]] >= max_doublings:
                ready[arm["id"]] = True
            else:
                lengths[arm["id"]] *= 2
                doublings[arm["id"]] += 1
    records: dict[str, list[dict[str, float]]] = {arm["id"]: [] for arm in arms}
    order: list[str] = []
    for _ in range(blocks):
        for arm in arms:
            elapsed = _timed_block(arm, lengths[arm["id"]])
            records[arm["id"]].append({"seconds": elapsed, "transitions": lengths[arm["id"]]})
            order.append(arm["id"])
    cells = {}
    for arm in arms:
        taken = records[arm["id"]]
        elapsed_s = float(sum(item["seconds"] for item in taken))
        joint_n = int(sum(item["transitions"] for item in taken))
        joint = elapsed_s / joint_n
        cells[arm["id"]] = {
            "batch_chains": arm["batch"], "scalar_logging": True,
            "block_seconds": [item["seconds"] for item in taken],
            "transitions_per_block": lengths[arm["id"]],
            "doublings": doublings[arm["id"]],
            "elapsed_seconds": elapsed_s,
            "joint_transitions": joint_n,
            "chain_transitions": joint_n * arm["batch"],
            "seconds_per_joint_step": joint,
            "seconds_per_chain_transition": joint / arm["batch"],
            "chain_transitions_per_wall_second": arm["batch"] / joint,
            "min_block_seconds": float(min(item["seconds"] for item in taken)),
            "scalar_checksum": float(arm["checksum"].detach().cpu()),
        }
    narrow, wide = arms[0]["id"], arms[1]["id"]
    return {
        "order": order, "blocks_per_cell": blocks, "min_seconds": min_seconds,
        "met_duration_rule": bool(min_seconds <= 0 or all(
            cell["min_block_seconds"] >= min_seconds for cell in cells.values())),
        "cells": cells, "narrow": narrow, "wide": wide,
        "cost_ratio_wide_over_narrow": (cells[wide]["seconds_per_joint_step"]
                                        / cells[narrow]["seconds_per_joint_step"]),
        "convention": ("widths alternate on one device; each width keeps its own synchronized "
                       "elapsed time, transition count, and block length; a joint step advances "
                       "every chain and wall time is not multiplied by the chain count; "
                       "scalar logging is inside the measured block; length-finding probes are excluded"),
        "hardware": hardware_record(arms[0]["device"]),
    }


COST_TASKS = (
    {"id": "deep_m0032_r0", "arch": "deep", "m": 32, "rep": 0, "kind": "baseline"},
    {"id": "deep_m0256_r0", "arch": "deep", "m": 256, "rep": 0, "kind": "baseline"},
)


def _ess_ratio(cfg: dict[str, Any], architecture: str, replicate: int) -> float:
    path = RC.output_root(cfg) / "break_even_ratios.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["architecture"] == architecture and int(row["replicate"]) == int(replicate):
                return float(row["ess_ratio"])
    raise FileNotFoundError(f"{path} has no {architecture} replicate {replicate}")


def cost_pilot(cfg: dict[str, Any], *, device: Optional[str] = None) -> dict[str, Any]:
    """Deep widths 32 and 256, replicate 0, one device, four-chain scalar-logged blocks."""
    dev = pick_device(device or cfg.get("device"))
    timing = cfg["timing"]
    h_dim = float(cfg["fixed_step"]["h_over_sigma2"])
    arms = []
    for task in COST_TASKS:
        built = build_target(cfg, task, dev)
        arms.append(_arm(built, cfg, h_dim * float(built["identity"]["sigma"]) ** 2, "cost-pilot"))
    report = interleaved_cost(
        arms, transitions=int(timing["transitions"]), blocks=int(timing["blocks"]),
        min_seconds=float(timing["min_seconds"]))
    threshold = _ess_ratio(cfg, "deep", 0)
    narrow_id, wide_id = report["narrow"], report["wide"]
    comparison = timed_width_comparison(
        [(item, report["cells"][narrow_id]["transitions_per_block"])
         for item in report["cells"][narrow_id]["block_seconds"]],
        [(item, report["cells"][wide_id]["transitions_per_block"])
         for item in report["cells"][wide_id]["block_seconds"]],
        threshold)
    if not report["met_duration_rule"]:
        comparison["decision"] = "inconclusive"
        comparison["favors_wide"] = None
        comparison["duration_rule"] = "a measured block was shorter than one second"
    else:
        comparison["duration_rule"] = "every measured block lasted at least one second"
    report["h_over_sigma2"] = h_dim
    report["h"] = h_dim * float(arms[0]["sigma"]) ** 2
    report["device"] = str(dev)
    report["break_even_ess_ratio"] = threshold
    report["comparison"] = comparison
    report["wide_wins_on_ess_per_second"] = comparison["favors_wide"]
    out = RC.output_root(cfg) / "cost_pilot_deep_r0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, clean_json(report))
    report["path"] = str(out)
    return report


def ou_control(sigma: float, h: float, *, dim: int = 8, n: int = 400, chains: int = 4, seed: int = 0) -> dict[str, Any]:
    """Prior-only PCNL on a standard Gaussian. Acceptance must be one and lag-1 must track eta."""
    dev = torch.device("cpu")
    theta0 = torch.zeros(dim, dtype=torch.float64)

    def pg(theta):
        return (torch.zeros(theta.shape[0], dtype=theta.dtype), torch.zeros_like(theta), None)

    sampler = PCNL(pg, theta0, sigma, h)
    rngs = [ChainRNG(seed + 17 * c, dev) for c in range(chains)]
    init = torch.stack([sigma * torch.randn(dim, generator=ChainRNG(seed + c, dev).torch, dtype=torch.float64)
                        for c in range(chains)])
    state = sampler.init_state(init)
    series = np.empty((chains, n))
    accepts = 0
    for t in range(n):
        acc, _lam = sampler.step(state, rngs)
        accepts += int(np.sum(acc))
        series[:, t] = state.theta[:, 0].detach().cpu().numpy()
    centered = series - series.mean(axis=1, keepdims=True)
    lag1 = float(np.mean(np.sum(centered[:, :-1] * centered[:, 1:], axis=1) / np.sum(centered * centered, axis=1)))
    eta = math.exp(-h / sigma ** 2)
    return {"acceptance": accepts / (n * chains), "lag1": lag1, "eta": eta, "abs_error": abs(lag1 - eta),
            "n": n, "chains": chains}


def run_task(cfg: dict[str, Any], stage: str, task: dict[str, Any], *, device: Optional[str] = None) -> dict[str, Any]:
    dev = pick_device(device or cfg.get("device"))
    built = build_target(cfg, task, dev)
    root = RC.output_root(cfg) / "runs" / task["id"] / stage
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "identity.json", clean_json(built["identity"]))
    h_dim = float(cfg.get("fixed_step", {}).get("h_over_sigma2", 0.01))
    sigma = float(built["model"].sigma)
    h = h_dim * sigma ** 2
    chains = list(range(int(cfg.get("fixed_step", {}).get("chains", 4))))
    if stage == "smoke":
        n = int(cfg.get("fixed_step", {}).get("transitions", 24))
        stride = int(cfg.get("fixed_step", {}).get("checkpoint_stride", 8))
    elif stage == "local-pilot":
        n = int(cfg.get("local_pilot", {}).get("transitions", 32))
        stride = int(cfg.get("local_pilot", {}).get("checkpoint_stride", 8))
    else:
        n = int(cfg.get("fixed_step", {}).get("transitions", 8192))
        stride = int(cfg.get("fixed_step", {}).get("checkpoint_stride", 256))
    summary: dict[str, Any] = {"task": task, "h": h, "sigma": sigma, "device": str(dev),
                               "hardware": hardware_record(dev)}
    if stage == "tune":
        summary["tuning"] = _tune(cfg, built, root, chains, sigma)
    elif stage == "prior-starts":
        summary["prior_starts"] = _prior_starts(cfg, built, root, sigma, h)
    elif stage == "prior-only":
        summary["run"] = run_pcnl(built, cfg, h=h, n=n, chains=chains, out=root / "prior_only",
                                  segment="production", stage=f"prior-only|{h}", prior_only=True,
                                  init="prior", checkpoint_stride=stride)
    else:
        summary["run"] = run_pcnl(built, cfg, h=h, n=n, chains=chains, out=root / "fixed",
                                  segment="production", stage=f"fixed|{h}", init="center",
                                  checkpoint_stride=stride)
        if stage in ("smoke", "local-pilot", "fixed-step"):
            if stage == "local-pilot":
                # A laptop forecast. The one-second block rule is applied by the cluster fixed-step stage.
                block_n, blocks, min_s = int(cfg.get("local_pilot", {}).get("timing_transitions", 16)), 2, 0.0
                summary["timing_label"] = "short-block laptop forecast; not the one-second cluster protocol"
            else:
                block_n = int(cfg.get("timing", {}).get("transitions", 8 if stage == "smoke" else 32))
                blocks = int(cfg.get("timing", {}).get("blocks", 2 if stage != "fixed-step" else 5))
                min_s = float(cfg.get("timing", {}).get("min_seconds", 0.0 if stage != "fixed-step" else 1.0))
            summary["timing"] = kernel_timing(built, cfg, h, transitions=block_n, blocks=blocks, min_seconds=min_s)
            try:
                loss = read_loss(root / "fixed", chains, "production")
                summary["loss_rate"] = scalar_rate(loss, seconds_per_transition=summary["timing"]["seconds_per_transition"])
            except Exception as exc:  # a partial smoke interrupt has no consolidated file
                summary["loss_rate"] = {"status": "unavailable", "reason": str(exc)}
    atomic_write_json(root / "summary.json", clean_json(summary))
    return summary


def _tune(cfg: dict[str, Any], built: dict[str, Any], root: Path, chains: list[int], sigma: float) -> dict[str, Any]:
    tune = cfg.get("tuning", {})
    candidates = [float(v) for v in tune.get("h_over_sigma2", [0.01, 0.02, 0.04])]
    pilot_n = int(tune.get("pilot_transitions", 32))
    scored = []
    for h_dim in candidates:
        h = h_dim * sigma ** 2
        warm = warmup_steps(h, sigma, time_over_sigma2=float(tune.get("warmup_time_over_sigma2", 20.0)))
        # The pilot window is the configured length after the fixed warm-up, capped so a smoke config stays short.
        total = warm + pilot_n
        run_pcnl(built, cfg, h=h, n=total, chains=chains, out=root / f"pilot_{h_dim}", segment="pilot",
                 stage=f"tune|{h_dim}", init="center", checkpoint_stride=max(1, total // 4))
        loss = read_loss(root / f"pilot_{h_dim}", chains, "pilot")[:, warm:]
        rate = scalar_rate(loss)
        ess_s = rate["ess_per_transition"] or float("nan")
        scored.append((h_dim, ess_s))
    choice = select_step(scored)
    return {"scores_ess_per_transition": scored, "selection": choice,
            "note": "smoke and short pilots score ESS/transition because a one-second block is a cluster measurement"}


def _prior_starts(cfg: dict[str, Any], built: dict[str, Any], root: Path, sigma: float, h: float) -> dict[str, Any]:
    spec = cfg.get("prior_starts", {})
    n_starts = int(spec.get("n_starts", 4))
    horizon = int(spec.get("horizon", 16))
    batch = int(spec.get("batch", 4))
    # One batch is enough to exercise the code path. Further batches are separate chain-id groups.
    groups = []
    done = 0
    while done < n_starts:
        take = min(batch, n_starts - done)
        chains = list(range(done, done + take))
        groups.append(run_pcnl(built, cfg, h=h, n=horizon, chains=chains, out=root / f"batch_{done}",
                               segment="from_prior", stage=f"prior-start|{h}", init="prior",
                               checkpoint_stride=max(1, horizon // 4)))
        done += take
    return {"batches": groups, "n_starts": n_starts, "horizon": horizon,
            "reference": "independent elliptical-slice reference means are not in this short run"}


def smoke_suite(cfg: dict[str, Any]) -> dict[str, Any]:
    """Analytic prior-only check plus one tiny timed chain. A second open must not add transitions."""
    root = RC.output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    ou = ou_control(1.0, 0.01, n=int(cfg.get("ou_length", 500)))
    task = tasks_for(cfg, "smoke")[0]
    summary = run_task(cfg, "smoke", task, device="cpu")
    built = build_target(cfg, task, torch.device("cpu"))
    n = int(cfg.get("fixed_step", {}).get("transitions", 24))
    chains = list(range(int(cfg.get("fixed_step", {}).get("chains", 4))))
    rerun = run_pcnl(built, cfg, h=summary["h"], n=n, chains=chains,
                     out=root / "runs" / task["id"] / "smoke" / "fixed", segment="production",
                     stage=f"fixed|{summary['h']}", init="center",
                     checkpoint_stride=int(cfg.get("fixed_step", {}).get("checkpoint_stride", 8)))
    report = {"ou_control": ou, "smoke_run": summary, "resume_record": rerun["record"],
              "slice": slice_bracket_smoke(cfg), "hardware": hardware_record(torch.device("cpu"))}
    atomic_write_json(root / "smoke_report.json", clean_json(report))
    return report


def refuse_foreign_checkpoint(path: Path, built: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Opening a finished trajectory with a different target hash must raise StorageError."""
    model = built["model"]
    foreign = dict(built)
    foreign["identity"] = dict(built["identity"])
    foreign["identity"]["target_hash"] = "0" * 64
    sampler = PCNL(model.V_and_grad, model.theta0, model.sigma, 0.01)
    rngs = [ChainRNG(1, model.theta0.device)]
    traj = Trajectory(path, kind="pcnl", sampler=sampler, rngs=rngs, chains=[0],
                      hashes={"target_hash": foreign["identity"]["target_hash"], "segment_family": "fixed|0.01"},
                      record=built["record"], names=built["names"], checkpoint_stride=8)
    try:
        traj.open(lambda: model.theta0.unsqueeze(0))
    except StorageError:
        return
    raise AssertionError("incompatible target hash was resumed")


def slice_bracket_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """A few elliptical-slice transitions on the smoke target, counting bracket evaluations."""
    task = tasks_for(cfg, "smoke")[0]
    built = build_target(cfg, task, torch.device("cpu"))
    model = built["model"]
    sampler = EllipticalSlice(model.V, model.theta0, model.sigma, guard=1000)
    rngs = [ChainRNG(100 + c, model.theta0.device) for c in range(2)]
    state, _ = sampler.init_state(prior_draws(model.theta0, model.sigma, rngs))
    # Fresh streams for the transitions so initialization does not consume the slice uniforms.
    rngs = [ChainRNG(200 + c, model.theta0.device) for c in range(2)]
    evals = [sampler.step(state, rngs) for _ in range(3)]
    stacked = np.stack(evals, axis=0)
    return {"transitions": 3, "chains": 2, "mean_bracket_evaluations": float(stacked.mean()),
            "p95_bracket_evaluations": float(np.quantile(stacked, 0.95)),
            "total_evaluations": int(stacked.sum())}
