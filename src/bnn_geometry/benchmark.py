"""Hardware benchmark and resource estimate for the widest production architectures (runbook §6.3)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import config as C
from .chains import prior_draws
from .context import TargetContext, pick_device
from .ellipse import EllipticalSlice
from .pcnl import PCNL
from .pipeline import ensure_data
from .randomness import ChainRNG
from .spectral import state_S
from .storage import write_trace_file


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time(fn, reps: int, dev: torch.device) -> float:
    fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / reps


def benchmark(cfg: dict[str, Any], root: Path, device=None) -> pd.DataFrame:
    dev = pick_device(device)
    ensure_data(cfg, root)
    rows = []
    s2 = float(cfg["model"]["sigma"]) ** 2
    r, dy = cfg["reference"], cfg["dynamics"]
    pre = r["burn_in_transitions"] + r["calibration_transitions"] + r["separation_transitions"]
    for arch in ("shallow", "deep"):
        m = int(cfg["model"][arch]["widths"][-1])
        ctx = TargetContext(cfg, C.Target(arch, m, int(cfg["replicates"][0])), root, dev)
        nch = len(ctx.chains)
        rngs = [ChainRNG(1000 + c, dev) for c in range(nch)]
        th = prior_draws(ctx.model.theta0, ctx.sigma, rngs)
        t_v = _time(lambda: ctx.model.V(th), 200, dev)
        t_vg = _time(lambda: ctx.model.V_and_grad(th), 200, dev)
        t_probe = _time(lambda: ctx.obs.probe_values_and_grad_sq(th[:1]), 20, dev)
        V, f = ctx.model.V(th)
        t_rec = _time(lambda: ctx.obs.record(th, V, f), 200, dev)
        t_svd = _time(lambda: state_S(th[:1], ctx.lay, ctx.a), 20, dev) if arch == "deep" else float("nan")
        ess = EllipticalSlice(ctx.model.V, ctx.model.theta0, ctx.sigma)
        st, _ = ess.init_state(th)
        for _ in range(200):
            ess.step(st, rngs)
        _sync(dev)
        t0, ev = time.perf_counter(), 0
        for _ in range(100):
            ev += int(ess.step(st, rngs).sum())
        _sync(dev)
        t_ess = (time.perf_counter() - t0) / 100
        pc = PCNL(ctx.model.V_and_grad, ctx.model.theta0, ctx.sigma, float(dy["h_over_sigma_squared_candidates"][0]) * s2)
        pst = pc.init_state(st.theta)
        t_pc = _time(lambda: pc.step(pst, rngs), 200, dev)
        with tempfile.TemporaryDirectory() as tmp:
            K = len(ctx.names)
            arr = {"scalars": np.zeros((1024, K)), "draw": np.arange(1024), "evals": np.ones(1024, dtype=np.int64),
                   "archive/draw": np.arange(0, 1024, 16), "archive/states": np.zeros((64, ctx.lay.p))}
            t0 = time.perf_counter()
            write_trace_file(Path(tmp) / "x.h5", arrays=arr, attrs={"chain": 0})
            t_io = time.perf_counter() - t0
        h0 = float(dy["h_over_sigma_squared_candidates"][0]) * s2
        T0 = float(dy["cumulative_retained_time_over_sigma_squared"][0]) * s2
        Tmax = float(dy["cumulative_retained_time_over_sigma_squared"][-1]) * s2
        disc = float(dy["production_discard_time_over_sigma_squared"]) * s2
        n_dyn0 = C.n_steps(T0 + disc, h0)
        n_dynmax = C.n_steps(Tmax + disc, float(dy["h_over_sigma_squared_candidates"][-1]) * s2 / 2)
        ref0 = (pre + r["cumulative_production_stages"][0]) * (t_ess + t_rec) + \
            r["cumulative_production_stages"][0] / 1024 * nch * t_io
        refmax = (pre + r["cumulative_production_stages"][-1]) * (t_ess + t_rec)
        static = int(cfg["static"]["selected_states_per_chain_stages"][0]) * nch * t_probe
        rows.append({"architecture": arch, "m": m, "p": ctx.lay.p, "device": str(dev), "chains_batched": nch,
                     "V_eval_s": t_v, "V_grad_eval_s": t_vg, "probe_11_gradients_s": t_probe, "svd_s": t_svd,
                     "scalar_record_s": t_rec, "ess_transition_s": t_ess, "ess_evals_per_chain_transition": ev / (100 * nch),
                     "pcnl_transition_s": t_pc, "chunk_write_s": t_io,
                     "est_reference_initial_h": ref0 / 3600, "est_reference_max_h": refmax / 3600,
                     "est_static_initial_h": static / 3600,
                     "est_dynamics_initial_h_at_h0": n_dyn0 * (t_pc + t_rec) / 3600,
                     "est_dynamics_worst_h": n_dynmax * (t_pc + t_rec) / 3600,
                     "cpu_threads": torch.get_num_threads(),
                     "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None})
    df = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "benchmark.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.T.to_string())
    print("Estimates are per target (4 chains batched on one device); worst dynamics = smallest candidate step "
          "halved by the refinement, at the maximal duration. Array tasks run targets in parallel.")
    return df
