"""Use the existing BNN target, kernels, scalar probes and resumable trajectories."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from .core import CHAINS, EXPECTED_CHI, PROBES, gate_decision, metrics_rates
from .core import needs_inline_check, parse_duration, production_seconds
from .core import remaining_joint_steps, require_comparative_device, segments_for
from .core import spectrum, task_matrix

HERE = Path(__file__).resolve().parent


def inputs(cfg, rep):
    from bnn_geometry.data import load_replicate
    from bnn_geometry.revision import config as rc

    data, bank, hashes = load_replicate(rc.baseline_root(cfg), rep)
    frozen = pd.read_csv(HERE / "baseline_identity.csv").set_index("target_id")
    key = frozen.loc[f"deep_m0256_r{rep}"]
    for field in ("hash_X", "hash_y", "hash_X_test", "hash_center_bank"):
        if hashes[field] != key[field]:
            raise ValueError(f"Replicate {rep} does not match the archived {field}")
    X = np.asarray(data["X"])
    summary = spectrum(X)
    if X.shape != (128, 32) or data["X_test"].shape != (1024, 32):
        raise ValueError(f"Unexpected baseline input shape for replicate {rep}")
    if bank.ndim != 2 or bank.shape[0] < 4096 or bank.shape[1] != 32:
        raise ValueError("The original first-layer center bank must include 4,096 rows")
    np.testing.assert_allclose([summary["row_norm_min"], summary["row_norm_max"]],
                               1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["chi"], EXPECTED_CHI[rep], rtol=1e-11)
    summary.update(rep=rep, archived_width256_target_hash=key.target_hash,
                   **{k: hashes[k] for k in ("hash_X", "hash_y", "hash_X_test",
                                             "hash_center_bank")})
    return data, summary


def export_inputs(cfg, out):
    path = out / "inputs"
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(3):
        data, summary = inputs(cfg, rep)
        np.savez_compressed(path / f"rep{rep}.npz", X=data["X"], y=data["y"],
                            X_test=data["X_test"])
        np.savetxt(path / f"X_train_r{rep}.csv", data["X"], delimiter=",", fmt="%.17g")
        rows.append(summary)
    pd.DataFrame(rows).to_csv(path / "input_geometry.csv", index=False)


@contextmanager
def lock(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sample(args, cfg, task):
    import torch
    from bnn_geometry.chains import Trajectory, prior_draws
    from bnn_geometry.context import pick_device
    from bnn_geometry.diagnostics import scalar_diagnostics
    from bnn_geometry.ellipse import EllipticalSlice
    from bnn_geometry.pcnl import PCNL
    from bnn_geometry.randomness import ChainRNG
    from bnn_geometry.revision.runner import build_target, _seeds
    from bnn_geometry.revision.timing import synchronize
    from bnn_geometry.storage import atomic_write_json, clean_json

    data, geometry = inputs(cfg, task["rep"])
    device = pick_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    require_comparative_device(args.action, device.type, device_name,
                               allow_any_device=args.allow_any_device)
    built = build_target(cfg, task, device)
    model, identity = built["model"], built["identity"]
    p_expected = task["m"] ** 2 + 33 * task["m"]
    if (identity["L"], identity["sigma"], identity["activation"], identity["p"]) != (
            3, 1.0, "tanh", p_expected):
        raise ValueError(f"Unexpected model identity: {identity}")
    if model.theta0.dtype != torch.float64 or identity["hash_X"] != geometry["hash_X"]:
        raise ValueError("Model dtype or input hash mismatch")
    np.testing.assert_array_equal(model.X.detach().cpu().numpy(), data["X"])
    if set(PROBES) - set(built["names"]):
        raise ValueError("Baseline probes missing")

    ref = args.kernel == "slice"
    h = args.step * model.sigma ** 2
    sampler = (EllipticalSlice(model.V, model.theta0, model.sigma) if ref
               else PCNL(model.V_and_grad, model.theta0, model.sigma, h))
    arm = "slice" if ref else f"pcnl_h_{args.step:g}"
    stage = f"deep-width-v1|{args.action}|{arm}"
    path = args.out / args.action / arm / task["id"]
    warmup, production = segments_for(args.kernel, args.level, args.step,
                                      args.pilot_steps if args.action == "pilot" else None)
    stride = 1024

    def rngs(stream):
        return [ChainRNG(_seeds(cfg, identity, c, stage, stream), device) for c in CHAINS]

    def write_json(file, value):
        atomic_write_json(file, clean_json(value))

    execution = clean_json(dict(
        device=device_name, dtype="float64", torch_version=str(torch.__version__),
        target=identity, input_geometry=geometry, arm=arm, chains=CHAINS,
        h_over_sigma2=None if ref else args.step, checkpoint_stride=stride,
        warmup=warmup, master_seed=int(cfg["master_seed"])))
    elapsed_path = path / "production_clock.json"
    with lock(path):
        execution_path = path / "execution.json"
        execution = json.loads(json.dumps(execution))
        if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
            raise ValueError("Execution manifest mismatch. Use a separate output directory.")
        write_json(execution_path, execution)
        traj = Trajectory(path, kind="ess" if ref else "pcnl", sampler=sampler,
                          rngs=rngs("sampler"), chains=CHAINS,
                          hashes=dict(target_hash=identity["target_hash"], segment_family=stage,
                                      master_seed=str(cfg["master_seed"])),
                          record=built["record"], names=built["names"], checkpoint_stride=stride)
        traj.open(lambda: prior_draws(model.theta0, model.sigma, rngs("init")))
        for name, steps in warmup:
            result = traj.run_segment(name, steps, archive_stride=0)
            if result["partial"] or result["n"] != steps:
                raise RuntimeError(f"Incomplete warmup {name}")
        ledger = json.loads(elapsed_path.read_text()) if elapsed_path.exists() else {}
        for name, steps in production:
            if traj.done(name):
                continue
            new = name not in ledger and traj.ck["current"] is None
            ledger[name] = None
            write_json(elapsed_path, ledger)
            synchronize(device)
            start = time.perf_counter()
            result = traj.run_segment(name, steps, archive_stride=0)
            synchronize(device)
            seconds = time.perf_counter() - start
            if result["partial"] or result["n"] != steps:
                raise RuntimeError(f"Incomplete production segment {name}")
            ledger[name] = seconds if new else None
            write_json(elapsed_path, ledger)
        counters = traj.ck["counters"]
        if any(counters["numerical_failures"]):
            raise RuntimeError(f"Sampler reported numerical failures: {counters}")

        elapsed = production_seconds(ledger, production)
        from a10_next.traces import four
        blocks, flags = [], []
        for name, steps in production:
            values, accepted = four(path, name, PROBES, steps, identity["target_hash"])
            if values.shape != (len(CHAINS), steps, len(PROBES)):
                raise ValueError(f"Unexpected trace shape {values.shape}")
            blocks.append(values)
            if not ref:
                if accepted is None or not np.isin(accepted, [0, 1]).all():
                    raise ValueError("Missing or invalid acceptance flags")
                mask = ~accepted[:, 1:].astype(bool)
                np.testing.assert_allclose(values[:, 1:, 0][mask], values[:, :-1, 0][mask],
                                           rtol=1e-10, atol=1e-10)
                flags.append(accepted)
        values = np.concatenate(blocks, axis=1)
        count = values.shape[1]
        acceptance = float(np.concatenate(flags, axis=1).mean()) if flags else None
        peak = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        reserved = (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
        rows = []
        for j, probe in enumerate(PROBES):
            row = scalar_diagnostics(probe, values[:, :, j]).row()
            row.update(cell=task["id"], width=task["m"], rep=task["rep"], probe=probe,
                       step=args.step if not ref else None, chains=len(CHAINS), steps=count,
                       chi=geometry["chi"], elapsed_seconds=elapsed, acceptance=acceptance,
                       device=device_name, target_hash=identity["target_hash"],
                       **metrics_rates(row["ess_mean"], elapsed, count))
            if args.action == "run":
                half = count // 2
                row["first_half_ess_mean"] = scalar_diagnostics(probe, values[:, :half, j]).row()["ess_mean"]
                row["second_half_ess_mean"] = scalar_diagnostics(probe, values[:, half:, j]).row()["ess_mean"]
            rows.append(row)
        metrics = path / f"metrics_{count}.csv"
        temporary = metrics.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(metrics)
        write_json(path / f"run_{count}.json", dict(counters=counters, seconds=elapsed,
                   segments=production, peak_cuda_allocated_bytes=peak,
                   peak_cuda_reserved_bytes=reserved,
                   checkpoint_bytes=traj.ck_path.stat().st_size,
                   likelihood_evaluations_total=sum(counters["evals"])))
        print(f"{task['id']} {arm} {count} retained per chain, {elapsed} production seconds")
        print(f"Peak CUDA allocated bytes: {peak}. Metrics: {metrics}")


def read_short_run(path, steps):
    """Read the short check written by ``sample``. Raise ``SystemExit`` when it is unusable."""
    clock_path = path / "production_clock.json"
    run_path = path / f"run_{steps}.json"
    metrics_path = path / f"metrics_{steps}.csv"
    missing = [str(item) for item in (clock_path, run_path, metrics_path) if not item.exists()]
    if missing:
        raise SystemExit("GATE_STOP: short run did not finish. Missing " + ", ".join(missing)
                         + ". Production was not started.")
    seconds = json.loads(clock_path.read_text()).get("pilot")
    run = json.loads(run_path.read_text())
    metrics = pd.read_csv(metrics_path)
    if (set(metrics["probe"]) != set(PROBES) or not metrics["steps"].eq(steps).all()
            or not metrics["chains"].eq(len(CHAINS)).all()
            or not np.isfinite(metrics["acceptance"]).all()
            or not np.isfinite(metrics["mean"]).all()):
        raise SystemExit("GATE_STOP: short-run metrics are incomplete or non-finite. "
                         "Production was not started.")
    acceptance = float(metrics["acceptance"].iloc[0])
    if not np.isclose(metrics["acceptance"], acceptance).all():
        raise SystemExit("GATE_STOP: acceptance differs across probes. Production was not started.")
    seconds_per_step = None
    if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0:
        seconds_per_step = float(seconds) / steps
    return dict(seconds_per_step=seconds_per_step, acceptance=acceptance,
                failures=list(run["counters"]["numerical_failures"]),
                peak_reserved_bytes=run.get("peak_cuda_reserved_bytes"))


def slurm_remaining_seconds():
    """Remaining allocation, or ``None`` when this process is not a running Slurm job."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        output = subprocess.check_output(
            ["squeue", "-h", "-j", job_id, "-o", "%L"], text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        print("GATE_TIME_UNKNOWN: squeue did not return a remaining time; continuing")
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        print("GATE_TIME_UNKNOWN: job is not visible to squeue; continuing")
        return None
    return parse_duration(lines[-1])


def gpu_capacity_bytes(device_name):
    if device_name != "cuda" and not str(device_name).startswith("cuda"):
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(0).total_memory)


def continue_after_short_check(args, task, info, *, remaining_seconds, total_gpu_bytes, ledger):
    warmup, production = segments_for(args.kernel, args.level, args.step)
    remaining = remaining_joint_steps(warmup, production, ledger)
    decision = gate_decision(
        seconds_per_step=info["seconds_per_step"], acceptance=info["acceptance"],
        failures=info["failures"], remaining_steps=remaining,
        remaining_seconds=remaining_seconds, peak_reserved_bytes=info["peak_reserved_bytes"],
        total_gpu_bytes=total_gpu_bytes)
    if not decision["ok"]:
        raise SystemExit("GATE_STOP: " + "; ".join(decision["reasons"])
                         + ". Production was not started.")
    projected = decision["projected_s"]
    projected_text = "n/a" if projected is None else f"{projected / 3600:.2f}h"
    low = " acceptance_low=true" if info["acceptance"] < 0.05 else ""
    print(f"GATE_PASS {task['id']} acceptance={info['acceptance']:.6f} "
          f"seconds_per_joint_step={info['seconds_per_step']:.6g} "
          f"remaining_steps={remaining} projected_with_headroom={projected_text}{low}")
    return decision


def short_check_then_run(args, cfg, task):
    import copy
    pilot_args = copy.copy(args)
    pilot_args.action = "pilot"
    sample(pilot_args, cfg, task)
    arm = f"pcnl_h_{args.step:g}"
    info = read_short_run(args.out / "pilot" / arm / task["id"], args.pilot_steps)
    production_path = args.out / "run" / arm / task["id"]
    ledger = None
    clock_path = production_path / "production_clock.json"
    if clock_path.exists():
        ledger = json.loads(clock_path.read_text())
    continue_after_short_check(
        args, task, info, remaining_seconds=slurm_remaining_seconds(),
        total_gpu_bytes=gpu_capacity_bytes(args.device), ledger=ledger)
    sample(args, cfg, task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "inputs", "pilot", "run"])
    parser.add_argument("--config", default="configs/revision.yaml")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--level", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--step", type=float, choices=[0.01, 0.04, 0.08, 0.16, 0.32], default=0.16)
    parser.add_argument("--kernel", choices=["pcnl", "slice"], default="pcnl")
    parser.add_argument("--out", type=Path, default=Path("results/deep_width"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot-steps", type=int, default=128)
    parser.add_argument("--no-gate", action="store_true",
                        help="Skip the in-job width-4096 check and start production directly")
    parser.add_argument("--allow-any-device", action="store_true",
                        help="Record the actual CUDA GPU instead of requiring an A10")
    args = parser.parse_args()
    tasks = task_matrix()
    if args.action == "list":
        for i, task in enumerate(tasks):
            print(i, task["id"])
        return
    if not 0 <= args.task < len(tasks) or args.pilot_steps < 16:
        parser.error("Use a listed task and at least 16 pilot steps")
    from bnn_geometry.revision import config as rc
    cfg = rc.load(args.config)
    if args.action == "inputs":
        export_inputs(cfg, args.out)
    elif needs_inline_check(args.action, args.kernel, tasks[args.task]["m"]) and not args.no_gate:
        short_check_then_run(args, cfg, tasks[args.task])
    else:
        sample(args, cfg, tasks[args.task])


if __name__ == "__main__":
    main()
