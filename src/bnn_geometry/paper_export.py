"""Export of the fixed-step training-loss traces for the two-figure paper package (figure guide §2, §5, §9.1).

Reads the completed campaign's h=0.01 dynamics trajectories directly from their HDF5 segment files, verifies their
identity (target/execution hashes, kernel, step, chain ids, contiguous unit-stride transitions, retained rejected
moves, end-state V recomputed from the final checkpoint) and writes one NPZ per target holding the last
``window`` retained production transitions of V for each chain. No sampling is performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import config as C
from .storage import StorageError, read_json, read_trace_file, segment_path, torch_load, validate_trace

OBSERVABLE = "summed_training_cross_entropy"
DEFAULT_WINDOW = 102400


def selection_rule(window: int) -> str:
    return f"last_{int(window)}_retained_production_transitions"


def sha256_file(path: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def h_dir_name(h: float) -> str:
    return "h_" + repr(float(h)).replace(".", "p").replace("-", "m").replace("+", "")


def _retained_segments(order: list[str]) -> list[str]:
    stages = [s for s in order if s.startswith("stage_")]
    if [s for s in order if s not in stages] != ["discard"] or order[0] != "discard":
        raise StorageError(f"unexpected segment order {order}")
    if stages != [f"stage_{i}" for i in range(1, len(stages) + 1)]:
        raise StorageError(f"retained stages not consecutive: {stages}")
    return stages


def export_target(cfg: dict[str, Any], root: Path, t: C.Target, out_dir: Path, *, h: float, window: int,
                  device: str = "cpu", recompute_end_state: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Export one target. Returns (log record, input-map rows). Raises StorageError on any integrity failure."""
    tdir = root / "targets" / t.target_id
    spec = read_json(tdir / "spec.json")
    traj = tdir / "dynamics" / h_dir_name(h)
    ck_path = traj / "checkpoint.pt"
    if not ck_path.exists():
        raise StorageError(f"{t.target_id}: no dynamics trajectory at h={h} ({traj})")
    ck = torch_load(ck_path)
    meta = ck.get("meta") or {}
    problems = []
    if ck.get("kind") != "pcnl":
        problems.append(f"kind={ck.get('kind')}")
    if meta.get("role") != "dynamics":
        problems.append(f"role={meta.get('role')}")
    if float(meta.get("h", float("nan"))) != float(h):
        problems.append(f"h={meta.get('h')}")
    if ck.get("chains") != [int(c) for c in cfg["chain_ids"]]:
        problems.append(f"chains={ck.get('chains')}")
    if ck["hashes"].get("target_hash") != spec["target_hash"]:
        problems.append("target_hash differs from spec.json")
    if ck.get("current") is not None:
        problems.append(f"unfinished segment {ck['current'].get('name')}")
    if ck.get("status") not in ("running", None):
        problems.append(f"status={ck.get('status')}")
    expect_exec = C.execution_hash(cfg, spec["target_hash"], "pcnl", step=h, role="dynamics")
    exec_ok = ck["hashes"].get("execution_hash") == expect_exec
    code_ok = C.sampler_code_hash() == spec.get("sampler_code_hash")
    if code_ok and not exec_ok:
        problems.append("execution_hash differs from the campaign kernel/settings at this step")
    if problems:
        raise StorageError(f"{t.target_id}: ineligible trajectory: {'; '.join(problems)}")

    order = list(ck["order"])
    stages = _retained_segments(order)
    s2 = float(cfg["model"]["sigma"]) ** 2
    n_discard = C.n_steps(float(cfg["dynamics"]["production_discard_time_over_sigma_squared"]) * s2, h)
    if ck["done"]["discard"]["n"] != n_discard or ck["done"]["discard"]["partial"]:
        raise StorageError(f"{t.target_id}: discard segment has {ck['done']['discard']['n']} != {n_discard}")
    runs = read_json(tdir / "analysis" / "dynamics_runs.json")
    run = runs.get(h_dir_name(h)) or {}
    # An endpoint trajectory first run as a fixed-length half-step check can hold more stages than the later
    # production analysis needed; every stage is the same continuous chain, so all of them are retained.
    fs = run.get("final_stage")
    if not isinstance(fs, int) or fs < 1 or fs > len(stages):
        raise StorageError(f"{t.target_id}: dynamics_runs final_stage {fs} inconsistent with {len(stages)} stages")
    if not (tdir / "analysis" / f"dynamics_{h_dir_name(h)}_stage_{len(stages)}.json").exists():
        raise StorageError(f"{t.target_id}: no campaign analysis for stage_{len(stages)}")
    hashes = {k: ck["hashes"][k] for k in ("target_hash", "execution_hash")}

    V_rows, A_rows, starts, ends, files, rows = [], [], [], [], [], []
    v_all, n_ret = [], None
    for c in ck["chains"]:
        parts_v, parts_a, offset = [], [], 0
        for seg in stages:
            rec = ck["done"][seg]
            if rec["partial"]:
                raise StorageError(f"{t.target_id}: segment {seg} is partial")
            p = segment_path(traj / f"chain_{c}", seg)
            arr, attrs = read_trace_file(p, ["scalars", "draw", "accepted"])
            validate_trace(arr, attrs, expect_first=0, expect_n=rec["n"], hashes=hashes, chain=c, where=str(p))
            if int(attrs.get("saved_stride", 0)) != 1:
                raise StorageError(f"{p}: saved_stride {attrs.get('saved_stride')} != 1")
            names = list(attrs["names"])
            if names[0] != "V" or names != list(spec["scalar_names"]):
                raise StorageError(f"{p}: scalar names do not match spec.json (V must be column 0)")
            if "accepted" not in arr or arr["accepted"].shape[0] != rec["n"]:
                raise StorageError(f"{p}: acceptance flags missing")
            parts_v.append(np.asarray(arr["scalars"][:, 0], dtype=np.float64))
            parts_a.append(np.asarray(arr["accepted"], dtype=np.int8))
            sc = arr["scalars"]
            rows.append({"logical_input": "dynamics_loss_trace", "target_id": t.target_id, "chain": c,
                         "segment": seg, "path": str(p.relative_to(root)), "dataset": "scalars[:, 0] (V)",
                         "shape": f"{sc.shape[0]}x{sc.shape[1]}", "dtype": str(sc.dtype),
                         "sha256": sha256_file(p), "first_transition": n_discard + offset + 1,
                         "last_transition": n_discard + offset + rec["n"],
                         "mapping": f"{t.target_id}/chain_{c}; post-transition V incl. rejections; h={h}"})
            files.append(str(p.relative_to(root)))
            offset += rec["n"]
        v = np.concatenate(parts_v)
        a = np.concatenate(parts_a)
        n_ret = v.shape[0] if n_ret is None else n_ret
        if v.shape[0] != n_ret:
            raise StorageError(f"{t.target_id}: chains have different retained lengths")
        if v.shape[0] < window:
            raise StorageError(f"{t.target_id}/chain_{c}: {v.shape[0]} retained transitions < window {window}")
        if not np.all(np.isfinite(v)):
            raise StorageError(f"{t.target_id}/chain_{c}: non-finite V")
        # a rejected transition repeats the current state, so V must be exactly unchanged there
        rej = np.flatnonzero(a[1:] == 0) + 1
        if rej.size and not np.array_equal(v[rej], v[rej - 1]):
            raise StorageError(f"{t.target_id}/chain_{c}: V changes at a rejected transition")
        v_all.append(v)
        V_rows.append(v[-window:])
        A_rows.append(a[-window:])
        starts.append(n_discard + v.shape[0] - window + 1)
        ends.append(n_discard + v.shape[0])
    V = np.stack(V_rows)
    if np.any(np.ptp(V, axis=1) == 0):
        raise StorageError(f"{t.target_id}: constant V window")

    # identity of the column with the analysed V: the campaign's own stage analysis recorded mean V over all
    # retained transitions of all chains
    an = read_json(tdir / "analysis" / f"dynamics_{h_dir_name(h)}_stage_{len(stages)}.json")
    vm = next(r for r in an["reference_comparison"] if r["observable"] == "V")["dynamics_mean"]
    full_mean = float(np.mean(np.stack(v_all)))
    if not math.isclose(full_mean, vm, rel_tol=1e-12, abs_tol=1e-12):
        raise StorageError(f"{t.target_id}: mean V {full_mean} != campaign analysis {vm}")

    # V really is the summed training cross-entropy of the final state (recomputed from the checkpoint)
    end_check = None
    if recompute_end_state:
        import torch
        from .context import TargetContext
        ctx = TargetContext(cfg, t, root, device=torch.device(device))
        th = ck["state"]["theta"].to(ctx.device, torch.float64)
        Vre = ctx.model.V(th)[0].detach().cpu().numpy()
        last = np.array([v[-1] for v in v_all])
        end_check = float(np.max(np.abs(Vre - last) / np.maximum(1.0, np.abs(last))))
        if end_check > 1e-9:
            raise StorageError(f"{t.target_id}: recomputed end-state V differs from the trace (rel {end_check:.2e})")

    exec_hash = hashes["execution_hash"]
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{t.target_id}.npz"
    np.savez_compressed(
        npz, V=V, accepted=np.stack(A_rows), chain_ids=np.asarray(ck["chains"], dtype=np.int64),
        iteration_start=np.asarray(starts, dtype=np.int64), iteration_end=np.asarray(ends, dtype=np.int64),
        save_stride=np.int64(1), h=np.float64(h), architecture=np.str_(t.arch), L=np.int64(t.L), m=np.int64(t.m),
        n=np.int64(cfg["data"]["train_size"]), d=np.int64(cfg["data"]["input_dimension"]),
        sigma=np.float64(cfg["model"]["sigma"]), replicate_id=np.int64(t.rep), observable=np.str_(OBSERVABLE),
        contains_rejected_transitions=np.bool_(True), target_hash=np.str_(spec["target_hash"]),
        sampler_code_hash=np.str_(spec.get("sampler_code_hash", "")), execution_hash=np.str_(exec_hash),
        source_run_ids=np.asarray([f"{exec_hash[:16]}/chain_{c}" for c in ck["chains"]]),
        source_filenames=np.asarray(files), selection_rule=np.str_(selection_rule(window)),
        discard_transitions=np.int64(n_discard), retained_transitions_per_chain=np.int64(n_ret),
        acceptance_window=np.asarray([float(np.mean(a)) for a in A_rows]))
    log = {"target_id": t.target_id, "architecture": t.arch, "m": t.m, "replicate": t.rep, "h": h,
           "trajectory": str(traj.relative_to(root)), "execution_hash": exec_hash,
           "execution_hash_matches_campaign": exec_ok, "sampler_code_hash_matches": code_ok,
           "segments": order, "production_analysis_final_stage": fs, "retained_stages": len(stages),
           "discard_transitions": n_discard, "retained_per_chain": n_ret, "window": window,
           "iteration_start": starts, "iteration_end": ends, "mean_V_matches_campaign_analysis": True,
           "end_state_V_max_rel_diff": end_check, "npz": str(npz.relative_to(root)), "npz_sha256": sha256_file(npz),
           "source_status": "recovered_existing_trace"}
    return log, rows


def export_all(cfg: dict[str, Any], root: Path, *, h: float = 0.01, window: int = DEFAULT_WINDOW,
               out: Optional[Path] = None, targets: Optional[list[str]] = None, device: str = "cpu",
               recompute_end_state: bool = True) -> int:
    out = Path(out) if out else root / "paper_figures"
    tr_dir = out / "loss_traces"
    logs, rows, fails = [], [], []
    for t in C.targets(cfg):
        if targets and t.target_id not in targets:
            continue
        try:
            lg, rw = export_target(cfg, root, t, tr_dir, h=h, window=window, device=device,
                                   recompute_end_state=recompute_end_state)
            logs.append(lg)
            rows += rw
            print(f"OK   {t.target_id}: retained {lg['retained_per_chain']} -> window "
                  f"[{lg['iteration_start'][0]}, {lg['iteration_end'][0]}]  end-state rel diff "
                  f"{lg['end_state_V_max_rel_diff']}", flush=True)
        except (StorageError, KeyError, OSError, StopIteration) as exc:
            fails.append({"target_id": t.target_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {t.target_id}: {exc}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    fn = out / "input_map_loss.csv"
    if rows:
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "loss_export_log.json").write_text(json.dumps(
        {"h": h, "window": window, "selection_rule": selection_rule(window), "observable": OBSERVABLE,
         "code_revision": C.code_revision(), "exported": logs, "failures": fails}, indent=2))
    print(f"exported {len(logs)} targets, {len(fails)} failures -> {tr_dir}")
    return 0 if not fails else 1
