"""Stage functions behind ``python -m bnn_geometry run`` (runbook §15.1) and the SLURM arrays."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional

from . import config as C
from .context import TargetContext, pick_device
from .data import freeze_replicate
from .storage import atomic_write_json, clean_json, read_json

TARGET_STAGES = ("reference", "production", "refine")
GLOBAL_STAGES = ("freeze-data", "select-step", "endpoint-decision", "refine-decision", "controls")


def freeze_data(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    info = {str(r): freeze_replicate(cfg, root, int(r)) for r in cfg["replicates"]}
    ids = [t.target_id for t in C.targets(cfg)]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate target ids")
    atomic_write_json(root / "data" / "summary.json", clean_json(info))
    return info


def ensure_data(cfg: dict[str, Any], root: Path) -> None:
    if not all((root / "data" / f"rep_{r}.npz").exists() for r in cfg["replicates"]):
        freeze_data(cfg, root)


def stage_reference(cfg: dict[str, Any], root: Path, tid: str, device=None) -> dict[str, Any]:
    """Array stage A: reference sampling with gates and static estimates, step calibration, predictive."""
    from .controls import predictive
    from .dynamics import step_calibration
    from .reference import run_reference
    ensure_data(cfg, root)
    ctx = TargetContext(cfg, C.parse_target(tid), root, device)
    st = run_reference(ctx)
    if (st.get("reference") or {}).get("status") == "reference_error":
        return st
    if (st.get("reference") or {}).get("stopping_stage"):
        predictive(ctx)
        step_calibration(ctx)
    return ctx.status()


def stage_production(cfg: dict[str, Any], root: Path, tid: str, device=None) -> dict[str, Any]:
    from .dynamics import production_stage
    ctx = TargetContext(cfg, C.parse_target(tid), root, device)
    if not (ctx.status().get("reference") or {}).get("stopping_stage"):
        return ctx.update_status(dynamics={"status": "not_run_no_reference_state"})
    steps = read_json(root / "dynamics_step.json")
    return production_stage(ctx, steps[ctx.t.arch]["h"])


def stage_refine(cfg: dict[str, Any], root: Path, tid: str, device=None) -> dict[str, Any]:
    from .dynamics import refinement_stage
    ctx = TargetContext(cfg, C.parse_target(tid), root, device)
    if not (ctx.status().get("reference") or {}).get("stopping_stage"):
        return ctx.status()
    return refinement_stage(ctx, read_json(root / "endpoint_decision_round1.json"))


def run_target_stage(cfg: dict[str, Any], root: Path, stage: str, tid: str, device=None) -> dict[str, Any]:
    fn = {"reference": stage_reference, "production": stage_production, "refine": stage_refine}[stage]
    return fn(cfg, root, tid, device)


def run_global_stage(cfg: dict[str, Any], root: Path, stage: str, device=None) -> Any:
    from .controls import prior_spectral
    from .dynamics import endpoint_decision, select_steps
    if stage == "freeze-data":
        return freeze_data(cfg, root)
    if stage == "select-step":
        return select_steps(cfg, root)
    if stage == "endpoint-decision":
        return endpoint_decision(cfg, root, 1)
    if stage == "refine-decision":
        return endpoint_decision(cfg, root, 2)
    if stage == "controls":
        return prior_spectral(cfg, root, device)
    raise ValueError(stage)


def run_all(cfg: dict[str, Any], root: Path, device=None, log=print) -> None:
    """Sequential campaign (single machine): the same stages the SLURM pipeline runs in parallel."""
    device = device or pick_device()
    freeze_data(cfg, root)
    tids = [t.target_id for t in C.targets(cfg)]
    for stage_name, stage in (("reference", "reference"),):
        for tid in tids:
            log(f"[{stage_name}] {tid}")
            _guard(lambda: run_target_stage(cfg, root, stage, tid, device), root, stage, tid)
    log("[select-step]")
    run_global_stage(cfg, root, "select-step", device)
    for tid in tids:
        log(f"[production] {tid}")
        _guard(lambda: run_target_stage(cfg, root, "production", tid, device), root, "production", tid)
    log("[endpoint-decision]")
    run_global_stage(cfg, root, "endpoint-decision", device)
    for tid in tids:
        log(f"[refine] {tid}")
        _guard(lambda: run_target_stage(cfg, root, "refine", tid, device), root, "refine", tid)
    log("[refine-decision]")
    run_global_stage(cfg, root, "refine-decision", device)
    log("[controls]")
    run_global_stage(cfg, root, "controls", device)


def _guard(fn, root: Path, stage: str, tid: str) -> Optional[Any]:
    """Record a deterministic exception in the target status instead of aborting the whole campaign."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        p = root / "targets" / tid / "errors.json"
        errs = read_json(p) if p.exists() else []
        errs.append({"stage": stage, "error": repr(exc), "traceback": traceback.format_exc()})
        atomic_write_json(p, errs)
        return None
