"""Turn per-task summary JSON into tables. Trace files stay on the cluster."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from . import config as RC
from .runner import tasks_for

SUBMITTED_STAGES = (
    "fixed-step", "tune", "prior-starts", "prior-only", "geometry-pilot",
    "relu", "depth4", "prior-scale", "concentrated", "nonlinear", "sin",
)

RUN_FIELDS = (
    "stage", "task_id", "arch", "m", "rep", "kind", "r", "h", "sigma", "device", "gpu",
    "n", "planned", "partial", "stop_reason", "elapsed_s", "accepted_fraction",
    "seconds_per_joint_step", "ess_bulk", "ess_per_transition", "ess_per_second", "rhat",
    "loss_status", "tune_winner", "tune_status", "summary_path",
)
TIMING_FIELDS = (
    "task_id", "arch", "m", "rep", "kind", "h", "device", "gpu",
    "transitions_per_block", "doublings", "total_seconds", "total_transitions",
    "seconds_per_joint_step", "seconds_per_chain_transition", "chain_transitions_per_wall_second",
    "min_block_seconds", "met_duration_rule", "batch_chains",
    "ess_bulk", "ess_tail", "ess_mean", "ess_per_transition", "ess_per_second", "tau", "rhat",
    "loss_status", "n_loss",
)
TUNE_FIELDS = (
    "task_id", "arch", "m", "rep", "h_over_sigma2", "ess_per_transition",
    "winner", "winner_score", "status", "boundary",
)
PRIOR_FIELDS = (
    "task_id", "arch", "m", "rep", "batch_index", "n", "planned", "partial",
    "stop_reason", "elapsed_s", "accepted_fraction",
)
COST_FIELDS = (
    "narrow", "wide", "device", "gpu", "ess_per_second_ratio", "decision", "spans_break_even",
    "cost_ratio", "break_even", "met_duration",
    "seconds_per_transition_narrow", "seconds_per_transition_wide",
)


def collect_revision(cfg: dict[str, Any]) -> dict[str, Any]:
    root = RC.output_root(cfg)
    runs = root / "runs"
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    tune_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    partial: list[str] = []
    for stage in SUBMITTED_STAGES:
        for task in tasks_for(cfg, stage):
            path = runs / task["id"] / stage / "summary.json"
            label = f"{stage}/{task['id']}"
            if not path.is_file():
                missing.append(label)
                continue
            summary = json.loads(path.read_text())
            run_rows.append(_run_row(stage, summary, path))
            if stage == "fixed-step":
                timing_rows.append(_timing_row(summary))
            if stage == "tune":
                tune_rows.extend(_tune_rows(summary))
            if stage == "prior-starts":
                prior_rows.extend(_prior_rows(summary))
            if _is_partial(summary):
                partial.append(label)
    cost_row, cost_missing = _cost_row(root / "cost_pilot_deep_r0.json")
    if cost_missing:
        missing.append(cost_missing)
    _write(tables / "runs.csv", RUN_FIELDS, run_rows)
    _write(tables / "fixed_step_timing.csv", TIMING_FIELDS, timing_rows)
    _write(tables / "tuning.csv", TUNE_FIELDS, tune_rows)
    _write(tables / "prior_starts.csv", PRIOR_FIELDS, prior_rows)
    _write(tables / "cost_pilot.csv", COST_FIELDS, [cost_row] if cost_row else [])
    manifest = {
        "runs": len(run_rows), "missing": missing, "partial": partial,
        "fixed_step_timing": len(timing_rows), "tuning_rows": len(tune_rows),
        "prior_start_batches": len(prior_rows), "cost_pilot": cost_row is not None,
        "tables": str(tables),
    }
    (tables / "collect_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["status"] = "ok" if not missing and not partial else "incomplete"
    return manifest


def _run_row(stage: str, summary: dict[str, Any], path: Path) -> dict[str, Any]:
    task = summary.get("task") or {}
    pieces = _pieces(summary)
    loss = summary.get("loss_rate") or {}
    timing = summary.get("timing") or {}
    selection = (summary.get("tuning") or {}).get("selection") or {}
    return {
        "stage": stage, "task_id": task.get("id"), "arch": task.get("arch"), "m": task.get("m"),
        "rep": task.get("rep"), "kind": task.get("kind"), "r": task.get("r"),
        "h": summary.get("h"), "sigma": summary.get("sigma"), "device": summary.get("device"),
        "gpu": (summary.get("hardware") or {}).get("gpu"),
        "n": pieces["n"], "planned": pieces["planned"], "partial": pieces["partial"],
        "stop_reason": pieces["stop_reason"], "elapsed_s": pieces["elapsed_s"],
        "accepted_fraction": pieces["accepted_fraction"],
        "seconds_per_joint_step": timing.get("seconds_per_joint_step"),
        "ess_bulk": loss.get("ess_bulk"), "ess_per_transition": loss.get("ess_per_transition"),
        "ess_per_second": loss.get("ess_per_second"), "rhat": loss.get("rhat"),
        "loss_status": loss.get("status"), "tune_winner": selection.get("winner"),
        "tune_status": selection.get("status"), "summary_path": str(path),
    }


def _timing_row(summary: dict[str, Any]) -> dict[str, Any]:
    task = summary.get("task") or {}
    timing = summary.get("timing") or {}
    loss = summary.get("loss_rate") or {}
    return {
        "task_id": task.get("id"), "arch": task.get("arch"), "m": task.get("m"),
        "rep": task.get("rep"), "kind": task.get("kind"), "h": summary.get("h"),
        "device": summary.get("device"), "gpu": (summary.get("hardware") or {}).get("gpu"),
        "transitions_per_block": timing.get("transitions_per_block"),
        "doublings": timing.get("doublings"), "total_seconds": timing.get("total_seconds"),
        "total_transitions": timing.get("total_transitions"),
        "seconds_per_joint_step": timing.get("seconds_per_joint_step"),
        "seconds_per_chain_transition": timing.get("seconds_per_chain_transition"),
        "chain_transitions_per_wall_second": timing.get("chain_transitions_per_wall_second"),
        "min_block_seconds": timing.get("min_seconds"),
        "met_duration_rule": timing.get("met_duration_rule"),
        "batch_chains": timing.get("batch_chains"),
        "ess_bulk": loss.get("ess_bulk"), "ess_tail": loss.get("ess_tail"),
        "ess_mean": loss.get("ess_mean"), "ess_per_transition": loss.get("ess_per_transition"),
        "ess_per_second": loss.get("ess_per_second"), "tau": loss.get("tau"), "rhat": loss.get("rhat"),
        "loss_status": loss.get("status"), "n_loss": loss.get("n_total"),
    }


def _tune_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    task = summary.get("task") or {}
    tuning = summary.get("tuning") or {}
    selection = tuning.get("selection") or {}
    rows = []
    for h_dim, score in tuning.get("scores_ess_per_transition") or []:
        rows.append({
            "task_id": task.get("id"), "arch": task.get("arch"), "m": task.get("m"), "rep": task.get("rep"),
            "h_over_sigma2": h_dim, "ess_per_transition": score,
            "winner": selection.get("winner"), "winner_score": selection.get("score"),
            "status": selection.get("status"), "boundary": selection.get("boundary"),
        })
    return rows


def _prior_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    task = summary.get("task") or {}
    batches = (summary.get("prior_starts") or {}).get("batches") or []
    rows = []
    for i, batch in enumerate(batches):
        record = batch.get("record") or {}
        rows.append({
            "task_id": task.get("id"), "arch": task.get("arch"), "m": task.get("m"), "rep": task.get("rep"),
            "batch_index": i, "n": record.get("n"), "planned": record.get("planned"),
            "partial": record.get("partial"), "stop_reason": record.get("stop_reason"),
            "elapsed_s": batch.get("elapsed_s"), "accepted_fraction": _accepted(batch.get("counters")),
        })
    return rows


def _cost_row(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not path.is_file():
        return None, "cost-pilot/deep_r0"
    report = json.loads(path.read_text())
    compared = report.get("comparison") or {}
    return {
        "narrow": report.get("narrow"), "wide": report.get("wide"), "device": report.get("device"),
        "gpu": (report.get("hardware") or {}).get("gpu"),
        "ess_per_second_ratio": compared.get("ess_per_second_ratio"),
        "decision": compared.get("decision"), "spans_break_even": compared.get("spans_break_even"),
        "cost_ratio": report.get("cost_ratio_wide_over_narrow"),
        "break_even": report.get("break_even_ess_ratio"),
        "met_duration": report.get("met_duration_rule"),
        "seconds_per_transition_narrow": compared.get("seconds_per_transition_narrow"),
        "seconds_per_transition_wide": compared.get("seconds_per_transition_wide"),
    }, None


def _pieces(summary: dict[str, Any]) -> dict[str, Any]:
    runs = _run_payloads(summary)
    if not runs:
        return {"n": None, "planned": None, "partial": None, "stop_reason": None,
                "elapsed_s": None, "accepted_fraction": None}
    records = [run.get("record") or {} for run in runs]
    elapsed = [run.get("elapsed_s") for run in runs if run.get("elapsed_s") is not None]
    return {
        "n": sum(int(rec.get("n") or 0) for rec in records),
        "planned": sum(int(rec.get("planned") or 0) for rec in records),
        "partial": any(bool(rec.get("partial")) for rec in records),
        "stop_reason": next((rec.get("stop_reason") for rec in records if rec.get("stop_reason")), None),
        "elapsed_s": sum(float(v) for v in elapsed) if elapsed else None,
        "accepted_fraction": _pooled_acceptance(runs),
    }


def _run_payloads(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if "run" in summary:
        return [summary["run"]]
    return list((summary.get("prior_starts") or {}).get("batches") or [])


def _is_partial(summary: dict[str, Any]) -> bool:
    return any(bool((run.get("record") or {}).get("partial")) for run in _run_payloads(summary))


def _pooled_acceptance(runs: list[dict[str, Any]]) -> Optional[float]:
    accepted = transitions = 0
    found = False
    for run in runs:
        counters = run.get("counters") or {}
        got_transitions = [int(v) for v in counters.get("transitions") or []]
        got_accepted = [int(v) for v in counters.get("accepted") or []]
        if not got_transitions or len(got_accepted) != len(got_transitions):
            continue
        found = True
        accepted += sum(got_accepted)
        transitions += sum(got_transitions)
    if not found or transitions <= 0:
        return None
    return accepted / transitions


def _accepted(counters: Optional[dict[str, Any]]) -> Optional[float]:
    if not counters:
        return None
    transitions = [int(v) for v in counters.get("transitions") or []]
    accepted = [int(v) for v in counters.get("accepted") or []]
    total = sum(transitions)
    if total <= 0 or len(accepted) != len(transitions):
        return None
    return sum(accepted) / total


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
