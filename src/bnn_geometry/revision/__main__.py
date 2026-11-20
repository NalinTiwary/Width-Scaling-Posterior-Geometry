"""Revision commands.

    python -m bnn_geometry.revision archive --config configs/revision.yaml
    python -m bnn_geometry.revision smoke --config configs/revision_smoke.yaml
    python -m bnn_geometry.revision run --config configs/revision.yaml --stage fixed-step --task-id 0
    python -m bnn_geometry.revision tasks --config configs/revision.yaml --stage tune
    python -m bnn_geometry.revision cost-pilot --config configs/revision.yaml --device cuda
    python -m bnn_geometry.revision prediction-acf --config configs/revision.yaml
"""

from __future__ import annotations

import argparse
import json

from . import config as RC
from .archive import run_archive
from .prediction import rebuild_test_prob_0
from .runner import cost_pilot, login_preflight, run_task, smoke_suite, tasks_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AISTATS revision experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("archive", "smoke", "tasks", "run", "cost-pilot", "prediction-acf", "preflight"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "preflight":
            p.add_argument("--require-cuda-build", action="store_true")
        if name in ("tasks", "run"):
            p.add_argument("--stage", required=True)
        if name == "run":
            p.add_argument("--task-id", type=int, default=None)
            p.add_argument("--device", default=None)
        if name in ("cost-pilot", "prediction-acf"):
            p.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = RC.load(args.config)
    if args.cmd == "preflight":
        report = login_preflight(cfg, require_cuda_build=args.require_cuda_build)
        print(f"PREFLIGHT status={report['status']} cuda_build={report.get('torch_cuda_build')} "
              f"cuda_visible_here={report.get('cuda_visible_here')} versions={report.get('versions')}")
        if report["status"] != "ok":
            print(json.dumps(report, indent=2))
            return 1
        print("PREFLIGHT_OK")
        return 0
    if args.cmd == "cost-pilot":
        report = cost_pilot(cfg, device=args.device)
        compared = report["comparison"]
        print(f"ess_per_second_ratio={compared['ess_per_second_ratio']:.6f} "
              f"decision={compared['decision']} spans_break_even={compared['spans_break_even']} "
              f"cost_ratio={report['cost_ratio_wide_over_narrow']:.6f} "
              f"break_even={report['break_even_ess_ratio']:.6f} "
              f"met_duration={report['met_duration_rule']} path={report['path']}")
        return 0
    if args.cmd == "prediction-acf":
        report = rebuild_test_prob_0(cfg, device=args.device)
        print(f"prediction_acf status={report['status']} device={report.get('device')} path={report['path']}")
        if report.get("reason"):
            print(report["reason"])
        if report.get("traceback"):
            print(report["traceback"])
        return 0 if report["status"] in ("ok", "raw_series_absent") else 1
    if args.cmd == "archive":
        manifest = run_archive(cfg)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.cmd == "smoke":
        report = smoke_suite(cfg)
        ou = report["ou_control"]
        print(f"smoke acceptance={ou['acceptance']:.3f} lag1={ou['lag1']:.4f} eta={ou['eta']:.4f} "
              f"resume_n={report['resume_record'].get('n')}")
        return 0
    tasks = tasks_for(cfg, args.stage)
    if args.cmd == "tasks":
        for i, task in enumerate(tasks):
            print(f"{i}\t{task['id']}")
        return 0
    if args.task_id is None:
        parser.error("run requires --task-id")
    if not 0 <= args.task_id < len(tasks):
        parser.error(f"task-id {args.task_id} outside 0..{len(tasks) - 1}")
    summary = run_task(cfg, args.stage, tasks[args.task_id], device=args.device)
    timing = summary.get("timing") or {}
    print(f"{tasks[args.task_id]['id']} stage={args.stage} seconds_per_transition="
          f"{timing.get('seconds_per_transition')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
