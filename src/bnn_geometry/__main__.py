"""Staged CLI (runbook §15). Every command accepts --dry-run."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import config as C


def _cfg(args) -> tuple[dict[str, Any], Path, Path]:
    path = Path(args.config)
    cfg = C.load(path)
    if getattr(args, "out", None):
        cfg = copy.deepcopy(cfg)
        cfg["output_root"] = str(Path(args.out).resolve())
    root = C.output_root(cfg)
    return cfg, (path if path.is_absolute() else C.ROOT / path), root


def _targets_for(cfg, args) -> list[str]:
    tids = [t.target_id for t in C.targets(cfg)]
    if getattr(args, "target", None):
        if args.target not in tids:
            raise SystemExit(f"unknown target {args.target}")
        return [args.target]
    if getattr(args, "task_id", None) is not None:
        return [tids[int(args.task_id)]]
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if tid is not None and getattr(args, "stage", None):
        return [tids[int(tid)]]
    return tids


def plan_lines(cfg: dict[str, Any], root: Path) -> list[str]:
    d = int(cfg["data"]["input_dimension"])
    r, dy, s2 = cfg["reference"], cfg["dynamics"], float(cfg["model"]["sigma"]) ** 2
    nch = len(cfg["chain_ids"])
    K = {"shallow": 2 + 8 + 8 + 4 + 4, "deep": 2 + 8 + 8 + 4 + 5}
    stages = r["cumulative_production_stages"]
    pre = r["burn_in_transitions"] + r["calibration_transitions"] + r["separation_transitions"]
    out = [f"campaign {cfg['campaign_id']}  output {root}",
           f"{'target':<20}{'p':>10}{'archive GB (init/max)':>24}{'scalar MB (max)':>17}"]
    tot0 = tot1 = 0.0
    for t in C.targets(cfg):
        p = t.p(d)
        a0 = nch * (stages[0] // r["parameter_archive_stride"]) * p * 8 / 1e9
        a1 = nch * (stages[-1] // r["parameter_archive_stride"]) * p * 8 / 1e9
        a1 += nch * (r["calibration_transitions"] // r["calibration_archive_stride"]) * p * 8 / 1e9
        sc = nch * (pre + stages[-1]) * (K[t.arch] + 2) * 8 / 1e6
        tot0 += a0
        tot1 += a1
        out.append(f"{t.target_id:<20}{p:>10}{a0:>12.2f}/{a1:<11.2f}{sc:>17.1f}")
    h0 = float(dy["h_over_sigma_squared_candidates"][0]) * s2
    Tm = float(dy["cumulative_retained_time_over_sigma_squared"][0]) * s2
    n_t = len(C.targets(cfg))
    out += [f"reference archive total: initial {tot0:.1f} GB, maximum {tot1:.1f} GB "
            f"(recommended free disk {cfg['limits']['recommended_free_disk_gb']} GB)",
            f"reference retained transitions: initial {n_t * nch * stages[0]:,}, maximum {n_t * nch * stages[-1]:,}; "
            f"including burn-in/calibration/separation: {n_t * nch * (pre + stages[0]):,} / {n_t * nch * (pre + stages[-1]):,}",
            f"dynamics at h={h0:g}, T={Tm:g}: {C.n_steps(Tm, h0):,} retained transitions per chain, "
            f"{n_t * nch * C.n_steps(Tm, h0):,} across targets (x2 with endpoint h/2 checks at equal duration)",
            f"limits: {r['max_likelihood_evaluations_per_chain']:,} likelihood evals per reference chain; "
            f"{dy['max_candidate_grad_evaluations_per_target_all_dynamics']:,} dynamics gradients per target; "
            f"{cfg['static']['selected_states_per_target_max']} static states per target; "
            f"{dy['architecture_wide_refinements_max']} architecture-wide refinement"]
    return out


def cmd_validate(args) -> int:
    cfg, path, root = _cfg(args)
    print(f"OK {path} sha256={C.config_sha256(path)} targets={len(C.targets(cfg))} output={root}")
    return 0


def cmd_plan(args) -> int:
    cfg, _, root = _cfg(args)
    print("\n".join(plan_lines(cfg, root)))
    return 0


def cmd_freeze(args) -> int:
    cfg, path, root = _cfg(args)
    if args.dry_run:
        print(f"[dry-run] would write {root}/data, centers, manifest.json, environment and test reports")
        return 0
    from .pipeline import freeze_data
    from .report import write_environment
    from .storage import atomic_write_json, clean_json
    info = freeze_data(cfg, root)
    write_environment(root)
    tests = {}
    if not args.skip_tests:
        (root / "tests").mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "BNN_CALIBRATION_OUT": str(root / "tests"), "PYTHONPATH": str(C.ROOT / "src")}
        for kind in ("unit", "integration"):
            rc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(C.ROOT / "tests" / kind),
                                 f"--junitxml={root / 'tests' / f'{kind}_report.xml'}"], env=env).returncode
            tests[kind] = rc
        if any(tests.values()):
            print(f"tests failed: {tests}", file=sys.stderr)
    from .data import load_replicate
    tg = []
    for t in C.targets(cfg):
        _, _, dh = load_replicate(root, t.rep)
        tg.append({"target_id": t.target_id, "target_hash": C.target_hash(cfg, t, dh), "arch": t.arch, "m": t.m,
                   "rep": t.rep, "p": t.p(int(cfg["data"]["input_dimension"]))})
    man = {"campaign_id": cfg["campaign_id"], "config": str(path), "config_sha256": C.config_sha256(path),
           "code_revision": C.code_revision(), "sampler_code_hash": C.sampler_code_hash(), "targets": tg,
           "replicates": info, "test_return_codes": tests, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "python_note": "Python 3.9 cluster environment (runbook suggests 3.11); version recorded in environment.json",
           "code_fixes": []}
    atomic_write_json(root / "manifest.json", clean_json(man))
    print(f"frozen {len(tg)} targets -> {root / 'manifest.json'}")
    return 1 if any(tests.values()) else 0


def cmd_run(args) -> int:
    cfg, _, root = _cfg(args)
    from . import pipeline as P
    from .context import pick_device
    if args.dry_run:
        print("\n".join(plan_lines(cfg, root)))
        print(f"[dry-run] stage={args.stage or 'all'} targets={_targets_for(cfg, args) if args.stage in P.TARGET_STAGES else 'n/a'}")
        return 0
    dev = pick_device(args.device)
    print(f"device {dev}", flush=True)
    if not args.stage:
        P.run_all(cfg, root, dev)
        return 0
    if args.stage in P.TARGET_STAGES:
        rc = 0
        for tid in _targets_for(cfg, args):
            t0 = time.time()
            print(f"[{args.stage}] {tid} start", flush=True)
            st = P._guard(lambda: P.run_target_stage(cfg, root, args.stage, tid, dev), root, args.stage, tid)
            if st is None:
                print(f"[{args.stage}] {tid} FAILED; traceback in {root / 'targets' / tid / 'errors.json'}", flush=True)
                rc = 1
                continue
            _mark(root / "targets" / tid / "stages", args.stage)
            print(f"[{args.stage}] {tid} done in {time.time() - t0:.0f}s: "
                  + json.dumps({k: (v.get('status') if isinstance(v, dict) else v) for k, v in st.items()}, default=str),
                  flush=True)
        return rc
    P.run_global_stage(cfg, root, args.stage, dev)
    _mark(root / "stages", args.stage)
    print(f"[{args.stage}] done", flush=True)
    return 0


def _mark(d: Path, stage: str) -> None:
    """Completion marker read by the SLURM continuation tasks (scripts/final_geometry)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stage}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


def cmd_analyze(args) -> int:
    cfg, _, root = _cfg(args)
    if args.dry_run:
        print(f"[dry-run] would write {root}/tables/*.csv")
        return 0
    from .analyze import analyze
    for k, v in analyze(cfg, root).items():
        print(f"tables/{k}.csv  {len(v)} rows")
    return 0


def cmd_figures(args) -> int:
    cfg, _, root = _cfg(args)
    if args.dry_run:
        print(f"[dry-run] would write {root}/figures/* and captions.tex from tables only")
        return 0
    from .figures import make_figures
    from .report import write_summary
    make_figures(cfg, root)
    write_summary(cfg, root)
    print(f"figures -> {root / 'figures'}; SUMMARY.md, captions.tex written")
    return 0


def cmd_audit(args) -> int:
    cfg, _, root = _cfg(args)
    if args.dry_run:
        print(f"[dry-run] would write {root}/audit.json")
        return 0
    from .report import audit
    res = audit(cfg, root)
    for c in res["checks"]:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"] + ("" if c["pass"] else f"  {c['detail']}"))
    print(f"archival_pass={res['archival_pass']} publication_ready={res['publication_ready']}")
    return 1 if (args.strict and not res["archival_pass"]) else 0


def cmd_export(args) -> int:
    cfg, path, root = _cfg(args)
    if args.dry_run:
        print(f"[dry-run] would write campaign.yaml, environment files and checksums.sha256 under {root}")
        return 0
    from .report import export
    print(export(cfg, path, root, heavy=not args.no_heavy))
    return 0


def cmd_benchmark(args) -> int:
    cfg, _, root = _cfg(args)
    if args.dry_run:
        print("[dry-run] would time V, V+grad, probe gradients, SVDs and I/O for the widest widths")
        return 0
    from .benchmark import benchmark
    benchmark(cfg, root, args.device)
    return 0


def cmd_fixture(args) -> int:
    fx = Path(args.fixture_config) if args.fixture_config else C.ROOT / "configs" / "fixture.yaml"
    out = Path(args.out).resolve()
    if args.dry_run:
        print(f"[dry-run] would run the complete pipeline on {fx} into {out}")
        return 0
    import pandas as pd
    from . import calibration as cal
    from . import pipeline as P
    from .analyze import analyze
    from .context import pick_device
    from .figures import make_figures
    from .report import audit, export, write_summary
    cfg = copy.deepcopy(C.load(fx))
    cfg["output_root"] = str(out)
    dev = pick_device(args.device)
    t0 = time.time()
    P.run_all(cfg, out, dev)
    (out / "tests").mkdir(parents=True, exist_ok=True)
    rows = []
    for s in (0.7, 1.0):
        rows += cal.ou_calibration(s, n=4096, reps=50) + cal.gaussian_entropy(s, n=4096, reps=50)
    pd.DataFrame(rows).assign(note="fixture-size calibration (not the §6.2 test)").to_csv(
        out / "tests" / "calibration_results.csv", index=False)
    analyze(cfg, out)
    make_figures(cfg, out)
    write_summary(cfg, out)
    res = audit(cfg, out)
    export(cfg, fx, out)
    print(f"fixture done in {time.time() - t0:.0f}s: archival_pass={res['archival_pass']} "
          f"publication_ready={res['publication_ready']} (fixture is never publication-ready)")
    return 0 if res["archival_pass"] else 1


def cmd_export_loss(args) -> int:
    from .paper_export import export_all
    cfg, _, root = _cfg(args)
    tids = [args.target] if args.target else None
    if args.dry_run:
        print(f"would export h={args.h} window={args.window} V traces for "
              f"{len(tids or C.targets(cfg))} targets from {root} -> {args.dest or root / 'paper_figures'}")
        return 0
    return export_all(cfg, root, h=args.h, window=args.window, out=Path(args.dest) if args.dest else None,
                      targets=tids, device=args.device or "cpu", recompute_end_state=not args.skip_end_state)


def cmd_paper_figures(args) -> int:
    from .paper_figures.build import STEPS, build
    cfg, _, root = _cfg(args)
    steps = tuple(s for s in args.steps.split(",") if s) if args.steps else STEPS
    if set(steps) - set(STEPS):
        raise SystemExit(f"unknown steps {set(steps) - set(STEPS)}; choose from {STEPS}")
    if args.dry_run:
        print(f"would build paper figures ({','.join(steps)}) from {root} -> {args.dest or root / 'paper_figures'}")
        return 0
    return build(cfg, root, Path(args.dest) if args.dest else None, steps=steps, window=args.window)


def cmd_paper_cleanup(args) -> int:
    from .paper_cleanup.build import LOCAL_STEPS, STEPS, build
    cfg, _, root = _cfg(args)
    steps = tuple(s for s in args.steps.split(",") if s) if args.steps else LOCAL_STEPS
    if set(steps) - set(STEPS):
        raise SystemExit(f"unknown steps {set(steps) - set(STEPS)}; choose from {STEPS}")
    if args.dry_run:
        print(f"would run paper cleanup ({','.join(steps)}) from {root} -> {args.dest or root / 'paper_cleanup'}")
        return 0
    return build(cfg, root, Path(args.dest) if args.dest else None, steps=steps, device=args.device or "cpu",
                 window=args.window, duration=args.duration, width_in=args.width_in,
                 targets=[args.target] if args.target else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bnn_geometry")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *, config=True):
        p = sub.add_parser(name)
        if config:
            p.add_argument("--config", default="configs/campaign.yaml")
            p.add_argument("--out", default=None, help="override output_root")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--device", default=None)
        p.set_defaults(fn=fn)
        return p

    add("validate-config", cmd_validate)
    add("plan", cmd_plan)
    f = add("freeze", cmd_freeze)
    f.add_argument("--skip-tests", action="store_true")
    r = add("run", cmd_run)
    r.add_argument("--resume", action="store_true", help="resume from checkpoints (always on; kept for the contract)")
    r.add_argument("--stage", default=None, choices=["reference", "production", "refine", "freeze-data", "select-step",
                                                     "endpoint-decision", "refine-decision", "controls"])
    r.add_argument("--target", default=None)
    r.add_argument("--task-id", default=None, type=int)
    add("analyze", cmd_analyze)
    add("figures", cmd_figures)
    a = add("audit", cmd_audit)
    a.add_argument("--strict", action="store_true")
    e = add("export", cmd_export)
    e.add_argument("--no-heavy", action="store_true", help="skip checksumming trace HDF5 files")
    add("benchmark", cmd_benchmark)
    el = add("export-loss-traces", cmd_export_loss)
    el.add_argument("--h", type=float, default=0.01)
    el.add_argument("--window", type=int, default=102400)
    el.add_argument("--target", default=None)
    el.add_argument("--dest", default=None, help="default <output_root>/paper_figures")
    el.add_argument("--skip-end-state", action="store_true", help="skip recomputing the final-state V")
    pf = add("paper-figures", cmd_paper_figures)
    pf.add_argument("--steps", default=None, help="comma list of selfcheck,spectral,acf,render,text (default all)")
    pf.add_argument("--window", type=int, default=102400)
    pf.add_argument("--dest", default=None, help="default <output_root>/paper_figures")
    pc = add("paper-cleanup", cmd_paper_cleanup)
    pc.add_argument("--steps", default=None, help="comma list of selfcheck,main,export-supplements,supplements,text "
                                                  "(default selfcheck,main,supplements,text)")
    pc.add_argument("--window", type=int, default=102400, help="S1 window (retained transitions per chain)")
    pc.add_argument("--duration", type=float, default=512.0, help="S2 matched algorithmic duration")
    pc.add_argument("--width-in", type=float, default=6.75)
    pc.add_argument("--target", default=None, help="restrict export-supplements to one target")
    pc.add_argument("--dest", default=None, help="default <output_root>/paper_cleanup")
    fx = add("fixture", cmd_fixture, config=False)
    fx.add_argument("--out", default="results/fixture")
    fx.add_argument("--fixture-config", default=None, help="alternative fixture config (default configs/fixture.yaml)")
    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
