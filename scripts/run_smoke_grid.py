#!/usr/bin/env python3
"""Build reduced smoke artifacts for figure validation (not the paper run)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.smoke.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-pcn", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    cfg = args.config
    device = ["--device", args.device] if args.device else []

    run([py, "scripts/00_preflight.py", "--config", cfg, "--write-data", "--skip-tests", *device])

    # ESS grid
    import yaml

    with open(ROOT / cfg) as f:
        conf = yaml.safe_load(f)
    for m in conf["widths"]:
        for seed in conf["prior"]["center_seeds"]:
            run(
                [
                    py,
                    "scripts/01_run_target.py",
                    "--config",
                    cfg,
                    "--m",
                    str(m),
                    "--seed",
                    str(seed),
                    "--kernel",
                    "ess",
                    *device,
                    "--overwrite",
                ]
            )

    if not args.skip_pcn:
        run(
            [
                py,
                "scripts/01_run_target.py",
                "--config",
                cfg,
                "--m",
                str(conf["pcn_crosscheck"]["width"]),
                "--seed",
                str(conf["pcn_crosscheck"]["center_seed"]),
                "--kernel",
                "pcn",
                *device,
                "--overwrite",
            ]
        )

    run([py, "scripts/02_extend_if_needed.py", "--config", cfg, *device])
    run([py, "scripts/03_curvature_states.py", "--config", cfg])
    run([py, "scripts/04_summarize.py", "--config", cfg])
    run([py, "scripts/05_make_figures.py", "--config", cfg])
    print("Smoke pipeline complete →", conf.get("artifacts_dir", "artifacts_smoke"))


if __name__ == "__main__":
    main()
