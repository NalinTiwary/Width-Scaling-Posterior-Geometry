#!/usr/bin/env python3
"""Run one (m, seed, kernel) sampling target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cylinder.config import artifacts_root, load_config  # noqa: E402
from cylinder.run import run_target  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="Prior-center seed")
    ap.add_argument("--kernel", choices=["ess", "pcn"], default="ess")
    ap.add_argument("--device", default=None)
    ap.add_argument("--chain_id", type=int, default=None)
    ap.add_argument("--n_retained", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Return immediately if this target already finished with matching identity",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    artifacts = artifacts_root(cfg, ROOT)

    if not (artifacts / "data.npz").exists():
        raise SystemExit("Missing artifacts/data.npz — run 00_preflight.py --write-data first")

    chain_ids = [args.chain_id] if args.chain_id is not None else None
    out = run_target(
        cfg=cfg,
        m=args.m,
        center_seed=args.seed,
        kernel=args.kernel,
        artifacts=artifacts,
        device_request=args.device,
        chain_ids=chain_ids,
        n_retained=args.n_retained,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
    )
    print(f"Wrote run directory: {out}")


if __name__ == "__main__":
    main()
