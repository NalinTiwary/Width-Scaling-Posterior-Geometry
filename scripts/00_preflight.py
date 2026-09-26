#!/usr/bin/env python3
"""Preflight: theorem table, optional data/center materialization, pytest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cylinder.centers import generate_center_bank, save_centers  # noqa: E402
from cylinder.config import artifacts_root, load_config  # noqa: E402
from cylinder.data import (  # noqa: E402
    generate_probes,
    generate_training_data,
    save_data_bundle,
)
from cylinder.device_utils import configure_dtype, resolve_device  # noqa: E402
from cylinder.theorem import assert_preflight  # noqa: E402


def write_frozen_data(cfg: dict, artifacts: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    train = generate_training_data(
        n=int(cfg["data"]["n"]),
        d=int(cfg["data"]["d"]),
        seed=int(cfg["data"]["seed"]),
    )
    probes = generate_probes(
        d=int(cfg["data"]["d"]),
        n_probes=int(cfg["data"]["n_probes"]),
        seed=int(cfg["data"]["probe_seed"]),
    )
    save_data_bundle(artifacts / "data.npz", train, probes)
    print(f"Wrote {artifacts / 'data.npz'}")
    print(f"  hash_X={train['hash_X']}")
    print(f"  hash_y={train['hash_y']}")
    print(f"  hash_X_probe={probes['hash_X_probe']}")

    for seed in cfg["prior"]["center_seeds"]:
        bank = generate_center_bank(
            n_pairs=int(cfg["prior"]["center_bank_pairs"]),
            d=int(cfg["architecture"]["input_dim"]),
            seed=int(seed),
            b0=float(cfg["prior"]["b0"]),
        )
        path = artifacts / f"centers_seed{seed}.npz"
        save_centers(path, bank)
        print(f"Wrote {path} hash_U={bank['hash_U']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--write-data", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    configure_dtype()
    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    req = args.device or cfg.get("device", "auto")
    device = resolve_device(req)
    print(f"device requested={req} effective={device}")

    report = assert_preflight()
    print(f"A_n = {report['A_n']:.7f}")
    for m, bun in report["widths"].items():
        print(
            f"  m={m}: B_m={bun['B_m']:.4f} D_th={bun['D_th']:.5f} "
            f"floor={bun['coverage_floor']:.6f}"
        )

    artifacts = artifacts_root(cfg, ROOT)
    if args.write_data:
        write_frozen_data(cfg, artifacts)

    if not args.skip_tests:
        print("Running pytest…")
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)
    print("Preflight OK.")


if __name__ == "__main__":
    main()
