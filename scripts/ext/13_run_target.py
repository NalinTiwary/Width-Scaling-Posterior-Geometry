#!/usr/bin/env python3
"""Run one extension target (4 unrestricted elliptical-slice chains + online curvature)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.ext.cli import base_parser, load  # noqa: E402
from cylinder.ext.run import run_ext_target  # noqa: E402
from cylinder.ext.targets import parse_name  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--target", required=True, help="e.g. orth_n128_m4096_r0 or fmnist_n256_m4096_r1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_retained", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg, _, artifacts, _ = load(args)
    t = parse_name(args.target)
    meta = run_ext_target(cfg=cfg, target=t, artifacts=artifacts, device_request=args.device,
                          n_retained=args.n_retained, overwrite=args.overwrite,
                          skip_existing=args.skip_existing)
    if "wall_time_sampling_s" in meta:
        print(f"{t.name}: T={meta['n_retained']} evals={meta['likelihood_evals']} "
              f"budget_hit={meta['hit_likelihood_budget']} sampling={meta['wall_time_sampling_s']:.1f}s "
              f"curvature={meta['wall_time_curvature_s']:.1f}s device={meta['effective_device']}")


if __name__ == "__main__":
    main()
