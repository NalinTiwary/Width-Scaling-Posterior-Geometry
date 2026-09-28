#!/usr/bin/env python3
"""Validate the reduced-coordinate sampler and the curvature numerics before the grid (addendum §7.1)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.ext.cli import base_parser, load  # noqa: E402
from cylinder.ext.validation import run_all  # noqa: E402
from cylinder.io import atomic_save_json  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--light", action="store_true", help="Shorter matched-sampling runs")
    args = ap.parse_args()
    _, _, artifacts, _ = load(args)
    res = run_all(light=args.light)
    atomic_save_json(artifacts / "validation.json", res)
    for name, r in res.items():
        if name == "all_ok":
            continue
        brief = {k: v for k, v in r.items() if isinstance(v, (int, float, bool, str))}
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {name}: {brief}")
    print("ALL VALIDATION CHECKS PASSED" if res["all_ok"] else "VALIDATION FAILED")
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
