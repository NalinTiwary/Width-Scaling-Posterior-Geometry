#!/usr/bin/env python3
"""Print extension target names (config execution order), one per line."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.ext.cli import base_parser, load  # noqa: E402
from cylinder.ext.targets import all_targets  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--filter", default=None, help="Regex on target names")
    ap.add_argument("--no-realdata", action="store_true")
    args = ap.parse_args()
    cfg, *_ = load(args)
    for t in all_targets(cfg, pattern=args.filter, include_realdata=not args.no_realdata):
        print(t.name)


if __name__ == "__main__":
    main()
