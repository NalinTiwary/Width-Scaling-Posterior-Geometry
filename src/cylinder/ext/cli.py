"""Shared helpers for scripts/ext/*.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..config import artifacts_root, load_config, results_root

ROOT = Path(__file__).resolve().parents[3]


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="config.ext.yaml")
    return ap


def load(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path]:
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    return cfg, cfg_path, artifacts_root(cfg, ROOT), results_root(cfg, ROOT)
