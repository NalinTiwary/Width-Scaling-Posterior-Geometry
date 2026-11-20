"""Revision config. This schema is separate from the frozen campaign schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .. import config as C

ROOT = C.ROOT


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text())
    if int(cfg.get("schema_version", -1)) != 1:
        raise ValueError("revision schema_version must be 1")
    for key in ("campaign_id", "master_seed", "output_root", "baseline_root", "baseline_config"):
        if key not in cfg:
            raise ValueError(f"revision config missing {key}")
    return cfg


def output_root(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["output_root"])
    return p if p.is_absolute() else ROOT / p


def baseline_root(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["baseline_root"])
    return p if p.is_absolute() else ROOT / p


def archive_zip(cfg: dict[str, Any]) -> Path | None:
    raw = cfg.get("archive_zip")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p
