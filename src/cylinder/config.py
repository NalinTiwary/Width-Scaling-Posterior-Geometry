"""Config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} is not a mapping")
    return cfg


def artifacts_root(cfg: dict[str, Any], base: str | Path | None = None) -> Path:
    root = Path(base) if base is not None else Path.cwd()
    return (root / cfg.get("artifacts_dir", "artifacts")).resolve()
