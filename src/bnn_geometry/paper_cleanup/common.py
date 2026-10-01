"""Paths, vendored-module access and provenance helpers for the cleanup/supplement package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIGURE_WIDTH_IN = 6.75          # AISTATS two-column full-width figure*
HISTORICAL_A = 2.5              # campaign normalizer of the stored S = T / 2.5 (inverted exactly once)
S1_LAGS = (25, 50, 100, 200)
S2_TIMES = (0.25, 0.5, 1.0, 2.0)
S2_DURATION = 512.0
WINDOW = 102400
PROTOCOL_GATES = {"rhat_max_exclusive": 1.01, "bulk_ess_min": 1000.0, "tail_ess_min": 400.0}


def vendored(name: str):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return __import__(name)


def clean_figures():
    return vendored("clean_figures")


def supplement_metrics():
    return vendored("supplement_metrics")


def run_selfcheck(out: Path) -> str:
    """Run the bundled selfcheck.py in its own directory; raise if it fails."""
    p = subprocess.run([sys.executable, "selfcheck.py"], cwd=HERE, capture_output=True, text=True)
    txt = (p.stdout + p.stderr).strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt + "\n")
    if p.returncode != 0:
        raise RuntimeError(f"bundled selfcheck failed:\n{txt}")
    return txt


class Layout:
    """``<output_root>/paper_cleanup``; heavy supplement traces live in ``supplement_traces`` (not committed)."""

    def __init__(self, out: Path):
        self.out = Path(out)
        self.figures = self.out / "figures"
        self.tables = self.out / "tables"
        self.exports = self.out / "exports"
        self.traces = self.out / "supplement_traces"
        self.diagnostics = self.out / "diagnostics"

    def mkdirs(self) -> None:
        for p in (self.figures, self.tables, self.exports, self.diagnostics):
            p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
