"""Shared paths, style and provenance helpers for the two-figure package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STYLE_PATH = HERE / "plot_style.json"
METRICS_PATH = HERE / "figure_metrics.py"

ANALYSIS_SCOPE = "fixed_step_discrete_sampler"
SPECTRAL_NORMALIZATION = "campaign_S_sigma_max_over_a_sqrt_m"  # spectral.normalized_norm: σ_max(W2)/(2.5√m)
OBSERVABLE = "summed_training_cross_entropy"

MPL_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def style() -> dict[str, Any]:
    return json.loads(STYLE_PATH.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def metrics():
    """The vendored array-only calculation module."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import figure_metrics  # noqa: E402
    return figure_metrics


def versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import pandas
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "matplotlib": matplotlib.__version__,
            "pandas": pandas.__version__}


class Layout:
    """Output layout under ``<output_root>/paper_figures``."""

    def __init__(self, out: Path):
        self.out = Path(out)
        self.traces = self.out / "loss_traces"
        self.tables = self.out / "tables"
        self.figures = self.out / "figures"
        self.diagnostics = self.out / "diagnostics"

    def mkdirs(self) -> None:
        for p in (self.tables, self.figures, self.diagnostics):
            p.mkdir(parents=True, exist_ok=True)
