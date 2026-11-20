"""Time-to-agreement rule, fixed before any prior-start run.

Primary observable is the ensemble mean of V. A checkpoint counts only when its whole
uncertainty interval lies inside ±0.25 posterior standard deviations. The reported time is
the first of three successive checkpoints that qualify. Otherwise the last time is returned
with a right-censoring mark.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

MARGIN = 0.25
SUCCESSIVE = 3


def interval_inside(estimate: float, half_width: float, margin: float = MARGIN) -> bool:
    if not (np.isfinite(estimate) and np.isfinite(half_width)):
        return False
    lo, hi = estimate - half_width, estimate + half_width
    return bool(lo >= -margin and hi <= margin)


def time_to_agreement(times: Sequence[float], estimates: Sequence[float], half_widths: Sequence[float],
                      *, margin: float = MARGIN, successive: int = SUCCESSIVE) -> dict[str, Any]:
    times = [float(t) for t in times]
    estimates = [float(v) for v in estimates]
    half_widths = [float(v) for v in half_widths]
    if not (len(times) == len(estimates) == len(half_widths)) or len(times) < successive:
        raise ValueError("times, estimates and half-widths must match and cover the successive rule")
    inside = [interval_inside(e, h, margin) for e, h in zip(estimates, half_widths)]
    hit = None
    for i in range(len(inside) - successive + 1):
        if all(inside[i:i + successive]):
            hit = i
            break
    if hit is None:
        return {"time": times[-1], "censored": True, "index": None, "inside": inside, "margin": margin,
                "rule": f"first {successive} successive checkpoints inside ±{margin} SD"}
    return {"time": times[hit], "censored": False, "index": hit, "inside": inside, "margin": margin,
            "rule": f"first {successive} successive checkpoints inside ±{margin} SD"}


def standardized_error(mean_t: float, mean_ref: float, sd_ref: float) -> float:
    if not sd_ref > 0:
        return float("nan")
    return (mean_t - mean_ref) / sd_ref
