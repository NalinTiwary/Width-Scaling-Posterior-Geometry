"""Rebuild one held-out prediction autocorrelation from the raw stage files.

The check is the chain-averaged ACF of ``test_prob_0`` on ``deep_m0032_r0``, compared with
the unrounded per-point export. It runs only when the campaign HDF5 trajectories are present.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..context import pick_device
from ..diagnostics import scalar_diagnostics
from ..paper_cleanup.supplement_export import _Ctx, read_trajectory
from ..paper_figures.figure_metrics import acf_replicate
from ..storage import StorageError, atomic_write_json, clean_json
from . import config as RC

WINDOW = 102_400
MAX_LAG = 400
TARGET = C.Target("deep", 32, 0)


def rebuild_test_prob_0(cfg: dict[str, Any], *, device: Optional[str] = None) -> dict[str, Any]:
    root = RC.baseline_root(cfg)
    campaign = C.load(cfg["baseline_config"])
    dev = pick_device(device or cfg.get("device"))
    stage = root / "targets" / TARGET.target_id / "dynamics" / "h_0p01" / "chain_0" / "stage_1.h5"
    if not stage.exists():
        report = {"status": "raw_series_absent", "target_id": TARGET.target_id,
                  "observable": "test_prob_0", "looked_for": str(stage), "device": str(dev)}
        return _write(cfg, report)
    try:
        ctxs = _Ctx(campaign, root, str(dev))
        indices = [int(j) for j in campaign["probes"]["test_indices"]]
        columns = ["V"] + [f"test_prob_{j}" for j in indices]
        loaded = read_trajectory(campaign, root, TARGET, 0.01, ctxs, columns)
    except StorageError as exc:
        return _write(cfg, {"status": "unreadable", "target_id": TARGET.target_id,
                            "observable": "test_prob_0", "reason": str(exc)})
    if indices[0] != 0:
        raise ValueError("test_prob_0 is not the first held-out index in the campaign probes")
    if int(loaded["n_retained"]) < WINDOW:
        return _write(cfg, {"status": "short_window", "retained": int(loaded["n_retained"])})
    window = loaded["X"][:, -WINDOW:, :]
    loss = np.load(root / "paper_figures" / "loss_traces" / f"{TARGET.target_id}.npz", allow_pickle=False)
    if not np.array_equal(loss["V"], window[:, :, 0]):
        return _write(cfg, {"status": "loss_window_mismatch", "target_id": TARGET.target_id})
    series = window[:, :, 1]
    curve = acf_replicate(series, MAX_LAG, WINDOW)["replicate_acf"]
    published = pd.read_csv(root / "paper_cleanup" / "tables" / "prediction_acf_per_point.csv.gz")
    rows = published[(published.architecture == "deep") & (published.width == 32)
                     & (published.replicate == 0) & (published.test_index == 0)].sort_values("lag")
    exported = rows.acf.to_numpy(dtype=np.float64)
    if exported.shape != curve.shape:
        return _write(cfg, {"status": "acf_length_mismatch", "recomputed": int(curve.size),
                            "exported": int(exported.size)})
    diag = scalar_diagnostics("test_prob_0", series)
    table = pd.read_csv(root / "paper_cleanup" / "tables" / "supplement_diagnostics.csv")
    ess_row = table[(table.target_id == TARGET.target_id) & (table.observable == "test_prob_0")].iloc[0]
    delta = float(np.max(np.abs(exported - curve)))
    ess_delta = float(diag.ess_bulk - float(ess_row.ess_bulk))
    report = {
        "status": "ok" if delta <= 1e-8 and abs(ess_delta) <= 1e-6 else "differs",
        "target_id": TARGET.target_id, "observable": "test_prob_0", "device": str(dev), "h": 0.01,
        "window": WINDOW, "max_abs_acf": delta, "lag1": float(curve[1]),
        "exported_lag1": float(exported[1]), "ess_bulk": float(diag.ess_bulk),
        "exported_ess_bulk": float(ess_row.ess_bulk), "ess_bulk_abs_diff": ess_delta,
        "source_files": loaded["files"],
    }
    return _write(cfg, report)


def _write(cfg: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    path = RC.output_root(cfg) / "prediction_acf_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, clean_json(report))
    report["path"] = str(path)
    return report
