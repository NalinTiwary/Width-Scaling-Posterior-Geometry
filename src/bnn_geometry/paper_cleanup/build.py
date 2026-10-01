"""Step runner for ``python -m bnn_geometry paper-cleanup``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .. import config as C
from .common import FIGURE_WIDTH_IN, S2_DURATION, WINDOW, Layout, run_selfcheck, write_json

STEPS = ("selfcheck", "main", "export-supplements", "supplements", "text")
LOCAL_STEPS = ("selfcheck", "main", "supplements", "text")


def build(cfg: dict[str, Any], root: Path, dest: Optional[Path], *, steps=LOCAL_STEPS, device: str = "cpu",
          window: int = WINDOW, duration: float = S2_DURATION, width_in: float = FIGURE_WIDTH_IN,
          targets=None) -> int:
    L = Layout(dest or root / "paper_cleanup")
    L.mkdirs()
    prov_p = L.out / "provenance.json"
    prov = json.loads(prov_p.read_text()) if prov_p.exists() else {"runs": []}
    run = {"steps": list(steps), "code_revision": C.code_revision(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "source_root": str(root), "window": window, "duration": duration, "figure_width_in": width_in}
    rc = 0
    if "selfcheck" in steps:
        run["selfcheck"] = run_selfcheck(L.diagnostics / "selfcheck_result.txt").splitlines()[-1]
        print("bundled selfcheck:", run["selfcheck"])
    if "main" in steps:
        from .main_figures import build_main
        from .supplements import appendix_table
        run["main"] = build_main(cfg, root, L, device=device, width_in=width_in)
        s1_diag = L.tables / "supplement_diagnostics.csv"
        if not s1_diag.exists():
            appendix_table(cfg, root, L, None)
        print("main figures written:", L.figures)
    if "export-supplements" in steps:
        from .supplement_export import export_all
        rc |= export_all(cfg, root, L, device=device, window=window, duration=duration, targets=targets)
    if "supplements" in steps:
        from .supplements import build_supplements
        if not (L.traces.exists() and any(L.traces.glob("s1_*.npz"))):
            print("supplements: no exported traces in", L.traces, "- run the export-supplements step first")
            rc |= 1
        else:
            run["supplements"] = build_supplements(cfg, root, L, width_in=width_in, window=window, duration=duration)
            for k in ("s1", "s2"):
                s = run["supplements"].get(k, {})
                print(f"{k}: all series pass protocol gates = {s.get('all_series_valid')} "
                      f"(invalid {s.get('n_invalid')}, drift flags {s.get('n_drift_flags')})")
    if "text" in steps:
        from .text import write_text
        write_text(L)
        print("captions and figure_results.md written:", L.out)
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run["status"] = "ok" if rc == 0 else "failures"
    prov["runs"].append(run)
    write_json(prov_p, prov)
    return rc
