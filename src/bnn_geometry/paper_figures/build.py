"""One command for the whole two-figure package once the loss NPZs exist (guide §12)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .. import config as C
from .common import HERE, METRICS_PATH, STYLE_PATH, Layout, sha256, versions

STEPS = ("selfcheck", "spectral", "acf", "render", "text")


def _guard(cfg: dict[str, Any], out: Path) -> None:
    paper = (C.ROOT / "results" / "final_geometry").resolve()
    is_fixture = cfg["campaign_id"].startswith("fixture")
    inside = paper == out.resolve() or paper in out.resolve().parents
    if is_fixture and inside:
        raise SystemExit("refusing to write fixture outputs into the paper campaign directory")


def selfcheck(L: Layout) -> None:
    r = subprocess.run([sys.executable, "selfcheck.py"], cwd=HERE, capture_output=True, text=True)
    (L.out / "selfcheck_result.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"vendored selfcheck.py failed:\n{r.stdout}{r.stderr}")


def _write_input_map(L: Layout, root: Path, spectral_rows: list[dict]) -> None:
    rows = list(spectral_rows)
    for name, logical, mapping in [("predictive_scores.csv", "predictive_check", "prior/posterior predictive NLS"),
                                   ("target_audit.csv", "reference_diagnostics", "per-target reference status")]:
        p = root / "tables" / name
        if p.exists():
            rows.append({"logical_input": logical, "path": str(p.relative_to(root)), "dataset": "csv rows",
                         "shape": "-", "dtype": "csv", "sha256": sha256(p), "mapping": mapping})
    lm = L.out / "input_map_loss.csv"
    if lm.exists():
        with open(lm) as f:
            for r in csv.DictReader(f):
                rows.append({"logical_input": r["logical_input"], "path": r["path"], "dataset": r["dataset"],
                             "shape": r["shape"], "dtype": r["dtype"], "sha256": r["sha256"],
                             "mapping": f"{r['mapping']}; transitions {r['first_transition']}-{r['last_transition']}"})
    if not rows:
        return
    with open(L.out / "input_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build(cfg: dict[str, Any], root: Path, out: Optional[Path] = None, *, steps: tuple[str, ...] = STEPS,
          window: int = 102400) -> int:
    from . import analysis, diagnostics, prepare, render, text
    out = Path(out) if out else root / "paper_figures"
    _guard(cfg, out)
    L = Layout(out)
    L.mkdirs()
    t0 = time.time()
    log = json.loads((L.out / "loss_export_log.json").read_text()) if (L.out / "loss_export_log.json").exists() else None
    spectral_rows: list[dict] = []
    done = []
    if "selfcheck" in steps:
        selfcheck(L)
        done.append("selfcheck")
    if "spectral" in steps:
        spectral_rows = prepare.prepare_spectral(cfg, root, L)
        analysis.spectral_tables(cfg, root, L)
        done.append("spectral")
    have_traces = L.traces.exists() and any(L.traces.glob("*.npz"))
    if "acf" in steps:
        if not have_traces:
            raise SystemExit(f"no loss-trace NPZs in {L.traces}: run export-loss-traces first (or omit the acf step)")
        if log is not None and log.get("window") != window:
            raise SystemExit(f"NPZs were exported with window {log.get('window')}, not {window}")
        st = analysis.acf_tables(cfg, root, L, window=window)
        analysis.appendix_table(cfg, root, L, log)
        diagnostics.sheets(cfg, L)
        done.append(f"acf(K={st['K']}, status={st['fixed_step_loss_acf_status']})")
    figs = []
    if "render" in steps:
        figs += render.figure1(L.tables, L.figures)
        if (L.tables / "acf_plot.csv").exists():
            figs += render.figure2(L.tables, L.figures)
        done.append("render")
    if "text" in steps:
        text.write_all(L, export_log=log)
        done.append("text")
    if "spectral" in steps:
        _write_input_map(L, root, spectral_rows)
    prov_p = L.out / "provenance.json"
    prov = json.loads(prov_p.read_text()) if prov_p.exists() else {}
    prov.update({
        "campaign_id": cfg["campaign_id"], "campaign_root": str(root), "code_revision": C.code_revision(),
        "versions": versions(), "figure_metrics_sha256": sha256(METRICS_PATH), "plot_style_sha256": sha256(STYLE_PATH),
        "selfcheck_sha256": sha256(HERE / "selfcheck.py"), "last_steps": done,
        "regenerate_figures_from_tables": "python -m bnn_geometry.paper_figures.render --tables "
                                          f"{L.tables} --figures {L.figures}",
        "full_build": "python -m bnn_geometry paper-figures",
        "outputs": sorted(str(p.relative_to(L.out)) for p in L.out.rglob("*") if p.is_file() and p.suffix != ".npz"),
        "loss_trace_npz_sha256": {p.name: sha256(p) for p in sorted(L.traces.glob("*.npz"))} if have_traces else {}})
    prov_p.write_text(json.dumps(prov, indent=2))
    print(f"paper figures: {', '.join(done)} in {time.time() - t0:.0f}s -> {L.out}")
    for f in figs:
        print(f"  {f}")
    return 0
