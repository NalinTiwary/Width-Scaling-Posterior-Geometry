"""Per-target diagnostics of the reference (elliptical-slice) chains: loss V and, for deep targets, T."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config as C
from .common import HISTORICAL_A, Layout


def reference_table(cfg: dict[str, Any], root: Path, L: Layout) -> pd.DataFrame:
    rd = pd.read_csv(root / "tables" / "reference_diagnostics.csv")
    sp = pd.read_csv(root / "tables" / "spectral_summary.csv")
    sp = sp[sp.chain_id.astype(str) == "all"].set_index("target_id")
    ref = cfg["reference"]
    rows = []
    for t in C.targets(cfg):
        st = json.loads((root / "targets" / t.target_id / "status.json").read_text())["reference"]
        seg = st["segments"]
        prod = [k for k in seg if k.startswith("production_s")]
        retained = sum(int(seg[k]["n"]) for k in prod)
        archived = sum(int(seg[k]["n"]) // int(seg[k]["archive_stride"]) for k in prod)
        d = rd[rd.target_id == t.target_id]
        v = d[d.observable == "V"].iloc[0]
        n_chains = len(st["likelihood_evals_per_chain"])
        row = {"target_id": t.target_id, "architecture": t.arch, "m": t.m, "replicate": t.rep,
               "sampler": ref["sampler"], "chains": n_chains, "stage": v.stage,
               "burn_in_per_chain": int(seg["burnin"]["n"]), "calibration_per_chain": int(seg["calibration"]["n"]),
               "retained_per_chain": retained, "retained_total": retained * n_chains,
               "saved_stride": int(v.saved_stride), "archived_states_per_chain": archived,
               "likelihood_evals_per_chain_max": int(max(st["likelihood_evals_per_chain"])),
               "V_mean": v["mean"], "V_sd": v.sd, "V_mcse": v.mcse, "V_rhat": v.rhat, "V_ess_bulk": v.ess_bulk,
               "V_ess_tail": v.ess_tail, "V_split_half_drift_pass": bool(v.drift_pass),
               "all_observables_rhat_max": float(d.rhat.max()), "all_observables_ess_bulk_min": float(d.ess_bulk.min()),
               "all_observables_ess_tail_min": float(d.ess_tail.min()), "n_observables": int(len(d)),
               "all_gates_pass": bool(d.gate_pass.all())}
        if t.arch == "deep":
            s = sp.loc[t.target_id]
            row.update({"T_inspected_states": int(s.inspected_states), "T_rhat": s.S_rhat, "T_ess_bulk": s.S_ess_bulk,
                        "T_ess_tail": s.S_ess_tail, **{f"T_q{q}": HISTORICAL_A * s[f"S_q{q}"]
                                                       for q in ("0.5", "0.95", "0.99")},
                        "T_svd_backend_check_ok": bool(s.backend_check_ok)})
        row["reference_pass"] = bool(json.loads((root / "targets" / t.target_id / "analysis" /
                                                 f"reference_stage_{len(prod)}.json").read_text())["reference_pass"]) \
            if (root / "targets" / t.target_id / "analysis").exists() else None
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(L.tables / "reference_chain_diagnostics.csv", index=False)
    return out
