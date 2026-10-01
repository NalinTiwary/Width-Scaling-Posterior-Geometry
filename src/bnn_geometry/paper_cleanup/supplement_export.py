"""Cluster-side export for the selective supplements (supplementary_experiments.md §3-§5); no sampling.

S1: the eight fixed held-out class-1 probabilities (and V) over the same final window of every h=0.01 trajectory
used for the main loss figure. S2: V over the final ``duration`` of algorithmic time for the deep endpoint widths at
the final step and at its half (both existing trajectories). Also a fresh direct-SVD unit check of
T = ||W2||op/sqrt(m) on archived reference matrices. Every series is verified before export: kernel/step/target
and execution hashes, the discard length, contiguous unit-stride stages, the stored-value repeat at every rejected
transition (including the discard/retained boundary), and all 27 recorded scalars of the final state recomputed
from the checkpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..paper_export import h_dir_name, sha256_file
from ..storage import StorageError, read_json, read_trace_file, segment_path, torch_load, validate_trace
from .common import HISTORICAL_A, S2_DURATION, WINDOW, Layout, write_json

END_ATOL, END_RTOL = 1e-11, 1e-9


class _Ctx:
    """Lazily built model context per target (torch only needed for the end-state recomputation)."""

    def __init__(self, cfg, root, device):
        self.cfg, self.root, self.device, self._c = cfg, root, device, {}

    def get(self, t):
        if t.target_id not in self._c:
            import torch
            from ..context import TargetContext
            self._c = {t.target_id: TargetContext(self.cfg, t, self.root, device=torch.device(self.device))}
        return self._c[t.target_id]


def read_trajectory(cfg: dict[str, Any], root: Path, t: C.Target, h: float, ctxs: _Ctx,
                    columns: list[str]) -> dict[str, Any]:
    """All retained transitions (after the discard segment) of one dynamics trajectory, fully verified."""
    tdir = root / "targets" / t.target_id
    spec = read_json(tdir / "spec.json")
    traj = tdir / "dynamics" / h_dir_name(h)
    ck = torch_load(traj / "checkpoint.pt")
    meta = ck.get("meta") or {}
    bad = []
    if ck.get("kind") != "pcnl" or meta.get("role") != "dynamics":
        bad.append(f"kind/role {ck.get('kind')}/{meta.get('role')}")
    if float(meta.get("h", float("nan"))) != float(h):
        bad.append(f"h={meta.get('h')}")
    if ck["hashes"].get("target_hash") != spec["target_hash"]:
        bad.append("target_hash")
    if ck.get("current") is not None or ck.get("chains") != [int(c) for c in cfg["chain_ids"]]:
        bad.append("unfinished segment or chain ids")
    exp = C.execution_hash(cfg, spec["target_hash"], "pcnl", step=h, role="dynamics")
    code_ok = C.sampler_code_hash() == spec.get("sampler_code_hash")
    if code_ok and ck["hashes"].get("execution_hash") != exp:
        bad.append("execution_hash")
    if bad:
        raise StorageError(f"{t.target_id} h={h}: ineligible trajectory: {'; '.join(bad)}")
    order = list(ck["order"])
    stages = [s for s in order if s.startswith("stage_")]
    if order[0] != "discard" or order[1:] != stages or stages != [f"stage_{i}" for i in range(1, len(stages) + 1)]:
        raise StorageError(f"{t.target_id} h={h}: unexpected segment order {order}")
    s2 = float(cfg["model"]["sigma"]) ** 2
    n_discard = C.n_steps(float(cfg["dynamics"]["production_discard_time_over_sigma_squared"]) * s2, h)
    if ck["done"]["discard"]["n"] != n_discard or ck["done"]["discard"]["partial"]:
        raise StorageError(f"{t.target_id} h={h}: discard {ck['done']['discard']['n']} != {n_discard}")
    hashes = {k: ck["hashes"][k] for k in ("target_hash", "execution_hash")}
    names = list(spec["scalar_names"])
    idx = [names.index(c) for c in columns]
    X, A, files, last_rows = [], [], [], []
    for c in ck["chains"]:
        parts, acc, prev_last = [], [], None
        for seg in ["discard"] + stages:
            rec = ck["done"][seg]
            if rec["partial"]:
                raise StorageError(f"{t.target_id} h={h}: partial segment {seg}")
            p = segment_path(traj / f"chain_{c}", seg)
            arr, attrs = read_trace_file(p, ["scalars", "draw", "accepted"])
            validate_trace(arr, attrs, expect_first=0, expect_n=rec["n"], hashes=hashes, chain=c, where=str(p))
            if int(attrs.get("saved_stride", 0)) != 1 or list(attrs["names"]) != names:
                raise StorageError(f"{p}: stride/names differ from spec")
            sc = np.asarray(arr["scalars"], dtype=np.float64)
            a = np.asarray(arr["accepted"], dtype=np.int8)
            # a rejected transition repeats the current state: every recorded scalar is unchanged
            full = sc if prev_last is None else np.vstack([prev_last[None], sc])
            fa = a if prev_last is None else np.concatenate([[1], a])
            rej = np.flatnonzero(fa[1:] == 0) + 1
            if rej.size and not np.array_equal(full[rej], full[rej - 1]):
                raise StorageError(f"{p}: a recorded scalar changes at a rejected transition")
            prev_last = sc[-1]
            if seg != "discard":
                parts.append(sc[:, idx])
                acc.append(a)
                files.append(str(p.relative_to(root)))
        X.append(np.concatenate(parts))
        A.append(np.concatenate(acc))
        last_rows.append(prev_last)
    n_ret = {x.shape[0] for x in X}
    if len(n_ret) != 1:
        raise StorageError(f"{t.target_id} h={h}: chains have different retained lengths")
    # all recorded scalars of the final state, recomputed from the checkpoint
    import torch
    ctx = ctxs.get(t)
    th = ck["state"]["theta"].to(ctx.device, torch.float64)
    V, f = ctx.model.V(th)
    rec_all = ctx.obs.record(th, V, f).detach().cpu().numpy()
    last = np.stack(last_rows)
    err = np.abs(rec_all - last) - (END_ATOL + END_RTOL * np.abs(last))
    if np.any(err > 0):
        j = int(np.argmax(err.max(axis=0)))
        raise StorageError(f"{t.target_id} h={h}: recomputed end-state {names[j]} differs from the trace")
    return {"X": np.stack(X), "accepted": np.stack(A), "n_discard": n_discard, "n_retained": n_ret.pop(),
            "segments": order, "files": files, "execution_hash": hashes["execution_hash"],
            "target_hash": spec["target_hash"], "sampler_code_hash": spec.get("sampler_code_hash", ""),
            "end_state_max_abs_diff": float(np.max(np.abs(rec_all - last))), "execution_hash_checked": code_ok}


# ---- S1 -----------------------------------------------------------------------------------------------
def export_s1(cfg, root, L: Layout, ctxs: _Ctx, *, window: int = WINDOW, targets=None) -> list[dict[str, Any]]:
    J = [int(j) for j in cfg["probes"]["test_indices"]]
    cols = ["V"] + [f"test_prob_{j}" for j in J]
    loss_dir = root / "paper_figures" / "loss_traces"
    logs = []
    for t in C.targets(cfg):
        if targets and t.target_id not in targets:
            continue
        r = read_trajectory(cfg, root, t, 0.01, ctxs, cols)
        if r["n_retained"] < window:
            raise StorageError(f"{t.target_id}: {r['n_retained']} retained < window {window}")
        W = r["X"][:, -window:, :]
        start = r["n_discard"] + r["n_retained"] - window + 1
        it = np.tile(np.arange(start, start + window, dtype=np.int64), (W.shape[0], 1))
        P = W[:, :, 1:]
        if not np.all(np.isfinite(P)) or P.min() < 0 or P.max() > 1:
            raise StorageError(f"{t.target_id}: probabilities outside [0,1]")
        # exact overlap with the accepted main loss export
        lz = np.load(loss_dir / f"{t.target_id}.npz", allow_pickle=False)
        if not (np.array_equal(lz["V"], W[:, :, 0]) and np.array_equal(lz["iteration_start"], it[:, 0])):
            raise StorageError(f"{t.target_id}: S1 window is not the main-figure loss window")
        degenerate = [(c, J[j]) for c in range(P.shape[0]) for j in range(P.shape[2]) if P[c, :, j].std() < 1e-10]
        L.traces.mkdir(parents=True, exist_ok=True)
        npz = L.traces / f"s1_{t.target_id}.npz"
        np.savez_compressed(npz, probabilities=P, V=W[:, :, 0], iteration=it, test_indices=np.asarray(J, np.int64),
                            accepted=r["accepted"][:, -window:])
        side = {"target_id": t.target_id, "architecture": t.arch, "L": t.L, "m": t.m, "replicate": t.rep, "h": 0.01,
                "window": window, "iteration_start": int(start), "iteration_end": int(start + window - 1),
                "n_discard": r["n_discard"], "retained_per_chain": r["n_retained"], "segments": r["segments"],
                "source_files": r["files"], "execution_hash": r["execution_hash"], "target_hash": r["target_hash"],
                "sampler_code_hash": r["sampler_code_hash"], "execution_hash_checked": r["execution_hash_checked"],
                "probability_convention": "sigmoid(sqrt(2) f(x_j))", "test_indices": J,
                "end_state_max_abs_diff_all_scalars": r["end_state_max_abs_diff"],
                "matches_main_loss_window": True, "numerically_degenerate": degenerate,
                "npz": str(npz.relative_to(L.out)), "npz_sha256": sha256_file(npz), "status": "reused_existing_trace"}
        write_json(npz.with_suffix(".json"), side)
        logs.append(side)
        print(f"OK   S1 {t.target_id}: window [{start}, {start + window - 1}]  end-state max diff "
              f"{r['end_state_max_abs_diff']:.1e}", flush=True)
    return logs


# ---- S2 -----------------------------------------------------------------------------------------------
def deep_final_h(root: Path) -> float:
    d = read_json(root / "endpoint_decision_round2.json")["deep"]
    return float(d["final_h"] if "final_h" in d else d["h"])


def export_s2(cfg, root, L: Layout, ctxs: _Ctx, *, duration: float = S2_DURATION, targets=None) -> list[dict]:
    h0 = deep_final_h(root)
    steps = [h0, h0 / 2.0]
    ends = C.endpoint_widths(cfg, "deep")
    logs = []
    for t in C.targets(cfg):
        if t.arch != "deep" or t.m not in ends or (targets and t.target_id not in targets):
            continue
        for h in steps:
            n = int(round(duration / h))
            if not math.isclose(n * h, duration, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"duration {duration} is not an integer number of steps at h={h}")
            r = read_trajectory(cfg, root, t, h, ctxs, ["V"])
            if r["n_retained"] < n:
                raise StorageError(f"{t.target_id} h={h}: {r['n_retained']} retained < {n} for duration {duration}")
            V = r["X"][:, -n:, 0]
            start = r["n_discard"] + r["n_retained"] - n + 1
            L.traces.mkdir(parents=True, exist_ok=True)
            npz = L.traces / f"s2_{t.target_id}_{h_dir_name(h)}.npz"
            np.savez_compressed(npz, V=V, accepted=r["accepted"][:, -n:], h=np.float64(h),
                                iteration=np.tile(np.arange(start, start + n, dtype=np.int64), (V.shape[0], 1)))
            side = {"target_id": t.target_id, "m": t.m, "replicate": t.rep, "h": h, "duration": duration,
                    "transitions": n, "iteration_start": int(start), "iteration_end": int(start + n - 1),
                    "n_discard": r["n_discard"], "retained_per_chain": r["n_retained"], "segments": r["segments"],
                    "source_files": r["files"], "execution_hash": r["execution_hash"],
                    "execution_hash_checked": r["execution_hash_checked"],
                    "end_state_max_abs_diff_all_scalars": r["end_state_max_abs_diff"],
                    "npz": str(npz.relative_to(L.out)), "npz_sha256": sha256_file(npz), "status": "reused_existing_trace"}
            write_json(npz.with_suffix(".json"), side)
            logs.append(side)
            print(f"OK   S2 {t.target_id} h={h}: last {n} of {r['n_retained']} retained", flush=True)
    return logs


# ---- fresh unit check of T --------------------------------------------------------------------------
def spectral_unit_check(cfg, root, L: Layout) -> pd.DataFrame:
    """First and last archived reference W2 of every chain: full float64 SVD of the raw matrix / sqrt(m) vs T."""
    from ..model import Layout as ML
    states = pd.read_csv(root / "paper_figures" / "tables" / "spectral_states.csv.gz",
                         usecols=["target_id", "m", "chain_id", "original_draw_index", "S"])
    rows = []
    for t in C.targets(cfg):
        if t.arch != "deep":
            continue
        lay = ML(t.arch, t.m, int(cfg["data"]["input_dimension"]))
        ref = root / "targets" / t.target_id / "reference"
        ck = torch_load(ref / "checkpoint.pt")
        segs = [s for s in ck["order"] if s.startswith("production_s")]
        k = int(segs[-1].split("_s")[1])
        segs = [f"production_s{i}" for i in range(1, k + 1)]
        off = np.cumsum([0] + [ck["done"][s]["n"] for s in segs])
        for c in ck["chains"]:
            for si, which in ((0, 0), (len(segs) - 1, -1)):
                p = segment_path(ref / f"chain_{c}", segs[si])
                import h5py
                with h5py.File(p, "r") as fh:
                    th = fh["archive/states"][which]
                    dr = int(fh["archive/draw"][which]) + int(off[si])
                W2 = _w2(th, lay)
                T_direct = float(np.linalg.svd(W2, compute_uv=False)[0] / math.sqrt(t.m))
                st = states[(states.target_id == t.target_id) & (states.chain_id == c) &
                            (states.original_draw_index == dr)]
                if len(st) != 1:
                    raise StorageError(f"{t.target_id} chain {c}: archived draw {dr} not in the spectral table")
                T_tab = HISTORICAL_A * float(st.S.iloc[0])
                rel = abs(T_direct - T_tab) / T_direct
                rows.append({"target_id": t.target_id, "m": t.m, "chain": c, "draw": dr, "T_direct": T_direct,
                             "T_table": T_tab, "rel_diff": rel, "pass": rel <= 1e-9})
    d = pd.DataFrame(rows)
    L.exports.mkdir(parents=True, exist_ok=True)
    d.to_csv(L.exports / "spectral_unit_check_direct.csv", index=False)
    return d


def _w2(theta: np.ndarray, lay) -> np.ndarray:
    import torch
    from ..model import unpack
    return unpack(torch.as_tensor(theta, dtype=torch.float64)[None], lay)["W2"][0].numpy()


def export_all(cfg, root, L: Layout, *, device="cpu", window=WINDOW, duration=S2_DURATION,
               targets=None, parts=("s1", "s2", "unit")) -> int:
    ctxs = _Ctx(cfg, root, device)
    log, fails = {"window": window, "duration": duration, "code_revision": C.code_revision()}, []
    for part, fn in (("s1", lambda: export_s1(cfg, root, L, ctxs, window=window, targets=targets)),
                     ("s2", lambda: export_s2(cfg, root, L, ctxs, duration=duration, targets=targets)),
                     ("unit", lambda: spectral_unit_check(cfg, root, L).to_dict("records"))):
        if part not in parts:
            continue
        try:
            log[part] = fn()
        except (StorageError, KeyError, OSError, ValueError) as exc:
            fails.append({"part": part, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {part}: {exc}", flush=True)
    log["failures"] = fails
    write_json(L.out / "supplement_export_log.json", log)
    n1, n2 = len(log.get("s1", [])), len(log.get("s2", []))
    print(f"supplement export: S1 {n1} targets, S2 {n2} target/step combinations, "
          f"unit check {len(log.get('unit', []))} matrices, {len(fails)} failures")
    return 0 if not fails else 1
