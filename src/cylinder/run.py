"""High-level runner for one (m, seed, kernel) target."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .centers import assemble_theta0, load_centers
from .data import generate_projections, load_data_bundle
from .device_utils import configure_dtype, device_metadata, resolve_device, synchronize
from .ess_slice import SliceStats, elliptical_slice_step
from .io import (
    atomic_save_json,
    atomic_save_npz,
    build_run_metadata,
    run_dir_name,
    validate_resume,
)
from .observables import (
    compute_observables_torch_full,
    curvature_state_indices,
    empty_observable_buffers,
    fill_draw,
)
from .pcn import PCNStats, pcn_step
from .theorem import theorem_bundle


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def run_target(
    *,
    cfg: dict[str, Any],
    m: int,
    center_seed: int,
    kernel: str = "ess",
    artifacts: Path,
    device_request: Optional[str] = None,
    chain_ids: Optional[list[int]] = None,
    n_retained: Optional[int] = None,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Path:
    """
    Run all (or selected) chains for one scientific target and write
    artifacts/runs/m{m}_seed{seed}_{kernel}/.
    """
    configure_dtype()
    req = device_request or cfg.get("device", "auto")
    device = resolve_device(req)
    dev_meta = device_metadata(req, device)

    data_path = artifacts / "data.npz"
    centers_path = artifacts / f"centers_seed{center_seed}.npz"
    data = load_data_bundle(data_path)
    bank = load_centers(centers_path)

    X_np = np.asarray(data["X"], dtype=np.float64)
    y_np = np.asarray(data["y"], dtype=np.float64)
    X_probe_np = np.asarray(data["X_probe"], dtype=np.float64)
    n, d = X_np.shape
    assert d == cfg["architecture"]["input_dim"]

    sigma = float(cfg["prior"]["sigma"])
    b0 = float(cfg["prior"]["b0"])
    theta0_np = assemble_theta0(bank["U"], m=m, b0=b0)
    bun = theorem_bundle(
        m=m,
        n=n,
        d=d,
        C=cfg["architecture"]["classes"],
        b0=b0,
        sigma=sigma,
        M2=float(cfg["theorem"]["M2"]),
        M4=float(cfg["theorem"]["M4"]),
        c2=float(cfg["theorem"]["c2"]),
    )

    p = theta0_np.size
    proj = generate_projections(
        p=p,
        n_proj=int(cfg["data"]["n_projections"]),
        seed=int(cfg["data"]["projection_seed"]),
    )
    # Save projections once per width (shared across seeds is fine; p depends on m)
    proj_path = artifacts / f"projections_m{m}.npz"
    if not proj_path.exists():
        atomic_save_npz(proj_path, U=proj["U"], seed=proj["seed"], p=p)

    out_dir = artifacts / "runs" / run_dir_name(m, center_seed, kernel)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"

    expected_identity = {
        "schema_version": 1,
        "m": m,
        "center_seed": center_seed,
        "sigma": sigma,
        "data_hash_X": str(data["hash_X"]),
        "data_hash_y": str(data["hash_y"]),
        "centers_hash_U": str(bank["hash_U"]),
        "loss_normalization": "sum",
        "kernel": kernel,
        "B_m": bun["B_m"],
    }
    n_chains = int(cfg["sampling"]["n_chains"])
    if chain_ids is None:
        chain_ids = list(range(n_chains))
    burnin = int(cfg["sampling"]["burnin"])
    if n_retained is None:
        n_retained = int(cfg["sampling"]["retained_initial"])

    if meta_path.exists() and not overwrite:
        import json

        with open(meta_path) as f:
            old = json.load(f)
        validate_resume(old, expected_identity)
        if (
            skip_existing
            and (out_dir / "observables.npz").exists()
            and int(old.get("n_retained", 0)) >= n_retained
            and sorted(old.get("chain_ids", [])) == list(range(n_chains))
        ):
            print(f"Skipping completed target {out_dir.name} (T={old['n_retained']})")
            return out_dir
    lik_cap = int(cfg["sampling"]["likelihood_eval_cap_per_target"])
    # Per-chain share of the target budget (conservative).
    per_chain_cap = max(1, lik_cap // n_chains)

    X = _to_torch(X_np, device)
    y = _to_torch(y_np, device)
    X_probe = _to_torch(X_probe_np, device)
    theta0 = _to_torch(theta0_np, device)
    U_proj = _to_torch(proj["U"], device)

    buf = empty_observable_buffers(
        n_chains=n_chains,
        n_draws=n_retained,
        n_train=n,
        n_probes=X_probe_np.shape[0],
        n_proj=proj["U"].shape[0],
    )
    # Full z at m=4096 is ~17 GB per target, so by default only the
    # prespecified curvature states (design §6.1) are kept.
    store_full_z = bool(cfg.get("curvature", {}).get("store_full_z", False))
    n_curv = int(cfg["curvature"]["states_per_chain"])
    curv_idx = curvature_state_indices(n_retained, n_curv)
    curv_slots: dict[int, list[int]] = {}
    for j, t in enumerate(curv_idx):
        curv_slots.setdefault(int(t), []).append(j)
    Z_curv = np.full((n_chains, n_curv, p), np.nan, dtype=np.float64)
    Z_store = (
        np.full((n_chains, n_retained, p), np.nan, dtype=np.float64)
        if store_full_z
        else None
    )

    # Merge with an existing observables file when running a chain shard.
    obs_path = out_dir / "observables.npz"
    if obs_path.exists() and not overwrite and chain_ids != list(range(n_chains)):
        prev = np.load(obs_path)
        prev_T = prev["H"].shape[1]
        use_T = min(prev_T, n_retained)
        for key in buf:
            if key in prev.files:
                buf[key][:, :use_T] = prev[key][:, :use_T]
        if "z_curv" in prev.files and prev_T == n_retained:
            Z_curv[:] = prev["z_curv"]
        if Z_store is not None and "z" in prev.files:
            Z_store[:, :use_T] = prev["z"][:, :use_T]

    def record_state(c: int, t: int, z: torch.Tensor) -> None:
        slots = curv_slots.get(t)
        if slots is None and Z_store is None:
            return
        z_np = z.detach().cpu().numpy()
        for j in slots or ():
            Z_curv[c, j] = z_np
        if Z_store is not None:
            Z_store[c, t] = z_np

    total_lik = 0
    total_time = 0.0
    chain_stats: list[dict[str, Any]] = []
    hit_budget = False

    master = int(cfg.get("seed_master", 0))
    for c in chain_ids:
        rng = np.random.Generator(
            np.random.PCG64(master + 10_000 * center_seed + 100 * m + c + (0 if kernel == "ess" else 7))
        )
        z0_np = rng.standard_normal(p).astype(np.float64)
        z = _to_torch(z0_np, device)

        if kernel == "ess":
            stats = SliceStats()
            # burn-in
            for _ in range(burnin):
                if stats.n_likelihood_evals >= per_chain_cap:
                    hit_budget = True
                    break
                z = elliptical_slice_step(
                    z, theta0, sigma, X, y, m, rng, stats=stats
                )
            # retained
            import time

            synchronize(device)
            t0 = time.perf_counter()
            retained_time_lik0 = stats.n_likelihood_evals
            for t in range(n_retained):
                if stats.n_likelihood_evals >= per_chain_cap:
                    hit_budget = True
                    break
                z = elliptical_slice_step(
                    z, theta0, sigma, X, y, m, rng, stats=stats
                )
                obs = compute_observables_torch_full(
                    z, theta0, sigma, X, y, X_probe, U_proj, m, bun["B_m"]
                )
                fill_draw(buf, c, t, obs)
                record_state(c, t, z)
            synchronize(device)
            retained_time = time.perf_counter() - t0
            stats.wall_time_s = retained_time
            total_lik += stats.n_likelihood_evals
            total_time += retained_time
            chain_stats.append(
                {
                    "chain": c,
                    "likelihood_evals": stats.n_likelihood_evals,
                    "likelihood_evals_retained": stats.n_likelihood_evals - retained_time_lik0,
                    "bracket_shrinks": stats.n_bracket_shrinks,
                    "wall_time_retained_s": retained_time,
                    "n_updates": stats.n_updates,
                }
            )
        elif kernel == "pcn":
            pcfg = cfg["pcn_crosscheck"]
            stats = PCNStats()
            beta = float(pcfg["beta0"])
            beta_clip = tuple(pcfg["beta_clip"])
            # burn-in with adaptation
            for t_b in range(burnin):
                if stats.n_likelihood_evals >= per_chain_cap:
                    hit_budget = True
                    break
                z, accepted = pcn_step(
                    z, theta0, sigma, X, y, m, beta, rng, stats=stats
                )
                log_beta = np.log(beta)
                log_beta += float(pcfg["adapt_scale"]) * ((t_b + 10) ** -0.5) * (
                    float(accepted) - float(pcfg["target_accept"])
                )
                beta = float(np.clip(np.exp(log_beta), beta_clip[0], beta_clip[1]))
            stats.beta_final = beta
            import time

            synchronize(device)
            t0 = time.perf_counter()
            retained_time_lik0 = stats.n_likelihood_evals
            for t in range(n_retained):
                if stats.n_likelihood_evals >= per_chain_cap:
                    hit_budget = True
                    break
                z, _ = pcn_step(
                    z, theta0, sigma, X, y, m, beta, rng, stats=stats
                )
                obs = compute_observables_torch_full(
                    z, theta0, sigma, X, y, X_probe, U_proj, m, bun["B_m"]
                )
                fill_draw(buf, c, t, obs)
                record_state(c, t, z)
            synchronize(device)
            retained_time = time.perf_counter() - t0
            stats.wall_time_s = retained_time
            total_lik += stats.n_likelihood_evals
            total_time += retained_time
            chain_stats.append(
                {
                    "chain": c,
                    "likelihood_evals": stats.n_likelihood_evals,
                    "likelihood_evals_retained": stats.n_likelihood_evals - retained_time_lik0,
                    "wall_time_retained_s": retained_time,
                    "n_updates": stats.n_updates,
                    "n_accepted": stats.n_accepted,
                    "beta_final": stats.beta_final,
                }
            )
        else:
            raise ValueError(f"Unknown kernel {kernel}")

    z_arrays = {"z_curv": Z_curv, "curv_indices": curv_idx}
    if Z_store is not None:
        z_arrays["z"] = Z_store
    atomic_save_npz(
        out_dir / "observables.npz",
        **buf,
        **z_arrays,
        B_m=np.array(bun["B_m"]),
        D_th=np.array(bun["D_th"]),
        m=np.array(m),
        center_seed=np.array(center_seed),
    )

    meta = build_run_metadata(
        m=m,
        center_seed=center_seed,
        sigma=sigma,
        B_m=bun["B_m"],
        D_th=bun["D_th"],
        kernel=kernel,
        data_hashes={
            "hash_X": str(data["hash_X"]),
            "hash_y": str(data["hash_y"]),
            "hash_X_probe": str(data["hash_X_probe"]),
        },
        centers_hash=str(bank["hash_U"]),
        device_info=dev_meta,
        n_chains=n_chains,
        burnin=burnin,
        n_retained=n_retained,
        likelihood_evals=total_lik,
        wall_time_s=total_time,
        extra={
            "chain_stats": chain_stats,
            "hit_likelihood_budget": hit_budget,
            "A_n": bun["A_n"],
            "coverage_floor": bun["coverage_floor"],
            "chain_ids": chain_ids,
            "projections_hash": proj["hash_U"],
        },
    )
    atomic_save_json(meta_path, meta)
    # Also stash theta0 for curvature replay convenience
    atomic_save_npz(out_dir / "theta0.npz", theta0=theta0_np)
    return out_dir
