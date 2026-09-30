"""Prior spectral comparator and prior/posterior predictive scores (runbook §12.3, §13.2)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from . import config as C
from . import diagnostics as dg
from .context import TargetContext, pick_device
from .randomness import numpy_rng, stream_seed, torch_gen
from .reference import boot_kw, completed_production, select_indices
from .spectral import normalized_norm
from .storage import atomic_write_json, clean_json


def prior_spectral(cfg: dict[str, Any], root: Path, device=None) -> pd.DataFrame:
    device = device or pick_device()
    sigma, a = float(cfg["model"]["sigma"]), float(cfg["domain"]["spectral_cutoff_a"])
    nmat = int(cfg["spectral"]["prior_iid_matrices_per_width"])
    qs = [float(q) for q in cfg["spectral"]["posterior_quantiles"]]
    reps = int(cfg["uncertainty"]["bootstrap_replicates"])
    pct = [float(x) for x in cfg["uncertainty"]["interval_percentiles"]]
    rows, values = [], {}
    for m in cfg["model"]["deep"]["widths"]:
        m = int(m)
        seed = stream_seed(int(cfg["master_seed"]), rep="shared", arch="deep", width=m, chain="shared",
                           stage="prior_spectral", stream="W2_prior")
        gen = torch_gen(seed, device)
        S = []
        bs = max(1, min(512, (1 << 24) // (m * m)))
        for i in range(0, nmat, bs):
            k = min(bs, nmat - i)
            W = sigma * torch.randn(k, m, m, generator=gen, device=device, dtype=torch.float64)
            S.append(normalized_norm(W, a).cpu().numpy())
        S = np.concatenate(S)
        values[m] = S
        rng = numpy_rng(stream_seed(int(cfg["master_seed"]), rep="shared", arch="deep", width=m, chain="shared",
                                    stage="prior_spectral", stream="bootstrap"))
        idx = rng.integers(0, nmat, size=(reps, nmat))
        row = {"architecture": "deep", "m": m, "sigma": sigma, "a": a, "n_matrices": nmat,
               "law": "iid_gaussian_prior_W2", "shared_across_replicates": True,
               "exits": int((S > 1.0).sum()), "occupancy": float((S <= 1.0).mean())}
        for q in qs:
            est = float(np.quantile(S, q, method=dg.QUANTILE_METHOD))
            bq = np.quantile(S[idx], q, axis=1, method=dg.QUANTILE_METHOD)
            lo, hi = np.percentile(bq, pct, method=dg.QUANTILE_METHOD)
            row.update({f"S_q{q:g}": est, f"S_q{q:g}_mcse": float(bq.std(ddof=1)), f"S_q{q:g}_mc_low": float(lo),
                        f"S_q{q:g}_mc_high": float(hi)})
        rows.append(row)
    df = pd.DataFrame(rows)
    (root / "controls").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "controls" / "prior_spectral.csv", index=False)
    np.savez(root / "controls" / "prior_spectral_values.npz", **{f"m{m}": v for m, v in values.items()})
    return df


def _log_probs(ctx: TargetContext, states: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    f = ctx.model.logits(states, X)
    z = math.sqrt(2.0) * f
    return torch.where(y.bool(), -torch.nn.functional.softplus(-z), -torch.nn.functional.softplus(z))


def nls(logp: np.ndarray) -> float:
    """logp (S, n_test): −mean_i log mean_s p_s(y_i|x_i), with log-sum-exp (no clipping)."""
    S = logp.shape[0]
    mx = logp.max(axis=0)
    lme = mx + np.log(np.exp(logp - mx).sum(axis=0)) - math.log(S)
    return float(-lme.mean())


def predictive(ctx: TargetContext) -> dict[str, Any]:
    cfg = ctx.cfg
    pc = cfg["predictive_check"]
    tr = ctx.open_reference()
    segs = completed_production(tr)
    out_path = ctx.dir / "predictive.h5"
    prior_path = ctx.root / "controls" / "prior_predictive" / f"{ctx.t.target_id}.h5"
    X = torch.as_tensor(ctx.D["X_test"], device=ctx.device)
    y = torch.as_tensor(ctx.D["y_test"], device=ctx.device)
    n_post = int(pc["posterior_states_per_target"])
    per_chain = n_post // len(ctx.chains)
    batch = max(1, min(256, (1 << 26) // (X.shape[0] * ctx.t.m * 8)))
    res: dict[str, Any] = {"target_id": ctx.t.target_id}
    post, post_draws = None, []
    if segs:
        M = tr.archive_count(segs)
        if M >= per_chain:
            idx = select_indices(M, per_chain)
            lp = []
            for c in ctx.chains:
                rows, dr = tr.archive_rows(c, segs, idx)
                post_draws.append(dr)
                for i in range(0, rows.shape[0], batch):
                    lp.append(_log_probs(ctx, torch.as_tensor(rows[i:i + batch], device=ctx.device), X, y).cpu().numpy())
            post = np.concatenate(lp)
    n_prior = int(pc["prior_iid_states_per_target"])
    gen = torch_gen(ctx.seed("shared", "prior_predictive", "prior_states"), ctx.device)
    lp = []
    for i in range(0, n_prior, batch):
        k = min(batch, n_prior - i)
        th = ctx.model.theta0 + ctx.sigma * torch.randn(k, ctx.lay.p, generator=gen, device=ctx.device)
        lp.append(_log_probs(ctx, th, X, y).cpu().numpy())
    prior = np.concatenate(lp)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = prior_path.with_name(f".{prior_path.name}.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("log_prob", data=prior)
        f.attrs["target_id"] = ctx.t.target_id
        f.attrs["target_hash"] = ctx.thash
    tmp.replace(prior_path)
    nls_prior = nls(prior)
    teacher = np.where(ctx.D["y_test"] == 1, np.log(ctx.D["p_teacher_test"]), np.log1p(-ctx.D["p_teacher_test"]))
    res.update({"nls_prior": nls_prior, "teacher_log_score": float(-teacher.mean()),
                "prior_accuracy": float(((np.exp(prior).mean(0) > 0.5) == (ctx.D["y_test"] == 1)).mean()),
                "n_prior_states": n_prior, "n_test": int(X.shape[0])})
    if post is None:
        res.update({"status": "unresolved_insufficient_archive", "nls_posterior": None, "improvement": None})
        ctx.save_analysis("predictive", res)
        return res
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("log_prob", data=post)
        f.create_dataset("draws", data=np.stack(post_draws))
        f.attrs["target_hash"] = ctx.thash
    tmp.replace(out_path)
    nls_post = nls(post)
    bk = boot_kw(cfg)
    chains_lp = [post[i * per_chain:(i + 1) * per_chain] for i in range(len(ctx.chains))]
    ll = np.stack([c.mean(axis=1) for c in chains_lp])
    iat = ll.size / dg.ess_mean_raw(ll)[0] if not dg.is_constant(ll) else 1.0
    prng = numpy_rng(ctx.seed("shared", "prior_predictive", "bootstrap_prior"))
    pidx = prng.integers(0, n_prior, size=(bk["reps"], n_prior))
    counter = {"i": -1}                          # first call is the point estimate (full prior sample)

    def stat(chs):
        i = counter["i"]
        counter["i"] += 1
        pr = prior if i < 0 else prior[pidx[i % bk["reps"]]]
        return nls(pr) - nls(np.concatenate(chs))

    br = dg.bootstrap_scalar(stat, chains_lp, iat, seed=ctx.seed("shared", "posterior_predictive", "bootstrap"), **bk)
    est = nls_prior - nls_post
    res.update({"status": "ok", "nls_posterior": nls_post, "improvement": est, "improvement_mcse": br.mcse,
                "improvement_mc_low": br.low, "improvement_mc_high": br.high, "block_length": br.block,
                "block_stable": br.stable, "n_posterior_states": int(post.shape[0]),
                "posterior_accuracy": float(((np.exp(post).mean(0) > 0.5) == (ctx.D["y_test"] == 1)).mean()),
                "posterior_draws_per_chain": per_chain})
    ctx.save_analysis("predictive", res)
    return res
