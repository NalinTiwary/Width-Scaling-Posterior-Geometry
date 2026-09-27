"""Chain-aware MCMC diagnostics and coverage uncertainty (§4.2, §5.2)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def batch_means_variance(
    indicators: np.ndarray,
    batch_len: Optional[int] = None,
) -> dict[str, Any]:
    """
    Chainwise batch-means variance for coverage indicators.

    indicators: shape (n_chains, T) with values in {0,1}.
    Returns Var-hat of the pooled mean estimator and per-chain details.
    If a chain has zero within-batch variation usable for estimation,
    mark that chain as non-estimable.
    """
    indicators = np.asarray(indicators, dtype=np.float64)
    C, T = indicators.shape
    if batch_len is None:
        batch_len = int(np.floor(np.sqrt(T)))
    batch_len = max(batch_len, 1)
    a = T // batch_len
    T_prime = a * batch_len
    if a < 2:
        return {
            "estimable": False,
            "reason": "fewer than 2 batches",
            "var_hat": np.nan,
            "se": np.nan,
            "batch_len": batch_len,
            "n_batches": a,
        }

    v_c = []
    estimable_chains = 0
    for c in range(C):
        x = indicators[c, :T_prime].reshape(a, batch_len).mean(axis=1)
        xbar = x.mean()
        if a > 1:
            vc = (batch_len / (a - 1)) * float(np.sum((x - xbar) ** 2))
        else:
            vc = np.nan
        v_c.append(vc)
        if np.isfinite(vc) and vc > 0:
            estimable_chains += 1

    # Design: Var(p̂) ≈ (1/16) sum_c v_c / T   with 4 chains → 1/C²
    # Generalize: (1/C²) sum_c (v_c / T)
    v_c_arr = np.asarray(v_c, dtype=np.float64)
    if estimable_chains == 0:
        return {
            "estimable": False,
            "reason": "no positive batch-means variance",
            "var_hat": np.nan,
            "se": np.nan,
            "v_c": v_c_arr,
            "batch_len": batch_len,
            "n_batches": a,
        }
    var_hat = float(np.nansum(v_c_arr) / (C**2 * T))
    return {
        "estimable": True,
        "var_hat": var_hat,
        "se": float(np.sqrt(var_hat)),
        "v_c": v_c_arr,
        "batch_len": batch_len,
        "n_batches": a,
    }


def coverage_summary(inside: np.ndarray) -> dict[str, Any]:
    """
    inside: (n_chains, T) int/bool.
    """
    inside = np.asarray(inside)
    C, T = inside.shape
    flat = inside.astype(np.float64)
    p_hat = float(flat.mean())
    exits = int(np.sum(1 - inside.astype(np.int64)))
    # Exit episodes: contiguous runs of outside
    episodes = 0
    chain_exits = []
    for c in range(C):
        row = inside[c].astype(np.int64)
        outside = 1 - row
        # count 0->1 transitions in outside
        ep = int(np.sum((outside[1:] == 1) & (outside[:-1] == 0)))
        if outside[0] == 1:
            ep += 1
        episodes += ep
        chain_exits.append(int(np.sum(outside)))

    bm = batch_means_variance(flat)
    all_inside = exits == 0
    return {
        "p_hat": p_hat,
        "n_chains": C,
        "T": T,
        "n_draws": C * T,
        "exit_count": exits,
        "exit_episodes": episodes,
        "chain_exit_counts": chain_exits,
        "all_inside": all_inside,
        "mcse_estimable": bool(bm["estimable"]) and not all_inside,
        "mcse": bm.get("se", np.nan) if (bm["estimable"] and not all_inside) else np.nan,
        "batch_means": bm,
    }


def exit_episodes_detail(inside: np.ndarray) -> list[dict[str, int]]:
    """List exit episodes as {chain, start, length}."""
    out = []
    C, T = inside.shape
    for c in range(C):
        row = 1 - inside[c].astype(np.int64)
        i = 0
        while i < T:
            if row[i] == 1:
                j = i
                while j < T and row[j] == 1:
                    j += 1
                out.append({"chain": c, "start": i, "length": j - i})
                i = j
            else:
                i += 1
    return out


def posterior_idata(data_vars: dict[str, np.ndarray]):
    """(chain, draw) arrays -> ArviZ posterior; 1.x takes a group dict, 0.x a kwarg."""
    import arviz as az

    if int(az.__version__.split(".")[0]) >= 1:
        return az.from_dict({"posterior": data_vars})
    return az.from_dict(posterior=data_vars)


def arviz_diagnostics(
    arrays: dict[str, np.ndarray],
    head_quantiles: Optional[list[float]] = None,
) -> dict[str, Any]:
    """
    Rank-normalized split/folded R-hat and bulk/tail ESS via ArviZ.

    arrays values: shape (chain, draw) for scalars, or (chain, draw, dim)
    for multivariate — each trailing dim is treated as a separate variable.
    """
    import arviz as az

    if head_quantiles is None:
        head_quantiles = [0.95, 0.99]

    data_vars = {}
    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2:
            data_vars[name] = arr
        elif arr.ndim == 3:
            for k in range(arr.shape[-1]):
                data_vars[f"{name}_{k}"] = arr[:, :, k]
        else:
            raise ValueError(f"Unsupported array ndim for {name}: {arr.ndim}")

    idata = posterior_idata(data_vars)
    summary = az.summary(
        idata,
        kind="diagnostics",
        round_to=None,
    )
    # summary index = variable names; columns include r_hat, ess_bulk, ess_tail
    rhat_max = float(summary["r_hat"].max()) if "r_hat" in summary else np.nan
    ess_bulk_min = float(summary["ess_bulk"].min()) if "ess_bulk" in summary else np.nan
    ess_tail_min = float(summary["ess_tail"].min()) if "ess_tail" in summary else np.nan

    # Localized quantile ESS for H if present
    quantile_ess = {}
    if "H" in arrays:
        for q in head_quantiles:
            try:
                qess = az.ess(idata, method="quantile", prob=q, var_names=["H"])
                # ArviZ >=1.3 returns a DataTree; older returns Dataset
                if hasattr(qess, "posterior"):
                    val = qess.posterior["H"].values
                else:
                    val = qess["H"].values if "H" in qess else np.asarray(qess["H"])
                quantile_ess[str(q)] = float(np.asarray(val).item())
            except Exception as exc:  # noqa: BLE001
                quantile_ess[str(q)] = np.nan
                quantile_ess[f"{q}_error"] = str(exc)

    return {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "quantile_ess_H": quantile_ess,
        "n_variables": len(data_vars),
        "summary_table": summary,
    }


def diagnostics_pass(
    diag: dict[str, Any],
    rhat_max: float = 1.01,
    bulk_ess_min: float = 1000,
    tail_ess_min: float = 400,
    quantile_ess_min: float = 400,
) -> dict[str, Any]:
    checks = {
        "rhat_ok": diag["rhat_max"] < rhat_max,
        "bulk_ess_ok": diag["ess_bulk_min"] >= bulk_ess_min,
        "tail_ess_ok": diag["ess_tail_min"] >= tail_ess_min,
    }
    qess = diag.get("quantile_ess_H", {})
    q_ok = True
    for q, val in qess.items():
        if q.endswith("_error"):
            continue
        if not (isinstance(val, (float, int)) and np.isfinite(val)):
            q_ok = False
            continue
        if val < quantile_ess_min:
            q_ok = False
    checks["quantile_ess_ok"] = q_ok
    checks["all_ok"] = all(checks.values())
    return checks


def half_chain_stability(
    values: np.ndarray,
) -> dict[str, float]:
    """Compare first vs second half means with a crude MCSE proxy."""
    values = np.asarray(values, dtype=np.float64)
    C, T = values.shape
    mid = T // 2
    if mid < 2:
        return {"delta": np.nan, "se_proxy": np.nan}
    m1 = values[:, :mid].mean()
    m2 = values[:, mid : 2 * mid].mean()
    # Independent-ish SE proxy using within-half variances / (C*mid)
    v1 = values[:, :mid].var(ddof=1)
    v2 = values[:, mid : 2 * mid].var(ddof=1)
    se = float(np.sqrt((v1 + v2) / (C * mid)))
    return {"mean_first": float(m1), "mean_second": float(m2), "delta": float(m2 - m1), "se_proxy": se}


def order_stat_quantile(x: np.ndarray, q: float = 0.95) -> float:
    """Empirical quantile via order statistic x_(ceil(q N)), no interpolation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    xs = np.sort(x)
    idx = int(np.ceil(q * xs.size)) - 1
    idx = min(max(idx, 0), xs.size - 1)
    return float(xs[idx])
