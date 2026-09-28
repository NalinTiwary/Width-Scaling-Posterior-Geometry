"""
Validation of the extension's new numerics against independent references (addendum §5, §7.1).

Each check returns a JSON-serializable dict with an ``ok`` flag.  `scripts/ext/11_validate.py`
runs them at full strength before the grid; the pytest suite runs lighter versions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..data import _qr_positive_diagonal
from .arrowhead import arrowhead_dense, arrowhead_min_eig
from .curvature import bracket_dense, bracket_orth
from .ess import ESSStats, ess_step
from .problem import Problem, forward_f, paired_theta0, potential
from .theory import ext_bundle


def tiny_pair(
    n: int = 5, d: int = 12, m: int = 6, seed: int = 0, sigma: float = 0.5, b0: float = 1.0,
    device: torch.device | None = None,
) -> tuple[Problem, Problem, np.ndarray]:
    """Matched (full-coordinate, reduced-coordinate) problems on orthonormal X_n (n×d)."""
    device = device or torch.device("cpu")
    rng = np.random.Generator(np.random.PCG64(seed))
    X_n = _qr_positive_diagonal(rng.standard_normal((d, d))).T[:n].copy()
    y = (rng.random(n) < 0.5).astype(np.float64)
    U = rng.standard_normal((m // 2, d))
    th = ext_bundle(n, m, sigma=sigma, b0=b0)
    common = dict(n=n, m=m, rep=0, y=torch.as_tensor(y, device=device), sigma=sigma,
                  theory=th, hashes={}, sub_idx=np.arange(n), p_full=m * (1 + d))
    full = Problem(kind="full", k=1 + d, X=torch.as_tensor(X_n, device=device),
                   theta0=torch.as_tensor(paired_theta0(U, m, b0), device=device), **common)
    red = Problem(kind="orth", k=1 + n, X=None,
                  theta0=torch.as_tensor(paired_theta0(U @ X_n.T, m, b0), device=device), **common)
    return full, red, X_n


def full_to_reduced(theta_full: torch.Tensor, full: Problem, X_n: np.ndarray) -> torch.Tensor:
    blocks = theta_full.view(full.m, full.k)
    Z = blocks[:, 1:] @ torch.as_tensor(X_n.T, dtype=blocks.dtype, device=blocks.device)
    return torch.cat([blocks[:, :1], Z], dim=1).reshape(-1)


def check_reduction_exact(n_draws: int = 50, seed: int = 1) -> dict[str, Any]:
    """Pilot full-network V(a, W; X_n) == reduced V(a, W X_nᵀ); θ0 maps to the projected prior mean.

    The reference is the pilot's independent NumPy implementation (`cylinder.model`).
    """
    from ..model import binary_potential, forward_f as pilot_forward_f

    full, red, X_n = tiny_pair(n=16, d=64, m=10, seed=seed)
    y = full.y.numpy()
    g = torch.Generator().manual_seed(seed)
    errs, ferrs = [], []
    for _ in range(n_draws):
        th = full.theta0 + full.sigma * torch.randn(full.p, generator=g) * 3.0
        tr = full_to_reduced(th, full, X_n)
        th_np = th.numpy()
        errs.append(abs(binary_potential(th_np, X_n, y, full.m) - float(potential(tr, red))))
        ferrs.append(float(np.abs(pilot_forward_f(th_np, X_n, full.m) - forward_f(tr, red).numpy()).max()))
    mean_err = float((full_to_reduced(full.theta0, full, X_n) - red.theta0).abs().max())
    f0 = float(forward_f(red.theta0, red).abs().max())
    ok = max(errs) < 1e-10 and max(ferrs) < 1e-12 and mean_err < 1e-12 and f0 < 1e-12
    return {"ok": ok, "max_V_err": max(errs), "max_f_err": max(ferrs),
            "prior_mean_map_err": mean_err, "paired_center_f_max": f0}


def check_arrowhead(m: int = 4000, n: int = 17, seed: int = 2) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(m, n, generator=g)
    b = 3 * torch.randn(m, n, generator=g)
    cases = {
        "random": (c, b),
        "zero_coupling": (torch.zeros_like(c), b),
        "partial_zero_coupling": (c * (torch.rand(m, n, generator=g) > 0.5), b),
        "all_zero": (torch.zeros_like(c), torch.zeros_like(b)),
        "repeated_poles": (c, torch.round(b)),
        "weak_coupling_1e-9": (1e-9 * c, b),
        "strong_coupling_1e3": (1e3 * c, b),
        "positive_poles": (c, b.abs() + 1),
    }
    out: dict[str, Any] = {}
    ok = True
    for name, (cc, bb) in cases.items():
        lam, vec, res = arrowhead_min_eig(cc, bb)
        A = arrowhead_dense(cc, bb)
        ref = torch.linalg.eigvalsh(A)[:, 0]
        scale = torch.clamp(torch.linalg.matrix_norm(A, ord=2), min=1.0)
        rel_err = float(((lam - ref).abs() / scale).max())
        dres = torch.linalg.vector_norm(torch.bmm(A, vec[:, :, None])[:, :, 0] - lam[:, None] * vec, dim=1)
        dres = float((dres / scale).max())
        case_ok = rel_err < 1e-12 and dres < 1e-12 and float(res.max()) < 1e-10
        ok &= case_ok
        out[name] = {"ok": case_ok, "max_rel_eig_err": rel_err, "max_rel_dense_residual": dres}
    return {"ok": ok, "cases": out}


def _dense_hessian(prob: Problem, theta: torch.Tensor) -> torch.Tensor:
    return torch.autograd.functional.hessian(lambda t: potential(t, prob), theta)


def check_brackets(n_states: int = 40, seed: int = 3) -> dict[str, Any]:
    """d₋ ≤ d_H ≤ d₊ against a dense autograd Hessian, for both bracket routines.

    States are drawn at inflated prior scale so that negative curvature actually occurs.
    Also checks that the reduced d_H equals the full-coordinate d_H.
    """
    rows = []
    g = torch.Generator().manual_seed(seed)
    ok = True
    for i in range(n_states):
        n, d, m = (5, 12, 6) if i % 2 == 0 else (8, 10, 4)
        full, red, X_n = tiny_pair(n=n, d=d, m=m, seed=seed + i)
        scale = [1.0, 3.0, 6.0][i % 3]
        th_full = full.theta0 + full.sigma * scale * torch.randn(full.p, generator=g)
        th_red = full_to_reduced(th_full, full, X_n)
        s2 = full.sigma**2
        lam_red = float(torch.linalg.eigvalsh(_dense_hessian(red, th_red))[0])
        lam_full = float(torch.linalg.eigvalsh(_dense_hessian(full, th_full))[0])
        dH_red = s2 * max(0.0, -lam_red)
        dH_full = s2 * max(0.0, -lam_full)
        bo = bracket_orth(th_red, red, n_blocks=3)
        bd = bracket_dense(th_full, full)
        tol = 1e-9
        row = {
            "n": n, "m": m, "scale": scale, "d_H_reduced": dH_red, "d_H_full": dH_full,
            "orth": [bo["d_minus"], bo["d_plus"]], "dense": [bd["d_minus"], bd["d_plus"]],
            "orth_ok": bo["d_minus"] - tol <= dH_red <= bo["d_plus"] + tol,
            "dense_ok": bd["d_minus"] - tol <= dH_full <= bd["d_plus"] + tol,
            "reduction_ok": abs(dH_red - dH_full) < 1e-9,
            "residual": max(bo["max_eigen_residual"], bd["max_eigen_residual"]),
        }
        ok &= row["orth_ok"] and row["dense_ok"] and row["reduction_ok"] and row["residual"] < 1e-8
        rows.append(row)
    n_neg = sum(r["d_H_reduced"] > 0 for r in rows)
    ok &= n_neg >= max(1, n_states // 4)  # the check must exercise negative curvature
    return {"ok": ok, "n_states": n_states, "n_with_negative_curvature": n_neg, "states": rows}


def check_decomposition(seed: int = 4) -> dict[str, Any]:
    """Dense Hessian == Jᵀ diag(h) J + blockdiag(arrowheads) in reduced coordinates."""
    from .curvature import _likelihood_terms

    _, red, _ = tiny_pair(n=6, d=12, m=4, seed=seed)
    g = torch.Generator().manual_seed(seed)
    th = red.theta0 + red.sigma * 3 * torch.randn(red.p, generator=g)
    a, phi, phi_p, phi_pp, q, h = _likelihood_terms(th, red)
    rs = 1 / math.sqrt(red.m)
    c = q[None] * phi_p * rs
    b = a[:, None] * q[None] * phi_pp * rs
    Rb = arrowhead_dense(c, b)
    R = torch.block_diag(*Rb)
    J = torch.zeros(red.n, red.p)
    for j in range(red.m):
        J[:, j * red.k] = phi[j] * rs
        J[:, j * red.k + 1 : (j + 1) * red.k] = torch.diag(a[j] * phi_p[j] * rs)
    H = J.T @ (h[:, None] * J) + R
    err = float((H - _dense_hessian(red, th)).abs().max())
    return {"ok": err < 1e-10, "max_abs_entry_err": err}


def check_dense_matches_pilot(seed: int = 5) -> dict[str, Any]:
    """Torch dense-block bracket reproduces the pilot's NumPy bracket (d = 32 case)."""
    from ..curvature import bracket_lambda_min

    full, _, X_n = tiny_pair(n=20, d=32, m=64, seed=seed)
    g = torch.Generator().manual_seed(seed)
    th = full.theta0 + full.sigma * 2 * torch.randn(full.p, generator=g)
    ours = bracket_dense(th, full)
    ref = bracket_lambda_min(th.numpy(), X_n, full.y.numpy(), full.m)
    e_ell = abs(ours["ell"] - ref["ell"])
    e_u = abs(ours["u"] - ref["u"])
    return {"ok": e_ell < 1e-10 and e_u < 1e-8, "ell_err": e_ell, "u_err": e_u,
            "ell": ours["ell"], "u": ours["u"]}


def _sample(prob: Problem, n_chains: int, burnin: int, T: int, seed: int) -> dict[str, np.ndarray]:
    from .problem import preacts

    out = {"V": np.zeros((n_chains, T)), "H": np.zeros((n_chains, T)),
           "f": np.zeros((n_chains, T, prob.n)), "pre": np.zeros((n_chains, T, 4)),
           "pre_ms": np.zeros((n_chains, T))}
    for c in range(n_chains):
        rng = np.random.Generator(np.random.PCG64([seed, c]))
        gen = torch.Generator(device=prob.device).manual_seed(seed * 100 + c)
        st = ESSStats()

        def ll(u: torch.Tensor) -> float:
            return -float(potential(prob.theta0 + prob.sigma * u, prob))

        u = torch.randn(prob.p, generator=gen, device=prob.device)
        lu = ll(u)
        for t in range(burnin + T):
            u, lu = ess_step(u, lu, ll, rng, st, gen=gen)
            if t >= burnin:
                th = prob.theta0 + prob.sigma * u
                out["V"][c, t - burnin] = -lu
                out["H"][c, t - burnin] = float(th.view(prob.m, prob.k)[:, 0].abs().max())
                out["f"][c, t - burnin] = forward_f(th, prob).cpu().numpy()
                _, pre = preacts(th, prob)  # hidden pre-activations x_iᵀw_j == z_ij
                out["pre"][c, t - burnin] = pre[:2, :2].reshape(-1).cpu().numpy()
                out["pre_ms"][c, t - burnin] = float((pre * pre).mean())
    return out


def check_matched_sampling(
    n_chains: int = 4, burnin: int = 500, T: int = 4000, seed: int = 6, perturb: float = 1.0
) -> dict[str, Any]:
    """Reduced-coordinate ESS vs full-coordinate ESS on one small matched target.

    Posterior means of V, H, every training prediction and hidden pre-activations must
    agree within 4 combined Monte Carlo standard errors (ArviZ mcse, chain-aware).
    ``perturb`` ≠ 1 scales the reduced prior means (a deliberately wrong reduction) for
    the negative control.
    """
    from ..diagnostics import posterior_idata
    import arviz as az

    full, red, _ = tiny_pair(n=6, d=16, m=8, seed=seed)
    if perturb != 1.0:
        th0 = red.theta0.view(red.m, red.k).clone()
        th0[:, 1:] *= perturb
        red.theta0 = th0.reshape(-1)
    sf = _sample(full, n_chains, burnin, T, seed)
    sr = _sample(red, n_chains, burnin, T, seed + 1)

    def mean_mcse(x: np.ndarray) -> tuple[float, float]:
        mc = az.mcse(posterior_idata({"x": x}), method="mean")
        mc = mc.posterior if hasattr(mc, "posterior") else mc
        return float(x.mean()), float(np.asarray(mc["x"]))

    rows = {}
    pairs = [("V", sf["V"], sr["V"]), ("H", sf["H"], sr["H"]), ("pre_ms", sf["pre_ms"], sr["pre_ms"])]
    pairs += [(f"f_{i}", sf["f"][:, :, i], sr["f"][:, :, i]) for i in range(full.n)]
    pairs += [(f"pre_{i}", sf["pre"][:, :, i], sr["pre"][:, :, i]) for i in range(4)]
    for name, xf, xr in pairs:
        mf, ef = mean_mcse(xf)
        mr, er = mean_mcse(xr)
        z = (mr - mf) / math.hypot(ef, er)
        rows[name] = {"full": mf, "reduced": mr, "z": z}
    max_z = max(abs(r["z"]) for r in rows.values())
    return {"ok": max_z < 4.0, "max_abs_z": max_z, "n_chains": n_chains, "T": T, "observables": rows}


def run_all(light: bool = False) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    kw = dict(n_chains=4, burnin=200, T=1000) if light else {}
    neg = check_matched_sampling(perturb=1.05, **kw)
    res = {
        "reduction_exact": check_reduction_exact(n_draws=10 if light else 50),
        "arrowhead_vs_dense": check_arrowhead(m=500 if light else 4000),
        "hessian_decomposition": check_decomposition(),
        "brackets_vs_dense_hessian": check_brackets(n_states=12 if light else 40),
        "dense_bracket_matches_pilot": check_dense_matches_pilot(),
        "matched_full_vs_reduced_sampling": check_matched_sampling(**kw),
        "matched_sampling_negative_control": {
            "ok": not neg["ok"],
            "description": "5% wrong reduced prior means must be detected",
            "max_abs_z": neg["max_abs_z"],
        },
    }
    res["all_ok"] = all(v["ok"] for v in res.values())
    return res
