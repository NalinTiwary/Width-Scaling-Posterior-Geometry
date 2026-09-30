"""Known-distribution calibration tests (runbook §6.2). Used by tests/integration and by Appendix S1(c,d)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.integrate
import scipy.stats
import torch

from .diagnostics import batch_means_mcse, bootstrap_scalar, bootstrap_vector
from .ellipse import EllipticalSlice
from .entropy import tilt_terms
from .pcnl import PCNL
from .randomness import ChainRNG, numpy_rng, stream_seed
from .relaxation import inefficiency, tau_hat, tau_ou

MASTER = 20260930
TILTS = (-1.0, -0.5, 0.5, 1.0)
CPU = torch.device("cpu")


def _seed(stage: str, stream: str, chain: Any = "shared") -> int:
    return stream_seed(MASTER, rep=0, arch="fixture", width="shared", chain=chain, stage=stage, stream=stream)


def _within(err: float, est: float, rel: float, se: float, k: float) -> bool:
    return abs(err) <= max(rel * abs(est), k * se)


# ---- 1. Gaussian OU ---------------------------------------------------------------------------------
def ou_calibration(sigma: float, *, dim: int = 16, chains: int = 4, n: int = 65536, reps: int = 400) -> list[dict]:
    h = 0.02 * sigma**2
    stage = f"ou_calibration|sigma={sigma!r}"
    theta0 = torch.zeros(dim, dtype=torch.float64)

    def pg(th):
        return torch.zeros(th.shape[0], dtype=th.dtype), torch.zeros_like(th), None

    rngs = [ChainRNG(_seed(stage, "sampler", c), CPU) for c in range(chains)]
    init = [ChainRNG(_seed(stage, "init", c), CPU) for c in range(chains)]
    th = torch.stack([sigma * torch.randn(dim, generator=r.torch, dtype=torch.float64) for r in init])
    s = PCNL(pg, theta0, sigma, h)
    st = s.init_state(th)
    X = np.empty((n, chains, dim))
    acc_all, lam_abs = 0, 0.0
    for i in range(n):
        a, lam = s.step(st, rngs)
        acc_all += int(a.sum())
        lam_abs += float(np.abs(lam).sum())
        X[i] = st.theta.numpy()
    X = X.transpose(1, 0, 2)                                    # (C, N, dim)
    rows = []
    mean_log_acc_err = lam_abs / (n * chains)
    exact_tau = tau_ou(h, sigma)
    # Coordinates are independent replicates of the same OU law: their spread is an independent error estimate.
    coord_tau = np.array([tau_hat(X[:, :, j], h) for j in range(dim)])
    indep_se = float(coord_tau.std(ddof=1))
    u = np.ones(dim) / math.sqrt(dim)
    probes = {"coord_0": X[:, :, 0], "unit_sum": X @ u}
    for name, x in probes.items():
        mcse_m = batch_means_mcse(x)
        mcse_v = batch_means_mcse(x * x)
        var_hat = float((x * x).mean())
        chs = [x[c] for c in range(chains)]
        br = bootstrap_scalar(lambda cs: tau_hat(np.stack(cs), h), chs, inefficiency(x), reps=reps,
                              seed=_seed(stage, f"bootstrap_{name}"))
        ok_tau = _within(br.estimate - exact_tau, exact_tau, 0.05, br.mcse, 3.0)
        rows.append({"test": "ou_calibration", "sigma": sigma, "probe": name, "h": h, "n_per_chain": n,
                     "chains": chains, "acceptance": acc_all / (n * chains), "mean_abs_log_ratio": mean_log_acc_err,
                     "mean": float(x.mean()), "mean_mcse": mcse_m, "mean_exact": 0.0,
                     "var": var_hat, "var_mcse": mcse_v, "var_exact": sigma**2,
                     "tau_hat": br.estimate, "tau_mcse": br.mcse, "tau_low": br.low, "tau_high": br.high,
                     "tau_exact_discrete": exact_tau, "tau_continuous": sigma**2,
                     "discrete_offset": exact_tau - sigma**2,
                     "pass_acceptance": mean_log_acc_err <= 1e-10 and acc_all == n * chains,
                     "pass_mean": abs(float(x.mean())) <= max(0.05 * sigma, 3.0 * mcse_m),
                     "pass_var": _within(var_hat - sigma**2, sigma**2, 0.05, mcse_v, 3.0),
                     "pass_tau": ok_tau,
                     "tau_independent_se": indep_se, "tau_coordinate_mean": float(coord_tau.mean()),
                     "tau_err_over_independent_se": (br.estimate - exact_tau) / indep_se,
                     "audit": ("" if ok_tau else
                               ("ordinary MC fluctuation (|err| <= 3 independent SE across coordinates)"
                                if abs(br.estimate - exact_tau) <= 3 * indep_se else "UNEXPLAINED"))})
    return rows


# ---- 2. Gaussian entropy ----------------------------------------------------------------------------
def gaussian_entropy(sigma: float, *, dim: int = 8, n: int = 65536, reps: int = 400) -> list[dict]:
    stage = f"gaussian_entropy|sigma={sigma!r}"
    g = numpy_rng(_seed(stage, "draws"))
    X = sigma * g.standard_normal((n, dim))
    cal = sigma * numpy_rng(_seed(stage, "calibration_draws")).standard_normal((2048, dim))
    u = np.ones(dim) / math.sqrt(dim)
    raw, raw_cal = X @ u, cal @ u
    mu, s = float(raw_cal.mean()), float(raw_cal.std(ddof=1))
    fz = (raw - mu) / s
    gz = np.full(n, 1.0 / s**2)                                  # ||u||^2 / s^2
    J = np.ones(n)
    rows = []
    for t in TILTS:
        st = tilt_terms(fz, gz, J, t)
        chs = [np.stack([fz, gz, J], axis=1)]
        br = bootstrap_scalar(lambda cs: tilt_terms(cs[0][:, 0], cs[0][:, 1], cs[0][:, 2], t)["R"], chs, 1.0,
                              reps=reps, seed=_seed(stage, f"bootstrap_t={t!r}"))
        rows.append({"test": "gaussian_entropy", "sigma": sigma, "t": t, "n": n, "R": st["R"], "R_mcse": br.mcse,
                     "R_low": br.low, "R_high": br.high, "R_exact": sigma**2, "R_over_sigma2": st["R"] / sigma**2,
                     "pass": _within(st["R"] - sigma**2, sigma**2, 0.03, br.mcse, 3.0)})
    return rows


# ---- 3. Quadratic likelihood ------------------------------------------------------------------------
def quadratic_problem(sigma: float = 0.7, dim: int = 8):
    g = numpy_rng(_seed("quadratic_fixture", "quadratic_fixture"))
    G = g.standard_normal((dim, dim))
    D = G / math.sqrt(dim) + np.diag(np.arange(1, dim + 1) / dim)
    b = g.standard_normal(dim)
    mu0 = np.arange(1, dim + 1) / 10.0
    prec = np.eye(dim) / sigma**2 + D.T @ D
    cov = np.linalg.inv(prec)
    mean = cov @ (mu0 / sigma**2 + D.T @ b)
    return D, b, mu0, cov, mean


def quadratic_fixture(*, sigma: float = 0.7, dim: int = 8, chains: int = 4, burn: int = 4096,
                      n: int = 32768) -> list[dict]:
    D, b, mu0, cov, mean = quadratic_problem(sigma, dim)
    Dt, bt = torch.as_tensor(D), torch.as_tensor(b)
    theta0 = torch.as_tensor(mu0)

    def V(th):
        r = th @ Dt.T - bt
        return 0.5 * (r * r).sum(-1), None

    def VG(th):
        r = th @ Dt.T - bt
        return 0.5 * (r * r).sum(-1), r @ Dt, None

    rows = []
    for name in ("ess", "pcnl"):
        stage = f"quadratic_fixture|{name}"
        rngs = [ChainRNG(_seed(stage, "sampler", c), CPU) for c in range(chains)]
        init = [ChainRNG(_seed(stage, "init", c), CPU) for c in range(chains)]
        th = torch.stack([theta0 + sigma * torch.randn(dim, generator=r.torch, dtype=torch.float64) for r in init])
        if name == "ess":
            s = EllipticalSlice(V, theta0, sigma)
            st, _ = s.init_state(th)
            step = lambda: s.step(st, rngs)  # noqa: E731
        else:
            s = PCNL(VG, theta0, sigma, 0.02 * sigma**2)
            st = s.init_state(th)
            step = lambda: s.step(st, rngs)  # noqa: E731
        for _ in range(burn):
            step()
        X = np.empty((n, chains, dim))
        for i in range(n):
            step()
            X[i] = st.theta.numpy()
        X = X.transpose(1, 0, 2)
        for j in range(dim):
            x = X[:, :, j]
            mm = batch_means_mcse(x)
            sq = (x - mean[j]) ** 2
            vm = batch_means_mcse(sq)
            rows.append({"test": "quadratic_fixture", "sampler": name, "coord": j, "mean": float(x.mean()),
                         "mean_exact": float(mean[j]), "mean_mcse": mm, "var": float(sq.mean()),
                         "var_exact": float(cov[j, j]), "var_mcse": vm,
                         "pass_mean": abs(float(x.mean()) - mean[j]) <= 4 * mm,
                         "pass_var": abs(float(sq.mean()) - cov[j, j]) <= 4 * vm})
    return rows


# ---- 4. Conditional entropy -------------------------------------------------------------------------
def conditional_entropy(*, n: int = 65536, cut: float = 1.5, reps: int = 400) -> list[dict]:
    stage = "conditional_entropy"
    x = numpy_rng(_seed(stage, "draws")).standard_normal(n)
    J = (np.abs(x) <= cut).astype(np.float64)
    gz = np.ones(n)
    P_exact = scipy.stats.norm.cdf(cut) - scipy.stats.norm.cdf(-cut)
    phi = scipy.stats.norm.pdf
    rows = []
    for t in TILTS:
        Zq = scipy.integrate.quad(lambda v: math.exp(t * v) * phi(v), -cut, cut, epsabs=1e-13)[0] / P_exact
        Mq = scipy.integrate.quad(lambda v: v * math.exp(t * v) * phi(v), -cut, cut, epsabs=1e-13)[0] / P_exact
        D_exact = t * Mq / Zq - math.log(Zq)
        I_exact = t * t
        st = tilt_terms(x, gz, J, t)

        def stat(cs, t=t):
            s2 = tilt_terms(cs[0][:, 0], cs[0][:, 1], cs[0][:, 2], t)
            return np.array([s2["D"], s2["I"], s2["R"], s2["P"]])

        br = bootstrap_vector(stat, [np.stack([x, gz, J], axis=1)], 1.0, reps=reps, seed=_seed(stage, f"bootstrap_t={t!r}"))
        ex = [D_exact, I_exact, 2 * D_exact / I_exact, P_exact]
        row = {"test": "conditional_entropy", "t": t, "n": n}
        ok = True
        for k, nm in enumerate(("D", "I", "R", "P")):
            est = st[nm]
            row.update({nm: est, f"{nm}_exact": ex[k], f"{nm}_mcse": br["mcse"][k]})
            se = br["mcse"][k] if np.isfinite(br["mcse"][k]) else 0.0
            ok &= abs(est - ex[k]) <= max(0.02 * abs(ex[k]), 4 * se)
        row["pass"] = bool(ok)
        rows.append(row)
    return rows


def run_all(fast: bool = False) -> list[dict]:
    kw = {"n": 8192} if fast else {}
    rows = []
    for s in (0.7, 1.0):
        rows += ou_calibration(s, **kw)
        rows += gaussian_entropy(s, **kw)
    rows += quadratic_fixture(**({"n": 8192, "burn": 1024} if fast else {}))
    rows += conditional_entropy(**kw)
    return rows
