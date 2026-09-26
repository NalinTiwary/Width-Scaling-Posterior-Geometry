"""Theorem cutoffs B_m and curvature envelope D_th (design §2.2–2.3, §3.5)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Analytic preflight table from the design document (§3.5). These are theorem
# calculations, not measurements. Unit tests must match these rows.
PREFLIGHT = {
    256: {
        "p": 8448,
        "B_m": 8.7397,
        "coverage_floor": 0.996094,
        "D_th": 0.96597,
        "curvature_floor": 0.03403,
    },
    1024: {
        "p": 33792,
        "B_m": 8.8348,
        "coverage_floor": 0.999023,
        "D_th": 0.48756,
        "curvature_floor": 0.51244,
    },
    4096: {
        "p": 135168,
        "B_m": 8.9287,
        "coverage_floor": 0.999756,
        "D_th": 0.24604,
        "curvature_floor": 0.75396,
    },
}

# Design reports A_n = 93.7348851 for the frozen configuration.
PREFLIGHT_A_N = 93.7348851


def c2_constant() -> float:
    return 4.0 / (3.0 * math.sqrt(3.0))


def evidence_bound_A_n(
    n: int,
    C: int,
    b0: float,
    sigma: float,
) -> float:
    """
    A_n = n ( log C + 2 sqrt(b0^2 + (C-1) σ^2) )
    under the paired-center assumptions.
    """
    return float(n * (math.log(C) + 2.0 * math.sqrt(b0**2 + (C - 1) * sigma**2)))


def cylinder_cutoff_B_m(
    m: int,
    b0: float,
    sigma: float,
    A_n: float,
    C: int = 2,
    s: int = 1,
) -> float:
    """
    With s=1 and δ = m^{-1}:

      B_m = b0 + σ [ sqrt(C-1) + sqrt(2 {A_n + 2 log m}) ]
    """
    if s != 1:
        raise NotImplementedError("Only s=1 is implemented in this protocol")
    return float(
        b0
        + sigma
        * (
            math.sqrt(C - 1)
            + math.sqrt(2.0 * (A_n + 2.0 * math.log(m)))
        )
    )


def theoretical_deficit_D_th(
    m: int,
    n: int,
    sigma: float,
    B_m: float,
    M2: float = 1.0,
    M4: float = 1.0,
    c2: float | None = None,
) -> float:
    """
    D_th(m) = σ^2 * sqrt(2n/m) * (M2 + c2 B_m M4^2)
    """
    if c2 is None:
        c2 = c2_constant()
    return float(
        (sigma**2)
        * math.sqrt(2.0 * n / m)
        * (M2 + c2 * B_m * (M4**2))
    )


def coverage_floor(m: int) -> float:
    return 1.0 - 1.0 / m


def theorem_bundle(
    m: int,
    n: int = 32,
    d: int = 32,
    C: int = 2,
    b0: float = 1.0,
    sigma: float = 0.5,
    M2: float = 1.0,
    M4: float = 1.0,
    c2: float | None = None,
) -> dict[str, Any]:
    if c2 is None:
        c2 = c2_constant()
    A_n = evidence_bound_A_n(n=n, C=C, b0=b0, sigma=sigma)
    B_m = cylinder_cutoff_B_m(m=m, b0=b0, sigma=sigma, A_n=A_n, C=C, s=1)
    D_th = theoretical_deficit_D_th(
        m=m, n=n, sigma=sigma, B_m=B_m, M2=M2, M4=M4, c2=c2
    )
    p = m * (1 + d)
    return {
        "m": m,
        "p": p,
        "A_n": A_n,
        "B_m": B_m,
        "D_th": D_th,
        "coverage_floor": coverage_floor(m),
        "curvature_floor": 1.0 - D_th,
        "c2": c2,
        "M2": M2,
        "M4": M4,
        "sigma": sigma,
        "b0": b0,
        "n": n,
        "C": C,
    }


def assert_preflight(
    rtol_B: float = 5e-5,
    rtol_D: float = 5e-5,
    rtol_A: float = 1e-8,
) -> dict[str, Any]:
    """Recompute analytic table and compare to design-doc preflight values."""
    A_n = evidence_bound_A_n(n=32, C=2, b0=1.0, sigma=0.5)
    if not math.isclose(A_n, PREFLIGHT_A_N, rel_tol=rtol_A, abs_tol=1e-7):
        raise AssertionError(f"A_n mismatch: got {A_n}, expected {PREFLIGHT_A_N}")

    report = {"A_n": A_n, "widths": {}}
    for m, row in PREFLIGHT.items():
        bun = theorem_bundle(m=m)
        if bun["p"] != row["p"]:
            raise AssertionError(f"p mismatch at m={m}: {bun['p']} vs {row['p']}")
        if not math.isclose(bun["B_m"], row["B_m"], rel_tol=rtol_B, abs_tol=1e-4):
            raise AssertionError(
                f"B_m mismatch at m={m}: {bun['B_m']} vs {row['B_m']}"
            )
        if not math.isclose(bun["D_th"], row["D_th"], rel_tol=rtol_D, abs_tol=1e-4):
            raise AssertionError(
                f"D_th mismatch at m={m}: {bun['D_th']} vs {row['D_th']}"
            )
        if not math.isclose(
            bun["coverage_floor"], row["coverage_floor"], rel_tol=1e-6, abs_tol=1e-6
        ):
            raise AssertionError(
                f"coverage floor mismatch at m={m}: "
                f"{bun['coverage_floor']} vs {row['coverage_floor']}"
            )
        report["widths"][m] = bun
    return report


def prior_cdf_H(
    b: np.ndarray | float,
    m: int,
    b0: float = 1.0,
    sigma: float = 0.5,
) -> np.ndarray:
    """
    Exact prior CDF of H = max_j |a_j| for binary paired centers
    (design §7.2), evaluated in a numerically stable way.
    """
    from scipy.stats import norm

    b = np.asarray(b, dtype=np.float64)
    # P(|a| <= b) for a ~ N(±1, σ^2) — identical for both centers.
    p1 = norm.cdf((b - b0) / sigma) - norm.cdf((-b - b0) / sigma)
    p1 = np.clip(p1, 0.0, 1.0)
    # log-space: m * log(p1)
    with np.errstate(divide="ignore"):
        log_p = m * np.log(p1)
    return np.exp(log_p)


def prior_quantile_H(
    q: float,
    m: int,
    b0: float = 1.0,
    sigma: float = 0.5,
    lo: float = 0.0,
    hi: float = 20.0,
    tol: float = 1e-10,
) -> float:
    """Invert the exact prior CDF of H via bisection."""
    target = q

    def cdf(b: float) -> float:
        return float(prior_cdf_H(b, m=m, b0=b0, sigma=sigma))

    a, c = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + c)
        if cdf(mid) < target:
            a = mid
        else:
            c = mid
        if c - a < tol:
            break
    return 0.5 * (a + c)
