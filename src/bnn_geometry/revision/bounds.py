"""Finite-width constants from Appendix B of the manuscript.

Equation numbers in comments follow the compiled appendix labels used by the experiment plan:
(35) head constants, (39) rho and q, (41) the matrix upper bound, (42) the comparison,
(44) positivity, (45) the half-curvature width, (46) the additive LSI bound, (72) the
input-geometry width, (74) the posterior-mass log bound.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def head_constants(n: int, *, C: float = 2.0, sigma: float = 1.0, a0: float = 1.0) -> tuple[float, float]:
    """(35): c0 and K."""
    c0 = math.log(C) + math.sqrt(2.0) * a0 + (sigma ** 2) * (C - 1.0) / (2.0 * C)
    K = (sigma ** 2) * (C - 1.0) + (a0 + sigma * math.sqrt(2.0 * n * c0)) ** 2
    return c0, K


def shallow_coefficients(chi: float, n: int, *, sigma: float = 1.0, a0: float = 1.0,
                         c_phi: float = 1.0) -> tuple[float, float, float]:
    G = float(chi)
    kappa1 = 2.0 * c_phi * a0 * G
    kappa2 = (sigma ** 2) * G * (c_phi * (1.0 + 2.0 * n) + 2.0 * n)
    return G, kappa1, kappa2


def _p_vectors(h: int, a: float, xbar: float) -> list[np.ndarray]:
    w = np.ones(h)
    w[0] = xbar
    out = []
    for ell in range(1, h + 1):
        p = np.zeros(h)
        for j in range(1, ell + 1):
            p[j - 1] = w[j - 1] * (a ** (ell - j))
        out.append(p)
    return out


def deep_propagation(L: int, a: float, *, xbar: float = 1.0, c_phi: float = 1.0) -> dict[str, float]:
    """Derivative coefficients S, T, T_mix. Hidden depth h = L - 1."""
    if L <= 2:
        raise ValueError("deep propagation is for L > 2")
    h = L - 1
    ps = _p_vectors(h, a, xbar)
    S = float(np.linalg.norm(ps[-1]))
    acc = np.zeros((h, h))
    for ell, p in enumerate(ps, start=1):
        acc += (a ** (h - ell)) * np.outer(p, p)
    T = float(c_phi * np.linalg.eigvalsh(acc)[-1])
    R = np.zeros((h, h))
    w = np.ones(h)
    w[0] = xbar
    for j in range(h):
        for ell in range(j + 1, h):
            R[j, ell] = R[ell, j] = w[j] * (a ** (h - 1 - j))
    T_mix = float(np.linalg.eigvalsh(R)[-1]) if h > 1 else 0.0
    return {"S": S, "T": T, "T_mix": T_mix, "h": h}


def deep_coefficients(n: int, L: int, a: float, K: float, *, sigma: float = 1.0, xbar: float = 1.0,
                      c_phi: float = 1.0) -> tuple[float, float, float, dict[str, float]]:
    prop = deep_propagation(L, a, xbar=xbar, c_phi=c_phi)
    G = n * prop["S"] ** 2
    root = math.sqrt(2.0 * K)
    kappa1 = n * root * prop["T"]
    kappa2 = n * root * prop["T_mix"] + 2.0 * (sigma ** 2) * n * G
    return G, kappa1, kappa2, prop


def rho_q(m: int, G: float, kappa1: float, kappa2: float, K: float, *, sigma: float = 1.0,
          n: int) -> tuple[float, float, float]:
    """(39)."""
    delta = kappa1 / math.sqrt(m) + kappa2 / m
    rho = sigma ** -2 - delta
    q = (G / m) * (2.0 * (sigma ** 2) * n + 0.5 * K)
    return delta, rho, q


def lower_cp(n: int, m: int, *, sigma: float = 1.0) -> float:
    """(42), left comparison. Defined whether or not rho is positive."""
    return 1.0 / (sigma ** -2 + n / (2.0 * m))


def lambda_max_M(rho: float, q: float, *, sigma: float = 1.0) -> float:
    """Largest eigenvalue of (41). Requires rho > 0."""
    a = (sigma ** 2) * (1.0 + q / rho)
    b = sigma * math.sqrt(q) / rho
    c = 1.0 / rho
    return 0.5 * (a + c + math.sqrt((a - c) ** 2 + 4.0 * b * b))


def additive_cls(delta: float, q: float, *, sigma: float = 1.0) -> float:
    """(46). Valid once the half-curvature width condition holds; reported whenever rho > 0 as a comparison."""
    return (sigma ** 2) + 2.0 * (sigma ** 4) * (delta + q) + 2.0 * (sigma ** 3) * math.sqrt(max(q, 0.0))


def positivity_width(kappa1: float, kappa2: float, *, sigma: float = 1.0) -> float:
    """(44): rho > 0 exactly when m exceeds this value."""
    inner = (sigma ** 2) * kappa1 + math.sqrt((sigma ** 4) * kappa1 ** 2 + 4.0 * (sigma ** 2) * kappa2)
    return 0.25 * inner ** 2


def half_curvature_width(kappa1: float, kappa2: float, *, sigma: float = 1.0) -> float:
    """(45)."""
    return max(1.0, 16.0 * (sigma ** 4) * kappa1 ** 2, 4.0 * (sigma ** 2) * kappa2)


def input_width(chi: float, n: int, *, sigma: float = 1.0, a0: float = 1.0, c_phi: float = 1.0) -> float:
    """(72), the shallow input-geometry sufficient width."""
    return max(1.0,
               64.0 * (sigma ** 4) * (c_phi ** 2) * (a0 ** 2) * (chi ** 2),
               4.0 * (sigma ** 4) * chi * (c_phi * (1.0 + 2.0 * n) + 2.0 * n))


def mass_log(L: int, n: int, c0: float, m: int, a: float, *, r0: float = 0.0, sigma: float = 1.0) -> Optional[float]:
    """Unclipped log of the (74) excluded-mass bound. None when a is inadmissible or L = 2."""
    if L <= 2 or not a > r0 + 2.0 * sigma:
        return None
    return math.log(L - 2) + n * c0 - m * (a - r0 - 2.0 * sigma) ** 2 / (2.0 * sigma ** 2)


def probability_upper(log_bound: Optional[float]) -> Optional[float]:
    """Clip the log at zero before exponentiating, so the reported probability is at most 1."""
    if log_bound is None:
        return None
    return math.exp(min(0.0, log_bound))


def bound_row(*, label: str, L: int, m: int, n: int, chi: Optional[float] = None, xbar: float = 1.0,
              C: float = 2.0, sigma: float = 1.0, a0: float = 1.0, c_phi: float = 1.0,
              a: Optional[float] = None, r0: float = 0.0, law: str = "pi") -> dict[str, Any]:
    """One target row. Deep rows leave the curvature upper bound empty when rho <= 0."""
    c0, K = head_constants(n, C=C, sigma=sigma, a0=a0)
    prop: dict[str, float] = {}
    if L == 2:
        if chi is None:
            raise ValueError("shallow bounds require chi")
        G, kappa1, kappa2 = shallow_coefficients(chi, n, sigma=sigma, a0=a0, c_phi=c_phi)
        chi_n = n * xbar ** 2
        G_n, k1_n, k2_n = shallow_coefficients(chi_n, n, sigma=sigma, a0=a0, c_phi=c_phi)
    else:
        if a is None:
            raise ValueError("deep bounds require a cutoff")
        G, kappa1, kappa2, prop = deep_coefficients(n, L, a, K, sigma=sigma, xbar=xbar, c_phi=c_phi)
        G_n = k1_n = k2_n = float("nan")
        chi_n = None
    delta, rho, q = rho_q(m, G, kappa1, kappa2, K, sigma=sigma, n=n)
    positive = bool(rho > 0.0)
    lam = lambda_max_M(rho, q, sigma=sigma) if positive else float("nan")
    additive = additive_cls(delta, q, sigma=sigma) if positive else float("nan")
    m_pos = positivity_width(kappa1, kappa2, sigma=sigma)
    m_half = half_curvature_width(kappa1, kappa2, sigma=sigma)
    m72 = input_width(chi, n, sigma=sigma, a0=a0, c_phi=c_phi) if L == 2 and chi is not None else float("nan")
    log_b = mass_log(L, n, c0, m, a, r0=r0, sigma=sigma) if a is not None else None
    prior = sigma ** 2
    within = bool(positive and lam <= 1.25 * prior)
    row = {
        "label": label, "law": law if L == 2 else f"pi(.|G_a={a})", "L": L, "m": m, "n": n,
        "C": C, "sigma": sigma, "xbar": xbar, "a0": a0, "c_phi": c_phi, "r0": r0,
        "chi": chi, "chi_nxbar2": chi_n, "a": a, "c0": c0, "K": K,
        "G": G, "kappa1": kappa1, "kappa2": kappa2,
        "kappa1_over_sqrt_m": kappa1 / math.sqrt(m), "kappa2_over_m": kappa2 / m,
        "delta_m": delta, "rho": rho, "q": q, "rho_positive": positive,
        "positivity_width": m_pos, "passes_positivity": bool(m > m_pos),
        "half_curvature_width": m_half, "passes_half_curvature": bool(m + 1e-12 >= m_half),
        "input_width_72": m72, "passes_input_width_72": bool(L == 2 and m + 1e-12 >= m72),
        "cp_lower": lower_cp(n, m, sigma=sigma),
        "lambda_max_M": lam, "lambda_max_M_over_sigma2": lam / prior if positive else float("nan"),
        "cls_additive": additive,
        "upper_within_25pct_of_prior": within,
        "mass_log_unclipped": log_b, "mass_probability_upper": probability_upper(log_b),
        "mass_applicable": log_b is not None,
        "S": prop.get("S"), "T": prop.get("T"), "T_mix": prop.get("T_mix"),
        "upper_bound_defined": positive,
        "missing": "",
    }
    if L > 2 and not positive:
        row["missing"] = "rho<=0; curvature upper bound undefined; mass bound still recorded"
        row["lambda_max_M"] = None
        row["cls_additive"] = None
    if L == 2:
        d2, r2, q2 = rho_q(m, G_n, k1_n, k2_n, K, sigma=sigma, n=n)
        row["delta_m_nxbar2"] = d2
        row["rho_nxbar2"] = r2
        row["q_nxbar2"] = q2
        row["lambda_max_M_nxbar2"] = lambda_max_M(r2, q2, sigma=sigma) if r2 > 0 else None
    return row


def n4_fixture() -> dict[str, Any]:
    """Orthonormal n=4, m=4096 fixture quoted in the plan."""
    return bound_row(label="fixture_n4_m4096_orthonormal", L=2, m=4096, n=4, chi=1.0)


def n16_widths(chi_values: tuple[float, ...] = (1.0, 4.0, 13.0)) -> list[dict[str, Any]]:
    rows = []
    for chi in chi_values:
        width = input_width(chi, 16)
        rows.append({"chi": chi, "n": 16, "sufficient_width_72": width,
                     "passes_at_1024": 1024 + 1e-9 >= width})
    return rows


def deep_mass_example() -> dict[str, Any]:
    """Original deep endpoint m=256, n=128, a=2.10. Curvature may be undefined; the mass log is the point."""
    return bound_row(label="original_deep_m256_a2.10", L=3, m=256, n=128, a=2.10, chi=None, law="pi(.|G_a)")
