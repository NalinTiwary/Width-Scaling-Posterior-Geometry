"""
Extension theorem quantities (addendum §3).

  A_n      = n (log 2 + 2 sqrt(1.25))
  B_{m,n}  = 1 + ½ [1 + sqrt(2 (A_n + 2 log m))]
  D_th     = σ² sqrt(2n/m) (M2 + c2 B_{m,n} M4²),   M2 = M4 = 1 for orthonormal rows

Same formulas as the pilot (`cylinder.theorem`), evaluated at general n.
"""

from __future__ import annotations

import math
from typing import Any

from ..theorem import (
    c2_constant,
    coverage_floor,
    cylinder_cutoff_B_m,
    evidence_bound_A_n,
    theoretical_deficit_D_th,
)

# Analytic table from the addendum (§3). Theorem calculations, not measurements.
EXT_PREFLIGHT: dict[tuple[int, int], dict[str, float]] = {
    (64, 1024): {"B": 11.5333, "D_th": 0.8731},
    (64, 4096): {"B": 11.6021, "D_th": 0.4389},
    (128, 1024): {"B": 15.4428, "D_th": 1.6110},
    (128, 4096): {"B": 15.4924, "D_th": 0.8079},
    (128, 16384): {"B": 15.5419, "D_th": 0.4051},
    (256, 4096): {"B": 21.0770, "D_th": 1.5225},
    (256, 16384): {"B": 21.1123, "D_th": 0.7625},
}


def ext_bundle(
    n: int,
    m: int,
    *,
    sigma: float = 0.5,
    b0: float = 1.0,
    C: int = 2,
    M2: float = 1.0,
    M4: float = 1.0,
) -> dict[str, Any]:
    c2 = c2_constant()
    A_n = evidence_bound_A_n(n=n, C=C, b0=b0, sigma=sigma)
    B = cylinder_cutoff_B_m(m=m, b0=b0, sigma=sigma, A_n=A_n, C=C, s=1)
    D_th = theoretical_deficit_D_th(m=m, n=n, sigma=sigma, B_m=B, M2=M2, M4=M4, c2=c2)
    return {
        "n": n,
        "m": m,
        "A_n": A_n,
        "B": B,
        "D_th": D_th,
        "curvature_margin": 1.0 - D_th,
        "coverage_floor": coverage_floor(m),
        "n_over_sqrt_m": n / math.sqrt(m),
        "M2": M2,
        "M4": M4,
        "c2": c2,
        "sigma": sigma,
        "b0": b0,
        "C": C,
    }


def assert_ext_preflight(abs_tol: float = 1e-4) -> dict[str, Any]:
    """Recompute the addendum's analytic table; raise on any mismatch."""
    rows = {}
    for (n, m), ref in EXT_PREFLIGHT.items():
        bun = ext_bundle(n, m)
        for key in ("B", "D_th"):
            if not math.isclose(bun[key], ref[key], rel_tol=0.0, abs_tol=abs_tol):
                raise AssertionError(
                    f"{key} mismatch at (n={n}, m={m}): computed {bun[key]:.6f}, table {ref[key]}"
                )
        rows[f"n{n}_m{m}"] = bun
    return rows
