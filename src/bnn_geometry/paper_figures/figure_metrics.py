"""Array-only calculations for the two-figure BNN reanalysis.

These functions do not load project files or verify posterior/sampler metadata.
The adapter must check those, including h=0.01, chronology, burn-in, and V identity.
No PI/LSI constant or integrated autocorrelation time is calculated here.
"""
from __future__ import annotations
import numpy as np


def _vector(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains nonfinite values")
    return a


def validate_iterations(indices, expected_count: int) -> None:
    a = np.asarray(indices)
    if a.ndim != 1 or a.size != expected_count:
        raise ValueError("Iteration indices have the wrong shape/count")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Iteration indices must have an integer dtype")
    if a.size > 1 and not np.all(np.diff(a) == 1):
        raise ValueError("Indices must be consecutive: gaps/duplicates/thinning found")


def acf_1d(values, max_lag: int = 400) -> np.ndarray:
    """Unadjusted, mean-centered ACF including lag zero, via zero-padded FFT.

    rho[k] = sum(z[:N-k] * z[k:]) / sum(z*z).
    Rejected transitions must remain in values as repeated post-transition states.
    """
    x = _vector(values, "values")
    if not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag must be a nonnegative integer")
    if x.size <= max_lag:
        raise ValueError("Trace must be longer than the requested lag range")
    z = x - x.mean(dtype=np.float64)
    denom = float(np.dot(z, z))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Trace variance is zero or numerically invalid")
    nfft = 1 << (2 * x.size - 1).bit_length()
    transformed = np.fft.rfft(z, n=nfft)
    raw = np.fft.irfft(transformed * transformed.conjugate(), n=nfft)
    rho = np.asarray(raw[:max_lag + 1] / denom, dtype=np.float64)
    if not np.isfinite(rho).all() or abs(rho[0] - 1) > 1e-10:
        raise ArithmeticError("Invalid FFT autocorrelation result")
    rho[0] = 1.0  # remove only roundoff at the known lag-zero normalization
    return rho


def acf_replicate(chains, max_lag: int = 400,
                  window: int = 102400) -> dict:
    """Last common window of four chains; normalize before averaging chains."""
    x = np.asarray(chains, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 4:
        raise ValueError("Expected shape (4, n_transitions) for one target")
    if not isinstance(window, (int, np.integer)) or window <= max_lag:
        raise ValueError("window must be an integer greater than max_lag")
    if x.shape[1] < window:
        raise ValueError("Insufficient retained transitions; recover missing data")
    selected = x[:, -window:]
    curves = np.stack([acf_1d(row, max_lag) for row in selected])
    return {"lag": np.arange(max_lag + 1, dtype=np.int64),
            "chain_acf": curves, "replicate_acf": curves.mean(axis=0),
            "n_per_chain": int(window)}


def aggregate_replicates(replicate_curves) -> dict:
    """Median and observed range of exactly three replicate-mean ACFs."""
    a = np.asarray(replicate_curves, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != 3 or a.shape[1] == 0:
        raise ValueError("Expected three replicate curves of equal length")
    if not np.isfinite(a).all():
        raise ValueError("Replicate ACFs contain nonfinite values")
    return {"median": np.median(a, axis=0),
            "replicate_min": np.min(a, axis=0),
            "replicate_max": np.max(a, axis=0)}


def spectral_target(normalized_s, guard: float = 1e-10) -> dict:
    """Summarize already-normalized S from one posterior target, all four chains.

    Do not combine different data/center replicates in normalized_s.
    The ambiguous guard represents numerical membership uncertainty.
    """
    s = _vector(normalized_s, "normalized_s")
    if np.any(s < 0):
        raise ValueError("A normalized spectral norm cannot be negative")
    if not np.isfinite(guard) or guard < 0 or guard >= 1:
        raise ValueError("guard must lie in [0,1)")
    inside = int(np.count_nonzero(s < 1 - guard))
    outside = int(np.count_nonzero(s > 1 + guard))
    ambiguous = int(s.size - inside - outside)
    q = np.quantile(s, [0.50, 0.95, 0.99], method="linear")
    return {"n_inspected": int(s.size), "n_inside": inside,
            "n_outside": outside, "n_ambiguous": ambiguous,
            "inside_fraction_lower": inside / s.size,
            "inside_fraction_upper": (inside + ambiguous) / s.size,
            "q50": float(q[0]), "q95": float(q[1]),
            "q99": float(q[2]), "max_S": float(s.max())}


def spectral_replicates(three_target_summaries) -> dict:
    """Aggregate per-target quantiles without mixing posterior distributions."""
    if len(three_target_summaries) != 3:
        raise ValueError("Exactly three target summaries are required")
    out = {}
    for key in ["q50", "q95", "q99"]:
        values = np.array([s[key] for s in three_target_summaries], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Invalid quantile")
        out[key] = {"median": float(np.median(values)),
                    "replicate_min": float(values.min()),
                    "replicate_max": float(values.max())}
    for key in ["n_inspected", "n_inside", "n_outside", "n_ambiguous"]:
        out[key] = sum(int(s[key]) for s in three_target_summaries)
    return out


def reconstruct_rejected_moves(initial_v: float, accepted, accepted_v) -> np.ndarray:
    """Reconstruct post-transition V with known initial V and complete accept flags.

    accepted_v contains the loss immediately after each accepted transition, in
    acceptance order. This convention must be confirmed against the real logger.
    """
    flags = np.asarray(accepted)
    values = np.asarray(accepted_v, dtype=np.float64)
    if flags.ndim != 1 or flags.dtype != np.bool_:
        raise ValueError("accepted must be a one-dimensional boolean array")
    if values.ndim != 1 or values.size != np.count_nonzero(flags):
        raise ValueError("One accepted-state value is required for each true flag")
    if not np.isfinite(initial_v) or not np.isfinite(values).all():
        raise ValueError("Nonfinite loss")
    if values.size == 0:
        return np.full(flags.size, float(initial_v), dtype=np.float64)
    index = np.cumsum(flags, dtype=np.int64) - 1
    out = np.full(flags.size, float(initial_v), dtype=np.float64)
    mask = index >= 0
    out[mask] = values[index[mask]]
    return out
