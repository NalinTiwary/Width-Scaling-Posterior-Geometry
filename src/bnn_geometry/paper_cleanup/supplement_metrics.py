"""Array-only calculations for the two selective supplements.

No sampler, archive adapter, convergence diagnostics, or campaign data included.
Input windows must already be selected and matched using the protocol.
"""
from __future__ import annotations
import numpy as np


def acf_1d(values, max_lag):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite nonempty one-dimensional series")
    if not isinstance(max_lag, (int, np.integer)) or not 0 <= max_lag < x.size:
        raise ValueError("max_lag must be an integer in [0,N-1]")
    z = x-x.mean()
    denom = np.dot(z, z)
    if denom <= 0 or not np.isfinite(denom):
        raise ValueError("Constant or invalid series")
    nfft = 1 << (2*x.size-1).bit_length()
    f = np.fft.rfft(z, nfft)
    corr = np.fft.irfft(f*np.conjugate(f), nfft)[:max_lag+1]/denom
    corr[0] = 1.0
    return corr


def probability_acfs(probabilities, max_lag=400):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] != 4 or p.shape[2] != 8:
        raise ValueError("Expected probabilities of shape (4,N,8)")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if (p.std(axis=1) < 1e-10).any():
        raise ValueError("Numerically degenerate probability; inspect logits; do not drop the point")
    per_chain = np.stack([np.stack([acf_1d(p[c, :, j], max_lag)
                                  for j in range(8)]) for c in range(4)])
    per_point = per_chain.mean(axis=0)
    return {"per_chain": per_chain, "per_point": per_point,
            "mean8": per_point.mean(axis=0)}


def loss_acfs(loss, max_lag):
    x = np.asarray(loss, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 4:
        raise ValueError("Expected loss of shape (4,N)")
    per_chain = np.stack([acf_1d(row, max_lag) for row in x])
    return {"per_chain": per_chain, "mean4": per_chain.mean(axis=0)}


def aggregate_replicates(curves):
    x = np.asarray(curves, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected three finite replicate curves")
    return {"median": np.median(x, axis=0), "low": x.min(axis=0), "high": x.max(axis=0)}


def exact_lag_indices(times, h, available_max_lag):
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or not np.isfinite(t).all() or (t < 0).any() or not np.isfinite(h) or h <= 0:
        raise ValueError("Invalid time grid or step size")
    ratio = t/h
    indices = np.rint(ratio).astype(np.int64)
    if not np.allclose(indices, ratio, atol=1e-9, rtol=0):
        raise ValueError("Requested algorithmic lag is not an integer sampler lag")
    if (indices > available_max_lag).any():
        raise ValueError("Requested lag not present in computed ACF")
    return indices


def duration_window(values, h, duration=512):
    n = int(exact_lag_indices([duration], h, 10**9)[0])
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 4 or x.shape[1] < n:
        raise ValueError("Insufficient four-chain history for the matched duration")
    if n < 2 or not np.isfinite(x[:, -n:]).all():
        raise ValueError("Invalid duration window")
    return x[:, -n:]
