"""Fixed design and analysis rules for the deep width sweep."""
import math
import numpy as np

WIDTHS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
REPS = (0, 1, 2)
EXPECTED_CHI = (8.53971569975893, 8.365186026535126, 8.271676480000686)
PROBES = ["V", *(f"test_prob_{i}" for i in range(0, 1024, 128)), "q_A", "q_W"]
PRODUCTION = (8192, 8192, 16384, 32768, 65536)
CHAINS = list(range(4))


def task_matrix():
    return [dict(id=f"deep_m{m:04d}_r{rep}", arch="deep", kind="baseline", m=m, rep=rep)
            for m in WIDTHS for rep in REPS]


def spectrum(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not X.size or not np.isfinite(X).all():
        raise ValueError("Expected a nonempty finite matrix with examples as rows")
    norms = np.linalg.norm(X, axis=1)
    eig = np.linalg.eigvalsh(X @ X.T)
    chi = float(eig[-1])
    sigma_squared = float(np.linalg.svd(X, compute_uv=False)[0] ** 2)
    np.testing.assert_allclose(chi, sigma_squared, rtol=1e-12, atol=1e-12)
    if eig[0] < -1e-12 * max(chi, 1.0):
        raise ValueError("Input Gram matrix failed the PSD check")
    return dict(chi=chi, svd_squared=sigma_squared, row_norm_min=float(norms.min()),
                row_norm_max=float(norms.max()), n=X.shape[0], d=X.shape[1])


def segments_for(kernel, level, h_dim, pilot_steps=None):
    if pilot_steps:
        return [("burnin", 4)], [("pilot", pilot_steps)]
    if kernel == "slice":
        warmup = [("burnin", 4096), ("calibration", 2048), ("separation", 1024)]
    else:
        warmup = [("burnin", max(1024, math.ceil(20 / h_dim)))]
    production = [(f"production_{i}", n) for i, n in enumerate(PRODUCTION[:level], 1)]
    return warmup, production


def production_seconds(ledger, segments):
    times = [ledger.get(name) for name, _ in segments]
    if any(t is None or not math.isfinite(t) or t <= 0 for t in times):
        return None
    return sum(times)


def metrics_rates(ess_mean, elapsed, steps):
    if steps <= 0:
        raise ValueError("Expected positive retained chain length")
    return dict(ess_mean_per_1000_transitions=1000 * ess_mean / (len(CHAINS) * steps),
                ess_mean_per_second=ess_mean / elapsed if elapsed else None,
                seconds_per_joint_step=elapsed / steps if elapsed else None)


def diagnostic_pass(row):
    numeric = ("rhat", "ess_bulk", "ess_tail", "mean", "sd", "mcse")
    return (row is not None and row["status"] == "ok"
            and all(math.isfinite(float(row[key])) for key in numeric)
            and float(row["rhat"]) < 1.01 and float(row["ess_bulk"]) >= 1000
            and float(row["ess_tail"]) >= 400 and bool(row["drift_pass"]))


def reference_agreement(row, reference):
    if not diagnostic_pass(reference):
        return False, None, None
    gap = abs(row["mean"] - reference["mean"])
    threshold = 3 * math.hypot(row["mcse"], reference["mcse"]) + 0.05 * reference["sd"]
    return bool(diagnostic_pass(row) and gap <= threshold), gap, threshold
