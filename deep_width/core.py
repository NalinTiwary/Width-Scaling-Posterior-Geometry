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


def needs_inline_check(action, kernel, width):
    """The width-4,096 Langevin production job checks itself, then continues."""
    return action == "run" and kernel == "pcnl" and int(width) == 4096


def parse_duration(text):
    """Parse a Slurm remaining-time field. ``None`` means the value does not limit the job."""
    text = str(text).strip()
    if text == "UNLIMITED":
        return None
    if not text or text in {"INVALID", "NOT_SET", "N/A"}:
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    elif len(parts) == 1:
        hours, minutes, seconds = 0, parts[0], 0
    else:
        raise ValueError(f"Unrecognised Slurm duration {text!r}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def slurm_time(seconds):
    seconds = int(math.ceil(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def remaining_joint_steps(warmup, production, ledger):
    """Joint steps still to run. A missing ledger means production has not started."""
    if ledger is None:
        return sum(n for _, n in warmup) + sum(n for _, n in production)
    remaining = 0
    for name, steps in production:
        elapsed = ledger.get(name)
        if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0:
            continue
        remaining += steps
    return remaining


def gate_decision(*, seconds_per_step, acceptance, failures, remaining_steps,
                  remaining_seconds, peak_reserved_bytes, total_gpu_bytes,
                  headroom=1.5, reserve_s=600, memory_fraction=0.80):
    """Decide whether the short check may continue into production in this allocation.

    A low but finite acceptance is a sampling result and does not stop the job.
    The gate stops for a broken run, a GPU allocation that is already near full,
    or a projected wall time that would be killed before the production clock closes.
    """
    reasons = []
    if any(failures):
        reasons.append(f"numerical failures {list(failures)}")
    if acceptance is None or not math.isfinite(acceptance) or not 0.0 <= acceptance <= 1.0:
        reasons.append(f"acceptance unusable ({acceptance})")
    if (seconds_per_step is None or not math.isfinite(seconds_per_step)
            or seconds_per_step <= 0):
        reasons.append("short-run clock is missing")
    projected = None
    clock_ok = not any(reason.startswith("short-run") for reason in reasons)
    if clock_ok and remaining_steps > 0 and remaining_seconds is not None:
        projected = seconds_per_step * remaining_steps * headroom + reserve_s
        if projected > remaining_seconds:
            reasons.append(
                f"projected {projected / 3600:.2f}h exceeds remaining "
                f"{remaining_seconds / 3600:.2f}h; resubmit with --time={slurm_time(projected)}")
    if (peak_reserved_bytes is not None and total_gpu_bytes
            and peak_reserved_bytes > memory_fraction * total_gpu_bytes):
        reasons.append(
            f"reserved GPU memory {peak_reserved_bytes} exceeds "
            f"{memory_fraction:.0%} of {total_gpu_bytes}")
    return dict(ok=not reasons, reasons=reasons, projected_s=projected,
                seconds_per_step=seconds_per_step, acceptance=acceptance)
