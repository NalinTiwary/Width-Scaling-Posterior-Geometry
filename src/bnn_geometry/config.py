"""Strict campaign schema, target enumeration, and immutable identity hashes (runbook §2–4)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from . import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]

# Every key of configs/campaign.yaml; the schema is closed (unknown or missing keys fail).
SCHEMA: dict[str, Any] = {
    "schema_version": int, "campaign_id": str, "master_seed": int, "theory_source": str,
    "output_root": str, "replicates": list, "chain_ids": list,
    "model": {
        "activation": str, "biases": bool, "dtype": str, "amp": bool, "tf32": bool, "sigma": float,
        "loss": str, "centered_logits": list, "class_one_probability": str,
        "first_layer_input_scaling": str, "hidden_scaling": str, "output_scaling": str,
        "sampled_blocks": str,
        "shallow": {"weight_layers": int, "widths": list},
        "deep": {"weight_layers": int, "widths": list},
    },
    "prior_centers": {
        "head": str, "head_operator_norm": float, "first_layer": str, "row_bank_size": int,
        "later_hidden": str, "paired_hidden_rows": bool, "prior_coordinate_variance": float,
    },
    "data": {
        "train_size": int, "test_size": int, "input_dimension": int, "input_distribution": str,
        "teacher": str, "teacher_log_odds_multiplier": float, "labels": str,
        "share_across_width_and_architecture": bool, "whiten": bool,
        "redraw_for_unfavorable_results": bool,
    },
    "domain": {
        "reference_sampling": str, "dynamics_sampling": str, "shallow_entropy": str,
        "deep_entropy": str, "static_pi_both_architectures": str, "spectral_cutoff_a": float,
        "constrain_head": bool, "constrain_first_layer": bool, "sampling_projection_or_clipping": bool,
    },
    "probes": {
        "train_indices": list, "test_indices": list, "entropy_train_indices": list,
        "interaction_neurons": list, "shallow_interaction_input_indices": list,
        "deep_interaction_direction": str, "linear_controls": list, "tilts": list,
        "standardization_source": str, "degenerate_sd_relative_threshold": float,
        "selection_chain_folds": list, "selection_procedure": str,
    },
    "reference": {
        "sampler": str, "initialization": str, "burn_in_transitions": int,
        "calibration_transitions": int, "separation_transitions": int,
        "cumulative_production_stages": list, "max_likelihood_evaluations_per_chain": int,
        "bracket_guard_evaluations": int, "scalar_save_stride": int, "parameter_archive_stride": int,
        "calibration_archive_stride": int, "checkpoint_stride": int, "continuation_gate": str,
    },
    "reference_gates": {
        "rhat_max_exclusive": float, "bulk_ess_min": int, "tail_ess_min": int,
        "split_mean_mcse_multiplier": float, "split_mean_reference_sd_allowance": float,
        "require_spectral_diagnostics_for_deep": bool, "zero_exit_indicator_ess": str,
    },
    "dynamics": {
        "sampler": str, "reverse_proposal_required": bool, "drift_gradient": str,
        "h_over_sigma_squared_candidates": list, "calibration_discard_time_over_sigma_squared": float,
        "calibration_measure_time_over_sigma_squared": float, "calibration_acceptance_min": float,
        "calibration_max_rejection_streak": int, "step_selection": str,
        "production_discard_time_over_sigma_squared": float,
        "cumulative_retained_time_over_sigma_squared": list, "scalar_save_stride": int,
        "parameter_history": str, "checkpoint_stride": int, "endpoint_checks": str,
        "endpoint_equal_physical_time": bool, "step_validation_relative_difference_max": float,
        "architecture_wide_refinements_max": int, "final_endpoint_step": str,
        "max_candidate_grad_evaluations_per_target_all_dynamics": int, "retain_rejected_transitions": bool,
    },
    "dynamics_gates": {
        "require_reference_pass": bool, "production_acceptance_per_chain_min": float,
        "production_max_rejection_streak": int, "rhat_max_exclusive": float, "bulk_ess_min": int,
        "tail_ess_min": int, "family_relative_mcse_max": float,
        "duration_over_max_integrated_time_min": float, "reference_mean_mcse_multiplier": float,
        "reference_mean_sd_allowance": float,
    },
    "static": {
        "selected_states_per_chain_stages": list, "selected_states_per_target_max": int,
        "selection": str, "gradient_coordinates": str, "probes_including_controls": int,
        "cumulative_unique_state_evaluations_per_target_max": int,
        "cumulative_probe_gradient_evaluations_per_target_max": int,
        "cache_by_target_hash_chain_draw_probe": bool, "conditional_estimator": str,
        "weight_clipping": bool, "denominator_floor": bool,
    },
    "static_gates": {
        "inside_states_per_chain_min": int, "max_normalized_tilt_weight": float,
        "iid_weight_concentration_per_fold_min": int, "normalizer_and_fisher_raw_mean_ess_per_fold_min": int,
        "entropy_relative_mcse_max": float, "fisher_relative_mcse_max": float,
        "ratio_relative_mcse_max": float, "selected_integrand_rhat_max_exclusive": float,
        "require_all_nondegenerate_family_candidates_valid": bool,
    },
    "uncertainty": {
        "method": str, "bootstrap_replicates": int, "interval_percentiles": list,
        "block_integrated_time_multiplier": float, "round_block_length_to_next_power_of_two": bool,
        "nonoverlapping_blocks_per_chain_min": int, "doubled_block_min_blocks": int,
        "doubled_block_mcse_relative_difference_max": float, "cross_replicate_summary": str,
        "pool_data_replicates_as_iid_draws": bool,
    },
    "spectral": {
        "method": str, "inspected_states": str, "normalized_boundary_guard": float,
        "independent_backend_check_states_per_target": int, "independent_backend_relative_tolerance": float,
        "posterior_quantiles": list, "prior_iid_matrices_per_width": int,
        "prior_reference_shared_across_replicates": bool, "occupancy_interval_min_exits": int,
        "occupancy_interval_min_chains_with_exits": int, "zero_exit_ci": str,
    },
    "predictive_check": {
        "posterior_states_per_target": int, "prior_iid_states_per_target": int, "test_points": str,
        "score": str, "average_probabilities_before_log": bool,
    },
    "limits": {
        "scientific_settings": int, "posterior_targets": int, "additional_scientific_experiments": int,
        "hardware_restarts_per_stage": int, "recommended_free_disk_gb": int, "extra_real_data_runs": bool,
        "fitted_scaling_exponents_in_paper": bool,
    },
    "outputs": {
        "main_figures": list, "appendix_figures": list, "formats": list, "png_dpi": int,
        "captions": str, "required_summary": str, "required_audit": str,
    },
}

# Optional, fixture-only section (runbook §6.3); never present in the production campaign.
FIXTURE_KEYS = {"force_reference_budget": dict, "label": str}

# Fixed scientific conventions that the implementation hard-codes; the config must agree.
FIXED_VALUES = {
    ("model", "activation"): "tanh", ("model", "biases"): False, ("model", "dtype"): "float64",
    ("model", "amp"): False, ("model", "tf32"): False,
    ("model", "loss"): "summed_binary_centered_cross_entropy",
    ("model", "class_one_probability"): "sigmoid(sqrt(2)*f)",
    ("model", "first_layer_input_scaling"): "none", ("model", "hidden_scaling"): "inverse_sqrt_width",
    ("model", "output_scaling"): "inverse_sqrt_width", ("model", "sampled_blocks"): "all",
    ("prior_centers", "head"): "alternating_sign_inverse_sqrt_width",
    ("prior_centers", "first_layer"): "nested_standard_gaussian_row_bank",
    ("prior_centers", "later_hidden"): "zero", ("prior_centers", "paired_hidden_rows"): False,
    ("data", "whiten"): False, ("data", "redraw_for_unfavorable_results"): False,
    ("domain", "constrain_head"): False, ("domain", "constrain_first_layer"): False,
    ("domain", "sampling_projection_or_clipping"): False,
    ("reference", "sampler"): "centered_full_state_elliptical_slice",
    ("dynamics", "sampler"): "metropolis_adjusted_gaussian_preserving_langevin",
    ("dynamics", "reverse_proposal_required"): True, ("dynamics", "drift_gradient"): "likelihood_only",
    ("static", "weight_clipping"): False, ("static", "denominator_floor"): False,
    ("uncertainty", "pool_data_replicates_as_iid_draws"): False,
}


class ConfigError(ValueError):
    pass


def _check(node: Any, schema: Any, path: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise ConfigError(f"{path or '<root>'}: expected mapping")
        missing = set(schema) - set(node)
        extra = set(node) - set(schema)
        if path == "" and "fixture" in extra:
            extra.discard("fixture")
            _check(node["fixture"], FIXTURE_KEYS, "fixture")
        if missing or extra:
            raise ConfigError(f"{path or '<root>'}: missing {sorted(missing)} unknown {sorted(extra)}")
        for k, sub in schema.items():
            _check(node[k], sub, f"{path}.{k}" if path else k)
        return
    if schema is float:
        ok = isinstance(node, (int, float)) and not isinstance(node, bool)
    elif schema is int:
        ok = isinstance(node, int) and not isinstance(node, bool)
    else:
        ok = isinstance(node, schema)
    if not ok:
        raise ConfigError(f"{path}: expected {schema.__name__}, got {type(node).__name__}")


def validate(cfg: dict[str, Any]) -> None:
    _check(cfg, SCHEMA, "")
    if cfg["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version {cfg['schema_version']} != {SCHEMA_VERSION}")
    for (sec, key), val in FIXED_VALUES.items():
        if cfg[sec][key] != val:
            raise ConfigError(f"{sec}.{key} must be {val!r} (implemented convention), got {cfg[sec][key]!r}")
    m = cfg["model"]
    for arch, L in (("shallow", 2), ("deep", 3)):
        if m[arch]["weight_layers"] != L:
            raise ConfigError(f"model.{arch}.weight_layers must be {L}")
        w = m[arch]["widths"]
        if w != sorted(set(w)) or any(int(x) <= 0 for x in w):
            raise ConfigError(f"model.{arch}.widths must be strictly increasing positive ints")
        if max(w) > cfg["prior_centers"]["row_bank_size"]:
            raise ConfigError("row_bank_size smaller than the widest width")
    sigma, a = float(m["sigma"]), float(cfg["domain"]["spectral_cutoff_a"])
    if not a > 0.0 + 2.0 * sigma:  # r0 = 0 for the zero W2 center
        raise ConfigError("spectral cutoff must satisfy a > r0 + 2 sigma")
    if abs(float(cfg["prior_centers"]["prior_coordinate_variance"]) - sigma**2) > 1e-15:
        raise ConfigError("prior_coordinate_variance must equal sigma^2")
    n, nt = cfg["data"]["train_size"], cfg["data"]["test_size"]
    p = cfg["probes"]
    for key, lim in (("train_indices", n), ("entropy_train_indices", n), ("test_indices", nt),
                     ("shallow_interaction_input_indices", n)):
        if any(not 0 <= int(i) < lim for i in p[key]):
            raise ConfigError(f"probes.{key} out of range [0, {lim})")
    if not set(p["entropy_train_indices"]) <= set(p["train_indices"]):
        raise ConfigError("entropy_train_indices must be recorded train_indices")
    if len(p["interaction_neurons"]) != len(p["shallow_interaction_input_indices"]):
        raise ConfigError("interaction neurons / input indices length mismatch")
    if min(min(m["shallow"]["widths"]), min(m["deep"]["widths"])) <= max(p["interaction_neurons"]):
        raise ConfigError("interaction neuron index exceeds the narrowest width")
    folds = p["selection_chain_folds"]
    if sorted(sum(folds, [])) != sorted(cfg["chain_ids"]) or len(folds) != 2:
        raise ConfigError("selection_chain_folds must partition chain_ids into two folds")
    if 0.0 in [float(t) for t in p["tilts"]]:
        raise ConfigError("tilts must be nonzero")
    r = cfg["reference"]
    st = r["cumulative_production_stages"]
    if st != sorted(st) or any(s % r["parameter_archive_stride"] for s in st):
        raise ConfigError("production stages must increase and be multiples of the archive stride")
    if r["calibration_transitions"] % r["calibration_archive_stride"]:
        raise ConfigError("calibration length must be a multiple of its archive stride")
    for seg in ("burn_in_transitions", "calibration_transitions", "separation_transitions"):
        if r[seg] % r["checkpoint_stride"] and r[seg] > r["checkpoint_stride"]:
            raise ConfigError(f"reference.{seg} must be a multiple of checkpoint_stride")
    if any(s % r["checkpoint_stride"] for s in st if s >= r["checkpoint_stride"]):
        raise ConfigError("production stages must be multiples of checkpoint_stride")
    d = cfg["dynamics"]
    hc = d["h_over_sigma_squared_candidates"]
    if hc != sorted(hc, reverse=True):
        raise ConfigError("step candidates must be in descending order")
    tt = d["cumulative_retained_time_over_sigma_squared"]
    if tt != sorted(tt):
        raise ConfigError("dynamics durations must increase")
    s = cfg["static"]
    if s["selected_states_per_chain_stages"] != sorted(s["selected_states_per_chain_stages"]):
        raise ConfigError("static selection stages must increase")
    if max(s["selected_states_per_chain_stages"]) * len(cfg["chain_ids"]) > s["selected_states_per_target_max"]:
        raise ConfigError("static selection exceeds selected_states_per_target_max")
    n_targets = len(targets(cfg))
    fixture = "fixture" in cfg
    if not fixture:
        if cfg["limits"]["posterior_targets"] != n_targets:
            raise ConfigError(f"limits.posterior_targets={cfg['limits']['posterior_targets']} but {n_targets} targets")
        if cfg["limits"]["scientific_settings"] != len(m["shallow"]["widths"]) + len(m["deep"]["widths"]):
            raise ConfigError("limits.scientific_settings does not match the width lists")


def load(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text())
    validate(cfg)
    return cfg


def config_sha256(path: Path | str) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_root(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["output_root"])
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Target:
    arch: str
    m: int
    rep: int

    @property
    def L(self) -> int:
        return 2 if self.arch == "shallow" else 3

    @property
    def target_id(self) -> str:
        return f"{self.arch}_m{self.m:04d}_r{self.rep}"

    def p(self, d: int) -> int:
        return self.m + self.m * d + (self.m * self.m if self.arch == "deep" else 0)


def targets(cfg: dict[str, Any]) -> list[Target]:
    out = []
    for arch in ("shallow", "deep"):
        for m in cfg["model"][arch]["widths"]:
            for r in cfg["replicates"]:
                out.append(Target(arch, int(m), int(r)))
    return out


def parse_target(tid: str) -> Target:
    arch, ms, rs = tid.split("_")
    return Target(arch, int(ms[1:]), int(rs[1:]))


def endpoint_widths(cfg: dict[str, Any], arch: str) -> tuple[int, int]:
    w = cfg["model"][arch]["widths"]
    return int(w[0]), int(w[-1])


def is_endpoint(cfg: dict[str, Any], t: Target) -> bool:
    return t.m in endpoint_widths(cfg, t.arch)


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def code_revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "src/bnn_geometry"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


SAMPLING_MODULES = ("randomness.py", "data.py", "model.py", "probes.py", "ellipse.py", "pcnl.py",
                    "storage.py", "spectral.py", "chains.py")


def sampler_code_hash() -> str:
    """Content hash of the modules that determine sampled trajectories (not analysis/plotting code)."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in SAMPLING_MODULES:
        f = here / name
        if f.exists():
            h.update(name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def target_hash(cfg: dict[str, Any], t: Target, data_hashes: dict[str, str]) -> str:
    """Scientific identity: excludes seeds of chains, paths and transition counts (runbook §3.2)."""
    m = cfg["model"]
    return _h({
        "schema": SCHEMA_VERSION, "arch": t.arch, "L": t.L, "m": t.m, "rep": t.rep,
        "n": cfg["data"]["train_size"], "d": cfg["data"]["input_dimension"],
        "data": {k: data_hashes[k] for k in sorted(data_hashes)},
        "sigma": m["sigma"], "activation": m["activation"], "logits": m["centered_logits"],
        "class_one_probability": m["class_one_probability"], "loss": m["loss"],
        "prior_centers": cfg["prior_centers"],
        "event": ({"a": cfg["domain"]["spectral_cutoff_a"], "norm": "op(W2)/sqrt(m)"} if t.arch == "deep" else None),
    })


def execution_hash(cfg: dict[str, Any], t_hash: str, sampler: str, step: Optional[float] = None,
                   role: str = "reference") -> str:
    return _h({
        "target_hash": t_hash, "sampler": sampler, "step": None if step is None else float(step).hex(),
        "role": role, "dtype": cfg["model"]["dtype"], "sampler_code": sampler_code_hash(),
        "probes": cfg["probes"],
        "numerics": cfg["reference"] if sampler == "ess" else cfg["dynamics"],
        "master_seed": cfg["master_seed"],
    })


def deep_copy(cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(cfg)


def n_steps(T: float, h: float) -> int:
    return int(math.ceil(T / h - 1e-12))
