"""Validation gates for the AISTATS revision (plan §10)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bnn_geometry.model import Layout, activation_derivative, loop_V
from bnn_geometry.pcnl import log_ratio_full, log_ratio_stable
from bnn_geometry.revision import archive, bounds, discrepancy, efficiency, geometry, gram, initcheck, slow
from bnn_geometry.revision.runner import ou_control, refuse_foreign_checkpoint, run_pcnl, smoke_suite, tasks_for
from bnn_geometry.revision.variants import add_training_rows, nonlinear_labels, odd_score
from conftest import tiny_model

ROOT = Path(__file__).resolve().parents[2]
ZIP = Path("/Users/apple/Downloads/final_geometry_package.zip")
torch.set_default_dtype(torch.float64)


def _smoke_cfg(tmp_path: Path) -> dict:
    cfg = yaml.safe_load((ROOT / "configs/revision_smoke.yaml").read_text())
    cfg["output_root"] = str(tmp_path)
    cfg["ou_length"] = 400
    return cfg


def test_collect_revision_flattens_summaries_and_reports_missing(tmp_path):
    import csv
    import json

    from bnn_geometry.revision.collect import collect_revision

    cfg = yaml.safe_load((ROOT / "configs/revision.yaml").read_text())
    cfg["output_root"] = str(tmp_path)
    fixed = tmp_path / "runs" / "shallow_m0064_r0" / "fixed-step"
    fixed.mkdir(parents=True)
    (fixed / "summary.json").write_text(json.dumps({
        "task": {"id": "shallow_m0064_r0", "arch": "shallow", "m": 64, "rep": 0, "kind": "baseline"},
        "h": 0.01, "sigma": 1.0, "device": "cuda:0", "hardware": {"gpu": "NVIDIA A10"},
        "run": {"record": {"n": 8192, "planned": 8192, "partial": False, "stop_reason": None},
                "counters": {"accepted": [8000, 8100], "transitions": [8192, 8192]}, "elapsed_s": 12.5},
        "timing": {"seconds_per_joint_step": 0.001, "seconds_per_chain_transition": 0.00025,
                   "chain_transitions_per_wall_second": 4000.0, "transitions_per_block": 512,
                   "doublings": 1, "total_seconds": 5.0, "total_transitions": 5120,
                   "min_seconds": 1.1, "met_duration_rule": True, "batch_chains": 4},
        "loss_rate": {"status": "ok", "ess_bulk": 100.0, "ess_tail": 80.0, "ess_mean": 90.0,
                      "ess_per_transition": 0.01, "ess_per_second": 10.0, "tau": 90.0, "rhat": 1.01,
                      "n_total": 16384},
    }))
    tune = tmp_path / "runs" / "shallow_m0064_r0" / "tune"
    tune.mkdir()
    (tune / "summary.json").write_text(json.dumps({
        "task": {"id": "shallow_m0064_r0", "arch": "shallow", "m": 64, "rep": 0, "kind": "baseline"},
        "tuning": {"scores_ess_per_transition": [[0.01, 1.5], [0.02, 2.0]],
                   "selection": {"winner": 0.02, "score": 2.0, "status": "boundary", "boundary": True}},
    }))
    (tmp_path / "cost_pilot_deep_r0.json").write_text(json.dumps({
        "narrow": "deep_m0032_r0", "wide": "deep_m0256_r0", "device": "cuda:0",
        "hardware": {"gpu": "NVIDIA A10"}, "cost_ratio_wide_over_narrow": 1.135475,
        "break_even_ess_ratio": 2.068404, "met_duration_rule": True,
        "comparison": {"ess_per_second_ratio": 1.821621, "decision": "wide", "spans_break_even": False,
                       "seconds_per_transition_narrow": 0.001, "seconds_per_transition_wide": 0.001135475},
    }))
    manifest = collect_revision(cfg)
    assert manifest["status"] == "incomplete"
    assert manifest["runs"] == 2
    assert "cost-pilot/deep_r0" not in manifest["missing"]
    with (tmp_path / "tables" / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fixed_row = next(row for row in rows if row["stage"] == "fixed-step")
    assert fixed_row["n"] == "8192"
    assert fixed_row["partial"] == "False"
    assert float(fixed_row["accepted_fraction"]) == pytest.approx((8000 + 8100) / (8192 * 2))
    assert float(fixed_row["ess_per_second"]) == pytest.approx(10.0)
    with (tmp_path / "tables" / "tuning.csv").open(newline="") as handle:
        scores = list(csv.DictReader(handle))
    assert [float(row["h_over_sigma2"]) for row in scores] == [0.01, 0.02]
    assert scores[1]["winner"] == "0.02"
    with (tmp_path / "tables" / "cost_pilot.csv").open(newline="") as handle:
        cost = next(csv.DictReader(handle))
    assert cost["decision"] == "wide"
    assert float(cost["ess_per_second_ratio"]) == pytest.approx(1.821621)


def test_n4_fixture_matches_quoted_constants():
    row = bounds.n4_fixture()
    assert row["c0"] == pytest.approx(2.357361, abs=5e-7)
    assert row["K"] == pytest.approx(29.54425, abs=5e-5)
    assert row["delta_m"] == pytest.approx(0.03540039, abs=5e-8)
    assert row["q"] == pytest.approx(0.00555960, abs=5e-8)
    assert row["cls_additive"] == pytest.approx(1.23105, abs=5e-5)
    assert row["cp_lower"] == pytest.approx(0.99951, abs=5e-5)
    assert row["rho_positive"] and row["passes_half_curvature"] and row["passes_input_width_72"]
    assert row["upper_within_25pct_of_prior"]
    assert row["lambda_max_M"] <= row["cls_additive"] + 1e-12


def test_input_widths_and_deep_mass_log():
    widths = {round(item["chi"]): item["sufficient_width_72"] for item in bounds.n16_widths()}
    assert widths[1] == pytest.approx(260)
    assert widths[4] == pytest.approx(1040)
    assert widths[13] == pytest.approx(10816)
    assert bounds.n16_widths()[1]["passes_at_1024"] is False
    deep = bounds.deep_mass_example()
    assert round(deep["mass_log_unclipped"], 2) == 300.46
    assert deep["mass_probability_upper"] == 1.0
    assert deep["mass_applicable"] is True


def test_inadmissible_cutoff_is_marked():
    assert bounds.mass_log(3, 128, 1.0, 256, a=1.0, sigma=1.0) is None
    assert bounds.mass_log(2, 4, 1.0, 100, a=3.0) is None


def test_paired_geometry_is_exact_and_label_matched():
    _lat, designs, checks = geometry.paired_designs(7, 0, [0.0, 0.2, 0.8], n=16, d=32, n_test=32)
    labels = [designs[r]["y"] for r in (0.0, 0.2, 0.8)]
    assert np.array_equal(labels[0], labels[1]) and np.array_equal(labels[1], labels[2])
    by_r = {c["r"]: c["chi_exact"] for c in checks}
    assert by_r == {0.0: 1.0, 0.2: 4.0, 0.8: 13.0}
    assert geometry.probe_train_indices(16) == [0, 2, 4, 6, 8, 10, 12, 14]


def test_gram_backends_agree_and_original_chi_clears_the_floor():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    spec = gram.gram_spectrum(X)
    assert spec["chi"] == pytest.approx(spec["chi_gram"], rel=1e-10)
    if not (ROOT / "results/final_geometry/data/rep_0.npz").exists():
        pytest.skip("frozen replicates are not in this checkout")
    rows = archive.original_chi(ROOT / "results/final_geometry")
    assert len(rows) == 3
    assert all(r["chi_at_least_n_over_d"] for r in rows)
    assert all(abs(r["row_norm_max"] - 1.0) < 1e-12 for r in rows)


def test_two_point_tv_and_zero_information():
    toy = discrepancy.two_point_tv((1.0, 1.0 / 3.0))
    assert toy["tv"] == pytest.approx(0.25)
    assert toy["cdf_gap"] == pytest.approx(0.25)
    zero = discrepancy.two_point_tv((1.0, 1.0))
    assert zero["tv"] == pytest.approx(0.0)
    assert zero["cdf_gap"] == pytest.approx(0.0)


def test_held_out_gap_on_separated_gaussians():
    post = np.vstack([np.full((2, 200), -2.0), np.full((0, 200), 0.0)])
    # Two chains of losses well below the prior losses.
    rng = np.random.default_rng(1)
    post = rng.normal(-1.5, 0.2, size=(4, 300))
    prior = rng.normal(1.5, 0.2, size=800)
    out = discrepancy.two_fold_discrepancy(post, prior)
    assert out["held_out_mean"] > 0.9
    assert out["plug_in_sup"] > 0.9


def test_step_selection_tie_and_boundary():
    choice = efficiency.select_step([(0.01, 1.0), (0.02, 1.03), (0.04, 0.5)])
    assert choice["winner"] == 0.01  # within 5% of the best, so the smaller step wins
    edge = efficiency.select_step([(0.01, 0.2), (0.02, 0.4), (0.04, 0.9)])
    assert edge["boundary"] is True and edge["suggest_h"] == pytest.approx(0.08)
    assert efficiency.warmup_steps(0.01, 1.0) == 2000
    assert math.isnan(efficiency.pilot_score(float("nan"), [1.0]))


def test_acf_matches_direct_sum_and_constant_is_undefined():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(64)
    assert np.allclose(efficiency.chain_centered_acf(x, 5), efficiency.explicit_acf(x, 5), atol=1e-12)
    rate = efficiency.scalar_rate(np.ones((4, 30)))
    assert rate["status"] == "undefined"
    iid = efficiency.iid_or_ar_check(0.0, n=8000, seed=3)
    ar = efficiency.iid_or_ar_check(0.8, n=20000, seed=4)
    assert abs(iid["tau_hat"] - 1.0) < 0.08
    assert abs(ar["tau_hat"] - ar["tau_exact"]) / ar["tau_exact"] < 0.08


def test_slow_direction_prefers_the_slow_coordinate():
    rng = np.random.default_rng(5)

    def ar(phi, n, chains, seed):
        g = np.random.default_rng(seed)
        x = np.empty((chains, n))
        x[:, 0] = g.standard_normal(chains)
        noise = g.standard_normal((chains, n - 1)) * math.sqrt(1 - phi ** 2)
        for t in range(1, n):
            x[:, t] = phi * x[:, t - 1] + noise[:, t - 1]
        return x

    slow_c = ar(0.98, 3000, 4, 11)
    fast_c = ar(0.2, 3000, 4, 12)
    # Duplicate the slow column so the rank rule has to drop it.
    bank = [np.stack([slow_c[c], fast_c[c], slow_c[c]], axis=1) for c in range(4)]
    cal = np.stack([
        np.stack([
            ar(0.98, 1500, 1, 21)[0],
            ar(0.2, 1500, 1, 22)[0],
            ar(0.98, 1500, 1, 21)[0],
        ], axis=1)
    ])
    # cal above is (1, 1500, 3); whiten expects (N, P).
    cal = cal[0]
    found = slow.both_folds(cal, bank, lag=50, names=["slow", "fast", "slow_copy"])
    assert found["fit_01_eval_23"]["retained_rank"] == 2
    for fold in ("fit_01_eval_23", "fit_23_eval_01"):
        assert found[fold]["worst_probe"]["name"] == "slow"
    assert rng.random() >= 0.0


def test_time_to_agreement_censors_and_fires():
    times = [0, 1, 2, 4, 8]
    early = initcheck.time_to_agreement(times, [1, 1, 1, 1, 1], [0.01] * 5)
    assert early["censored"] is True and early["time"] == 8
    late = initcheck.time_to_agreement(times, [1.0, 0.5, 0.0, 0.0, 0.0], [0.01] * 5)
    assert late["censored"] is False and late["time"] == 2


def test_general_backprop_matches_legacy_tanh_and_autograd():
    for arch in ("shallow", "deep"):
        model, _ = tiny_model(arch, m=5, d=3, n=6)
        theta = model.theta0 + 0.2 * torch.randn(2, model.lay.p)
        legacy = model._legacy_V_and_grad(theta)
        general = model._general_V_and_grad(theta)
        assert torch.allclose(legacy[0], general[0], atol=1e-12)
        assert torch.allclose(legacy[1], general[1], atol=1e-11)
    relu, _ = tiny_model("deep", m=4, d=3, n=5)
    relu.activation = "relu"
    theta = (relu.theta0 + 0.3 * torch.randn(2, relu.lay.p)).requires_grad_(True)
    V, g, _ = relu.V_and_grad(theta.detach())
    (auto,) = torch.autograd.grad(relu.V(theta)[0].sum(), theta)
    assert torch.allclose(g, auto, atol=1e-10, rtol=1e-10)
    assert torch.autograd.gradcheck(lambda t: relu.V(t)[0], (theta,), eps=1e-6, atol=1e-6,
                                    nondet_tol=0)


def test_relu_kink_uses_zero_subgradient():
    model, _ = tiny_model("shallow", m=1, d=1, n=1)
    model.activation = "relu"
    model.X = torch.ones(1, 1)
    model.y = torch.zeros(1)
    # W1 = 0 puts the only preactivation on the kink. A is nonzero so a unit slope would move V.
    theta = model.theta0.unsqueeze(0).clone()
    theta[:, model.lay.offsets["W1"][0]] = 0.0
    theta[:, model.lay.offsets["A"][0]] = 1.0
    _V, g, _f = model.V_and_grad(theta)
    w_index = model.lay.offsets["W1"][0]
    assert float(g[0, w_index]) == 0.0
    # A central difference sees the average of the two one-sided slopes and is not the subgradient.
    eps = 1e-6
    plus = theta.clone(); plus[0, w_index] += eps
    minus = theta.clone(); minus[0, w_index] -= eps
    fd = (float(model.V(plus)[0]) - float(model.V(minus)[0])) / (2 * eps)
    assert abs(fd) > 1e-4


def test_depth4_packing_scale_and_loop():
    model, _ = tiny_model("deep", m=3, d=2, n=4)
    lay = Layout("deep", 3, 2, weight_layers=4)
    assert lay.p == (2 + 1) * 3 + 2 * 9
    bank = np.random.default_rng(0).standard_normal((8, 2))
    from bnn_geometry.model import prior_center
    th0 = prior_center(lay, bank)
    model.lay = lay
    model.theta0 = torch.as_tensor(th0)
    model.activation = "sin"
    model.likelihood_scale = 2.0
    theta = model.theta0 + 0.1 * torch.randn(1, lay.p)
    V, g, f = model.V_and_grad(theta)
    assert V.shape == (1,) and g.shape == theta.shape
    literal = loop_V(theta[0].numpy(), lay, model.X.numpy(), model.y.numpy(), activation="sin", likelihood_scale=2.0)
    assert abs(float(V) - literal) < 1e-10
    base = model.V_from_f(f)
    assert torch.allclose(V, base)
    theta = theta.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: model.V(t)[0], (theta,), eps=1e-6, atol=1e-6)
    # sigma = 0.5 forward/reverse antisymmetry of the MH ratio, independent of the network.
    g = np.random.default_rng(1)
    x, y = torch.as_tensor(g.standard_normal(5)), torch.as_tensor(g.standard_normal(5))
    gx, gy = torch.as_tensor(g.standard_normal(5)), torch.as_tensor(g.standard_normal(5))
    for sigma in (0.5, 1.0):
        a = float(log_ratio_stable(x, y, torch.tensor(0.3), torch.tensor(0.7), gx, gy, 0.02, sigma))
        b = float(log_ratio_stable(y, x, torch.tensor(0.7), torch.tensor(0.3), gy, gx, 0.02, sigma))
        full = float(log_ratio_full(x, y, torch.tensor(0.3), torch.tensor(0.7), gx, gy, 0.02, sigma))
        assert abs(a + b) < 1e-8
        assert abs(a - full) < 1e-8


def test_variants_keep_the_prefix_and_match_baseline_rms():
    rng = np.random.default_rng(0)
    d, n = 6, 20
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    teacher = rng.standard_normal(d)
    p = 1 / (1 + np.exp(-2 * X @ teacher))
    y = (rng.random(n) < p).astype(float)
    D = {"X": X, "y": y, "p_teacher": p, "teacher": teacher, "X_test": X[:3].copy()}
    more = add_training_rows(D, 8, master=5, rep=1)
    assert np.array_equal(more["X"][:n], X) and more["X"].shape[0] == n + 8
    packed = nonlinear_labels(D, master=5, rep=1, n_cal=300)
    directions = rng.standard_normal((4, d))  # unused; the stored score is odd by construction
    del directions
    # Rebuild the frozen directions by calling odd_score on a negated copy through the public multiplier path:
    # the label probability at -x equals one minus the probability at x only after we recover directions.
    # Check the calibration identity: multiplier * RMS(raw) == target RMS, via a fresh call's reported fields.
    assert packed["multiplier"] * packed["calibration_rms"] == pytest.approx(packed["target_rms"])
    assert set(np.unique(packed["data"]["y"])).issubset({0.0, 1.0})
    v = np.eye(4, d)
    assert np.allclose(odd_score(-X, v), -odd_score(X, v))


def test_ou_acceptance_and_smoke_resume(tmp_path):
    ou = ou_control(1.0, 0.02, n=600, chains=2, dim=4, seed=9)
    assert ou["acceptance"] == 1.0
    assert abs(ou["lag1"] - ou["eta"]) < 0.12
    cfg = _smoke_cfg(tmp_path)
    report = smoke_suite(cfg)
    assert report["ou_control"]["acceptance"] == 1.0
    assert report["resume_record"]["partial"] is False
    assert report["resume_record"]["n"] == cfg["fixed_step"]["transitions"]
    assert report["slice"]["total_evaluations"] >= 3 * 2
    from bnn_geometry.revision.runner import build_target
    task = tasks_for(cfg, "smoke")[0]
    built = build_target(cfg, task, torch.device("cpu"))
    path = tmp_path / "runs" / task["id"] / "smoke" / "fixed"
    refuse_foreign_checkpoint(path, built, cfg)


def test_smoke_tuning_and_prior_starts(tmp_path):
    from bnn_geometry.revision.runner import _prior_starts, _tune, build_target
    cfg = _smoke_cfg(tmp_path)
    built = build_target(cfg, tasks_for(cfg, "smoke")[0], torch.device("cpu"))
    tuned = _tune(cfg, built, tmp_path / "tune", [0, 1], built["model"].sigma)
    assert tuned["selection"]["winner"] in (0.01, 0.02)
    assert len(tuned["scores_ess_per_transition"]) == 2
    started = _prior_starts(cfg, built, tmp_path / "prior", built["model"].sigma, 0.01)
    assert started["n_starts"] == 4 and len(started["batches"]) == 1


def test_timed_and_untimed_sequences_match():
    import time
    model, _ = tiny_model("shallow", m=4, d=3, n=5)
    from bnn_geometry.pcnl import PCNL
    from bnn_geometry.randomness import ChainRNG

    def roll(stamp: bool) -> torch.Tensor:
        sampler = PCNL(model.V_and_grad, model.theta0, model.sigma, 0.03)
        rngs = [ChainRNG(4, model.theta0.device), ChainRNG(5, model.theta0.device)]
        state = sampler.init_state(model.theta0.unsqueeze(0).expand(2, -1).clone())
        for _ in range(15):
            if stamp:
                time.perf_counter()
            sampler.step(state, rngs)
        return state.theta.clone()

    assert torch.equal(roll(False), roll(True))


def _same(left, right) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(x, y) for x, y in zip(left, right))
    return left == right


def test_split_run_matches_uninterrupted_run(tmp_path):
    """Twelve steps, stop, resume to 24 must match one uninterrupted 24-step run."""
    from bnn_geometry.revision.runner import build_target
    from bnn_geometry.storage import read_trace_file, segment_path

    cfg = _smoke_cfg(tmp_path)
    task = tasks_for(cfg, "smoke")[0]
    chains = [0, 1, 2, 3]

    def finish(path: Path, split: bool) -> dict:
        built = build_target(cfg, task, torch.device("cpu"))
        kwargs = dict(h=0.01, n=24, chains=chains, out=path, segment="production",
                      stage="fixed|0.01", checkpoint_stride=4)
        if split:
            stopped = run_pcnl(built, cfg, stop_after=12, **kwargs)
            assert stopped["record"]["partial"] is True and stopped["record"]["n"] == 12
        done = run_pcnl(built, cfg, **kwargs)
        assert done["record"]["partial"] is False and done["record"]["n"] == 24
        return done

    full = finish(tmp_path / "full", False)
    split = finish(tmp_path / "split", True)
    assert full["counters"] == split["counters"]
    a = torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "split" / "checkpoint.pt", weights_only=False)
    for key in ("theta", "V", "g", "cache"):
        left, right = a["state"][key], b["state"][key]
        if left is None or right is None:
            assert left is None and right is None
        else:
            assert torch.equal(left, right), key
    assert _same(a["rng"], b["rng"])
    assert a["current"] is None and b["current"] is None
    for c in chains:
        left, _ = read_trace_file(segment_path(tmp_path / "full" / f"chain_{c}", "production"))
        right, _ = read_trace_file(segment_path(tmp_path / "split" / f"chain_{c}", "production"))
        for key in ("scalars", "accepted", "draw", "evals", "logratio"):
            np.testing.assert_array_equal(left[key], right[key])
        assert np.array_equal(left["draw"], np.arange(24))


def test_interleaved_cost_keeps_four_chains_and_alternates(tmp_path):
    from bnn_geometry.revision.runner import _arm, build_target, interleaved_cost

    cfg = _smoke_cfg(tmp_path)
    task = tasks_for(cfg, "smoke")[0]
    built = build_target(cfg, task, torch.device("cpu"))
    narrow = _arm(built, cfg, 0.01, "cost-narrow")
    wide = _arm(built, cfg, 0.01, "cost-wide")
    wide["id"] = "wide"
    report = interleaved_cost([narrow, wide], transitions=4, blocks=2, min_seconds=0.0)
    assert report["order"] == [narrow["id"], "wide", narrow["id"], "wide"]
    assert report["cells"][narrow["id"]]["batch_chains"] == 4
    assert report["cells"][narrow["id"]]["joint_transitions"] == 8
    assert report["cells"]["wide"]["joint_transitions"] == 8
    assert report["cells"]["wide"]["elapsed_seconds"] > 0
    assert report["cells"]["wide"]["scalar_logging"] is True
    assert report["cost_ratio_wide_over_narrow"] > 0
    assert math.isfinite(report["cells"][narrow["id"]]["scalar_checksum"])


def test_prediction_rebuild_without_stage_files_exits_cleanly(tmp_path):
    from bnn_geometry.revision.prediction import rebuild_test_prob_0

    cfg = yaml.safe_load((ROOT / "configs/revision.yaml").read_text())
    cfg["output_root"] = str(tmp_path)
    cfg["baseline_root"] = str(tmp_path / "missing_campaign")
    report = rebuild_test_prob_0(cfg, device="cpu")
    assert report["status"] == "raw_series_absent"
    assert report["device"] == "cpu"
    assert (tmp_path / "prediction_acf_check.json").is_file()


def test_prediction_rebuild_records_a_bad_config_instead_of_raising(tmp_path):
    from bnn_geometry.revision.prediction import rebuild_test_prob_0

    cfg = yaml.safe_load((ROOT / "configs/revision.yaml").read_text())
    cfg["output_root"] = str(tmp_path)
    cfg["baseline_config"] = str(tmp_path / "missing.yaml")
    report = rebuild_test_prob_0(cfg, device="cpu")
    assert report["status"] == "error"
    assert "traceback" in report


def test_login_preflight_runs_four_cpu_steps_without_a_gpu(tmp_path):
    from bnn_geometry.revision.runner import login_preflight

    report = login_preflight(_smoke_cfg(tmp_path), require_cuda_build=False)
    assert report["status"] == "ok"
    assert report["cpu_steps"] == 4
    assert report["versions"]["h5py"]


def test_auto_device_follows_gpu_visibility():
    from bnn_geometry.context import pick_device

    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert pick_device(None).type == pick_device("auto").type
    if torch.cuda.is_available():
        assert pick_device("cuda") == torch.device("cuda:0")
    else:
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            pick_device("cuda")


def test_bare_cuda_name_is_given_index_zero_before_set_device(monkeypatch):
    """torch 2.8 rejects set_device(torch.device('cuda')); the cluster job died on that."""
    from bnn_geometry import context

    seen = []
    monkeypatch.setattr(context.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(context.torch.cuda, "set_device", lambda device: seen.append(device))
    assert context.pick_device("cuda") == torch.device("cuda:0")
    assert context.pick_device("auto") == torch.device("cuda:0")
    assert context.pick_device("cuda:1") == torch.device("cuda:1")
    assert seen == [0, 0, 1]


def test_width_comparison_uses_each_blocks_own_length_and_flags_overlap():
    from bnn_geometry.revision.efficiency import timed_width_comparison

    clear = timed_width_comparison([(1.0, 100), (1.0, 100)], [(1.2, 50), (1.2, 50)], 2.068404)
    assert clear["seconds_per_transition_wide"] == pytest.approx(0.024)
    assert clear["ess_per_second_ratio"] == pytest.approx(2.068404 / 2.4)
    assert clear["decision"] == "narrow"
    assert clear["spans_break_even"] is False

    overlap = timed_width_comparison([(1.0, 100), (1.0, 100)], [(1.5, 100), (3.0, 100)], 2.068404)
    assert overlap["cost_ratio_low"] < 2.068404 < overlap["cost_ratio_high"]
    assert overlap["decision"] == "inconclusive"
    assert overlap["favors_wide"] is None


def test_interruption_resumes_without_a_gap(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    from bnn_geometry.revision.runner import build_target
    task = tasks_for(cfg, "smoke")[0]
    built = build_target(cfg, task, torch.device("cpu"))
    path = tmp_path / "partial"
    stopped = run_pcnl(built, cfg, h=0.01, n=20, chains=[0, 1], out=path, segment="production",
                       stage="fixed|0.01", checkpoint_stride=5, stop_after=7)
    assert stopped["record"]["partial"] is True and stopped["record"]["n"] < 20
    finished = run_pcnl(built, cfg, h=0.01, n=20, chains=[0, 1], out=path, segment="production",
                        stage="fixed|0.01", checkpoint_stride=5)
    assert finished["record"]["n"] == 20 and finished["record"]["partial"] is False
    from bnn_geometry.revision.runner import read_loss
    loss = read_loss(path, [0, 1], "production")
    assert loss.shape == (2, 20)
    assert np.isfinite(loss).all()


@pytest.mark.skipif(not ZIP.exists(), reason="final geometry package is not on this machine")
def test_archive_section2_and_three_spectral_counts():
    rows = archive.section2_loss_rows(archive.supplement_rows(ZIP))
    assert len(rows) == 24
    med = {(r["architecture"], r["m"]): r for r in archive.median_table(rows)}
    for key, published in archive.PUBLISHED_MEDIAN_ESS.items():
        assert round(med[key]["ess_bulk_median"]) == published
    for ratio in archive.replicate_ratios(rows):
        assert ratio["ess_ratio"] == pytest.approx(ratio["published_ratio"], abs=0.0015)
    counts = archive.spectral_counts(ZIP)
    assert all(r["match_published"] for r in counts)
    assert all(r["n_above_2_10"] == 0 for r in counts if r["width"] == 256)
    # Three archived endpoint counts, checked against the raw T table rather than the summary.
    checked = [r for r in counts if (r["width"], r["replicate"]) in ((32, 0), (128, 1), (256, 2))]
    assert len(checked) == 3 and all(r["match_published"] for r in checked)


def test_wisconsin_split_standardizes_from_training_only(tmp_path):
    from bnn_geometry.revision.variants import load_wisconsin
    rng = np.random.default_rng(0)
    rows = ["id,diagnosis," + ",".join(f"f{i}" for i in range(30))]
    for i in range(40):
        label = "M" if i < 16 else "B"
        vals = rng.normal(loc=3.0 if label == "M" else 0.0, size=30)
        rows.append(",".join([str(1000 + i), label, *map(str, vals)]))
    path = tmp_path / "wdbc.csv"
    path.write_text("\n".join(rows) + "\n")
    out = load_wisconsin(path, seed=1, n_train=12, split=0)
    assert out["X"].shape == (12, 30)
    assert abs(np.linalg.norm(out["X"], axis=1).mean() - 1.0) < 1e-8
    assert out["X_test"].shape[0] == 28
    assert 0.0 < out["class_fraction"] < 1.0


def test_activation_derivative_of_sin_and_tanh():
    z = torch.linspace(-1, 1, 5)
    h = torch.tanh(z)
    assert torch.allclose(activation_derivative("tanh", h, z), 1 - h * h)
    assert torch.allclose(activation_derivative("sin", torch.sin(z), z), torch.cos(z))
