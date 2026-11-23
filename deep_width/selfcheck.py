"""Run without a GPU or the private experiment checkout."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .core import PROBES, gate_decision, metrics_rates, needs_inline_check
from .core import parse_duration, production_seconds, reference_agreement
from .core import remaining_joint_steps, segments_for, spectrum, task_matrix
from .report import HERE, collect, figure, pairs


def main():
    tasks = task_matrix()
    assert len({t["id"] for t in tasks}) == 24
    assert [tasks[i]["m"] for i in (9, 15, 21)] == [256, 1024, 4096]
    assert [needs_inline_check("run", "pcnl", tasks[i]["m"]) for i in (9, 21)] == [False, True]
    assert not needs_inline_check("run", "slice", 4096)
    assert not needs_inline_check("pilot", "pcnl", 4096)
    assert parse_duration("01:30:00") == 5400
    assert parse_duration("1-02:00:00") == 93600
    assert parse_duration("UNLIMITED") is None
    warmup, production = segments_for("pcnl", 1, 0.16)
    assert remaining_joint_steps(warmup, production, None) == 1024 + 8192
    assert remaining_joint_steps(warmup, production, {"production_1": 10.0}) == 0
    assert remaining_joint_steps(warmup, production, {"production_1": None}) == 8192
    healthy = dict(seconds_per_step=0.5, acceptance=0.01, failures=[0, 0, 0, 0],
                   remaining_steps=9216, remaining_seconds=8 * 3600,
                   peak_reserved_bytes=2_000_000_000, total_gpu_bytes=24_000_000_000)
    assert gate_decision(**healthy)["ok"]
    too_slow = dict(healthy, seconds_per_step=5.0)
    stopped = gate_decision(**too_slow)
    assert not stopped["ok"] and any("projected" in reason for reason in stopped["reasons"])
    full = dict(healthy, peak_reserved_bytes=22_000_000_000)
    assert not gate_decision(**full)["ok"]
    assert not gate_decision(**dict(healthy, acceptance=float("nan")))["ok"]
    X = np.array([[1., 0.], [1., 0.], [0., 1.]])
    assert math.isclose(spectrum(X)["chi"], 2)
    assert (8192 + 8192 + 16384) == sum(n for _, n in segments_for("pcnl", 3, .16)[1])
    assert production_seconds({"a": 4, "b": None}, [("a", 1), ("b", 2)]) is None
    assert production_seconds({"a": 4, "b": 3}, [("a", 1), ("b", 2)]) == 7
    assert metrics_rates(800, 20, 100)["ess_mean_per_second"] == 40
    row = dict(status="ok", rhat=1.001, ess_bulk=1400, ess_tail=700, mean=1, sd=2,
               mcse=.03, drift_pass=True)
    assert reference_agreement(row, row)[0]
    assert not reference_agreement(row, dict(row, mean=5))[0]
    assert not reference_agreement(dict(row, ess_bulk=999), row)[0]
    old = pd.read_csv(HERE / "archived_reference.csv")
    assert len(old) == 132 and not old.duplicated(["target_id", "name"]).any()

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        import json
        frozen = pd.read_csv(HERE / "baseline_identity.csv").set_index("target_id")
        for m, mean_ess, elapsed in [(256, 800, 20), (4096, 1600, 80)]:
            cell = f"deep_m{m:04d}_r0"
            path = root / "run" / "pcnl_h_0.16" / cell
            path.mkdir(parents=True)
            key = frozen.loc["deep_m0256_r0"]
            geo = {k: key[k] for k in ("hash_X", "hash_y", "hash_X_test", "hash_center_bank")}
            ident = dict(n=128,d=32,L=3,sigma=1.,m=m,arch="deep",target_hash=f"hash{m}")
            info = dict(input_geometry=geo,target=ident)
            (path / "execution.json").write_text(json.dumps(info))
            data = []
            for probe in PROBES:
                # Use archived reference means to exercise the matching/qualification path.
                mean = old.loc[old.target_id.eq("deep_m0256_r0") & old.name.eq(probe), "mean"].iloc[0]
                data.append(dict(row, mean=mean, width=m, rep=0, chi=8.54, steps=8192,
                                 device="NVIDIA A10",chains=4, probe=probe,
                                 elapsed_seconds=elapsed,ess_mean=mean_ess,target_hash=f"hash{m}",
                                 **metrics_rates(mean_ess, elapsed, 8192)))
            pd.DataFrame(data).to_csv(path / "metrics_8192.csv", index=False)
        metrics = collect(root, .16, 8192)
        assert metrics.loc[metrics.width.eq(256), "timed_pass"].all()
        assert not metrics.loc[metrics.width.eq(4096), "timed_pass"].any()
        ratio = pairs(metrics)
        np.testing.assert_allclose(ratio.rate_ratio, .5)
        np.testing.assert_allclose(ratio.ess_gain_per_transition, 2)
        np.testing.assert_allclose(ratio.cost_ratio, 4)
        figure(root, metrics[metrics.probe.eq("V")].assign(plot_pass=lambda x: x.timed_pass),
               [("ess_mean_per_second", "Rate", "plot_pass")], "fixture", "Test caption.")
        assert (root / "fixture.pdf").read_bytes().startswith(b"%PDF")
    print("Design, input eigenvalue, rate and clock checks, reference gates, ratios and figure tests passed.")


if __name__ == "__main__":
    main()
