#!/usr/bin/env python3
"""
Profile short runs before launching the grid (addendum §5: "Profile one n=128, m=4096
target before launching the grid"). Measures seconds per likelihood evaluation,
evaluations per update, and seconds per curvature state, then extrapolates the cost of
each configured setting at T = retained_initial (extensions can add up to 4.5x more).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cylinder.ext.cli import base_parser, load  # noqa: E402
from cylinder.ext.run import run_ext_target  # noqa: E402
from cylinder.ext.targets import Target, all_targets  # noqa: E402
from cylinder.io import atomic_save_json  # noqa: E402


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--updates", type=int, default=150, help="Updates per profiled setting")
    ap.add_argument("--all-settings", action="store_true", help="Profile every setting, not just n=128,m=4096")
    args = ap.parse_args()
    cfg, _, artifacts, _ = load(args)

    settings = sorted({(t.kind, t.n, t.m) for t in all_targets(cfg)})
    if not args.all_settings:
        first = cfg["controlled"]["settings"][0]
        settings = [("orth", int(first[0]), int(first[1]))]
    burn = args.updates // 3
    T = args.updates - burn
    sc = cfg["sampling"]
    n_states = int(cfg["curvature"]["states_per_chain"]) * int(sc["n_chains"])
    rows = []
    for kind, n, m in settings:
        meta = run_ext_target(cfg=cfg, target=Target(kind, n, m, 0), artifacts=artifacts,
                              device_request=args.device, n_retained=T, burnin=burn, n_chains=1,
                              write=False)
        cs = meta["chain_stats"][0]
        evals = cs["likelihood_evals_retained"]
        s_per_update = cs["wall_time_retained_s"] / max(cs["retained_done"], 1)
        n_curv = sum(1 for _ in meta["curvature_indices"])
        s_per_state = cs["curvature_time_s"] / max(n_curv, 1)
        updates = int(sc["n_chains"]) * (int(sc["burnin"]) + int(sc["retained_initial"]))
        est = updates * s_per_update + n_states * s_per_state
        row = {
            "setting": f"{kind}_n{n}_m{m}", "kind": kind, "n": n, "m": m,
            "p": meta["p"], "device": meta["effective_device"],
            "evals_per_update": evals / max(cs["retained_done"], 1),
            "s_per_eval": cs["wall_time_retained_s"] / max(evals, 1),
            "s_per_update": s_per_update, "s_per_curvature_state": s_per_state,
            "est_target_minutes_T_initial": est / 60,
        }
        rows.append(row)
        print(f"{row['setting']:22s} p={row['p']:>9d}  {row['evals_per_update']:.1f} evals/update  "
              f"{1e3 * row['s_per_eval']:.2f} ms/eval  {row['s_per_curvature_state']:.2f} s/curv-state  "
              f"~{row['est_target_minutes_T_initial']:.1f} min/target (T={sc['retained_initial']})")
    reps: dict[tuple[str, int, int], int] = {}
    for t in all_targets(cfg):
        reps[(t.kind, t.n, t.m)] = reps.get((t.kind, t.n, t.m), 0) + 1
    if args.all_settings:
        total = sum(r["est_target_minutes_T_initial"] * reps[(r["kind"], r["n"], r["m"])] for r in rows)
        print(f"Estimated grid total at T={sc['retained_initial']}: {total / 60:.1f} h "
              "(before extensions; evals/update at burn-in may understate posterior cost)")
    atomic_save_json(artifacts / "profile.json", {"updates_per_setting": args.updates, "settings": rows})


if __name__ == "__main__":
    main()
