"""Internal 3-by-4 integrity sheets of the selected V windows (guide §9.4): not paper figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config as C  # noqa: E402
from .common import Layout  # noqa: E402


def sheets(cfg: dict[str, Any], L: Layout) -> list[Path]:
    chain = pd.read_csv(L.tables / "acf_chain.csv.gz")
    st = json.loads((L.tables / "acf_settings.json").read_text())
    reps = [int(r) for r in cfg["replicates"]]
    out = []
    L.diagnostics.mkdir(parents=True, exist_ok=True)
    for arch in ("shallow", "deep"):
        ws = [int(m) for m in cfg["model"][arch]["widths"]]
        for m in (min(ws), max(ws)):
            fig, ax = plt.subplots(2 * len(reps), 4, figsize=(13, 2.0 * 2 * len(reps)), squeeze=False)
            for i, r in enumerate(reps):
                t = C.Target(arch, m, r)
                z = np.load(L.traces / f"{t.target_id}.npz", allow_pickle=False)
                V, start = z["V"], z["iteration_start"]
                for c in range(4):
                    a = ax[2 * i, c]
                    x = np.arange(V.shape[1]) + int(start[c])
                    a.plot(x, V[c], lw=0.2, color="#333333", rasterized=True)
                    a.set_title(f"{t.target_id} chain {c}: V trace", fontsize=7)
                    a.tick_params(labelsize=6)
                    g = chain[(chain.architecture == arch) & (chain.m == m) & (chain.replicate == r) & (chain.chain == c)]
                    b = ax[2 * i + 1, c]
                    b.plot(g.lag, g.rho, lw=0.8, color="#0072B2")
                    b.axhline(0, color="#777777", lw=0.6)
                    b.set_xlim(0, st["K"])
                    b.set_title(f"chain {c}: ACF of V (lag in transitions)", fontsize=7)
                    b.tick_params(labelsize=6)
            fig.suptitle(f"Integrity sheet (not a paper figure): {arch} m={m}, h={st['h']}, last {st['window']} "
                         f"retained transitions per chain", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            p = L.diagnostics / f"loss_integrity_{arch}_m{m:04d}.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            out.append(p)
    return out
