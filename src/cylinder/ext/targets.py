"""Enumerate extension targets in the addendum's execution order."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Target:
    kind: str  # "orth" | "fmnist"
    n: int
    m: int
    rep: int

    @property
    def name(self) -> str:
        return f"{self.kind}_n{self.n}_m{self.m}_r{self.rep}"


def all_targets(cfg: dict[str, Any], pattern: Optional[str] = None, include_realdata: bool = True) -> list[Target]:
    out: list[Target] = []
    ctl = cfg["controlled"]
    for n, m in ctl["settings"]:
        for r in ctl["replicates"]:
            out.append(Target("orth", int(n), int(m), int(r)))
    rd = cfg.get("realdata", {})
    if include_realdata and rd.get("enabled", False):
        n = 2 * int(rd["n_per_class"])
        for m in rd["widths"]:
            for r in rd["replicates"]:
                out.append(Target("fmnist", n, int(m), int(r)))
    if pattern:
        rx = re.compile(pattern)
        out = [t for t in out if rx.search(t.name)]
    return out


def parse_name(name: str) -> Target:
    mt = re.fullmatch(r"(orth|fmnist)_n(\d+)_m(\d+)_r(\d+)", name)
    if not mt:
        raise ValueError(f"bad target name {name!r}")
    return Target(mt.group(1), int(mt.group(2)), int(mt.group(3)), int(mt.group(4)))
