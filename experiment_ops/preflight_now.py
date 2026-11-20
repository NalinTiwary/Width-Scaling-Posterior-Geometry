"""Check the files and imports the next parallel jobs need. No GPU and no submission."""
import importlib
import sys
from pathlib import Path

PILOT = [
    "geometry_r0_m0064_r0",
    "geometry_r0_m4096_r0",
    "geometry_r0.2_m0064_r0",
    "geometry_r0.2_m4096_r0",
    "geometry_r0.8_m0064_r0",
    "geometry_r0.8_m4096_r0",
]
ENDPOINTS = [
    "shallow_m0064_r0", "shallow_m0064_r1", "shallow_m0064_r2",
    "shallow_m4096_r0", "shallow_m4096_r1", "shallow_m4096_r2",
    "deep_m0032_r0", "deep_m0032_r1", "deep_m0032_r2",
    "deep_m0256_r0", "deep_m0256_r1", "deep_m0256_r2",
]


def missing_files(root: Path) -> list[str]:
    needed = []
    for cell in PILOT:
        for chain in range(4):
            needed.append(root / "results/revision/runs" / cell / "geometry-pilot/fixed" / f"chain_{chain}" / "production.h5")
    for cell in ENDPOINTS:
        needed.append(root / "results/revision/runs" / cell / "prior-starts/batch_0/chain_0/from_prior.h5")
    needed.append(root / "a10_next/a10_revision_attachment.zip")
    needed.append(root / "a10_next/Prior_Start_Reference_Values.csv")
    needed.append(root / "results/revision_next/winners.csv")
    return [str(path) for path in needed if not path.is_file()]


def main() -> int:
    root = Path(".").resolve()
    failed = []
    for name in ("h5py", "yaml", "numpy", "pandas", "scipy", "torch", "bnn_geometry"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")
    sys.path.insert(0, str(root / "a10_next"))
    try:
        importlib.import_module("traces")
    except Exception as exc:
        failed.append(f"traces: {exc}")
    if failed:
        print("IMPORT FAILURE")
        print("\n".join(failed))
        return 1
    absent = missing_files(root)
    if absent:
        print("STOP: a file these jobs read is not in this checkout:")
        print(absent[0])
        print(f"{len(absent)} required files are missing. Nothing was submitted.")
        return 1
    print("PREFLIGHT_OK")
    print("imports ok")
    print("geometry traces for tasks 0,9,12,21,24,33 ok")
    print("prior-start traces ok")
    print("winners.csv ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
