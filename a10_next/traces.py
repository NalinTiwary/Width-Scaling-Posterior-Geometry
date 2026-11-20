"""Revision HDF5 scalar reads. Never load parameter archives to compute scalar diagnostics."""
from pathlib import Path
import json
import h5py
import numpy as np

PROBES = ["V"] + [f"test_prob_{i}" for i in range(0, 1024, 128)] + ["q_A", "q_W"]

def one(path, chain, columns, n=None, target_hash=None, rows=slice(None)):
    with h5py.File(path, "r") as f:
        names = json.loads(f.attrs["names"])
        assert int(f.attrs["chain"]) == chain and int(f.attrs["saved_stride"]) == 1, path
        assert str(f.attrs["segment"]) == Path(path).stem, path
        if target_hash is not None:
            assert f.attrs["target_hash"] == target_hash, path
        draws = f["draw"][:]
        if n is not None:
            assert len(draws)==n, path
        assert len(draws)>0 and np.array_equal(draws, np.arange(len(draws))), path
        assert f["scalars"].shape == (len(draws), len(names)), path
        data = np.asarray(f["scalars"][rows,:])[:, [names.index(k) for k in columns]]
        assert np.isfinite(data).all(), path
        accepted = f["accepted"][rows] if "accepted" in f else None
        return data, accepted

def four(root, segment, columns, n=None, target_hash=None, rows=slice(None)):
    records = [one(root/f"chain_{c}"/(segment+".h5"),c,columns,n,target_hash,rows) for c in range(4)]
    return np.stack([r[0] for r in records]), np.stack([r[1] for r in records]) if records[0][1] is not None else None
