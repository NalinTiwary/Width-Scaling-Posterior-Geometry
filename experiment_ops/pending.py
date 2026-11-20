"""Print missing and currently unclaimed array indices for one stage, to stdout."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

N = [8192, 16384, 32768, 65536, 131072]
PROBES = {"V","q_A","q_W",*(f"test_prob_{i}" for i in range(0,1024,128))}

def indices(spec):
    result=set()
    for term in spec.split(","):
        a,*rest=term.strip().split("-")
        b=rest[0] if rest else a
        if len(rest)>1 or int(a)>int(b) or int(a)<0: raise ValueError(spec)
        result.update(range(int(a),int(b)+1))
    return result

def valid_metrics(root,n):
    file=root/f"metrics_{n}.csv"
    if not file.exists(): return False
    with file.open() as f: rows=list(csv.DictReader(f))
    if len(rows)!=11 or {r['name'] for r in rows}!=PROBES or any(int(r['chains'])!=4 or int(r['steps'])!=n for r in rows):
        raise ValueError(f"Unexpected metrics in {file}. Inspect the files before retrying.")
    return True

def complete(root,job,scope,cell,n):
    base=root/job/scope/cell
    if job=="cold-zero":
        path=base/"verified_zero.csv"
        if not path.is_file(): return False
        with path.open() as f: rows=list(csv.DictReader(f))
        return (len(rows)==32 and {int(r["chain"]) for r in rows}==set(range(32))
                and all(r["cell"]==cell and int(r["t"])==0 for r in rows))
    if job=="iid":
        return (base/"prior_loss.npy").is_file() and (base/"identity.json").is_file()
    if job=="finalists": arms=["h_0.08","h_0.16","h_0.32"]; n=8192
    elif job=="confirm":
        arch,wm,_=cell.split("_")
        with (root/"winners.csv").open() as f: rows=list(csv.DictReader(f))
        winner,=[float(r["h_over_sigma2"]) for r in rows if r["arch"]==arch and int(float(r["m"]))==int(wm[1:])]
        arms=["h_0.01",f"h_{winner:g}"]
    elif job=="reference": arms=["slice"]
    else: arms=["h_0.01"]
    if job=="reuse": n=6192
    for arm in arms:
        p=base/arm
        levels=[n] if job in ("reuse","finalists") else [x for x in N if x>=n]
        completed=False
        for level in levels:
            if not valid_metrics(p,level): continue
            if job!="reuse":
                audit=p/f"run_{level}.json"
                if not audit.exists(): continue
                if any(json.loads(audit.read_text())["counters"]["numerical_failures"]):
                    raise ValueError(f"Numerical failure in {audit}")
            completed=True
            break
        if not completed: return False
        if job!="reference":
            timing=p/"kernel_timing.json"
            if not timing.exists(): return False
            rec=json.loads(timing.read_text())
            if rec["batch_chains"]!=4 or not rec["met_duration_rule"]:
                raise ValueError(f"Invalid timing in {timing}")
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("job",choices=["cold-zero","finalists","confirm","compare","reference","reuse","iid"])
    p.add_argument("scope");p.add_argument("ids");p.add_argument("level",type=int,choices=range(1,6),nargs="?",default=1)
    p.add_argument("--out",type=Path,default=Path("results/revision_next"))
    p.add_argument("--ledger",type=Path,default=Path("results/revision_next/active.csv"))
    a=p.parse_args()
    listing=subprocess.check_output([sys.executable,"a10_next/cluster.py","list","--scope",a.scope],text=True)
    tasks={int(parts[0]):parts[1] for line in listing.splitlines() if (parts:=line.split()) and parts[0].isdigit()}
    wanted=indices(a.ids)
    if not wanted<=tasks.keys(): raise ValueError(f"Unknown task indices {wanted-tasks.keys()}")
    ledger=[]
    if a.ledger.is_file():
        with a.ledger.open() as f: ledger=list(csv.DictReader(f))
    active=set()
    for row in ledger:
        if row["job"]==a.job and row["scope"]==a.scope:
            active.update(indices(row["ids"]))
    todo=[i for i in sorted(wanted-active) if not complete(a.out,a.job,a.scope,tasks[i],N[a.level-1])]
    print(",".join(map(str,todo)))
if __name__=="__main__": main()
