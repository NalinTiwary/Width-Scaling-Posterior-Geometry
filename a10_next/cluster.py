"""Follow-up jobs using the project's target, scalar record, kernels, RNG and checkpoint code."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bnn_geometry.chains import Trajectory, prior_draws
from bnn_geometry.context import pick_device
from bnn_geometry.data import load_replicate
from bnn_geometry.diagnostics import scalar_diagnostics
from bnn_geometry.ellipse import EllipticalSlice
from bnn_geometry.pcnl import PCNL
from bnn_geometry.randomness import ChainRNG
from bnn_geometry.revision import config as rc
from bnn_geometry.revision.efficiency import warmup_steps
from bnn_geometry.revision.runner import build_target, tasks_for, kernel_timing, _seeds
from bnn_geometry.revision.timing import synchronize, hardware_record
from bnn_geometry.storage import atomic_write_json, clean_json
from traces import PROBES, four, one

HERE=Path(__file__).resolve().parent
STEPS=(.08,.16,.32)

def tasks(cfg,scope):
    if scope=="endpoint": return tasks_for(cfg,"tune")
    if scope=="geometry": return tasks_for(cfg,"geometry")
    out=[]
    for t in tasks_for(cfg,scope):
        for r in range(3):
            out.append(dict(t,rep=r,id=t["id"][:-1]+str(r)))
    return out

@contextmanager
def locked(path):
    path.mkdir(parents=True,exist_ok=True)
    with (path/".task.lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def dump(path,obj): atomic_write_json(path,clean_json(obj))

def sample(a,cfg,task,built,h_dim,out,dev):
    model=built["model"]; ident=built["identity"]; cell=task["id"]
    ref=a.job=="reference"; h=h_dim*model.sigma**2
    phase=f"next-v1|{a.job}|{cell}"
    family=f"{phase}|{h_dim:g}"
    chains=list(range(4))
    def rng(stream,tag): return [ChainRNG(_seeds(cfg,ident,c,tag,stream),dev) for c in chains]
    if ref:
        sampler=EllipticalSlice(model.V,model.theta0,model.sigma)
    else:
        sampler=PCNL(model.V_and_grad,model.theta0,model.sigma,h)
    path=out/("slice" if ref else f"h_{h_dim:g}")
    with locked(path):
        traj=Trajectory(path,kind="ess" if ref else "pcnl",sampler=sampler,rngs=rng("sampler",family),
            chains=chains,hashes={"target_hash":ident["target_hash"],"segment_family":family,
                                  "master_seed":str(cfg["master_seed"])},
            record=built["record"],names=built["names"],checkpoint_stride=256)
        traj.open(lambda:(model.theta0.unsqueeze(0).expand(4,-1).clone() if a.job=="compare"
                          else prior_draws(model.theta0,model.sigma,rng("init",phase))))
        dump(path/"identity.json",ident)
        if ref: early=[("burnin",4096),("calibration",2048),("separation",1024)]
        else: early=[("burnin",warmup_steps(h,model.sigma,time_over_sigma2=20))]
        for name,n in early:
            rec=traj.run_segment(name,n)
            if rec["partial"] or rec["n"]!=n: raise RuntimeError(f"incomplete {name}")
        # A new segment contributes only the extra draws required to double the pooled window.
        segments=[("production_1",8192),("production_2",8192),("production_3",16384),
                  ("production_4",32768),("production_5",65536)][:a.level]
        if a.job=="finalists": segments=segments[:1]
        ledger=path/"production_clock.json"
        times=json.loads(ledger.read_text()) if ledger.exists() else {}
        for name,n in segments:
            if traj.done(name):
                continue
            uninterrupted=(name not in times and traj.ck["current"] is None)
            times[name]=None
            dump(ledger,times)
            synchronize(dev); started=time.perf_counter()
            rec=traj.run_segment(name,n)
            synchronize(dev); elapsed=time.perf_counter()-started
            if rec["partial"] or rec["n"]!=n: raise RuntimeError(f"incomplete {name}")
            times[name]=elapsed if uninterrupted else None
            dump(ledger,times)
        wall=sum(times[name] for name,n in segments) if all(times.get(name) is not None for name,n in segments) else None
        if any(traj.ck["counters"]["numerical_failures"]): raise RuntimeError("numerical failures: inspect checkpoint")
        diagnostics=[]; blocks=[]; acc_blocks=[]
        for name,n in segments:
            x,acc=four(path,name,PROBES,n,ident["target_hash"])
            blocks.append(x)
            if not ref: acc_blocks.append(acc)
        v=np.concatenate(blocks,axis=1); count=v.shape[1]
        clock=None
        if not ref:
            timing_file=path/"kernel_timing.json"
            if not timing_file.exists():
                clock=kernel_timing(built,cfg,h,transitions=512,blocks=5,min_seconds=1.0)
                dump(timing_file,clock)
            else: clock=json.loads(timing_file.read_text())
            if not clock["met_duration_rule"] or clock["batch_chains"]!=4: raise ValueError("timing failed")
        for j,probe in enumerate(PROBES):
            row=scalar_diagnostics(probe,v[:,:,j]).row()
            row.update(cell=cell,arch=task["arch"],m=task["m"],rep=task["rep"],h_over_sigma2=h_dim if not ref else None,
                       steps=count,chains=4,acceptance=np.concatenate(acc_blocks,axis=1).mean() if not ref else None,
                       joint_step_s=clock["seconds_per_joint_step"] if clock else None)
            row["measured_production_s"]=wall
            row["raw_ess_per_production_s"]=row["ess_mean"]/wall if wall else None
            row["bulk_ess_per_production_s"]=row["ess_bulk"]/wall if wall else None
            if clock:
                row["raw_ess_s"]=row["ess_mean"]/(count*row["joint_step_s"])
                row["bulk_ess_s"]=row["ess_bulk"]/(count*row["joint_step_s"])
            diagnostics.append(row)
        pd.DataFrame(diagnostics).to_csv(path/f"metrics_{count}.csv",index=False)
        dump(path/f"run_{count}.json",dict(hardware=hardware_record(dev),seed=cfg["master_seed"],segments=segments,
             warmup=early,kind=traj.kind,job=a.job,scope=a.scope,prior_initialization=(a.job!="compare"),
             cumulative_active_sampler_seconds=traj.ck["elapsed_s"],counters=traj.ck["counters"]))
        if any(traj.ck["counters"]["numerical_failures"]): raise RuntimeError("numerical failures: inspect counters")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("job",choices=["list","finalists","confirm","reference","compare","iid","cold-zero","reuse"])
    p.add_argument("--scope",default="endpoint",choices=["endpoint","geometry","relu","sin","nonlinear","depth4","concentrated","prior-scale"])
    p.add_argument("--config",default="configs/revision.yaml"); p.add_argument("--task",type=int,default=0)
    p.add_argument("--level",type=int,choices=range(1,6),default=1)
    p.add_argument("--out",type=Path,default=Path("results/revision_next")); p.add_argument("--device",default="cuda")
    a=p.parse_args(); cfg=rc.load(a.config); ts=tasks(cfg,a.scope)
    if a.job=="list":
        for i,t in enumerate(ts): print(i,t["id"])
        return
    if not 0<=a.task<len(ts): p.error("task outside scope")
    if a.job in ("finalists","confirm","cold-zero") and a.scope!="endpoint": p.error("use endpoint scope")
    task=ts[a.task]; dev=pick_device(a.device)
    if dev.type!="cuda" or torch.cuda.get_device_name(dev)!="NVIDIA A10":
        raise RuntimeError("Comparative runs require an NVIDIA A10. Check your allocation.")
    built=build_target(cfg,task,dev); out=a.out/a.job/a.scope/task["id"]; out.mkdir(parents=True,exist_ok=True)
    if a.scope=="endpoint":
        key=pd.read_csv(HERE/"baseline_identity.csv").set_index("target_id").loc[task["id"]]
        _,_,hashes=load_replicate(rc.baseline_root(cfg),task["rep"])
        for name in ("hash_X","hash_y","hash_X_test","hash_center_bank"):
            if hashes[name]!=key[name]: raise ValueError(f"frozen dataset mismatch: {name}")
        for name in ("n","d","L","m","sigma","arch"):
            if built["identity"][name]!=key[name]: raise ValueError(f"frozen model mismatch: {name}")
    if a.job=="cold-zero":
        model=built["model"]; h=float(cfg["fixed_step"]["h_over_sigma2"])*model.sigma**2
        src=rc.output_root(cfg)/"runs"/task["id"]/"prior-starts"
        old_id=json.loads((src/"identity.json").read_text())
        if old_id["target_hash"]!=built["identity"]["target_hash"]: raise ValueError("prior-start target mismatch")
        cols=[built["names"].index(p) for p in PROBES]; rows=[]
        for i in range(0,32,4):
            cs=list(range(i,i+4))
            def make_rng(tag,stream):
                return [ChainRNG(_seeds(cfg,built["identity"],c,tag,stream),dev) for c in cs]
            theta=prior_draws(model.theta0,model.sigma,make_rng("revision_init","prior"))
            k=PCNL(model.V_and_grad,model.theta0,model.sigma,h); st=k.init_state(theta)
            zero=built["record"](st).detach().cpu().numpy()[:,cols]
            flags,_=k.step(st,make_rng(f"prior-start|{h}","sampler"))
            first=built["record"](st).detach().cpu().numpy()[:,cols]
            for j,c in enumerate(cs):
                path=src/f"batch_{i}"/f"chain_{c}"/"from_prior.h5"
                x,acc=one(path,c,PROBES,8192,old_id["target_hash"],slice(0,1))
                np.testing.assert_allclose(x[0],first[j],atol=1e-10,rtol=1e-10)
                if int(flags[j])!=int(acc[0]): raise ValueError("first acceptance mismatch")
                rows.append(dict(cell=task["id"],chain=c,t=0,**dict(zip(PROBES,zero[j]))))
        pd.DataFrame(rows).to_csv(out/"verified_zero.csv",index=False)
        return
    if a.job=="reuse":
        if a.scope=="endpoint": raise ValueError("use only for geometry or robustness")
        cell=task["id"]; src=rc.output_root(cfg)/"runs"/cell
        source=next((src/st for st in (("geometry-pilot","geometry") if a.scope=="geometry" else (a.scope,))
                     if (src/st/"fixed"/"chain_0"/"production.h5").exists()),None)
        if not source: raise FileNotFoundError(f"missing original production: {src}")
        stored=json.loads((source/"identity.json").read_text())
        if stored["target_hash"]!=built["identity"]["target_hash"]: raise ValueError("stored target mismatch")
        cfg_h=float(cfg["fixed_step"]["h_over_sigma2"]); h=cfg_h*built["model"].sigma**2
        skip=warmup_steps(h,built["model"].sigma); n=int(cfg["fixed_step"]["transitions"])
        data,acc=four(source/"fixed","production",PROBES,n,stored["target_hash"],slice(skip,None))
        dest=out/f"h_{cfg_h:g}"
        with locked(dest):
            tm=kernel_timing(built,cfg,h,transitions=512,blocks=5,min_seconds=1.)
            if not tm["met_duration_rule"] or tm["batch_chains"]!=4: raise ValueError("timing failed")
            dump(dest/"kernel_timing.json",tm); dump(dest/"identity.json",stored)
            rows=[]
            for j,pname in enumerate(PROBES):
                d=scalar_diagnostics(pname,data[:,:,j]).row(); clock=tm["seconds_per_joint_step"]
                rows.append(dict(d,cell=cell,arch=task["arch"],m=task["m"],rep=task["rep"],h_over_sigma2=cfg_h,
                       steps=n-skip,chains=4,acceptance=acc.mean(),joint_step_s=clock,
                       raw_ess_s=d["ess_mean"]/((n-skip)*clock),bulk_ess_s=d["ess_bulk"]/((n-skip)*clock)))
            pd.DataFrame(rows).to_csv(dest/f"metrics_{n-skip}.csv",index=False)
        return
    if a.job=="iid":
        # Independent Gaussian draws, generated in small batches. This stream is disjoint from cold starts.
        with locked(out):
            ident=built["identity"]; model=built["model"]
            rng=ChainRNG(_seeds(cfg,ident,0,"next-iid|"+task["id"],"prior"),dev).torch
            parts=[]
            with torch.no_grad():
                for _ in range(512):
                    noise=torch.randn((8,model.theta0.numel()),generator=rng,device=dev,dtype=model.theta0.dtype)
                    values,_=model.V(model.theta0[None,:]+model.sigma*noise)
                    parts.append(values.detach().cpu().numpy())
            v=np.concatenate(parts); assert v.shape==(4096,) and np.isfinite(v).all()
            np.save(out/"prior_loss.npy",v); dump(out/"identity.json",ident)
        return
    hs=STEPS if a.job=="finalists" else (.01,)
    if a.job=="confirm":
        win=pd.read_csv(a.out/"winners.csv")
        sel=win[(win.arch==task["arch"])&(win.m==task["m"])]
        if len(sel)!=1: raise ValueError("Missing or duplicate selection for this architecture/width")
        h=float(sel.iloc[0].h_over_sigma2)
        if h not in STEPS: raise ValueError("Unknown finalist")
        hs=(.01,h)
    for h in hs:
        sample(a,cfg,task,built,h,out,dev)

if __name__=="__main__": main()
