#!/usr/bin/env python3
"""Causal wall-history conditioning on a leakage-free periodic-hill time split.

Evidence level 2 only: a frozen familyclass closure maps current/past resolved coarse velocity to
signed wall stress; the stress history conditions a direct field-space posterior.  No flow solver
consumes the stress.  The three public Case3 spanwise slices are split at the same physical times so
that a test frame cannot have a nearly identical same-time slice in training.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="case3_grid_64x192_full.npz")
ap.add_argument("--families", default="diffusion,flow_matching")
ap.add_argument("--steps_train", type=int, default=18000)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--L", type=int, default=8)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--train_end", type=int, default=480)
ap.add_argument("--test_start", type=int, default=590)
ap.add_argument("--seq_len", type=int, default=960)
ap.add_argument("--B", type=int, default=4000)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.steps_train, args.batch, args.L, args.steps, args.B = 30, 4, 2, 4, 100
    args.train_end, args.test_start = 24, 28

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJ = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "closure"))
from wall_closure import predict_eqwm_tau, predict_tau_w_from_history  # noqa: E402

# Reuse the exact direct-field backbone/samplers, changing only the number of conditioning channels.
old_argv = sys.argv
try:
    sys.argv = [os.path.join(HERE, "eval_wallstress_cond.py")]
    spec = importlib.util.spec_from_file_location("wallstress_base", os.path.join(HERE, "eval_wallstress_cond.py"))
    WS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(WS)
finally:
    sys.argv = old_argv
WS.N_COND = 5  # [tau(t), tau(t-1), support, observed_fraction, domain]

dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
def log(*x): print(*x, flush=True)

z = np.load(os.path.join(ROOT, "data", args.data))
fields_np = np.asarray(z["fields"], dtype=np.float32)
mask_np = np.asarray(z["mask"], dtype=bool)
wall_row = np.asarray(z["wall_row"], dtype=int)
mu = np.asarray(z["mu"], dtype=np.float64)
sd = np.asarray(z["sd"], dtype=np.float64)
xg = np.asarray(z["grid_x"], dtype=np.float64)
yg = np.asarray(z["grid_y"], dtype=np.float64)
h, Lx, Ly = float(z["h"]), float(z["Lx"]), float(z["Ly"])
T, C, H, W = fields_np.shape
if T % args.seq_len or T // args.seq_len != 3:
    raise ValueError(f"expected three contiguous Case3 slices of {args.seq_len}, got T={T}")
if not (2 < args.train_end < args.test_start < args.seq_len):
    raise ValueError("invalid shared-time split")
n_slice = T // args.seq_len
train_idx_np = np.concatenate([s*args.seq_len + np.arange(1, args.train_end) for s in range(n_slice)])
test_times = np.arange(args.test_start, args.seq_len)
if args.smoke:
    test_times = test_times[:6]
test_idx_np = np.concatenate([s*args.seq_len + test_times for s in range(n_slice)])
n_time = len(test_times)

fields = torch.tensor(fields_np, device=dev)
DM = torch.tensor(mask_np, device=dev)
train_idx = torch.tensor(train_idx_np, device=dev)
test_idx = torch.tensor(test_idx_np, device=dev)
Xtr, Xte = fields[train_idx], fields[test_idx]
Xmean = Xtr.mean(0, keepdim=True)

# Physical time follows the primary source: 10 flow-through times, Lx=9h, 960 learning frames.
Ub = 0.2
Re_h = 2800.0
nu = Ub*h/Re_h
dt = (10.0*(Lx/Ub))/float(args.seq_len)
dy = Ly/H
Uphys = fields_np[:, 0]*sd[0] + mu[0]
Vphys = fields_np[:, 1]*sd[1] + mu[1]
yb = 0.5*dy
tau_true = nu*Uphys[:, wall_row, np.arange(W)]/yb
tau_mu = float(tau_true[train_idx_np].mean())
tau_sd = float(tau_true[train_idx_np].std() + 1e-12)
tau_true_t = torch.tensor(tau_true, dtype=torch.float32, device=dev)

rows = np.arange(H)[:, None]
support_np = (rows == wall_row[None, :]) & mask_np
support = torch.tensor(support_np, device=dev)
unobs = DM & (~support)
obsfrac = float(support_np.sum()/mask_np.sum())

# Integral time from one slice; inference clusters the three same-time slices in each bootstrap unit.
e0 = ((fields[:args.seq_len]-fields[:args.seq_len].mean(0, keepdim=True))[:, :, DM]**2).mean((1,2)).cpu().numpy()
e0 -= e0.mean()
ac = np.array([1.0]+[float(np.corrcoef(e0[:-k], e0[k:])[0,1]) for k in range(1, min(200,len(e0)//2))])
zc = np.argmax(ac < 0) if np.any(ac < 0) else len(ac)
tau_time = int(round(max(1.0, 1.0+2.0*ac[1:zc].sum())))
block = min(n_time, max(1, 2*tau_time))
n_eff = n_time/float(max(1, 2*tau_time))

log(f"[data] Case3 {T}={n_slice}x{args.seq_len}, shared-time train< {args.train_end}, "
    f"test>={args.test_start}; n_train={len(train_idx_np)} n_test={len(test_idx_np)}")
log(f"[time] dt={dt:.8g}s tau={tau_time} block={block} clustered N_eff={n_eff:.2f}")

def std_tau(a): return (np.asarray(a, dtype=np.float64)-tau_mu)/tau_sd

def make_cond(tau_now, tau_prev, jitter=False, seed=0):
    if torch.is_tensor(tau_now):
        now = (tau_now.to(device=dev,dtype=torch.float32)-tau_mu)/tau_sd
    else:
        now = torch.as_tensor(std_tau(tau_now), dtype=torch.float32, device=dev)
    if torch.is_tensor(tau_prev):
        prev = (tau_prev.to(device=dev,dtype=torch.float32)-tau_mu)/tau_sd
    else:
        prev = torch.as_tensor(std_tau(tau_prev), dtype=torch.float32, device=dev)
    if now.ndim == 1: now = now[None]
    if prev.ndim == 1: prev = prev[None]
    n = now.shape[0]; k = support[None,None]
    a = now[:,None,None,:].expand(n,1,H,W)*k
    b = prev[:,None,None,:].expand(n,1,H,W)*k
    if jitter:
        gen = torch.Generator(device=dev).manual_seed(seed)
        a = a + 0.03*torch.randn(a.shape, generator=gen, device=dev)*k
        b = b + 0.03*torch.randn(b.shape, generator=gen, device=dev)*k
    kc = k.expand(n,1,H,W).float()
    of = torch.full((n,1,H,W), obsfrac, device=dev)
    dm = DM[None,None].expand(n,1,H,W).float()
    return torch.cat([a,b,kc,of,dm],1)

def empty_cond(n):
    q = torch.zeros(n,1,H,W,device=dev)
    return torch.cat([q,q,q,q,DM[None,None].expand(n,1,H,W).float()],1)

def train(family, seed=1234):
    net = WS.CondUNet(C,ch=64).to(dev)
    ema = WS.CondUNet(C,ch=64).to(dev); ema.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(),lr=2e-4,betas=(0.9,0.99))
    gen = torch.Generator(device=dev).manual_seed(seed)
    dmloss = DM[None,None].float()
    for it in range(args.steps_train):
        pos = torch.randint(0,len(train_idx_np),(args.batch,),generator=gen,device=dev)
        idx = train_idx[pos]; xb = fields[idx]
        now = tau_true_t[idx]
        prev = tau_true_t[idx-1]
        r = float(torch.rand(1,generator=gen,device=dev))
        if r < 0.25:
            cond = empty_cond(args.batch)
        elif r < 0.55:
            cond = make_cond(now, torch.full_like(prev,tau_mu), jitter=True, seed=seed+it)
        else:
            cond = make_cond(now,prev,jitter=True,seed=seed+it)
        if family == "diffusion":
            lnsig = WS.P_MEAN+WS.P_STD*torch.randn(args.batch,device=dev,generator=gen)
            sig = lnsig.exp(); xn = xb+torch.randn(xb.shape,device=dev,generator=gen)*sig[:,None,None,None]
            weight = (sig**2+WS.SIGMA_DATA**2)/((sig*WS.SIGMA_DATA)**2)
            pred = WS.denoise(net,xn,sig,cond)
            loss = (weight[:,None,None,None]*dmloss*(pred-xb)**2).sum()/(dmloss.sum()*args.batch*C)
        else:
            tt = torch.rand(args.batch,device=dev,generator=gen).clamp(1e-3,1.0)
            eps = torch.randn(xb.shape,device=dev,generator=gen)
            xt = (1-tt)[:,None,None,None]*xb+tt[:,None,None,None]*eps
            pred = net(xt,tt,cond)
            loss = (dmloss*(pred-(eps-xb))**2).sum()/(dmloss.sum()*args.batch*C)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        with torch.no_grad():
            for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(0.9995).add_(pn,alpha=0.0005)
            for be,bn in zip(ema.buffers(),net.buffers()): be.copy_(bn)
        if (it+1)%max(1,args.steps_train//10)==0:
            log(f"[train:{family}] {it+1}/{args.steps_train} loss={float(loss):.5f} wall={time.time()-t0:.0f}s")
    ema.eval(); return ema, sum(p.numel() for p in ema.parameters())

def runtime_series(indices):
    cache = {}; failures=[]
    for kk,ti in enumerate(sorted(set(int(q) for q in indices))):
        try:
            tc,meta = predict_tau_w_from_history(
                Uphys[ti],Uphys[ti-1],Vphys[ti],xg,yg,wall_row,mask_np,
                dt=dt,nu=nu,x_stride=2,y_stride=2,edge_smooth=5,
            )
            teq_st = predict_eqwm_tau(meta["Um"],meta["ym"],nu)
            teq = np.interp(xg,meta["x"],teq_st)
            cache[ti]=(tc,teq,meta)
        except Exception as exc:
            failures.append({"index":ti,"error":repr(exc)})
        if (kk+1)%max(1,len(set(indices))//8)==0:
            log(f"[extract] {kk+1}/{len(set(indices))} failures={len(failures)}")
    if failures: raise RuntimeError(f"runtime extraction failures: {failures[:3]}")
    return cache

needed = np.r_[test_idx_np,test_idx_np-1]
runtime = runtime_series(needed)
tau_cl = np.stack([runtime[int(i)][0] for i in test_idx_np])
tau_cl_prev = np.stack([runtime[int(i-1)][0] for i in test_idx_np])
tau_eq = np.stack([runtime[int(i)][1] for i in test_idx_np])
tau_eq_prev = np.stack([runtime[int(i-1)][1] for i in test_idx_np])
tau_oracle = tau_true[test_idx_np]
tau_oracle_prev = tau_true[test_idx_np-1]
tau_pass = np.broadcast_to(tau_true[train_idx_np].mean(0),tau_oracle.shape)
rng = np.random.default_rng(9); perm = rng.permutation(W)
ARMS = ["correct_history","correct_current","closure_history","closure_current","passthrough",
        "eqwm_history","wrong","random","no_wall"]
arm_pair = {
    "correct_history":(tau_oracle,tau_oracle_prev),
    "correct_current":(tau_oracle,np.full_like(tau_oracle,tau_mu)),
    "closure_history":(tau_cl,tau_cl_prev),
    "closure_current":(tau_cl,np.full_like(tau_cl,tau_mu)),
    "passthrough":(tau_pass,tau_pass),
    "eqwm_history":(tau_eq,tau_eq_prev),
    "wrong":(-tau_pass,-tau_pass),
    "random":(tau_oracle[:,perm],tau_oracle_prev[:,perm]),
}

def wall_metrics(pred):
    p=np.asarray(pred); q=tau_oracle; good=np.isfinite(p)&np.isfinite(q)
    return {"corr_instantaneous":float(np.corrcoef(p[good],q[good])[0,1]),
            "corr_mean":float(np.corrcoef(p.mean(0),q.mean(0))[0,1]),
            "relRMSE":float(np.sqrt(np.mean((p[good]-q[good])**2))/(np.sqrt(np.mean(q[good]**2))+1e-12)),
            "sign_agreement":float(np.mean(np.sign(p[good])==np.sign(q[good])))}

def r2_components(pred,truth,centre):
    p=pred[:,:,unobs]; q=truth[:,:,unobs]
    return (((p-q)**2).sum((1,2))).cpu().numpy(),(((q-centre)**2).sum((1,2))).cpu().numpy()

def pooled_r2(pred,truth):
    p=pred[:,:,unobs]; q=truth[:,:,unobs]
    return float(1-((p-q)**2).sum()/(((q-q.mean())**2).sum()+1e-12))

def field_r2(a,b):
    aa=a[unobs]; bb=b[unobs]
    return float(1-((aa-bb)**2).sum()/(((bb-bb.mean())**2).sum()+1e-12))

def cluster_components(comp):
    # Slice-major ordering -> sum the three same-time planes into one causal-time bootstrap unit.
    return {k:(v[0].reshape(n_slice,n_time).sum(0),v[1].reshape(n_slice,n_time).sum(0)) for k,v in comp.items()}

def bootstrap(comp):
    comp=cluster_components(comp); rng=np.random.default_rng(0)
    nbl=int(np.ceil(n_time/block)); starts=rng.integers(0,n_time,size=(args.B,nbl))
    idx=((starts[:,:,None]+np.arange(block)[None,None,:])%n_time).reshape(args.B,-1)[:,:n_time]
    draws={k:1-v[0][idx].sum(1)/(v[1][idx].sum(1)+1e-12) for k,v in comp.items()}
    def rec(v): return {"mean":float(v.mean()),"ci95":[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]}
    out={k:rec(v) for k,v in draws.items()}
    for a,b in [("closure_history","closure_current"),("closure_history","no_wall"),
                ("closure_history","eqwm_history"),("closure_history","passthrough"),
                ("closure_history","wrong"),("closure_history","random"),
                ("correct_history","correct_current"),("correct_history","no_wall")]:
        d=draws[a]-draws[b]; rr=rec(d); rr["ci_positive"]=bool(rr["ci95"][0]>0); out[f"{a}_minus_{b}"]=rr
    return out

result_path=os.path.join(OUT,"causal_wall_history_results.json")
result={"_meta":{"script":"eval_causal_wall_history.py","evidence_level":2,
                 "claim":"offline closure-conditioned reconstruction; not solver-coupled WMLES",
                 "mechanism":"causal signed-stress history from a velocity-only unsteady momentum-residual bridge",
                 "data":args.data,"data_shape":[T,C,H,W],"slice_count":n_slice,"sequence_length":args.seq_len,
                 "split":"shared physical time across all three slices","train_end":args.train_end,
                 "test_start":args.test_start,"n_train":len(train_idx_np),"n_test":len(test_idx_np),
                 "n_test_times":n_time,"tau_time":tau_time,"block":block,"n_eff_clustered":n_eff,
                 "dt_seconds":dt,"dt_source":"CoNFiLD: 10 flow-through times; periodic-hill Lx=9h; 960 frames/slice",
                 "nu":nu,"Ub":Ub,"Re_h":Re_h,"L":args.L,"steps":args.steps,
                 "steps_train":args.steps_train,"B":args.B,"device":dev,"arms":ARMS},
        "wall_models_alone":{"closure":wall_metrics(tau_cl),"eqwm":wall_metrics(tau_eq),
                             "passthrough":wall_metrics(tau_pass)},"families":{}}

for family in [q.strip() for q in args.families.split(",") if q.strip()]:
    model,n_par=train(family)
    ck=os.path.join(OUT,f"causal_history_{family}.pt")
    torch.save({"ema":model.state_dict(),"n_par":n_par,"family":family,"split":result["_meta"]["split"]},ck)
    arms={}; ct={}; cf={}; Xte_f=Xte-Xmean; centre_t=Xte[:,:,unobs].mean(); centre_f=Xte_f[:,:,unobs].mean()
    for arm in ARMS:
        cond=empty_cond(len(test_idx_np)) if arm=="no_wall" else make_cond(*arm_pair[arm],jitter=True,seed=4242)
        with torch.no_grad():
            ens=WS.sample(model,cond,family,H,W,C,DM.float(),args.L,args.steps)
            pm=ens.mean(1); pmf=pm-Xmean
        ct[arm]=r2_components(pm,Xte,centre_t); cf[arm]=r2_components(pmf,Xte_f,centre_f)
        gm=pm.mean(0); dm=Xte.mean(0); guu=pm[:,0].var(0); duu=Xte[:,0].var(0); gvv=pm[:,1].var(0); dvv=Xte[:,1].var(0)
        arms[arm]={"R2_total":pooled_r2(pm,Xte),"R2_fluct":pooled_r2(pmf,Xte_f),
                   "meanU_R2":field_r2(gm[0],dm[0]),"uu_R2":field_r2(guu,duu),"vv_R2":field_r2(gvv,dvv),
                   "finite_fraction":float(torch.isfinite(ens).float().mean())}
        log(f"[{family}:{arm}] total={arms[arm]['R2_total']:+.4f} fluct={arms[arm]['R2_fluct']:+.4f} uu={arms[arm]['uu_R2']:+.3f}")
        del ens,pm,pmf,cond
        if dev=="cuda": torch.cuda.empty_cache()
    bt,bf=bootstrap(ct),bootstrap(cf)
    result["families"][family]={"checkpoint":os.path.basename(ck),"n_par":n_par,"arms":arms,
                                "bootstrap_total":bt,"bootstrap_fluct":bf,
                                "gates":{"history_load_bearing_total":bt["closure_history_minus_closure_current"]["ci_positive"],
                                         "history_load_bearing_fluct":bf["closure_history_minus_closure_current"]["ci_positive"],
                                         "closure_beats_no_wall_total":bt["closure_history_minus_no_wall"]["ci_positive"],
                                         "closure_beats_eqwm_total":bt["closure_history_minus_eqwm_history"]["ci_positive"],
                                         "correct_beats_no_wall_total":bt["correct_history_minus_no_wall"]["ci_positive"],
                                         "all_finite":all(v["finite_fraction"]==1 for v in arms.values())}}
    result["_meta"]["wall_s"]=time.time()-t0
    txt=json.dumps(result,indent=2,sort_keys=True); open(result_path,"w").write(txt)
    log(f"[out] incremental {result_path} md5={hashlib.md5(txt.encode()).hexdigest()}")

result["_gates"]={"both_families_complete":set(result["families"])=={"diffusion","flow_matching"},
                  "shared_time_split":True,"same_time_cross_slice_leakage":False,
                  "solver_coupled_claim":False,
                  "history_load_bearing_any":any(v["gates"]["history_load_bearing_total"] for v in result["families"].values()),
                  "closure_beats_no_wall_both":all(v["gates"]["closure_beats_no_wall_total"] for v in result["families"].values())}
result["_meta"]["wall_s"]=time.time()-t0
txt=json.dumps(result,indent=2,sort_keys=True); open(result_path,"w").write(txt)
log(f"=== done === md5={hashlib.md5(txt.encode()).hexdigest()} gates={result['_gates']} wall={time.time()-t0:.1f}s")
