#!/usr/bin/env python3
"""Leakage-free closure-conditioned reconstruction on the held-out Case1 test split.

This is evidence level 2 (offline serial composition), not solver-coupled WMLES.  For every held-out
snapshot, closure inputs are extracted from a spatially coarsened resolved outer-velocity snapshot;
no target time/ensemble mean enters the closure arm.  The frozen familyclass closure predicts signed
tau_w, which enters the already-trained direct-tau CondUNet feature channel.  The full arm battery is
correct / passthrough / runtime_closure / EQWM / wrong / random / no_wall.

The script retrains nothing and writes incrementally so completed family results survive interruption.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="case1_grid_48x80.npz")
ap.add_argument("--families", default="diffusion,flow_matching")
ap.add_argument("--L", type=int, default=8)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--x_stride", type=int, default=2)
ap.add_argument("--y_stride", type=int, default=2)
ap.add_argument("--B", type=int, default=4000)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.L, args.steps, args.B = 2, 6, 100

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJ = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "closure"))
from wall_closure import (  # noqa: E402
    build_direct_tau_condition,
)
from flux_bottleneck import (  # noqa: E402
    direct_tau_support,
    make_flux_arms,
    paired_block_r2,
    runtime_flux_condition,
)

# Reuse the exact trained architecture, sampler and metric kernels without invoking its main driver.
old_argv = sys.argv
try:
    sys.argv = [os.path.join(HERE, "eval_wallstress_cond.py")]
    spec = importlib.util.spec_from_file_location("wallstress_frozen", os.path.join(HERE, "eval_wallstress_cond.py"))
    WS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(WS)
finally:
    sys.argv = old_argv

dev = "cuda" if torch.cuda.is_available() else "cpu"
t_start = time.time()
def log(*x): print(*x, flush=True)

z = np.load(os.path.join(ROOT, "data", args.data))
fields_np = np.asarray(z["fields"], dtype=np.float32)
mask_np = np.asarray(z["mask"], dtype=bool)
wall_row = np.asarray(z["wall_row"], dtype=int)
mu = np.asarray(z["mu"], dtype=np.float64); sd = np.asarray(z["sd"], dtype=np.float64)
x = np.asarray(z["grid_x"], dtype=np.float64); y = np.asarray(z["grid_y"], dtype=np.float64)
h, Ly = float(z["h"]), float(z["Ly"])
T, C, H, W = fields_np.shape; dy = Ly / H
fields = torch.tensor(fields_np, device=dev)
DM = torch.tensor(mask_np, device=dev)

# Frozen train/test split and physical viscosity exactly as in eval_wallstress_cond.py.
Xmean_all = fields.mean(0, keepdim=True)
Ef = ((fields - Xmean_all)[:, :, DM] ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
Ef -= Ef.mean()
ac = np.array([1.0] + [float(np.corrcoef(Ef[:-k], Ef[k:])[0, 1]) for k in range(1, min(200, len(Ef)//2))])
zc = np.argmax(ac < 0.0) if np.any(ac < 0.0) else len(ac)
tau_time = round(max(1.0, 1.0 + 2.0 * ac[1:zc].sum()))
n_tr = int(0.76 * T); gap = max(80, 3 * tau_time)
te_idx_np = np.arange(n_tr + gap, T)
if args.smoke:
    te_idx_np = te_idx_np[:6]
Xte = fields[torch.tensor(te_idx_np, device=dev)]
Xmean = fields[:n_tr].mean(0, keepdim=True)
n_test = len(te_idx_np); block = max(1, 2 * tau_time); n_eff = n_test / (2.0 * tau_time)

old_res = json.load(open(os.path.join(OUT, "wallstress_cond_case1_bh1_results.json")))
tau_mu = float(old_res["closure"]["tau_mu"]); tau_sd = float(old_res["closure"]["tau_sd"])
nu = float(old_res["closure"]["nu"])
old_fields = np.load(os.path.join(OUT, "wallstress_cond_case1_bh1_fields.npz"))
tau_train_mean = np.asarray(old_fields["tau_mean"], dtype=np.float64)

# Instantaneous oracle wall stress from the held-out target band (evaluation/control only).
Uphys = fields_np[:, 0] * sd[0] + mu[0]
yb = 0.5 * dy
tau_true = np.zeros((n_test, W), dtype=np.float64)
for jj in range(W):
    tau_true[:, jj] = nu * Uphys[te_idx_np, wall_row[jj], jj] / yb

# Runtime resolved-coarse extraction: one snapshot at a time; no temporal aggregation.
tau_closure = np.empty_like(tau_true); tau_eqwm = np.empty_like(tau_true)
extract_failures = []
t_extract = time.time()
for ii, ti in enumerate(te_idx_np):
    try:
        flux = runtime_flux_condition(
            Uphys[ti], x, y, wall_row, mask_np,
            nu=nu, tau_mu=tau_mu, tau_sd=tau_sd,
            x_stride=args.x_stride, y_stride=args.y_stride, edge_smooth=5,
        )
        tau_closure[ii] = flux.tau_closure
        tau_eqwm[ii] = flux.tau_eqwm
    except Exception as exc:
        extract_failures.append({"test_index": int(ti), "error": repr(exc)})
        tau_closure[ii] = np.nan; tau_eqwm[ii] = np.nan
    if (ii + 1) % max(1, n_test // 8) == 0:
        log(f"[extract] {ii+1}/{n_test} failures={len(extract_failures)} wall={time.time()-t_extract:.1f}s")
if extract_failures:
    raise RuntimeError(f"runtime extraction failed on {len(extract_failures)}/{n_test} snapshots: {extract_failures[:3]}")
extract_wall_s = time.time() - t_extract

arms_tau = make_flux_arms(
    tau_true, tau_closure, tau_eqwm, tau_train_mean,
    wrong_value=float(tau_train_mean.mean()), seed=9,
)
ARMS = list(arms_tau)

def wall_metrics(pred):
    good = np.isfinite(pred) & np.isfinite(tau_true)
    p, q = pred[good], tau_true[good]
    pm, qm = pred.mean(0), tau_true.mean(0)
    neg_p, neg_q = pm < 0, qm < 0
    def extent(neg):
        if not neg.any(): return {"x_sep_over_h": None, "x_reatt_over_h": None, "n_reversed": 0}
        ids = np.flatnonzero(neg)
        return {"x_sep_over_h": float((x[ids[0]] - x.min()) / h),
                "x_reatt_over_h": float((x[ids[-1]] - x.min()) / h), "n_reversed": int(len(ids))}
    load_p = float(np.trapz(pm, x)); load_q = float(np.trapz(qm, x))
    return {
        "corr_tau_instantaneous": float(np.corrcoef(p, q)[0, 1]),
        "corr_tau_mean": float(np.corrcoef(pm, qm)[0, 1]),
        "relRMSE_tau": float(np.sqrt(np.mean((p-q)**2)) / (np.sqrt(np.mean(q**2)) + 1e-12)),
        "sign_agreement": float(np.mean(np.sign(p) == np.sign(q))),
        "finite_fraction": float(good.mean()),
        "mean_tau_pred": float(p.mean()), "mean_tau_dns": float(q.mean()),
        "integrated_skin_friction": load_p, "integrated_skin_friction_dns": load_q,
        "load_relative_error": float(abs(load_p-load_q) / (abs(load_q)+1e-12)),
        "separation_pred": extent(neg_p), "separation_dns": extent(neg_q),
        "reversed_jaccard": float((neg_p & neg_q).sum() / max(1, (neg_p | neg_q).sum())),
    }

wall = {"closure": wall_metrics(tau_closure), "eqwm": wall_metrics(tau_eqwm),
        "passthrough": wall_metrics(np.broadcast_to(tau_train_mean[None], tau_true.shape))}
log(f"[wall] closure corr_inst={wall['closure']['corr_tau_instantaneous']:+.3f} "
    f"corr_mean={wall['closure']['corr_tau_mean']:+.3f} relRMSE={wall['closure']['relRMSE_tau']:.3f}; "
    f"EQWM corr_inst={wall['eqwm']['corr_tau_instantaneous']:+.3f}")

support = direct_tau_support(wall_row, mask_np)
unobs = DM & (~torch.tensor(support, device=dev))

def make_cond(tau, seed=4242):
    a = build_direct_tau_condition(tau, tau_mu, tau_sd, wall_row, mask_np, band_h=1)
    out = torch.tensor(a, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    out[:, 0:1] += 0.03 * torch.randn(out[:, 0:1].shape, generator=gen, device=dev) * out[:, 1:2]
    return out

def empty_cond(n):
    out = torch.zeros(n, 4, H, W, device=dev)
    out[:, 3] = DM.float()
    return out

def r2_components(pred, truth, region, centre):
    p = pred[:, :, region]; q = truth[:, :, region]
    return (((p-q)**2).sum((1,2))).cpu().numpy(), (((q-centre)**2).sum((1,2))).cpu().numpy()

def pooled_r2(pred, truth, region):
    p, q = pred[:, :, region], truth[:, :, region]
    return float(1.0 - ((p-q)**2).sum() / (((q-q.mean())**2).sum()+1e-12))

def field_r2(a, b, region):
    return float(1.0 - ((a-b)[region]**2).sum() / (((b-b[region].mean())[region])**2).sum().clamp_min(1e-12))

def stats_metrics(pm):
    gm, dm = pm.mean(0), Xte.mean(0)
    guu, duu = pm[:,0].var(0), Xte[:,0].var(0)
    gvv, dvv = pm[:,1].var(0), Xte[:,1].var(0)
    row = H // 2
    def slope(q):
        q = q[:,0,row]; q = q-q.mean(1,keepdim=True)
        e = (torch.fft.rfft(q,dim=1).abs()**2).mean(0)
        kk = torch.arange(2, min(24, e.numel()), device=dev)
        return float(np.polyfit(np.log(kk.cpu().numpy()), np.log(e[kk].cpu().numpy()+1e-20), 1)[0])
    return {"meanU_R2": field_r2(gm[0], dm[0], unobs), "uu_R2": field_r2(guu, duu, unobs),
            "vv_R2": field_r2(gvv, dvv, unobs), "spectrum_slope_gen": slope(pm),
            "spectrum_slope_dns": slope(Xte)}

def bootstrap(comp):
    # Full regime has n_test >> block.  Smoke mode deliberately uses six frames;
    # cap only there so the integration path runs without presenting its interval
    # as inferential evidence.
    return paired_block_r2(comp, block=min(block, n_test), n_boot=args.B, seed=0)

result_path = os.path.join(OUT, "runtime_closure_conditioning_results.json")
result = {
    "_meta": {"script":"eval_runtime_closure_conditioning.py", "evidence_level":2,
              "claim":"closure-conditioned offline generative reconstruction; NOT solver-coupled WMLES",
              "runtime_input":"one spatially coarsened resolved snapshot; no target DNS time/ensemble mean",
              "data":args.data,"dev":dev,"T":T,"H":H,"W":W,"n_test":n_test,"n_eff":n_eff,
              "tau":tau_time,"block":block,"L":args.L,"steps":args.steps,"B":args.B,
              "x_stride":args.x_stride,"y_stride":args.y_stride,"extract_wall_s":extract_wall_s,
              "arms":ARMS,"closure_weight_sha256":"63e4941f682996ef2f95f8882363d73ca440199cd7b5ccee61002f1c5fa99280",
              "checkpoint_provenance":"codes/closure/provenance.json"},
    "wall_models_alone": wall, "families": {},
}

for fam in [q.strip() for q in args.families.split(",") if q.strip()]:
    ck_path = os.path.join(OUT, f"wscond_case1_bh1_{fam}_A1.pt")
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    model = WS.CondUNet(C, ch=64).to(dev); model.load_state_dict(ck["ema"]); model.eval()
    fam_start = time.time(); arm_res={}; comp_total={}; comp_fluct={}; finite={}
    centre_t = Xte[:, :, unobs].mean(); Xte_f = Xte-Xmean; centre_f = Xte_f[:,:,unobs].mean()
    for arm in ARMS:
        cond = empty_cond(n_test) if arm=="no_wall" else make_cond(arms_tau[arm])
        with torch.no_grad():
            ens = WS.sample(model, cond, fam, H, W, C, DM.float(), args.L, args.steps)
            pm = ens.mean(1); pm_f = pm-Xmean
        finite[arm] = float(torch.isfinite(ens).float().mean())
        comp_total[arm] = r2_components(pm,Xte,unobs,centre_t)
        comp_fluct[arm] = r2_components(pm_f,Xte_f,unobs,centre_f)
        arm_res[arm]={"R2_total":pooled_r2(pm,Xte,unobs),"R2_fluct":pooled_r2(pm_f,Xte_f,unobs),
                      "RMSE":float(((pm-Xte)[:,:,unobs]**2).mean().sqrt()),
                      "statistics":stats_metrics(pm),"finite_fraction":finite[arm]}
        log(f"[{fam}:{arm}] total={arm_res[arm]['R2_total']:+.4f} fluct={arm_res[arm]['R2_fluct']:+.4f} "
            f"uu={arm_res[arm]['statistics']['uu_R2']:+.3f} finite={finite[arm]:.6f}")
        del ens, pm, pm_f, cond
        if dev=="cuda": torch.cuda.empty_cache()
    bt=bootstrap(comp_total); bf=bootstrap(comp_fluct)
    result["families"][fam]={"checkpoint":os.path.basename(ck_path),"n_par":int(ck["n_par"]),
                             "arms":arm_res,"bootstrap_total":bt,"bootstrap_fluct":bf,
                             "runtime_s":time.time()-fam_start,
                             "gates":{"closure_beats_no_wall_total":bt["closure_minus_no_wall"]["ci_positive"],
                                      "closure_beats_eqwm_total":bt["closure_minus_eqwm"]["ci_positive"],
                                      "correct_beats_no_wall_total":bt["correct_minus_no_wall"]["ci_positive"],
                                      "all_samples_finite":all(v==1.0 for v in finite.values())}}
    result["_meta"]["wall_s"] = time.time()-t_start
    txt=json.dumps(result,indent=2,sort_keys=True)
    open(result_path,"w").write(txt)
    log(f"[out] {result_path} md5={hashlib.md5(txt.encode()).hexdigest()}")

result["_gates"]={"runtime_features_no_target_mean":True,"extract_failures":len(extract_failures),
                  "both_families_complete":set(result["families"])=={"diffusion","flow_matching"},
                  "solver_coupled_claim":False,
                  "closure_beats_no_wall_any":any(v["gates"]["closure_beats_no_wall_total"] for v in result["families"].values())}
result["_meta"]["wall_s"] = time.time()-t_start
txt=json.dumps(result,indent=2,sort_keys=True); open(result_path,"w").write(txt)
log(f"=== done === md5={hashlib.md5(txt.encode()).hexdigest()} gates={result['_gates']} wall={time.time()-t_start:.1f}s")
