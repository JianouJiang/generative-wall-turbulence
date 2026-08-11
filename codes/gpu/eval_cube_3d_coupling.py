#!/usr/bin/env python3
"""Genuinely-3D cube-array oracle-wall propagation experiment.

The domain is the complete Coceal periodic cube cell, not a homogeneous spanwise
extrusion.  A conditional rectified-flow posterior is trained on full 3-D
velocity volumes and the actual floor/cube/top near-wall band.  Correct,
no-wall and far-time wrong-wall interventions share model, sampler noise and
test fields.  All scores exclude the imposed band and use circular
block-bootstrap confidence intervals.

Evidence boundary: the contemporaneous band is supplied by held-out LES.  This
is an oracle propagation test, not a closure-conditioned or solver-coupled wall
model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument(
    "--stack", default=os.environ.get("GWT_CUBE_STACK", "controlled_raw/cube_les/stack")
)
P.add_argument(
    "--case", default=os.environ.get("GWT_CUBE_CASE", "controlled_raw/cube_les")
)
P.add_argument("--smoke", action="store_true")
P.add_argument("--steps-train", type=int, default=30000)
P.add_argument("--sample-steps", type=int, default=16)
P.add_argument("--members", type=int, default=8)
P.add_argument("--boot", type=int, default=4000)
A = P.parse_args()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "cube3d_coupling_results.json"
FIG = RESULTS / "fig_cube3d_coupling.png"
CKPT = RESULTS / "cube3d_rectified_flow.pt"
STACK = Path(A.stack)
CASE = Path(A.case)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(20260721)
np.random.seed(20260721)
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True

if A.smoke:
    A.steps_train, A.sample_steps, A.members, A.boot = 8, 3, 2, 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def integral_tau(x: np.ndarray, maxlag=200) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    if np.allclose(x.std(), 0):
        return 1.0
    vals = [1.0]
    for k in range(1, min(maxlag, len(x) // 2)):
        vals.append(float(np.corrcoef(x[:-k], x[k:])[0, 1]))
    ac = np.nan_to_num(np.asarray(vals), nan=0.0)
    stop = np.where(ac < 0)[0]
    m = int(stop[0]) if len(stop) else len(ac)
    return float(max(1.0, 1.0 + 2.0 * ac[1:m].sum()))


def manifest_records():
    mf = STACK / "manifest.jsonl"
    if not mf.exists():
        raise FileNotFoundError(mf)
    rec = [json.loads(line) for line in mf.read_text().splitlines() if line.strip()]
    # Deduplicate by physical time (a resumed watcher may encounter a retained restart field).
    by_t = {}
    for r in rec:
        p = STACK / r["npy"]
        if p.exists():
            by_t[round(float(r["time"]), 8)] = (float(r["time"]), p)
    out = [by_t[k] for k in sorted(by_t)]
    if len(out) < (12 if A.smoke else 1000):
        raise RuntimeError(f"need >=1000 post-production rasters, found {len(out)}")
    return out


def build_memmap(records):
    post = [(t, p) for t, p in records if t >= (0 if A.smoke else 40.0)]
    if len(post) < (12 if A.smoke else 1000):
        raise RuntimeError(f"need >=1000 snapshots after t=40, found {len(post)}")
    if A.smoke:
        post = post[:12]
    path = CASE / "cube_ds2_float16.npy"
    times_path = CASE / "cube_ds2_times.npy"
    expected = (len(post), 3, 48, 96, 48)
    rebuild = True
    if path.exists() and times_path.exists():
        try:
            old = np.load(times_path)
            mm = np.load(path, mmap_mode="r")
            rebuild = mm.shape != expected or not np.allclose(old, [t for t, _ in post])
        except Exception:
            rebuild = True
    if rebuild:
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=expected)
        for i, (t, p) in enumerate(post):
            vol = np.load(p, mmap_mode="r")
            if vol.shape != (3, 96, 192, 96):
                raise ValueError(f"{p} shape {vol.shape}")
            # Cell-centred volume average, not point subsampling.
            v = np.asarray(vol, np.float32).reshape(3, 48, 2, 96, 2, 48, 2).mean((2, 4, 6))
            mm[i] = v.astype(np.float16)
            if (i + 1) % 100 == 0:
                print(f"[preprocess] {i+1}/{len(post)}", flush=True)
        mm.flush()
        os.replace(tmp, path)
        np.save(times_path, np.array([t for t, _ in post], np.float64))
    return path, np.array([t for t, _ in post], np.float64), post


def geometry():
    nx, ny, nz = 48, 96, 48
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) * 4.0 / ny
    z = (np.arange(nz) + 0.5) * 2.0 / nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    solid = ((X >= .5) & (X <= 1.5) & (Y <= 1.0) & (Z >= .5) & (Z <= 1.5))
    fluid = ~solid
    # Euclidean distance to the cube box plus floor/top walls.  x/z boundaries
    # are periodic and therefore never treated as walls.
    dx = np.maximum.reduce([.5 - X, np.zeros_like(X), X - 1.5])
    dy = np.maximum.reduce([0.0 - Y, np.zeros_like(Y), Y - 1.0])
    dz = np.maximum.reduce([.5 - Z, np.zeros_like(Z), Z - 1.5])
    dcube = np.sqrt(dx * dx + dy * dy + dz * dz)
    dist = np.minimum.reduce([Y, 4.0 - Y, dcube])
    band = fluid & (dist <= 2.01 * (4.0 / ny))
    return fluid, band, dist, x, y, z


def noise_embed(t, dim=128):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Block3D(nn.Module):
    def __init__(self, ci, co, ce=128):
        super().__init__()
        groups_i = min(8, ci)
        groups_o = min(8, co)
        self.n1 = nn.GroupNorm(groups_i, ci)
        self.c1 = nn.Conv3d(ci, co, 3, padding=1)
        self.e = nn.Linear(ce, co)
        self.n2 = nn.GroupNorm(groups_o, co)
        self.c2 = nn.Conv3d(co, co, 3, padding=1)
        self.skip = nn.Conv3d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.e(e)[:, :, None, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class FlowUNet3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.inc = nn.Conv3d(8, base, 3, padding=1)
        self.b0 = Block3D(base, base)
        self.d1 = nn.Conv3d(base, 2 * base, 4, 2, 1)
        self.b1 = Block3D(2 * base, 2 * base)
        self.d2 = nn.Conv3d(2 * base, 4 * base, 4, 2, 1)
        self.m1 = Block3D(4 * base, 4 * base)
        self.m2 = Block3D(4 * base, 4 * base)
        self.u1 = nn.ConvTranspose3d(4 * base, 2 * base, 4, 2, 1)
        self.ub1 = Block3D(4 * base, 2 * base)
        self.u0 = nn.ConvTranspose3d(2 * base, base, 4, 2, 1)
        self.ub0 = Block3D(2 * base, base)
        self.out = nn.Conv3d(base, 3, 3, padding=1)

    def forward(self, x, t, cond, cmask, fluid):
        e = self.temb(noise_embed(t))
        h0 = self.b0(self.inc(torch.cat([x, cond, cmask, fluid], 1)), e)
        h1 = self.b1(self.d1(h0), e)
        m = self.m2(self.m1(self.d2(h1), e), e)
        u1 = self.ub1(torch.cat([self.u1(m), h1], 1), e)
        u0 = self.ub0(torch.cat([self.u0(u1), h0], 1), e)
        return self.out(F.silu(u0)) * fluid


def prepare_stats(mm, train_idx, fluid):
    # Streaming sufficient statistics over fluid cells only.
    count = np.zeros(3, np.float64)
    sm = np.zeros(3, np.float64)
    ss = np.zeros(3, np.float64)
    mean_field = np.zeros((3, 48, 96, 48), np.float64)
    for j, i in enumerate(train_idx):
        x = np.asarray(mm[i], np.float32)
        mean_field += x
        for c in range(3):
            v = x[c][fluid]
            count[c] += v.size; sm[c] += v.sum(dtype=np.float64)
            ss[c] += np.square(v, dtype=np.float64).sum(dtype=np.float64)
    mean_field /= len(train_idx)
    mu = sm / count
    sd = np.sqrt(np.maximum(ss / count - mu * mu, 1e-8))
    return mu.astype(np.float32), sd.astype(np.float32), mean_field.astype(np.float32)


def block_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]


def train(model, mm, train_idx, mu, sd, fluid, band):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .99), weight_decay=1e-4)
    ema = FlowUNet3D().to(DEV)
    ema.load_state_dict(model.state_dict())
    start = 0
    losses = []
    if CKPT.exists() and not A.smoke:
        q = torch.load(CKPT, map_location=DEV)
        if q.get("complete", False):
            ema.load_state_dict(q["ema"]); ema.eval()
            print(f"[resume] completed model {CKPT}", flush=True)
            return ema, q["train_meta"]
        model.load_state_dict(q["model"]); ema.load_state_dict(q["ema"])
        opt.load_state_dict(q["opt"]); start = int(q["step"]); losses = q.get("losses", [])
        print(f"[resume] step {start}", flush=True)
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=DEV)
    bt = torch.tensor(band[None, None], dtype=torch.float32, device=DEV)
    muv = torch.tensor(mu[None, :, None, None, None], device=DEV)
    sdv = torch.tensor(sd[None, :, None, None, None], device=DEV)
    rng = np.random.default_rng(7701)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    t0 = time.time()
    model.train()
    batch = 1 if A.smoke else 2
    for it in range(start, A.steps_train):
        ids = rng.choice(train_idx, size=batch, replace=True)
        arr = np.stack([np.asarray(mm[i], np.float32) for i in ids])
        x0 = torch.from_numpy(arr).to(DEV, non_blocking=True)
        x0 = ((x0 - muv) / sdv) * ft
        z = torch.randn_like(x0) * ft
        tt = torch.rand(batch, device=DEV)
        xt = (1 - tt[:, None, None, None, None]) * x0 + tt[:, None, None, None, None] * z
        keep = (torch.rand(batch, device=DEV) > .25).float()[:, None, None, None, None]
        cm = bt.expand(batch, -1, -1, -1, -1) * keep
        cond = x0 * cm
        target = (z - x0) * ft
        with torch.autocast(device_type=DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            pred = model(xt, tt, cond, cm, ft.expand(batch, -1, -1, -1, -1))
            loss = (((pred - target) ** 2) * ft).sum() / (ft.sum() * batch * 3)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            decay = .9995
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(p, alpha=1 - decay)
        if (it + 1) % max(1, A.steps_train // 30) == 0:
            losses.append([it + 1, float(loss.detach())])
            print(f"[train] {it+1}/{A.steps_train} loss={loss.item():.5f} wall={time.time()-t0:.0f}s", flush=True)
        if (it + 1) % 5000 == 0 and not A.smoke:
            torch.save({"complete": False, "step": it + 1, "model": model.state_dict(),
                        "ema": ema.state_dict(), "opt": opt.state_dict(), "losses": losses}, CKPT)
    meta = {"steps": A.steps_train, "batch": batch, "losses": losses,
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "wall_seconds": round(time.time() - t0, 1)}
    ema.eval()
    if not A.smoke:
        torch.save({"complete": True, "step": A.steps_train, "ema": ema.state_dict(),
                    "train_meta": meta}, CKPT)
    return ema, meta


@torch.no_grad()
def posterior_mean(model, truth, donor, mu, sd, fluid, band, arm, members, steps, seed):
    """Posterior mean for one truth batch; output is physical velocity."""
    b = truth.shape[0]
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=DEV)
    bt = torch.tensor(band[None, None], dtype=torch.float32, device=DEV)
    muv = torch.tensor(mu[None, :, None, None, None], device=DEV)
    sdv = torch.tensor(sd[None, :, None, None, None], device=DEV)
    tr = torch.from_numpy(truth).to(DEV)
    trn = ((tr - muv) / sdv) * ft
    if arm == "correct":
        obs = trn
        cm = bt.expand(b, -1, -1, -1, -1)
    elif arm == "wrong_wall":
        dn = (torch.from_numpy(donor).to(DEV) - muv) / sdv
        obs = dn * ft
        cm = bt.expand(b, -1, -1, -1, -1)
    elif arm == "no_wall":
        obs = torch.zeros_like(trn)
        cm = torch.zeros_like(bt).expand(b, -1, -1, -1, -1)
    else:
        raise KeyError(arm)
    obs = obs * cm
    obs = obs.repeat_interleave(members, 0)
    cm = cm.repeat_interleave(members, 0)
    fluid_b = ft.expand(b * members, -1, -1, -1, -1)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((b * members, 3, 48, 96, 48), generator=g, device=DEV) * fluid_b
    zobs = x.clone()
    for j in range(steps, 0, -1):
        t = j / steps
        tnext = (j - 1) / steps
        tv = torch.full((b * members,), t, device=DEV)
        v = model(x, tv, obs, cm, fluid_b)
        x = (x + (tnext - t) * v) * fluid_b
        bridge = (1 - tnext) * obs + tnext * zobs
        x = x * (1 - cm) + bridge * cm
    pm = x.reshape(b, members, 3, 48, 96, 48).mean(1)
    pm = (pm * sdv + muv) * ft
    return pm.float().cpu().numpy()


def evaluate(model, mm, test_idx, mu, sd, mean_field, fluid, band, dist, block):
    # Evenly retain at most 160 held-out times for the posterior tournament.
    nmax = 12 if A.smoke else 160
    if len(test_idx) > nmax:
        sel = np.linspace(0, len(test_idx) - 1, nmax).round().astype(int)
        test_idx = np.asarray(test_idx)[sel]
    else:
        test_idx = np.asarray(test_idx)
    donor_idx = np.roll(test_idx, len(test_idx) // 2)
    regions = {
        "full_support_excluded": fluid & (~band),
        "near_support_excluded_d_le_0p5h": fluid & (~band) & (dist <= .5),
        "outer_d_gt_0p5h": fluid & (dist > .5),
    }
    # Shared SST per frame/region on physical fluctuations, channel-standardised.
    truth_all = np.stack([np.asarray(mm[i], np.float32) for i in test_idx])
    tf = (truth_all - mean_field[None]) / sd[None, :, None, None, None]
    tbar = {k: float(tf[:, :, m].mean()) for k, m in regions.items()}
    sst = {k: np.square(tf[:, :, m] - tbar[k]).sum((1, 2)) for k, m in regions.items()}
    comps = {arm: {k: [] for k in regions} for arm in ("correct", "no_wall", "wrong_wall")}
    representative = {}
    batch = 1 if A.smoke else 4
    for arm in comps:
        for j in range(0, len(test_idx), batch):
            ids = test_idx[j:j + batch]
            dids = donor_idx[j:j + batch]
            truth = np.stack([np.asarray(mm[i], np.float32) for i in ids])
            donor = np.stack([np.asarray(mm[i], np.float32) for i in dids])
            pm = posterior_mean(model, truth, donor, mu, sd, fluid, band, arm,
                                A.members, A.sample_steps, seed=9100 + j)
            pf = (pm - mean_field[None]) / sd[None, :, None, None, None]
            tt = (truth - mean_field[None]) / sd[None, :, None, None, None]
            for k, m in regions.items():
                comps[arm][k].extend(np.square(pf[:, :, m] - tt[:, :, m]).sum((1, 2)).tolist())
            if j == 0:
                representative[arm] = pm[0]
        print(f"[eval] arm={arm} n={len(test_idx)}", flush=True)

    rng = np.random.default_rng(44)
    # Evaluation times were evenly thinned; scale the raw block accordingly.
    stride = max(1, int(round(np.median(np.diff(test_idx))))) if len(test_idx) > 1 else 1
    b_eval = max(1, int(round(block / stride)))
    bix = block_indices(len(test_idx), b_eval, A.boot, rng)
    result = {"arms": {}, "deltas": {}, "n_eval": len(test_idx), "eval_block": b_eval}
    boots = {}
    for arm in comps:
        result["arms"][arm] = {}
        boots[arm] = {}
        for k in regions:
            se = np.asarray(comps[arm][k])
            st = np.asarray(sst[k])
            point = float(1 - se.sum() / (st.sum() + 1e-12))
            br = 1 - se[bix].sum(1) / (st[bix].sum(1) + 1e-12)
            boots[arm][k] = br
            result["arms"][arm][k] = {"R2_fluct_balanced": point,
                "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]}
    for other in ("no_wall", "wrong_wall"):
        result["deltas"][f"correct_minus_{other}"] = {}
        for k in regions:
            d = boots["correct"][k] - boots[other][k]
            point = (result["arms"]["correct"][k]["R2_fluct_balanced"] -
                     result["arms"][other][k]["R2_fluct_balanced"])
            result["deltas"][f"correct_minus_{other}"][k] = {
                "point": float(point),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "ci_positive": bool(np.percentile(d, 2.5) > 0),
            }
    return result, representative, truth_all[0], test_idx


def make_figure(eval_result, rep, truth, fluid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zmid = 24
    fields = [("LES truth", truth), ("correct wall", rep["correct"]),
              ("no wall", rep["no_wall"]), ("wrong wall", rep["wrong_wall"])]
    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=True)
    vals = []
    for _, f in fields:
        q = np.sqrt(np.square(f).sum(0))[:, :, zmid].T
        vals.append(q)
    vmax = np.percentile(vals[0][fluid[:, :, zmid].T], 99)
    for a, (title, _), q in zip(ax.ravel(), fields, vals):
        im = a.imshow(q, origin="lower", aspect="auto", extent=[0, 2, 0, 4],
                      cmap="magma", vmin=0, vmax=vmax)
        a.set_title(title); a.set_xlabel("x/h"); a.set_ylabel("y/h")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=.8, label=r"$|u|/U_b$")
    fig.savefig(FIG, dpi=220)
    plt.close(fig)


def main():
    t0 = time.time()
    records = manifest_records()
    data_path, times, post = build_memmap(records)
    mm = np.load(data_path, mmap_mode="r")
    fluid, band, dist, xg, yg, zg = geometry()
    # Kinetic-energy decorrelation determines split gap and bootstrap blocks.
    energy = np.empty(len(mm), float)
    for i in range(len(mm)):
        a = np.asarray(mm[i], np.float32)
        energy[i] = np.square(a[:, fluid]).mean()
    tau = integral_tau(energy)
    # Freeze a chronological split from the correlation audit before any arm is
    # evaluated.  The original 70% train + 3*tau gap allocation was impossible
    # for the harvested 1,101-field record (it left only 38 test fields).  A
    # one-integral-time exclusion gap removes adjacent train/test states, while
    # a 60% training block leaves >3 integral times for honest held-out scoring.
    ntr = int(.60 * len(mm))
    gap = max(8, int(math.ceil(tau)))
    train_idx = np.arange(ntr)
    test_idx = np.arange(min(len(mm) - 1, ntr + gap), len(mm))
    min_test = 4 if A.smoke else max(120, int(math.ceil(3 * tau)))
    if len(test_idx) < min_test:
        raise RuntimeError(
            f"insufficient held-out times after gap: {len(test_idx)} < {min_test}")
    block = max(1, int(math.ceil(tau)))
    mu, sd, mean_field = prepare_stats(mm, train_idx, fluid)
    # Stationarity audit: compare middle and final thirds of post-spinup energy.
    thirds = np.array_split(energy, 3)
    drift = float((thirds[-1].mean() - thirds[-2].mean()) / (thirds[-2].mean() + 1e-12))
    model = FlowUNet3D().to(DEV)
    ema, train_meta = train(model, mm, train_idx, mu, sd, fluid, band)
    eval_result, rep, truth, eval_idx = evaluate(ema, mm, test_idx, mu, sd, mean_field,
                                                 fluid, band, dist, block)
    make_figure(eval_result, rep, truth, fluid)
    yplus_path = CASE / "yplus_preflight.json"
    out = {
        "_meta": {
            "script": Path(__file__).name,
            "device": str(DEV),
            "regime": "genuinely-3D Coceal aligned cube array, lambda_p=0.25, pitch=2h, Re_h=5000",
            "evidence_level": "held-out LES oracle near-wall-band propagation; not closure-conditioned and not solver-coupled",
            "source_grid": [96, 192, 96],
            "model_grid_full_domain": [48, 96, 48],
            "n_raw_rasters": len(records),
            "n_post_spinup": len(mm),
            "time_minmax_post_spinup": [float(times[-len(mm)]), float(times[-1])],
            "n_train": len(train_idx),
            "split_rule": "first 60% train; one integral-correlation-time gap; chronological remainder test",
            "split_gap_snapshots": gap,
            "n_test_available": len(test_idx),
            "test_span_in_integral_times": float(len(test_idx) / tau),
            "tau_integral_snapshots": tau,
            "bootstrap_block_snapshots": block,
            "effective_independent_post_spinup": float(len(mm) / tau),
            "band_cells": int(band.sum()),
            "fluid_cells": int(fluid.sum()),
            "band_thickness_h": float(2 * 4 / 96),
            "support_excluded_scoring": True,
            "stationarity_energy_drift_middle_to_final": drift,
            "wall_seconds_total": round(time.time() - t0, 1),
            "data_memmap_sha256": sha256(data_path),
            "checkpoint_sha256": sha256(CKPT) if CKPT.exists() else None,
            "yplus_preflight": json.loads(yplus_path.read_text()) if yplus_path.exists() else None,
        },
        "training": train_meta,
        "evaluation": eval_result,
        "gates": {
            "snapshot_count_ge_1000": bool(len(mm) >= 1000),
            "correct_beats_no_wall_support_excluded": eval_result["deltas"]["correct_minus_no_wall"]["full_support_excluded"]["ci_positive"],
            "correct_beats_wrong_wall_support_excluded": eval_result["deltas"]["correct_minus_wrong_wall"]["full_support_excluded"]["ci_positive"],
            "correct_near_skill_positive": bool(eval_result["arms"]["correct"]["near_support_excluded_d_le_0p5h"]["ci95"][0] > 0),
        },
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    digest = sha256(OUT)
    (RESULTS / "cube3d_coupling_results.sha256").write_text(f"{digest}  {OUT.name}\n")
    np.savez_compressed(RESULTS / "cube3d_representative_fields.npz", truth=truth.astype(np.float16),
                        correct=rep["correct"].astype(np.float16), no_wall=rep["no_wall"].astype(np.float16),
                        wrong_wall=rep["wrong_wall"].astype(np.float16), fluid=fluid, band=band,
                        eval_index=int(eval_idx[0]), x=xg, y=yg, z=zg)
    print(f"\n=== done ===\n[out] {OUT} sha256={digest}\n[fig] {FIG}", flush=True)


if __name__ == "__main__":
    main()
