#!/usr/bin/env python3
"""
eval_e2_cube_reachgated.py -- node_008 REPAIR-3 DECISIVE PRODUCER.

WHAT THIS REPAIRS, AND WHY IT IS NOT A REPEAT
=============================================
Repair-1 (traction conditioning, grouped periodic-hill protocol) and repair-2
(Reichardt band lift, same unit) both returned FAIL.  Their shared diagnostic is
NOT "the wall channel carries nothing": it is that on the separating-hill record
the experiment had no power to detect anything at all.  In repair-2 the *true DNS
near-wall velocity band* -- the richest wall information physically obtainable --
also failed to beat absence (-0.05918 flow matching).  An experiment whose
positive control fails cannot report a negative about the treatment.

REPAIR-3 therefore changes three things, in this order:

  (1) POWER GATE.  The oracle near-wall band is carried as an explicit arm and a
      prospectively frozen ADMISSIBILITY gate: unless the record demonstrably
      transmits wall information under this protocol, no interface verdict is
      issued from it.  This is the instrument calibration that repair-1/2 lacked.

  (2) REGIME VALIDITY.  The closure is an equilibrium-law-plus-bounded-correction
      model.  Its physical basis is a near-equilibrium wall layer.  The hill unit
      is massively separated, where that basis is invalid by construction, and
      the record additionally has N_eff ~ 3.3.  The decisive test is moved to the
      wall-resolved cube record, the only retained record whose viscous wall
      layer is resolved AND whose oracle control transmits.  CoNFiLD Case4 was
      examined first and EXCLUDED by measurement (no resolved viscous sublayer,
      U(y_1) = 0.047 != 0, unknown Re, two-parameter Reichardt calibration
      degenerate at R2 = 0.869 with Re_tau = 10) -- see the preregistration.

  (3) REACH-SCOPED ESTIMAND.  A wall boundary condition acts over a finite
      wall-normal reach.  Averaging its effect over a domain that is mostly
      outside that reach measures dilution, not propagation.  The near-wall
      region d <= 0.5 (co-primary with the legacy whole-field region, which is
      retained unchanged so nothing is hidden) is the physically identified
      estimand and is the region the project's frozen SC4 near/outer criterion
      already names.

DESIGN
------
* Record          cube LES, 1101 frames, 48 x 96 x 48, wall-resolved.
* Split           REVERSED IN TIME relative to every prior cube producer, so the
                  decisive unit is a window that has never been a test unit and
                  is not in this run's training set.  TRAIN = [420, 1100],
                  DEV = [390, 419] (sampler budget only), GAP = [300, 389],
                  TEST = [0, 299] strided -> the decisive unit.
* Closure L_A     fitted on TRAIN frames only, validated on a TRAIN-internal
                  window, frozen; predicts signed wall-on-fluid traction from
                  matching-height state at 2.5 Delta and 4.5 Delta.  It never
                  reads a wall-adjacent cell.
* Generators      one model per (family, seed); families F = flow matching and
                  G = denoising diffusion.  Conditioning mixture is FROZEN and
                  includes the oracle band, so every scored arm -- traction and
                  band alike -- is in-distribution for the same propagator and
                  the channel comparison is not confounded by model identity.
* Arms            absent | tau_closure | tau_native | tau_eqwm | tau_shuffle |
                  tau_fartime | band_oracle
* Scoring         R2_fluct of the 8-member posterior mean; band cells AND
                  closure-read cells excluded from every region, so no arm can
                  score its own conditioning support.

Everything that could tune the outcome is fixed before the decisive unit is
touched.  The only quantity selected from data is the diffusion sampler budget,
selected on DEV by a frozen rule that scores the BASELINE arm only.

Writes only new names (tag e2_cube_reachgated*).  Never rewrites a node-004/005/
006/007/008 artifact.  Run only through cloud/gpu_run.sh --target foshan.
"""
import argparse, hashlib, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

P = argparse.ArgumentParser()
P.add_argument("--case", default="/root/autodl-tmp/cube_les")
P.add_argument("--data-name", default="cube_ds2_float16.complete.npy")
P.add_argument("--base", type=int, default=32)
P.add_argument("--steps-train", type=int, default=20000)
P.add_argument("--closure-steps", type=int, default=12000)
P.add_argument("--batch", type=int, default=4)
P.add_argument("--fm-steps", type=int, default=32)
P.add_argument("--diff-grid", default="32,64,128")
P.add_argument("--members", type=int, default=8)
P.add_argument("--boot", type=int, default=4000)
P.add_argument("--seeds", type=int, default=2)
P.add_argument("--test-stride", type=int, default=12)
P.add_argument("--dev-stride", type=int, default=4)
P.add_argument("--nmax", type=int, default=0)
P.add_argument("--smoke", action="store_true")
P.add_argument("--tag", default="e2_cube_reachgated")
A = P.parse_args()

if A.smoke:
    A.steps_train, A.closure_steps = 20, 40
    A.fm_steps, A.diff_grid = 2, "2,4"
    A.members, A.boot, A.seeds, A.base = 2, 200, 1, 8
    A.batch = 1
    A.nmax = A.nmax or 40

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ---- frozen physical constants (byte-identical to eval_e2_closure_composition) --
NU = 2.0e-4
DELTA = 1.0 / 24.0
D_ANCHOR = 0.5 * DELTA
CELL_AREA = DELTA * DELTA
KAPPA, C_REICH = 0.41, 7.8
M_OFF1, M_OFF2 = 2, 4

# ---- FROZEN reversed-time split contract ---------------------------------------
TRAIN_W = (420, 1100)
DEV_W = (390, 419)
GAP_W = (300, 389)
TEST_W = (0, 299)

# ---- FROZEN conditioning mixture (band included so every arm is in-distribution)
MIXTURE = (("tau_native", 0.30), ("tau_closure", 0.30),
           ("band_oracle", 0.20), ("absent", 0.20))

ARMS_FULL = ("absent", "tau_closure", "tau_native", "band_oracle",
             "tau_eqwm", "tau_shuffle", "tau_fartime")
ARMS_CORE = ("absent", "tau_closure", "tau_native", "band_oracle")
SHUFFLE_SEED = 20260801

TAG = A.tag
OUT = RESULTS / f"{TAG}_results.json"
COMP = RESULTS / f"{TAG}_components.npz"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


# ==============================================================================
# Geometry (byte-identical to the frozen cube producers)
# ==============================================================================
def geometry():
    nx, ny, nz = 48, 96, 48
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) * 4.0 / ny
    z = (np.arange(nz) + 0.5) * 2.0 / nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    solid = ((X >= .5) & (X <= 1.5) & (Y <= 1.0) & (Z >= .5) & (Z <= 1.5))
    fluid = ~solid
    dx = np.maximum.reduce([.5 - X, np.zeros_like(X), X - 1.5])
    dy = np.maximum.reduce([0.0 - Y, np.zeros_like(Y), Y - 1.0])
    dz = np.maximum.reduce([.5 - Z, np.zeros_like(Z), Z - 1.5])
    dcube = np.sqrt(dx * dx + dy * dy + dz * dz)
    dist = np.minimum.reduce([Y, 4.0 - Y, dcube])
    band = fluid & (dist <= 2.01 * (4.0 / ny))
    return fluid, band, dist, x, y, z


def wall_faces(fluid):
    ii, jj, kk = np.arange(48), np.arange(96), np.arange(48)
    I, J, K = np.meshgrid(ii, jj, kk, indexing="ij")
    cube_i = (I >= 12) & (I <= 35)
    cube_j = (J >= 0) & (J <= 23)
    cube_k = (K >= 12) & (K <= 35)
    faces = []

    def add(name, mask, normal):
        m = mask & fluid
        if m.sum() > 0:
            faces.append({"name": name, "mask": m,
                          "n": np.asarray(normal, np.float32),
                          "step": tuple(int(v) for v in normal)})

    add("floor", (J == 0), (0.0, 1.0, 0.0))
    add("cube_top", (J == 24) & cube_i & cube_k, (0.0, 1.0, 0.0))
    add("cube_xlo", (I == 11) & cube_j & cube_k, (-1.0, 0.0, 0.0))
    add("cube_xhi", (I == 36) & cube_j & cube_k, (1.0, 0.0, 0.0))
    add("cube_zlo", (K == 11) & cube_i & cube_j, (0.0, 0.0, -1.0))
    add("cube_zhi", (K == 36) & cube_i & cube_j, (0.0, 0.0, 1.0))
    return faces


def traction_masks(faces):
    any_mask = np.zeros((48, 96, 48), bool)
    nface = np.zeros((48, 96, 48), np.float32)
    for f in faces:
        any_mask |= f["mask"]
        nface += f["mask"].astype(np.float32)
    return any_mask, nface


def face_sampling_plan(faces):
    shape = (48, 96, 48)
    idx_wall, idx_m1, idx_m2, normals, faceid = [], [], [], [], []
    for fi, f in enumerate(faces):
        ci, cj, ck = np.nonzero(f["mask"])
        sx, sy, sz = f["step"]
        for off, store in ((M_OFF1, idx_m1), (M_OFF2, idx_m2)):
            gi, gj, gk = ci + sx * off, cj + sy * off, ck + sz * off
            assert gi.min() >= 0 and gi.max() < 48
            assert gj.min() >= 0 and gj.max() < 96
            assert gk.min() >= 0 and gk.max() < 48
            store.append(np.ravel_multi_index((gi, gj, gk), shape))
        idx_wall.append(np.ravel_multi_index((ci, cj, ck), shape))
        normals.append(np.repeat(f["n"][None], len(ci), 0))
        faceid.append(np.full(len(ci), fi, np.int64))
    return (np.concatenate(idx_wall), np.concatenate(idx_m1), np.concatenate(idx_m2),
            np.concatenate(normals, 0).astype(np.float32), np.concatenate(faceid))


# ==============================================================================
# L_A: physics-grounded wall closure (byte-identical physics)
# ==============================================================================
def reichardt_uplus(yp):
    return ((1.0 / KAPPA) * torch.log1p(KAPPA * yp)
            + C_REICH * (1.0 - torch.exp(-yp / 11.0) - (yp / 11.0) * torch.exp(-yp / 3.0)))


def utau_equilibrium(umag, y, nu=NU, iters=60):
    lo = torch.full_like(umag, 1e-8)
    hi = torch.full_like(umag, 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pred = mid * reichardt_uplus(mid * y / nu)
        too_small = pred < umag
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


class WallClosure(nn.Module):
    A_MAX = 1.5
    TH_MAX = math.pi / 3.0

    def __init__(self, n_feat, n_face, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(n_face, 8)
        self.net = nn.Sequential(
            nn.Linear(n_feat + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats, faceid):
        o = self.net(torch.cat([feats, self.emb(faceid)], -1))
        return self.A_MAX * torch.tanh(o[..., 0]), self.TH_MAX * torch.tanh(o[..., 1])


def make_context_op(idx_m1_np):
    shape = (48, 96, 48)
    idx_t = torch.from_numpy(idx_m1_np).to(DEV).long()
    ones = torch.zeros(48 * 96 * 48, device=DEV)
    ones.scatter_add_(0, idx_t, torch.ones(len(idx_m1_np), device=DEV))
    k = torch.ones((1, 1, 3, 3, 3), device=DEV)
    cnt = F.conv3d(ones.view(1, 1, *shape), k, padding=1).reshape(-1).clamp_min(1.0)

    def op(idx, mag):
        acc = torch.zeros(48 * 96 * 48, device=DEV)
        acc.scatter_add_(0, idx_t, mag)
        sm = F.conv3d(acc.view(1, 1, *shape), k, padding=1).reshape(-1) / cnt
        return torch.log(mag.clamp_min(1e-8) / sm[idx_t].clamp_min(1e-8))
    return op


def closure_features(field_t, plan):
    idx_w, idx_m1, idx_m2, nrm, fid, ctx_op = plan
    flat = field_t.reshape(3, -1)
    n = nrm

    def tangential(idx):
        u = flat[:, idx].T
        un = (u * n).sum(-1, keepdim=True)
        return u - un * n, un.squeeze(-1)

    ut1, un1 = tangential(idx_m1)
    ut2, un2 = tangential(idx_m2)
    m1 = ut1.norm(dim=-1).clamp_min(1e-8)
    m2 = ut2.norm(dim=-1).clamp_min(1e-8)
    e1 = ut1 / m1[:, None]
    e2 = torch.cross(n, e1, dim=-1)
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    y1 = (M_OFF1 + 0.5) * DELTA
    utau = utau_equilibrium(m1, torch.full_like(m1, y1))
    ctx = ctx_op(idx_m1, m1)
    cos12 = (ut1 * ut2).sum(-1) / (m1 * m2)
    sin12 = (torch.cross(e1, ut2, dim=-1) * n).sum(-1) / m2
    feats = torch.stack([torch.log1p(y1 * utau / NU), m2 / m1, un1 / m1, un2 / m1,
                         cos12, sin12, torch.log(m1.clamp_min(1e-8)), ctx], -1).float()
    return feats, e1, e2, utau, fid


def closure_traction(model, field_t, plan, eqwm=False):
    feats, e1, e2, utau, fid = closure_features(field_t, plan)
    if eqwm:
        a = torch.zeros_like(utau); th = torch.zeros_like(utau)
    else:
        a, th = model(feats, fid)
    mag = (utau ** 2) * torch.exp(a)
    direction = torch.cos(th)[:, None] * e1 + torch.sin(th)[:, None] * e2
    return (-mag[:, None] * direction).T.contiguous()


# ==============================================================================
# Generator: one architecture, two families
# ==============================================================================
def noise_embed(t, dim=128):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Block3D(nn.Module):
    def __init__(self, ci, co, ce=128):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, ci), ci)
        self.c1 = nn.Conv3d(ci, co, 3, padding=1)
        self.e = nn.Linear(ce, co)
        self.n2 = nn.GroupNorm(min(8, co), co)
        self.c2 = nn.Conv3d(co, co, 3, padding=1)
        self.skip = nn.Conv3d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.e(e)[:, :, None, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    def __init__(self, base=32):
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


def vp_alpha_bar(t):
    s = 0.008
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    return (f / f0).clamp(1e-5, 1.0)


class Trainer:
    """F = flow matching (frozen node-005/006 family).  G = VP denoising diffusion
    with the cosine schedule, identical network and identical training budget."""

    def __init__(self, family):
        assert family in ("F", "G")
        self.family = family

    def loss(self, model, x0, cond, cm, fb, ft):
        b = x0.shape[0]
        z = torch.randn_like(x0) * ft
        tt = torch.rand(b, device=x0.device)
        tv = tt.view(b, 1, 1, 1, 1)
        if self.family == "F":
            xt = (1 - tv) * x0 + tv * z
            target = (z - x0) * ft
        else:
            ab = vp_alpha_bar(tt).view(b, 1, 1, 1, 1)
            xt = (ab.sqrt() * x0 + (1 - ab).sqrt() * z) * ft
            target = z * ft
        pred = model(xt, tt, cond, cm, fb)
        return (((pred - target) ** 2) * ft).sum() / (ft.sum() * b * 3)

    @torch.no_grad()
    def sample(self, model, shape, cond, cm, fb, steps, generator, device):
        x = torch.randn(shape, generator=generator, device=device) * fb
        b = shape[0]
        if self.family == "F":
            for j in range(steps, 0, -1):
                t, tn = j / steps, (j - 1) / steps
                v = model(x, torch.full((b,), t, device=device), cond, cm, fb)
                x = (x + (tn - t) * v) * fb
            return x
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for j in range(steps):
            t, tn = ts[j], ts[j + 1]
            ab = vp_alpha_bar(t.view(1)); abn = vp_alpha_bar(tn.view(1))
            eps = model(x, t.expand(b), cond, cm, fb)
            x0h = (((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)) * fb).clamp(-6., 6.)
            if j < steps - 1:
                beta = (1 - ab / abn).clamp(1e-8, .999)
                sig = (beta * (1 - abn) / (1 - ab).clamp_min(1e-8)).sqrt()
                noise = torch.randn(shape, generator=generator, device=device) * fb
                x = (abn.sqrt() * x0h + (1 - abn - sig ** 2).clamp_min(0).sqrt() * eps
                     + sig * noise) * fb
            else:
                x = x0h
        return x


# ==============================================================================
# Conditioning
# ==============================================================================
class Conditioner:
    SOURCE = {"tau_native": "tau", "tau_closure": "tau_c", "tau_eqwm": "tau_e",
              "tau_shuffle": "tau", "tau_fartime": "tau"}

    def __init__(self, tau, tau_c, tau_e, pair_idx, tau_mask_t, band_t, tau_sd_t, perm):
        self.tau, self.tau_c, self.tau_e = tau, tau_c, tau_e
        self.idx = pair_idx
        self.tau_mask_t, self.band_t, self.tau_sd_t = tau_mask_t, band_t, tau_sd_t
        self.perm = perm

    def _scatter(self, cells):
        b = cells.shape[0]
        out = torch.zeros((b, 3, 48 * 96 * 48), device=DEV, dtype=torch.float32)
        out.scatter_add_(2, self.idx.view(1, 1, -1).expand(b, 3, -1),
                         cells / self.tau_sd_t.view(1, 3, 1))
        return out.view(b, 3, 48, 96, 48)

    def build(self, arm, ids, donor_ids, field_norm):
        b = field_norm.shape[0]
        if arm == "absent":
            return (torch.zeros_like(field_norm),
                    torch.zeros((b, 1, 48, 96, 48), device=DEV))
        if arm == "band_oracle":
            m = self.band_t.unsqueeze(0).expand(b, -1, -1, -1, -1)
            return field_norm * m, m
        src = getattr(self, self.SOURCE[arm])
        if arm == "tau_fartime":
            cells = src[donor_ids]
        elif arm == "tau_shuffle":
            cells = src[ids][:, :, self.perm]
        else:
            cells = src[ids]
        m = self.tau_mask_t.unsqueeze(0).expand(b, -1, -1, -1, -1)
        return self._scatter(cells.contiguous()), m


# ==============================================================================
# Statistics
# ==============================================================================
def integral_tau(x, maxlag=200):
    x = np.asarray(x, float); x = x - x.mean()
    if np.allclose(x.std(), 0):
        return 1.0
    vals = [1.0]
    for k in range(1, min(maxlag, len(x) // 2)):
        vals.append(float(np.corrcoef(x[:-k], x[k:])[0, 1]))
    ac = np.nan_to_num(np.asarray(vals), nan=0.0)
    stop = np.where(ac < 0)[0]
    m = int(stop[0]) if len(stop) else len(ac)
    return float(max(1.0, 1.0 + 2.0 * ac[1:m].sum()))


def block_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]


def prepare_stats(mm, train_idx, fluid):
    count = np.zeros(3, np.float64); sm = np.zeros(3, np.float64); ss = np.zeros(3, np.float64)
    mean_field = np.zeros((3, 48, 96, 48), np.float64)
    for i in train_idx:
        x = np.asarray(mm[i], np.float32)
        mean_field += x
        for c in range(3):
            v = x[c][fluid]
            count[c] += v.size
            sm[c] += v.sum(dtype=np.float64)
            ss[c] += np.square(v, dtype=np.float64).sum(dtype=np.float64)
    mean_field /= len(train_idx)
    mu = sm / count
    sd = np.sqrt(np.maximum(ss / count - mu * mu, 1e-8))
    return mu.astype(np.float32), sd.astype(np.float32), mean_field.astype(np.float32)


# ==============================================================================
@torch.no_grad()
def score_arms(trainer, ema, mm, unit_idx, donor_idx, mu, sd, mean_field, fluid,
               regions_t, sst, cond_builder, faces_t, steps, members, arms,
               seed, out, prefix):
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=DEV)
    muv = torch.tensor(mu[None, :, None, None, None], device=DEV)
    sdv = torch.tensor(sd[None, :, None, None, None], device=DEV)
    mft = torch.from_numpy(mean_field).to(DEV)
    sdt = torch.from_numpy(sd).to(DEV)
    bs = 1
    r2 = {}
    for arm in arms:
        sse = {k: [] for k in regions_t}
        fx_l, crps_l = [], []
        for b0 in range(0, len(unit_idx), bs):
            ids = np.asarray(unit_idx[b0:b0 + bs])
            dids = np.asarray(donor_idx[b0:b0 + bs])
            truth = np.stack([np.asarray(mm[i], np.float32) for i in ids])
            tr = torch.from_numpy(truth).to(DEV)
            trn = ((tr - muv) / sdv) * ft
            idt = torch.from_numpy(ids).to(DEV).long()
            dnt = torch.from_numpy(dids).to(DEV).long()
            cond, cm = cond_builder.build(arm, idt, dnt, trn)
            b = len(ids)
            condr = cond.repeat_interleave(members, 0)
            cmr = cm.repeat_interleave(members, 0)
            fb = ft.expand(b * members, -1, -1, -1, -1)
            g = torch.Generator(device=DEV).manual_seed(9100 + int(b0) + 7919 * int(seed))
            x = trainer.sample(ema, (b * members, 3, 48, 96, 48), condr, cmr, fb,
                               steps, g, DEV)
            x = x.reshape(b, members, 3, 48, 96, 48)
            phys_x = (x * sdv.unsqueeze(1) + muv.unsqueeze(1)) * ft.unsqueeze(1)
            pm = phys_x.mean(1)
            pn = (pm - mft.unsqueeze(0)) / sdt.view(1, 3, 1, 1, 1)
            tn = (tr - mft.unsqueeze(0)) / sdt.view(1, 3, 1, 1, 1)
            for k, sel in regions_t.items():
                r = (pn.reshape(b, 3, -1)[:, :, sel] - tn.reshape(b, 3, -1)[:, :, sel])
                sse[k].append(np.square(r.float().cpu().numpy()).sum((1, 2)))
            mem = ((phys_x - mft.unsqueeze(0).unsqueeze(0)) / sdt.view(1, 1, 3, 1, 1, 1))
            selp = regions_t["near_srcex"]
            mm_ = mem.reshape(b, members, 3, -1)[:, :, :, selp]
            tt_ = tn.reshape(b, 3, -1)[:, :, selp]
            ad = (mm_ - tt_.unsqueeze(1)).abs().mean(dim=(1, 2, 3))
            pair = (mm_.unsqueeze(1) - mm_.unsqueeze(2)).abs().mean(dim=(3, 4))
            crps_l.append((ad - 0.5 * pair.mean(dim=(1, 2))).float().cpu().numpy())
            fx = torch.zeros(b, device=DEV)
            for (m_t, n_t) in faces_t:
                u = pm.reshape(b, 3, -1)[:, :, m_t]
                un = (n_t.view(1, 3, 1) * u).sum(1, keepdim=True)
                ut = u - n_t.view(1, 3, 1) * un
                fx += (-NU / D_ANCHOR) * ut[:, 0].sum(1) * CELL_AREA
            fx_l.append(fx.float().cpu().numpy())
        for k in regions_t:
            out[f"{prefix}|sse|{arm}|{k}"] = np.concatenate(sse[k])
        out[f"{prefix}|crps|{arm}"] = np.concatenate(crps_l)
        out[f"{prefix}|fx|{arm}"] = np.concatenate(fx_l)
        r2[arm] = {k: float(1 - np.concatenate(sse[k]).sum() / sst[k].sum())
                   for k in regions_t}
        log(f"[score {prefix}] {arm:12s} near={r2[arm]['near_srcex']:+.5f} "
            f"full={r2[arm]['full_srcex']:+.5f} outer={r2[arm]['outer_srcex']:+.5f}")
    return r2


def train_model(family, seed, mm, train_idx, mu, sd, fluid, cond_builder, ckpt, metas):
    trainer = Trainer(family)
    if ckpt.exists() and not A.smoke:
        q = torch.load(ckpt, map_location=DEV)
        if q.get("complete", False):
            ema = UNet3D(A.base).to(DEV); ema.load_state_dict(q["ema"]); ema.eval()
            metas[f"{family}s{seed}"] = q["train_meta"]
            log(f"[resume] {ckpt.name}")
            return trainer, ema
    torch.manual_seed(seed)
    model = UNet3D(A.base).to(DEV)
    ema = UNet3D(A.base).to(DEV); ema.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .99), weight_decay=1e-4)
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=DEV)
    muv = torch.tensor(mu[None, :, None, None, None], device=DEV)
    sdv = torch.tensor(sd[None, :, None, None, None], device=DEV)
    rng = np.random.default_rng(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    names = [k for k, _ in MIXTURE]
    probs = np.array([p for _, p in MIXTURE], float); probs /= probs.sum()
    losses, t0, B = [], time.time(), A.batch
    model.train()
    for it in range(A.steps_train):
        ids = rng.choice(train_idx, size=B, replace=True)
        arr = np.stack([np.asarray(mm[i], np.float32) for i in ids])
        x0 = ((torch.from_numpy(arr).to(DEV) - muv) / sdv) * ft
        idt = torch.from_numpy(np.asarray(ids)).to(DEV).long()
        pick = rng.choice(len(names), size=B, p=probs)
        cond = torch.zeros_like(x0)
        cm = torch.zeros((B, 1, 48, 96, 48), device=DEV)
        for ci, nm in enumerate(names):
            sel = np.flatnonzero(pick == ci)
            if not len(sel):
                continue
            st = torch.from_numpy(sel).to(DEV).long()
            c, m = cond_builder.build(nm, idt[st], idt[st], x0[st])
            cond[st] = c; cm[st] = m
        fb = ft.expand(B, -1, -1, -1, -1)
        with torch.autocast(device_type=DEV.type, dtype=torch.bfloat16,
                            enabled=DEV.type == "cuda"):
            loss = trainer.loss(model, x0, cond, cm, fb, ft)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.mul_(.9995).add_(p, alpha=5e-4)
        if (it + 1) % max(1, A.steps_train // 20) == 0:
            losses.append([it + 1, float(loss.detach())])
            log(f"[train {family}s{seed}] {it+1}/{A.steps_train} loss={loss.item():.5f} "
                f"wall={time.time()-t0:.0f}s")
    meta = {"family": family, "seed": seed, "steps": A.steps_train, "batch": B,
            "base": A.base, "losses": losses, "mixture": [list(m) for m in MIXTURE],
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "wall_seconds": round(time.time() - t0, 1)}
    metas[f"{family}s{seed}"] = meta
    ema.eval()
    if not A.smoke:
        torch.save({"complete": True, "ema": ema.state_dict(), "train_meta": meta}, ckpt)
    return trainer, ema


# ==============================================================================
def main():
    t_start = time.time()
    data_path = Path(A.case) / A.data_name
    mm = np.load(data_path, mmap_mode="r")
    n = len(mm) if not A.nmax else min(len(mm), A.nmax)
    global TRAIN_W, DEV_W, GAP_W, TEST_W
    if A.smoke:                       # compressed replica of the frozen contract
        TRAIN_W, DEV_W, GAP_W, TEST_W = (28, 39), (24, 27), (12, 23), (0, 11)
        A.test_stride, A.dev_stride = 6, 2
    res = {"_meta": {}, "gates": {}, "closure": {}, "cells": {},
           "sampler_selection": {}, "arms": {}, "deltas": {}}
    comps, metas = {}, {}

    fluid, band, dist, xg, yg, zg = geometry()
    faces = wall_faces(fluid)
    tau_mask, nface = traction_masks(faces)
    idx_w, idx_m1, idx_m2, nrm_np, fid_np = face_sampling_plan(faces)
    npair = len(idx_w)
    Y = np.broadcast_to(yg[None, :, None], (48, 96, 48))
    band_phys = band & (~(Y > 2.0))

    # ---------------- FROZEN reversed-time split ----------------------------
    train_idx = np.arange(TRAIN_W[0], min(n, TRAIN_W[1] + 1))
    dev_idx = np.arange(DEV_W[0], DEV_W[1] + 1)[::A.dev_stride]
    test_idx = np.arange(TEST_W[0], TEST_W[1] + 1)[::A.test_stride]
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.intersect1d(train_idx, dev_idx)) == 0
    assert len(np.intersect1d(dev_idx, test_idx)) == 0
    min_gap = 4 if A.smoke else 90
    assert train_idx.min() - test_idx.max() >= min_gap, "train->test temporal gap too small"

    energy = np.empty(n, float)
    for i in range(n):
        energy[i] = np.square(np.asarray(mm[i], np.float32)[:, fluid]).mean()
    tau_all = integral_tau(energy)
    tau_unit = integral_tau(energy[TEST_W[0]:TEST_W[1] + 1])
    # block length in units of the STRIDED decisive unit
    block = int(max(1, math.ceil(1.2551 * tau_unit / A.test_stride)))
    block = int(min(block, max(1, (len(test_idx) - 1) // 3)))
    res["_meta"] = {
        "script": os.path.basename(__file__), "script_sha256": sha256(Path(__file__)),
        "data": str(data_path), "data_sha256": sha256(data_path),
        "tag": TAG, "device": str(DEV), "smoke": bool(A.smoke),
        "n_frames": int(n),
        "split_contract": {"protocol": "reversed_time_frame_index",
                           "train": list(TRAIN_W), "dev": list(DEV_W),
                           "gap": list(GAP_W), "test": list(TEST_W),
                           "test_stride": int(A.test_stride),
                           "dev_stride": int(A.dev_stride)},
        "n_train": int(len(train_idx)), "n_dev": int(len(dev_idx)),
        "n_test": int(len(test_idx)),
        "gap_train_minus_test_frames": int(train_idx.min() - test_idx.max()),
        "tau_integral_record": float(tau_all),
        "tau_integral_test_window": float(tau_unit),
        "n_effective_test": float((TEST_W[1] - TEST_W[0] + 1) / max(1.0, tau_unit)),
        "block_strided": int(block),
        "arms_full": list(ARMS_FULL), "arms_core": list(ARMS_CORE),
        "mixture": [list(m) for m in MIXTURE],
        "members": int(A.members), "seeds": int(A.seeds),
        "steps_train": int(A.steps_train), "base": int(A.base),
    }
    log(f"[split] n={n} train={len(train_idx)} dev={len(dev_idx)} test={len(test_idx)} "
        f"gap={train_idx.min()-test_idx.max()} tau_rec={tau_all:.1f} "
        f"tau_unit={tau_unit:.1f} block={block}")

    mu, sd, mean_field = prepare_stats(mm, train_idx, fluid)

    # ---------------- oracle native traction --------------------------------
    tau_pairs = np.zeros((n, 3, npair), np.float32)
    off = 0; face_slices = []
    for f in faces:
        cnt = int(f["mask"].sum())
        face_slices.append((f["name"], off, off + cnt, f["n"])); off += cnt
    for i in range(n):
        flat = np.asarray(mm[i], np.float32).reshape(3, -1)
        for (_, s, e, nvec) in face_slices:
            u = flat[:, idx_w[s:e]]
            un = (nvec[:, None] * u).sum(0)
            ut = u - nvec[:, None] * un[None, :]
            tau_pairs[i, :, s:e] = (-NU / D_ANCHOR) * ut
    uniq_flat = np.flatnonzero(tau_mask.reshape(-1))
    pair_to_uniq = np.searchsorted(uniq_flat, idx_w)
    tcu = np.zeros((len(train_idx), 3, len(uniq_flat)), np.float32)
    for (_, s, e, _) in face_slices:
        tcu[:, :, pair_to_uniq[s:e]] += tau_pairs[train_idx][:, :, s:e]
    tau_sd = np.sqrt(np.maximum(np.square(tcu.astype(np.float64)).mean((0, 2)),
                                1e-16)).astype(np.float32)
    del tcu

    probe = np.asarray(mm[int(train_idx[0])], np.float32).reshape(3, -1)
    up = probe[:, idx_w]
    unp = (nrm_np.T * up).sum(0)
    utp = up - nrm_np.T * unp[None, :]
    dotp = (tau_pairs[int(train_idx[0])] * utp).sum(0)
    res["gates"]["traction"] = {
        "n_pairs": int(npair), "n_unique_wall_cells": int(tau_mask.sum()),
        "n_multiface_cells": int((nface > 1).sum()),
        "tau_dot_ut_max": float(dotp.max()),
        "sign_gate_local_dissipative": bool(dotp.max() <= 1e-20),
        "fx_viscous_wall_on_fluid_train_mean": float(
            (tau_pairs[train_idx][:, 0].sum(1) * CELL_AREA).mean()),
        "note": ("first-cell VISCOUS shear surrogate, wall-on-fluid sign; "
                 "EXCLUDES pressure/form drag"),
    }
    if not res["gates"]["traction"]["sign_gate_local_dissipative"]:
        raise RuntimeError("VOID: signed traction convention gate failed")
    log(f"[traction] pairs={npair} sign_gate=OK")

    # ---------------- closure L_A, TRAIN frames only ------------------------
    ctx_op = make_context_op(idx_m1)
    plan = (torch.from_numpy(idx_w).to(DEV).long(),
            torch.from_numpy(idx_m1).to(DEV).long(),
            torch.from_numpy(idx_m2).to(DEV).long(),
            torch.from_numpy(nrm_np).to(DEV),
            torch.from_numpy(fid_np).to(DEV).long(), ctx_op)
    torch.manual_seed(4242)
    closure = WallClosure(8, len(faces)).to(DEV)
    tau_sd_t = torch.from_numpy(tau_sd).to(DEV)
    n_fit = int(0.90 * len(train_idx))
    cl_fit, cl_val = train_idx[:n_fit], train_idx[n_fit:]
    ck_cl = RESULTS / f"{TAG}_closure_LA.pt"
    if ck_cl.exists() and not A.smoke and torch.load(ck_cl, map_location="cpu").get("complete", False):
        q = torch.load(ck_cl, map_location=DEV)
        closure.load_state_dict(q["model"]); closure_meta = q["meta"]
        log("[resume] closure L_A")
    else:
        opt = torch.optim.AdamW(closure.parameters(), lr=3e-3, weight_decay=1e-5)
        rng = np.random.default_rng(913)
        t0, hist, best = time.time(), [], {"loss": float("inf"), "state": None, "it": -1}
        for it in range(A.closure_steps):
            i = int(rng.choice(cl_fit))
            fld = torch.from_numpy(np.asarray(mm[i], np.float32)).to(DEV)
            pred = closure_traction(closure, fld, plan)
            tgt = torch.from_numpy(tau_pairs[i]).to(DEV)
            loss = (((pred - tgt) / tau_sd_t[:, None]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(closure.parameters(), 1.0)
            opt.step()
            if (it + 1) % max(1, A.closure_steps // 10) == 0:
                with torch.no_grad():
                    vs, ve = [], []
                    for j in cl_val[::4]:
                        fj = torch.from_numpy(np.asarray(mm[j], np.float32)).to(DEV)
                        tj = torch.from_numpy(tau_pairs[j]).to(DEV)
                        vs.append(float((((closure_traction(closure, fj, plan) - tj)
                                          / tau_sd_t[:, None]) ** 2).mean()))
                        ve.append(float((((closure_traction(closure, fj, plan, True) - tj)
                                          / tau_sd_t[:, None]) ** 2).mean()))
                vm = float(np.mean(vs))
                hist.append([it + 1, float(loss.detach()), vm, float(np.mean(ve))])
                if vm < best["loss"]:
                    best = {"loss": vm, "it": it + 1,
                            "state": {k: v.detach().clone() for k, v in closure.state_dict().items()}}
                log(f"[closure] {it+1}/{A.closure_steps} train={loss:.5f} val={vm:.5f} "
                    f"val_eqwm={np.mean(ve):.5f} best@{best['it']}")
        if best["state"] is not None:
            closure.load_state_dict(best["state"])
        closure_meta = {"steps": A.closure_steps, "history": hist,
                        "selected_step": best["it"], "selected_val": best["loss"],
                        "fit_frames": [int(cl_fit[0]), int(cl_fit[-1])],
                        "val_frames": [int(cl_val[0]), int(cl_val[-1])],
                        "n_parameters": sum(p.numel() for p in closure.parameters()),
                        "wall_seconds": round(time.time() - t0, 1)}
        if not A.smoke:
            torch.save({"complete": True, "model": closure.state_dict(),
                        "meta": closure_meta}, ck_cl)
    closure.eval()

    tau_cl = np.zeros_like(tau_pairs); tau_eq = np.zeros_like(tau_pairs)
    with torch.no_grad():
        for i in range(n):
            fld = torch.from_numpy(np.asarray(mm[i], np.float32)).to(DEV)
            tau_cl[i] = closure_traction(closure, fld, plan).cpu().numpy()
            tau_eq[i] = closure_traction(closure, fld, plan, True).cpu().numpy()

    def apriori(idxs):
        nt = tau_pairs[idxs]; c = tau_cl[idxs]; e = tau_eq[idxs]
        o = {}
        # CENTRED R^2 (referee-requested label repair): the reference is the
        # held-out mean traction field, not zero.
        ref = nt.mean(0, keepdims=True)
        sst = np.square((nt - ref) / tau_sd[None, :, None]).sum()
        sst0 = np.square(nt / tau_sd[None, :, None]).sum()
        for nm, v in (("closure", c), ("eqwm", e)):
            sse = np.square((v - nt) / tau_sd[None, :, None]).sum()
            o[f"R2c_{nm}"] = float(1 - sse / sst)       # centred
            o[f"skill0_{nm}"] = float(1 - sse / sst0)   # zero-reference (legacy label)
        o["corr_closure"] = float(np.corrcoef(c.ravel(), nt.ravel())[0, 1])
        o["corr_eqwm"] = float(np.corrcoef(e.ravel(), nt.ravel())[0, 1])
        return o

    res["closure"] = {"meta": closure_meta, "apriori_train": apriori(train_idx),
                      "apriori_dev": apriori(dev_idx), "apriori_test": apriori(test_idx)}
    log(f"[closure a-priori TEST] R2c={res['closure']['apriori_test']['R2c_closure']:+.4f} "
        f"eqwm={res['closure']['apriori_test']['R2c_eqwm']:+.4f}")
    comps["tau_native_test"] = tau_pairs[test_idx]
    comps["tau_closure_test"] = tau_cl[test_idx]
    comps["tau_eqwm_test"] = tau_eq[test_idx]
    comps["tau_sd"] = tau_sd
    comps["test_idx"] = test_idx
    comps["dev_idx"] = dev_idx
    comps["train_idx"] = train_idx

    # ---------------- regions ------------------------------------------------
    src = np.zeros(48 * 96 * 48, bool); src[idx_m1] = True; src[idx_m2] = True
    src = src.reshape(48, 96, 48)
    regions = {
        "full_srcex": fluid & (~band) & (~src),
        "near_srcex": fluid & (~band) & (~src) & (dist <= .5),
        "outer_srcex": fluid & (~band) & (~src) & (dist > .5),
        "near015_srcex": fluid & (~band) & (~src) & (dist <= .15),
        "full_support_excluded": fluid & (~band),
    }
    res["_meta"]["region_sizes"] = {k: int(v.sum()) for k, v in regions.items()}
    log(res["_meta"]["region_sizes"])
    regions_t = {k: torch.from_numpy(v.reshape(-1)).to(DEV) for k, v in regions.items()}

    def sst_of(idxs):
        acc = {k: [] for k in regions}
        for i in idxs:
            t = (np.asarray(mm[i], np.float32) - mean_field) / sd.reshape(3, 1, 1, 1)
            for k, m in regions.items():
                acc[k].append(np.square(t[:, m] - t[:, m].mean()).sum())
        return {k: np.asarray(v) for k, v in acc.items()}

    sst_test = sst_of(test_idx)
    sst_dev = sst_of(dev_idx)
    for k in regions:
        comps[f"sst_test|{k}"] = sst_test[k]

    perm = torch.from_numpy(np.random.default_rng(SHUFFLE_SEED).permutation(npair)).to(DEV).long()
    cb = Conditioner(torch.from_numpy(tau_pairs).to(DEV),
                     torch.from_numpy(tau_cl).to(DEV),
                     torch.from_numpy(tau_eq).to(DEV),
                     torch.from_numpy(idx_w).to(DEV).long(),
                     torch.tensor(tau_mask[None], dtype=torch.float32, device=DEV),
                     torch.tensor(band_phys[None], dtype=torch.float32, device=DEV),
                     tau_sd_t, perm)
    faces_t = []
    for (_, s, e, nvec) in face_slices:
        faces_t.append((torch.from_numpy(idx_w[s:e]).to(DEV).long(),
                        torch.from_numpy(nvec).to(DEV)))

    # ---------------- sampler budget: DEV window, BASELINE arm rule ----------
    grid = [int(s) for s in A.diff_grid.split(",")]
    donor_dev = np.roll(dev_idx, len(dev_idx) // 2)
    chosen = {"F": A.fm_steps}
    seeds_of = {"F": [8801, 8802], "G": [9901, 9902]}

    trainers, emas = {}, {}
    for fam in ("F", "G"):
        for si, sd_seed in enumerate(seeds_of[fam][:A.seeds]):
            ck = RESULTS / f"{TAG}_{fam}_s{sd_seed}.pt"
            trainers[(fam, sd_seed)], emas[(fam, sd_seed)] = train_model(
                fam, sd_seed, mm, train_idx, mu, sd, fluid, cb, ck, metas)

    fam, s0 = "G", seeds_of["G"][0]
    sweep = {}
    for s in grid:
        tmp = {}
        r2 = score_arms(trainers[(fam, s0)], emas[(fam, s0)], mm, dev_idx, donor_dev,
                        mu, sd, mean_field, fluid, regions_t, sst_dev, cb, faces_t,
                        s, A.members, ("absent", "tau_native"), 0, tmp, f"G|dev|s{s}")
        sweep[s] = {k: v["full_srcex"] for k, v in r2.items()}
        log(f"[dev sweep G] steps={s} absent={sweep[s]['absent']:+.5f} "
            f"native={sweep[s]['tau_native']:+.5f}")
    bestv = max(sweep[s]["absent"] for s in grid)
    chosen["G"] = min(s for s in grid if sweep[s]["absent"] >= bestv - 0.01)
    res["sampler_selection"] = {
        "flow_matching_steps": int(chosen["F"]),
        "flow_matching_rule": "frozen at the published node-005/006 value; not selected here",
        "diffusion_grid": grid, "diffusion_dev_R2_full_srcex": {str(k): v for k, v in sweep.items()},
        "diffusion_selected_steps": int(chosen["G"]),
        "diffusion_rule": "smallest budget whose DEV absent-arm R2 is within 0.01 of the best",
    }
    log(f"[dev sweep G] SELECTED steps={chosen['G']}")

    # ---------------- DECISIVE evaluation ------------------------------------
    donor_test = np.roll(test_idx, len(test_idx) // 2)
    for fam in ("F", "G"):
        for si, sd_seed in enumerate(seeds_of[fam][:A.seeds]):
            arms = ARMS_FULL if si == 0 else ARMS_CORE
            r2 = score_arms(trainers[(fam, sd_seed)], emas[(fam, sd_seed)], mm,
                            test_idx, donor_test, mu, sd, mean_field, fluid,
                            regions_t, sst_test, cb, faces_t, chosen[fam],
                            A.members, arms, si, comps, f"{fam}|test|{sd_seed}")
            res["arms"][f"{fam}|s{sd_seed}"] = r2
        res["cells"][fam] = {
            "family": "F_flow_matching" if fam == "F" else "G_denoising_diffusion",
            "regime": "wall_mounted_cube_LES_3D_wall_resolved",
            "protocol": "reversed_time_frame_split",
            "sampler_steps": int(chosen[fam]),
            "seeds": [int(s) for s in seeds_of[fam][:A.seeds]],
            "n_test": int(len(test_idx)), "block_strided": int(block),
        }

    # ---------------- paired block-bootstrap deltas --------------------------
    rng = np.random.default_rng(20260801)
    bidx = block_indices(len(test_idx), block, A.boot, rng)

    def r2_boot(arm, fam, seed, key):
        sse = comps[f"{fam}|test|{seed}|sse|{arm}|{key}"]
        st = sst_test[key]
        return 1.0 - sse[bidx].sum(1) / st[bidx].sum(1)

    def pooled(arm, fam, key):
        ss = [seed for seed in seeds_of[fam][:A.seeds]
              if f"{fam}|test|{seed}|sse|{arm}|{key}" in comps]
        return np.mean([r2_boot(arm, fam, s, key) for s in ss], 0), ss

    for fam in ("F", "G"):
        for key in regions:
            for arm in ARMS_FULL:
                if f"{fam}|test|{seeds_of[fam][0]}|sse|{arm}|{key}" not in comps:
                    continue
                d, ss = pooled(arm, fam, key)
                res["deltas"].setdefault(f"{fam}|{key}", {})[f"level|{arm}"] = {
                    "mean": float(d.mean()),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "seeds": [int(x) for x in ss]}
            base, _ = pooled("absent", fam, key)
            for arm in ARMS_FULL:
                if arm == "absent" or f"{fam}|test|{seeds_of[fam][0]}|sse|{arm}|{key}" not in comps:
                    continue
                a, ss = pooled(arm, fam, key)
                dd = a - base
                res["deltas"][f"{fam}|{key}"][f"delta|{arm}-absent"] = {
                    "mean": float(dd.mean()),
                    "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                    "seeds": [int(x) for x in ss]}
            if (f"{fam}|test|{seeds_of[fam][0]}|sse|tau_shuffle|{key}" in comps):
                for ctrl in ("tau_shuffle", "tau_fartime"):
                    c, _ = pooled(ctrl, fam, key)
                    cl, _ = pooled("tau_closure", fam, key)
                    dd = cl - c
                    res["deltas"][f"{fam}|{key}"][f"delta|tau_closure-{ctrl}"] = {
                        "mean": float(dd.mean()),
                        "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}

    # wall-force endpoint (engineering consequence, near-wall)
    fx_true = np.array([float(tau_pairs[i, 0].sum()) * CELL_AREA for i in test_idx])
    comps["fx_target_test"] = fx_true
    wf = {}
    for fam in ("F", "G"):
        s0 = seeds_of[fam][0]
        for arm in ARMS_FULL:
            k = f"{fam}|test|{s0}|fx|{arm}"
            if k not in comps:
                continue
            v = comps[k]
            wf[f"{fam}|{arm}"] = {
                "corr": float(np.corrcoef(v, fx_true)[0, 1]),
                "rel_rmse": float(np.sqrt(np.square(v - fx_true).mean())
                                  / np.sqrt(np.square(fx_true).mean()))}
    res["wall_force"] = wf

    res["train_meta"] = metas
    res["wall_seconds"] = round(time.time() - t_start, 1)
    res["gpu_hours"] = round((time.time() - t_start) / 3600.0, 3)
    np.savez_compressed(COMP, **comps)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    (RESULTS / f"{TAG}_results.sha256").write_text(
        f"{sha256(OUT)}  {OUT.name}\n{sha256(COMP)}  {COMP.name}\n")
    log(f"[write] {OUT.name} {COMP.name}")
    log("=== done ===")


if __name__ == "__main__":
    main()
