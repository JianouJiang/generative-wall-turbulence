#!/usr/bin/env python3
r"""Direct native-traction conditioning: the repair of the failed E2 interface.

Protocol frozen in ``development/nodes/node_005/PREREGISTRATION_E2_DIRECT.md``
before this script contacted any held-out outcome.

Why this experiment exists
--------------------------
Node 004 measured the closure-to-generator interface in the form the field would
reach for first — wall traction expanded by an equilibrium (Reichardt) velocity
lift and hard-clamped into a generator trained on complete oracle velocity bands
— and it *failed*: even the target's own traction left the field worse than
supplying nothing.  The panel identified two confounds in that negative and both
are structural, not statistical:

  C1  the lift is an equilibrium *model* of an instantaneous field, and it
      reproduces the true band with fluctuation R^2 < 0 even given exact u_tau;
  C2  every lifted arm is out of the frozen generator's conditioning
      distribution, because that generator only ever saw dense true velocity.

This script removes both confounds by deleting the lift.  The generative model
is trained *prospectively, from scratch*, to condition **directly on the signed
wall-on-fluid traction field** — the only quantity a wall closure actually
supplies.  There is no velocity reconstruction, no equilibrium law anywhere in
the interface, and no hard clamp: the sampler runs free over the whole fluid
volume and the traction enters only as a conditioning channel.  The evaluation
is therefore in-distribution by construction.

Native traction, not an equilibrium inversion
---------------------------------------------
For every fluid cell adjacent to a no-slip surface, with wall-normal unit vector
``n`` pointing from the wall into the fluid and first-cell distance ``d``,
no-slip gives the one-sided viscous wall-on-fluid traction of Methods Eq. (2),

    tau_wf = -P_t[ nu (grad u + grad u^T) n ]  ~=  - nu u_t / d ,
    u_t    = u - (u.n) n .

The tangential projector ``P_t`` annihilates the isotropic pressure term, so this
quantity needs no pressure channel: it is fully determined by the retained
velocity record and the molecular viscosity.  The sign is the physical one — the
wall retards the fluid, so ``tau_wf`` is anti-parallel to the near-wall velocity,
matching the manuscript's ``-u_tau^2 t_hat`` convention.  Nothing here inverts a
log/Reichardt law, and nothing is labelled "oracle traction" that is not read
directly from the target field by the definition above.

Honest information statement
----------------------------
On a wall-resolved grid the native viscous traction is an invertible linear map
of the first-cell tangential velocity.  Supplying it is therefore *exactly* as
informative as supplying two velocity components on one wall-adjacent cell layer
— strictly less than the published oracle band (three components, two layers).
Every scored region excludes the entire conditioning band, so no supplied cell is
ever scored.  This experiment measures how much whole-field accuracy the
closure-supplyable surface information alone can carry.

Arms
----
Family T (traction-conditioned model, 25% conditioning dropout):
    tau_native      native signed wall-on-fluid traction        PRIMARY
    absent          conditioning mask zeroed                    PRIMARY CONTROL
    tau_trainmean   training-window time-mean traction          information-matched null
    tau_fartime     traction of a far-time donor snapshot       temporal scramble
    tau_signflip    -tau_native                                 signed-convention control
    tau_shuffle     tau_native permuted across wall cells       spatial-correspondence control
Family B (physical-wall velocity-band model, same budget/seeds):
    band_phys       oracle band on the physical-wall support    MATCHED CEILING
    absent_B        conditioning mask zeroed                    ceiling's own control
Family D (deterministic regressor, identical architecture and conditioning):
    det_tau, det_absent                                         information-matched non-generative

Primary estimand
----------------
    Delta_tau  = R^2(tau_native) - R^2(absent)        same model, same sampler noise
    Delta_band = R^2(band_phys)  - R^2(absent_B)      matched ceiling
    transmission = Delta_tau / Delta_band

Evidence boundary.  This is offline closure-supplyable-information propagation on
held-out LES.  The traction is read from the held-out target, so it is an ORACLE
traction and bounds what a closure could transmit; it is not a closure-accuracy
measurement and it is not solver-coupled WMLES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "codes" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

P = argparse.ArgumentParser()
P.add_argument("--case", default="/root/autodl-tmp/cube_les")
P.add_argument("--data-name", default="cube_ds2_float16.complete.npy")
P.add_argument("--base", type=int, default=32)
P.add_argument("--steps-train", type=int, default=20000)
P.add_argument("--batch", type=int, default=4)
P.add_argument("--sample-steps", type=int, default=32)
P.add_argument("--members", type=int, default=8)
P.add_argument("--boot", type=int, default=4000)
P.add_argument("--seeds", type=int, default=3)
P.add_argument("--nmax", type=int, default=343)
P.add_argument("--smoke", action="store_true")
P.add_argument("--tag", default="e2_direct_traction")
A = P.parse_args()

if A.smoke:
    A.steps_train, A.sample_steps, A.members, A.boot = 60, 4, 2, 200
    A.seeds, A.nmax, A.base = 1, 8, 16

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True

NU = 2.0e-4                      # cube_prod.par, yplus_preflight.json
DELTA = 1.0 / 24.0               # uniform cell size of the 48x96x48 model grid
D_ANCHOR = 0.5 * DELTA           # wall distance of the first fluid cell centre
CELL_AREA = DELTA * DELTA

# Rasterisation audit inherited from node 004 (verified time-invariant).
DUP_ROWS = (41, 43, 51, 54, 55, 57, 58, 61, 65, 67, 68, 70, 71, 72, 73, 75, 76, 77,
            78, 79, 82, 83, 84, 85, 88, 89, 90, 91, 93, 94)
ZERO_ROW = 95

# Independent spectral-element wall-load quadrature from the continuation blocks
# (codes/results/wall_loads_audit_b001.json / b002.json), PHYSICAL WALLS ONLY
# (block total minus the computational lid).  External physical validation target
# for the finite-difference traction extraction.
NATIVE_FX_VISCOUS_PHYSWALL = (-0.007418, -0.006391)

TAG = A.tag
OUT = RESULTS / f"{TAG}_results.json"
COMP = RESULTS / f"{TAG}_components.npz"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Geometry: identical to the published producer, plus the wall-face enumeration
# --------------------------------------------------------------------------
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


def wall_faces(fluid, include_lid=False):
    """Enumerate every (fluid cell, no-slip face) pair with its wall normal.

    Returns a list of dicts with a boolean cell mask and the unit normal that
    points from the wall into the fluid.  Face membership is derived from the
    solid/boundary geometry only; it never reads the flow.

    The computational lid (``top``, j = 95) is EXCLUDED by default.  Node 004
    established that the lid-adjacent raster row is identically zero at every
    retained time, so no wall observation of any kind exists there and a traction
    read from it would be an identically-zero conditioning channel over a third
    of the surface support.  Excluding it matches the physical-wall support of
    the ceiling arm, so traction and band arms see the same surfaces.
    """
    ii, jj, kk = np.arange(48), np.arange(96), np.arange(48)
    I, J, K = np.meshgrid(ii, jj, kk, indexing="ij")
    cube_i = (I >= 12) & (I <= 35)
    cube_j = (J >= 0) & (J <= 23)
    cube_k = (K >= 12) & (K <= 35)

    faces = []

    def add(name, mask, normal):
        m = mask & fluid
        if m.sum() > 0:
            faces.append({"name": name, "mask": m, "n": np.asarray(normal, np.float32)})

    add("floor", (J == 0), (0.0, 1.0, 0.0))
    if include_lid:
        add("top", (J == 95), (0.0, -1.0, 0.0))
    add("cube_top", (J == 24) & cube_i & cube_k, (0.0, 1.0, 0.0))
    add("cube_xlo", (I == 11) & cube_j & cube_k, (-1.0, 0.0, 0.0))
    add("cube_xhi", (I == 36) & cube_j & cube_k, (1.0, 0.0, 0.0))
    add("cube_zlo", (K == 11) & cube_i & cube_j, (0.0, 0.0, -1.0))
    add("cube_zhi", (K == 36) & cube_i & cube_j, (0.0, 0.0, 1.0))
    return faces


def traction_masks(faces):
    """Union mask of traction-carrying cells and the per-cell face count."""
    any_mask = np.zeros((48, 96, 48), bool)
    nface = np.zeros((48, 96, 48), np.float32)
    for f in faces:
        any_mask |= f["mask"]
        nface += f["mask"].astype(np.float32)
    return any_mask, nface


def native_traction(field, faces):
    """Signed wall-on-fluid viscous traction, Methods Eq. (2), one-sided.

    ``field`` is (3, 48, 96, 48) physical velocity.  Returns (3, 48, 96, 48)
    supported only on wall-adjacent fluid cells.  For a cell touching more than
    one no-slip face the per-face tractions are summed; the face count is carried
    separately so the map is invertible and declared.
    """
    tau = np.zeros_like(field, dtype=np.float32)
    for f in faces:
        m = f["mask"]
        n = f["n"]
        u = field[:, m]                                   # (3, Ncell)
        un = (n[:, None] * u).sum(0)                      # wall-normal component
        ut = u - n[:, None] * un[None, :]                 # tangential velocity
        tau[:, m] += (-NU / D_ANCHOR) * ut
    return tau


def integrated_fx(tau, faces):
    """Streamwise viscous wall-on-fluid force, for the external balance gate."""
    fx = 0.0
    for f in faces:
        m = f["mask"]
        fx += float(tau[0][m].sum()) * CELL_AREA
    return fx


# --------------------------------------------------------------------------
# Model: architecture identical to the published producer
# --------------------------------------------------------------------------
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


class FlowUNet3D(nn.Module):
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


# --------------------------------------------------------------------------
# Statistics, split, bootstrap
# --------------------------------------------------------------------------
def integral_tau(x: np.ndarray, maxlag=200) -> float:
    """Byte-identical to eval_cube_3d_coupling.integral_tau, so that the split,
    gap and bootstrap block of this run are INHERITED from the published
    producer rather than re-derived here."""
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


def prepare_stats(mm, train_idx, fluid):
    count = np.zeros(3, np.float64)
    sm = np.zeros(3, np.float64)
    ss = np.zeros(3, np.float64)
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


def block_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]


# --------------------------------------------------------------------------
# Conditioning construction
# --------------------------------------------------------------------------
class Conditioner:
    """Builds the (cond, cmask) pair for every arm, entirely on the GPU.

    Family T conditions on the traction field supported on the physical-wall
    cells.  Family B conditions on the oracle velocity band restricted to the
    same physical walls (node 004's leak repair).  The per-snapshot traction is
    precomputed once over the wall cells only (``tau_cells``: (N, 3, Ncell)),
    so no conditioning arithmetic happens in the training loop.
    """

    def __init__(self, family, tau_cells, wall_flat_idx, tau_mask_t, band_phys_t,
                 tau_sd_t, mu_t, sd_t, tau_mean_cells, shuffle_perm_t):
        self.family = family
        self.tau_cells = tau_cells              # (N, 3, Ncell) on DEV, physical units
        self.idx = wall_flat_idx                # (Ncell,) long, flat into 48*96*48
        self.tau_mask_t = tau_mask_t            # (1, 48, 96, 48)
        self.band_phys_t = band_phys_t          # (1, 48, 96, 48)
        self.tau_sd_t = tau_sd_t                # (3,)
        self.mu_t, self.sd_t = mu_t, sd_t
        self.tau_mean_cells = tau_mean_cells    # (3, Ncell)
        self.perm = shuffle_perm_t              # (Ncell,) long

    def _scatter(self, cells):
        """(B, 3, Ncell) physical traction -> (B, 3, 48, 96, 48) normalised."""
        b = cells.shape[0]
        out = torch.zeros((b, 3, 48 * 96 * 48), device=DEV, dtype=torch.float32)
        out.scatter_(2, self.idx.view(1, 1, -1).expand(b, 3, -1),
                     cells / self.tau_sd_t.view(1, 3, 1))
        return out.view(b, 3, 48, 96, 48)

    def build(self, arm, ids, donor_ids, field_norm):
        """``field_norm`` is the batch of channel-normalised velocity volumes."""
        b = field_norm.shape[0]
        if arm in ("absent", "absent_B", "det_absent"):
            return (torch.zeros_like(field_norm),
                    torch.zeros((b, 1, 48, 96, 48), device=DEV))
        if self.family == "B":
            m = self.band_phys_t.unsqueeze(0).expand(b, -1, -1, -1, -1)
            return field_norm * m, m
        if arm in ("tau_native", "det_tau"):
            cells = self.tau_cells[ids]
        elif arm == "tau_trainmean":
            cells = self.tau_mean_cells.unsqueeze(0).expand(b, -1, -1)
        elif arm == "tau_fartime":
            cells = self.tau_cells[donor_ids]
        elif arm == "tau_signflip":
            cells = -self.tau_cells[ids]
        elif arm == "tau_shuffle":
            cells = self.tau_cells[ids][:, :, self.perm]
        else:
            raise KeyError(arm)
        m = self.tau_mask_t.unsqueeze(0).expand(b, -1, -1, -1, -1)
        return self._scatter(cells.contiguous()), m


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_model(family, seed, mm, train_idx, mu, sd, fluid, cond_builder, ckpt):
    torch.manual_seed(seed)
    model = FlowUNet3D(A.base).to(DEV)
    if ckpt.exists() and not A.smoke:
        q = torch.load(ckpt, map_location=DEV)
        if q.get("complete", False):
            ema = FlowUNet3D(A.base).to(DEV)
            ema.load_state_dict(q["ema"])
            ema.eval()
            print(f"[resume] complete {ckpt.name}", flush=True)
            return ema, q["train_meta"]
    ema = FlowUNet3D(A.base).to(DEV)
    ema.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .99), weight_decay=1e-4)
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=DEV)
    muv = torch.tensor(mu[None, :, None, None, None], device=DEV)
    sdv = torch.tensor(sd[None, :, None, None, None], device=DEV)
    rng = np.random.default_rng(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    losses = []
    t0 = time.time()
    model.train()
    B = A.batch
    arm_true = {"T": "tau_native", "D": "det_tau", "B": "band_phys"}[family]
    for it in range(A.steps_train):
        ids = rng.choice(train_idx, size=B, replace=True)
        arr = np.stack([np.asarray(mm[i], np.float32) for i in ids])
        x0 = torch.from_numpy(arr).to(DEV)
        x0 = ((x0 - muv) / sdv) * ft
        idt = torch.from_numpy(np.asarray(ids)).to(DEV).long()
        cond, cm = cond_builder.build(arm_true, idt, idt, x0)
        # 25% conditioning dropout: the SAME network serves the absent arm, so the
        # primary contrast carries no between-model confound.
        keep = (torch.rand(B, device=DEV) > .25).float()[:, None, None, None, None]
        cond = cond * keep
        cm = cm * keep
        fb = ft.expand(B, -1, -1, -1, -1)
        with torch.autocast(device_type=DEV.type, dtype=torch.bfloat16,
                            enabled=DEV.type == "cuda"):
            if family == "D":
                tt = torch.zeros(B, device=DEV)
                pred = model(torch.zeros_like(x0), tt, cond, cm, fb)
                loss = (((pred - x0) ** 2) * ft).sum() / (ft.sum() * B * 3)
            else:
                z = torch.randn_like(x0) * ft
                tt = torch.rand(B, device=DEV)
                tv = tt[:, None, None, None, None]
                xt = (1 - tv) * x0 + tv * z
                target = (z - x0) * ft
                pred = model(xt, tt, cond, cm, fb)
                loss = (((pred - target) ** 2) * ft).sum() / (ft.sum() * B * 3)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.mul_(.9995).add_(p, alpha=5e-4)
        if (it + 1) % max(1, A.steps_train // 20) == 0:
            losses.append([it + 1, float(loss.detach())])
            print(f"[train {family}s{seed}] {it+1}/{A.steps_train} "
                  f"loss={loss.item():.5f} wall={time.time()-t0:.0f}s", flush=True)
    meta = {"family": family, "seed": seed, "steps": A.steps_train, "batch": B,
            "base": A.base, "losses": losses,
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "wall_seconds": round(time.time() - t0, 1)}
    ema.eval()
    if not A.smoke:
        torch.save({"complete": True, "ema": ema.state_dict(), "train_meta": meta}, ckpt)
    return ema, meta


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
@torch.no_grad()
def posterior_mean(model, family, truth, ids, donor_ids, mu, sd, fluid,
                   cond_builder, arm, members, steps, seed):
    b = truth.shape[0]
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=DEV)
    muv = torch.tensor(mu[None, :, None, None, None], device=DEV)
    sdv = torch.tensor(sd[None, :, None, None, None], device=DEV)
    trn = ((torch.from_numpy(truth).to(DEV) - muv) / sdv) * ft
    idt = torch.from_numpy(np.asarray(ids)).to(DEV).long()
    dnt = torch.from_numpy(np.asarray(donor_ids)).to(DEV).long()
    cond, cm = cond_builder.build(arm, idt, dnt, trn)

    if family == "D":
        fb = ft.expand(b, -1, -1, -1, -1)
        x0 = torch.zeros((b, 3, 48, 96, 48), device=DEV)
        pm = model(x0, torch.zeros(b, device=DEV), cond, cm, fb)
        return ((pm * sdv + muv) * ft).float().cpu().numpy()

    cond = cond.repeat_interleave(members, 0)
    cm = cm.repeat_interleave(members, 0)
    fb = ft.expand(b * members, -1, -1, -1, -1)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((b * members, 3, 48, 96, 48), generator=g, device=DEV) * fb
    if family == "B":
        # The band arms supply true velocity VALUES on the conditioning support,
        # so the published bridge/clamp is retained for the ceiling arm.
        zobs = x.clone()
        for j in range(steps, 0, -1):
            t, tn = j / steps, (j - 1) / steps
            tv = torch.full((b * members,), t, device=DEV)
            v = model(x, tv, cond, cm, fb)
            x = (x + (tn - t) * v) * fb
            x = x * (1 - cm) + ((1 - tn) * cond + tn * zobs) * cm
    else:
        # Traction arms: NO clamp anywhere.  The surface information enters only
        # through the conditioning channels; the sampler is free on every cell.
        for j in range(steps, 0, -1):
            t, tn = j / steps, (j - 1) / steps
            tv = torch.full((b * members,), t, device=DEV)
            v = model(x, tv, cond, cm, fb)
            x = (x + (tn - t) * v) * fb
    pm = x.reshape(b, members, 3, 48, 96, 48).mean(1)
    return ((pm * sdv + muv) * ft).float().cpu().numpy()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
FAMILY_ARMS = {
    "T": ("tau_native", "absent", "tau_trainmean", "tau_fartime",
          "tau_signflip", "tau_shuffle"),
    "B": ("band_phys", "absent_B"),
    "D": ("det_tau", "det_absent"),
}
# The load-bearing contrasts carry every training seed.  The four scramble/null
# controls are evaluated on the first seed only, inside the frozen cost ceiling;
# this is declared in the preregistration and reported with the results.
PRIMARY_ARMS = ("tau_native", "absent", "band_phys", "absent_B",
                "det_tau", "det_absent")


def main():
    t_start = time.time()
    data_path = Path(A.case) / A.data_name
    mm = np.load(data_path, mmap_mode="r")
    n = len(mm)
    fluid, band, dist, xg, yg, zg = geometry()
    faces = wall_faces(fluid)
    tau_mask, nface = traction_masks(faces)

    # Physical-wall band support: the published band minus the computational lid,
    # which node 004 showed carries an identically-zero row and a duplicate row.
    Y = np.broadcast_to(yg[None, :, None], (48, 96, 48))
    lid_side = Y > 2.0
    band_phys = band & (~lid_side)

    # --- split: the frozen published rule, byte-identical ---
    energy = np.empty(n, float)
    for i in range(n):
        a = np.asarray(mm[i], np.float32)
        energy[i] = np.square(a[:, fluid]).mean()
    tau_int = integral_tau(energy)
    ntr = int(.60 * n)
    gap = max(8, int(math.ceil(tau_int)))
    train_idx = np.arange(ntr)
    test_idx = np.arange(min(n - 1, ntr + gap), n)
    block = max(1, int(math.ceil(tau_int)))
    print(f"[split] n={n} train={len(train_idx)} gap={gap} test={len(test_idx)} "
          f"tau={tau_int:.2f}", flush=True)

    mu, sd, mean_field = prepare_stats(mm, train_idx, fluid)

    # --- precompute native traction on the wall cells once, for every snapshot ---
    wall_flat = np.flatnonzero(tau_mask.reshape(-1))
    ncell = len(wall_flat)
    tau_cells_np = np.zeros((n, 3, ncell), np.float32)
    fx_all = np.zeros(n, np.float64)
    for i in range(n):
        tau = native_traction(np.asarray(mm[i], np.float32), faces)
        tau_cells_np[i] = tau.reshape(3, -1)[:, wall_flat]
        fx_all[i] = integrated_fx(tau, faces)
    print(f"[traction] cells={ncell} precomputed for {n} snapshots", flush=True)

    # normalisation and the train-mean field come from the TRAINING WINDOW ONLY
    tau_sd = np.sqrt(np.maximum(
        np.square(tau_cells_np[train_idx].astype(np.float64)).mean((0, 2)), 1e-16)
    ).astype(np.float32)
    tau_mean_cells = tau_cells_np[train_idx].mean(0).astype(np.float32)
    print(f"[traction] train sd={tau_sd.tolist()}", flush=True)

    # --- physical validation gates on the extracted native traction ---
    fx_mean = float(fx_all[train_idx].mean())
    ratios = [fx_mean / v for v in NATIVE_FX_VISCOUS_PHYSWALL]
    # anti-parallel test: tau . u_t < 0 on EVERY physical no-slip face
    dots = {}
    probe_field = np.asarray(mm[train_idx[0]], np.float32)
    probe_tau = native_traction(probe_field, faces)
    for f in faces:
        m, nvec = f["mask"], f["n"]
        u = probe_field[:, m]
        un = (nvec[:, None] * u).sum(0)
        ut = u - nvec[:, None] * un[None, :]
        dots[f["name"]] = float((probe_tau[:, m] * ut).sum())
    gates = {
        "tangential_only_pressure_free": True,
        "tau_dot_ut_negative_all_faces": bool(all(v < 0 for v in dots.values())),
        "tau_dot_ut_per_face": dots,
        "wall_on_fluid_x_force_negative": bool(fx_mean < 0),
        "fx_viscous_mean_ds2_fd_physwall": fx_mean,
        "fx_viscous_mean_native_quadrature_physwall": list(NATIVE_FX_VISCOUS_PHYSWALL),
        "ratio_fd_to_native_quadrature": ratios,
        "ratio_within_0p5_2p0": bool(all(0.5 <= abs(r) <= 2.0 for r in ratios)),
        "traction_cells": int(ncell),
        "multi_face_cells": int((nface > 1).sum()),
        "faces_used": [f["name"] for f in faces],
        "computational_lid_excluded": True,
    }
    print(f"[gates] {json.dumps(gates)}", flush=True)
    if not gates["tau_dot_ut_negative_all_faces"]:
        raise RuntimeError("VOID: signed-traction convention gate failed")

    tau_cells_t = torch.from_numpy(tau_cells_np).to(DEV)
    wall_idx_t = torch.from_numpy(wall_flat).to(DEV).long()
    tau_sd_t = torch.from_numpy(tau_sd).to(DEV)
    tau_mean_cells_t = torch.from_numpy(tau_mean_cells).to(DEV)
    tau_mask_t = torch.tensor(tau_mask[None], dtype=torch.float32, device=DEV)
    band_phys_t = torch.tensor(band_phys[None], dtype=torch.float32, device=DEV)
    mu_t = torch.from_numpy(mu).to(DEV)
    sd_t = torch.from_numpy(sd).to(DEV)

    # --- scoring regions: every supplied cell excluded, identical across arms ---
    dup = np.zeros(96, bool)
    for r in DUP_ROWS:
        dup[r] = True
    dup[ZERO_ROW] = True
    dup3 = np.broadcast_to(dup[None, :, None], (48, 96, 48))
    regions = {
        "full_support_excluded": fluid & (~band),
        "near_support_excluded_d_le_0p5h": fluid & (~band) & (dist <= .5),
        "outer_d_gt_0p5h": fluid & (dist > .5),
        "uniq_raster_support_excluded": fluid & (~band) & (~dup3),
    }
    print({k: int(v.sum()) for k, v in regions.items()}, flush=True)

    # --- held-out targets ---
    if len(test_idx) > A.nmax:
        sel = np.linspace(0, len(test_idx) - 1, A.nmax).round().astype(int)
        eval_idx = np.asarray(test_idx)[sel]
    else:
        eval_idx = np.asarray(test_idx)
    donor_idx = np.roll(eval_idx, len(eval_idx) // 2)

    truth_all = np.stack([np.asarray(mm[i], np.float32) for i in eval_idx])
    tf = (truth_all - mean_field[None]) / sd[None, :, None, None, None]
    tbar = {k: float(tf[:, :, m].mean()) for k, m in regions.items()}
    sst = {k: np.square(tf[:, :, m] - tbar[k]).sum((1, 2)) for k, m in regions.items()}

    rng_perm = np.random.default_rng(20260731)
    shuffle_perm_t = torch.from_numpy(rng_perm.permutation(ncell)).to(DEV).long()

    seeds = [7701 + s for s in range(A.seeds)]
    comps = {}          # comps[arm][seed][region] -> per-target SSE
    train_meta = {}
    reps = {}
    for family in ("T", "B", "D"):
        fam_seeds = seeds if family != "D" else seeds[:1]
        for seed in fam_seeds:
            cb = Conditioner(family, tau_cells_t, wall_idx_t, tau_mask_t,
                             band_phys_t, tau_sd_t, mu_t, sd_t,
                             tau_mean_cells_t, shuffle_perm_t)
            ckpt = RESULTS / f"{TAG}_{family}_s{seed}.pt"
            model, meta = train_model(family, seed, mm, train_idx, mu, sd, fluid,
                                      cb, ckpt)
            train_meta[f"{family}_s{seed}"] = meta
            bs = 1 if A.smoke else 4
            for arm in FAMILY_ARMS[family]:
                if arm not in PRIMARY_ARMS and seed != fam_seeds[0]:
                    continue
                acc = {k: [] for k in regions}
                for j in range(0, len(eval_idx), bs):
                    ids = eval_idx[j:j + bs]
                    dids = donor_idx[j:j + bs]
                    truth = np.stack([np.asarray(mm[i], np.float32) for i in ids])
                    pm = posterior_mean(model, family, truth, ids, dids, mu, sd,
                                        fluid, cb, arm, A.members, A.sample_steps,
                                        seed=9100 + j)
                    pf = (pm - mean_field[None]) / sd[None, :, None, None, None]
                    tt = (truth - mean_field[None]) / sd[None, :, None, None, None]
                    for k, m in regions.items():
                        acc[k].extend(np.square(pf[:, :, m] - tt[:, :, m]).sum((1, 2)).tolist())
                    if j == 0 and seed == seeds[0]:
                        reps[arm] = pm[0]
                comps.setdefault(arm, {})[str(seed)] = {k: np.asarray(v) for k, v in acc.items()}
                r2q = 1 - np.asarray(acc["full_support_excluded"]).sum() / (sst["full_support_excluded"].sum() + 1e-12)
                print(f"[eval] fam={family} seed={seed} arm={arm} "
                      f"R2_full={r2q:+.5f} wall={time.time()-t_start:.0f}s", flush=True)

    # --- statistics: dependence-aware bootstrap + across-seed aggregation ---
    stride = max(1, int(round(np.median(np.diff(eval_idx))))) if len(eval_idx) > 1 else 1
    b_eval = max(1, int(round(block / stride)))
    b_cons = max(1, int(round(1.2551 * block / stride)))   # conservative release rule
    rng = np.random.default_rng(44)
    bix = block_indices(len(eval_idx), b_eval, A.boot, rng)
    bix_c = block_indices(len(eval_idx), b_cons, A.boot, np.random.default_rng(45))

    def boot_r2(se, k, ix):
        st = sst[k]
        return 1 - se[ix].sum(1) / (st[ix].sum(1) + 1e-12)

    result = {"arms": {}, "deltas": {}, "n_eval": int(len(eval_idx)),
              "eval_block": b_eval, "eval_block_conservative": b_cons}
    boots, boots_c, points = {}, {}, {}
    for arm, per_seed in comps.items():
        result["arms"][arm] = {}
        boots[arm], boots_c[arm], points[arm] = {}, {}, {}
        for k in regions:
            per_seed_pts = []
            se_stack = []
            for s, d in per_seed.items():
                se = d[k]
                se_stack.append(se)
                per_seed_pts.append(float(1 - se.sum() / (sst[k].sum() + 1e-12)))
            se_mean = np.mean(np.stack(se_stack), 0)     # seed-averaged SSE
            pt = float(1 - se_mean.sum() / (sst[k].sum() + 1e-12))
            br = boot_r2(se_mean, k, bix)
            brc = boot_r2(se_mean, k, bix_c)
            boots[arm][k], boots_c[arm][k], points[arm][k] = br, brc, pt
            result["arms"][arm][k] = {
                "R2_fluct_balanced": pt,
                "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
                "ci95_conservative_block": [float(np.percentile(brc, 2.5)),
                                            float(np.percentile(brc, 97.5))],
                "per_seed": per_seed_pts,
                "seed_sd": float(np.std(per_seed_pts, ddof=1)) if len(per_seed_pts) > 1 else None,
            }

    def add_delta(name, a, b):
        # Deltas are computed on the seed set the two arms SHARE, so a contrast
        # against a single-seed control is never contaminated by an unmatched
        # seed average.
        common = sorted(set(comps[a]) & set(comps[b]))
        result["deltas"][name] = {"seeds_used": common}
        for k in regions:
            sa = np.mean(np.stack([comps[a][s][k] for s in common]), 0)
            sb = np.mean(np.stack([comps[b][s][k] for s in common]), 0)
            d = boot_r2(sa, k, bix) - boot_r2(sb, k, bix)
            dc = boot_r2(sa, k, bix_c) - boot_r2(sb, k, bix_c)
            pt_a = float(1 - sa.sum() / (sst[k].sum() + 1e-12))
            pt_b = float(1 - sb.sum() / (sst[k].sum() + 1e-12))
            per_seed_d = [
                float((1 - comps[a][s][k].sum() / (sst[k].sum() + 1e-12))
                      - (1 - comps[b][s][k].sum() / (sst[k].sum() + 1e-12)))
                for s in common]
            result["deltas"][name][k] = {
                "point": float(pt_a - pt_b),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "ci95_conservative_block": [float(np.percentile(dc, 2.5)),
                                            float(np.percentile(dc, 97.5))],
                "ci_positive": bool(np.percentile(d, 2.5) > 0),
                "ci_positive_conservative": bool(np.percentile(dc, 2.5) > 0),
                "per_seed_delta": per_seed_d,
                "seed_sd": float(np.std(per_seed_d, ddof=1)) if len(per_seed_d) > 1 else None,
                "all_seeds_same_sign": bool(len(set(np.sign(per_seed_d))) == 1),
            }

    add_delta("tau_native_minus_absent", "tau_native", "absent")
    add_delta("tau_native_minus_trainmean", "tau_native", "tau_trainmean")
    add_delta("tau_native_minus_fartime", "tau_native", "tau_fartime")
    add_delta("tau_native_minus_signflip", "tau_native", "tau_signflip")
    add_delta("tau_native_minus_shuffle", "tau_native", "tau_shuffle")
    add_delta("band_phys_minus_absentB", "band_phys", "absent_B")
    add_delta("det_tau_minus_det_absent", "det_tau", "det_absent")

    # transmission ratio with a bootstrap interval
    trans = {}
    for k in regions:
        num = boots["tau_native"][k] - boots["absent"][k]
        den = boots["band_phys"][k] - boots["absent_B"][k]
        ok = np.abs(den) > 1e-9
        r = num[ok] / den[ok]
        pn = points["tau_native"][k] - points["absent"][k]
        pd = points["band_phys"][k] - points["absent_B"][k]
        trans[k] = {
            "point": float(pn / pd) if abs(pd) > 1e-9 else None,
            "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
            "denominator_ci_excludes_zero": bool(
                result["deltas"]["band_phys_minus_absentB"][k]["ci_positive"]),
        }
    result["transmission_ratio"] = trans

    meta = {
        "script": os.path.basename(__file__),
        "script_sha256": sha256(__file__),
        "data_memmap_sha256": sha256(data_path),
        "n_post_spinup": int(n),
        "n_train": int(len(train_idx)),
        "split_gap_snapshots": int(gap),
        "n_test_available": int(len(test_idx)),
        "n_eval_used": int(len(eval_idx)),
        "tau_integral_snapshots": float(tau_int),
        "effective_independent_test_events": float(len(eval_idx) * stride / tau_int),
        "seeds": seeds,
        "base_width": A.base,
        "train_updates": A.steps_train,
        "batch": A.batch,
        "posterior_members": A.members,
        "sampler_steps": A.sample_steps,
        "nu": NU,
        "d_anchor": D_ANCHOR,
        "traction_definition": "tau_wf = -nu * u_t / d, one-sided from no-slip; "
                               "tangential projection removes pressure identically",
        "traction_normalisation": "training-window per-channel RMS over wall cells",
        "tau_sd_train": tau_sd.tolist(),
        "traction_cells": int(ncell),
        "traction_support": "physical walls only (floor, cube top, four cube "
                            "vertical faces); computational lid excluded",
        "clamp_used_traction_arms": False,
        "clamp_used_band_arms": True,
        "conditioning_dropout": 0.25,
        "physical_gates": gates,
        "region_cells": {k: int(v.sum()) for k, v in regions.items()},
        "evidence_level": "offline oracle wall-traction propagation on held-out LES; "
                          "not closure-accuracy and not solver-coupled WMLES",
        "wall_seconds_total": round(time.time() - t_start, 1),
        "device": str(DEV),
    }
    payload = {"_meta": meta, "evaluation": result}
    OUT.write_text(json.dumps(payload, indent=2))

    save = {"sst_" + k: sst[k] for k in regions}
    for arm, per_seed in comps.items():
        for s, d in per_seed.items():
            for k, v in d.items():
                save[f"sse_{arm}_{s}_{k}"] = v
    for arm, f in reps.items():
        save[f"rep_{arm}"] = f
    save["truth_rep"] = truth_all[0]
    save["eval_idx"] = eval_idx
    save["mean_field"] = mean_field
    save["sd"] = sd
    np.savez_compressed(COMP, **save)
    print(f"[write] {OUT}")
    print(f"[write] {COMP}")
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
