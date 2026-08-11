#!/usr/bin/env python3
"""
eval_e2_closure_composition.py -- SOURCE-VALID closure-to-generator composition (E2).

WHAT IS NEW HERE (and why the previous two E2 runs did not establish it)
-----------------------------------------------------------------------
Node 004 fed the wall signal through an equilibrium velocity-band lift and a hard
clamp; the panel returned that negative as confounded by the adapter.
Node 005 deleted the adapter and conditioned the generator directly on wall
traction -- but that traction was read from the held-out target's own first fluid
cell.  It is therefore an ORACLE interface-feasibility result: it bounds what a
closure could transmit, and does not evaluate a closure.

This producer closes the missing link L_A: q^coarse -> tau_w.  A physics-grounded
wall closure is fitted PROSPECTIVELY on the training window only, frozen by hash,
and then used to PREDICT the signed wall traction of held-out snapshots from
matching-height outer-flow state alone.  The predicted traction -- never the
target's wall-adjacent state -- is what the generative model consumes.

    q^coarse(y_m)  --L_A(frozen closure)-->  tau_hat_w  --L_B(generator)-->  field

EVIDENCE BOUNDARY (stated here so no downstream artefact can drift from it)
--------------------------------------------------------------------------
* The closure input is the RESOLVED-LES velocity at wall distance 2.5*Delta and
  4.5*Delta -- the standard WMLES matching-height seam, i.e. state a coarse-grid
  solver carries.  This is an A-PRIORI (offline) wall-model protocol.  It is NOT
  a solver-coupled deployment: nothing here advances a momentum equation.
* Every cell the closure reads is EXCLUDED from the primary scoring region, in
  addition to every cell whose traction is supplied.  The decision rule runs on
  `full_srcex`.
* `tau_native` remains an ORACLE arm and is labelled as the traction ceiling, not
  as a closure result.

PROSPECTIVE DISCIPLINE
----------------------
* Closure development/model selection used a validation window INSIDE the
  training span (snapshots 600-659).  No held-out frame was scored during
  development; the smoke path (`--smoke`) evaluates TRAIN frames only.
* The decision rule is registered in PREREGISTRATION_E2_CLOSURE.md and evaluated
  FIRST on `CONFIRM_STRICT` = the 103 test frames that node 005 never scored
  (machine-verified set difference against its retained `eval_idx`).
* `CONFIRM_ALL` = node 005's exact 240 evaluation frames, retained so the
  mandated same-unit non-regression table can be computed.

Run only through cloud/gpu_run.sh --target foshan.
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
P.add_argument("--sample-steps", type=int, default=32)
P.add_argument("--members", type=int, default=8)
P.add_argument("--boot", type=int, default=4000)
P.add_argument("--seeds", type=int, default=3)
P.add_argument("--smoke", action="store_true")
P.add_argument("--tag", default="e2_closure_composition")
A = P.parse_args()

if A.smoke:
    A.steps_train, A.closure_steps, A.sample_steps = 60, 200, 4
    A.members, A.boot, A.seeds, A.base = 2, 200, 1, 16

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True

NU = 2.0e-4                      # cube_prod.par, yplus_preflight.json
DELTA = 1.0 / 24.0               # uniform cell size of the 48x96x48 model grid
D_ANCHOR = 0.5 * DELTA           # wall distance of the first fluid cell centre
CELL_AREA = DELTA * DELTA
KAPPA, C_REICH = 0.41, 7.8       # Reichardt constants (familyclass champion values)

# Matching-height offsets, in cells, measured from the wall-adjacent cell.
# offset m puts the sample at wall distance (m + 0.5) * DELTA.
M_OFF1, M_OFF2 = 2, 4

# Rasterisation audit inherited from node 004 (verified time-invariant).
DUP_ROWS = (41, 43, 51, 54, 55, 57, 58, 61, 65, 67, 68, 70, 71, 72, 73, 75, 76, 77,
            78, 79, 82, 83, 84, 85, 88, 89, 90, 91, 93, 94)
ZERO_ROW = 95

# Independent spectral-element wall-load quadrature from the continuation blocks
# (codes/results/wall_loads_audit_b001.json / b002.json), PHYSICAL WALLS ONLY.
NATIVE_FX_VISCOUS_PHYSWALL = (-0.007418, -0.006391)

# node 005 retained components -- used ONLY for (a) the machine-verified strict
# uncontacted split and (b) the frozen non-regression comparison.  Never for
# tuning anything in this run.
PRIOR_COMPONENTS = RESULTS / "e2_direct_traction_components.npz"

TAG = A.tag
OUT = RESULTS / f"{TAG}_results.json"
COMP = RESULTS / f"{TAG}_components.npz"
CLOSURE_CKPT = RESULTS / f"{TAG}_closure_LA.pt"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Geometry (identical to the frozen node-005 producer)
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
    """(fluid cell, no-slip face) pairs with the unit normal pointing into the fluid.

    The computational lid (j = 95) is excluded: node 004 established that its
    raster row is identically zero at every retained time.  ``step`` is the index
    increment that walks from the wall-adjacent cell into the fluid.
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
            faces.append({"name": name, "mask": m,
                          "n": np.asarray(normal, np.float32),
                          "step": tuple(int(v) for v in normal)})

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
    any_mask = np.zeros((48, 96, 48), bool)
    nface = np.zeros((48, 96, 48), np.float32)
    for f in faces:
        any_mask |= f["mask"]
        nface += f["mask"].astype(np.float32)
    return any_mask, nface


def native_traction(field, faces):
    """Signed wall-on-fluid viscous traction, Methods Eq. (2), one-sided."""
    tau = np.zeros_like(field, dtype=np.float32)
    for f in faces:
        m, n = f["mask"], f["n"]
        u = field[:, m]
        un = (n[:, None] * u).sum(0)
        ut = u - n[:, None] * un[None, :]
        tau[:, m] += (-NU / D_ANCHOR) * ut
    return tau


def integrated_fx_per_face(field, faces):
    """Streamwise viscous wall-on-fluid force, each face contribution counted ONCE.

    node 005's `integrated_fx` summed the already-face-summed cell field again for
    every face, double counting the 96 cells that touch two no-slip faces.  This
    replacement accumulates each (cell, face) pair exactly once and is the value
    compared against the independent spectral-element quadrature.
    """
    fx = 0.0
    for f in faces:
        m, n = f["mask"], f["n"]
        u = field[:, m]
        un = (n[:, None] * u).sum(0)
        ut = u - n[:, None] * un[None, :]
        fx += float((-NU / D_ANCHOR) * ut[0].sum()) * CELL_AREA
    return fx


# --------------------------------------------------------------------------
# L_A: physics-grounded wall closure on the matching-height seam
# --------------------------------------------------------------------------
def face_sampling_plan(faces):
    """Per (cell, face) pair: flat wall index, the two matching-height flat indices,
    the wall normal, and the face id.  Everything geometric, never flow-dependent."""
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


def reichardt_uplus(yp):
    return ((1.0 / KAPPA) * torch.log1p(KAPPA * yp)
            + C_REICH * (1.0 - torch.exp(-yp / 11.0) - (yp / 11.0) * torch.exp(-yp / 3.0)))


def utau_equilibrium(umag, y, nu=NU, iters=60):
    """Invert the Reichardt profile for u_tau: umag = u_tau * f(y u_tau / nu).

    Monotone in u_tau, so plain bisection on a generous bracket is unconditionally
    convergent and free of the divergence modes of Newton on this law.
    """
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
    """tau_hat = -(u_tau_eq)^2 * exp(a) * R_theta(t_hat)  -- equilibrium law plus a
    bounded, learned non-equilibrium magnitude and direction correction.

    The equilibrium term is the physics; the network only supplies a bounded
    correction, so the closure degenerates to the classical equilibrium wall model
    when the correction is switched off (`tau_eqwm`).  The sign convention (the
    wall retards the fluid) is ARCHITECTURAL: |theta| <= pi/3 < pi/2 guarantees
    tau . u_t < 0 for every cell and every parameter value.
    """

    A_MAX = 1.5
    TH_MAX = math.pi / 3.0

    def __init__(self, n_feat, n_face, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(n_face, 8)
        self.net = nn.Sequential(
            nn.Linear(n_feat + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)          # starts EXACTLY at equilibrium

    def forward(self, feats, faceid):
        h = torch.cat([feats, self.emb(faceid)], -1)
        o = self.net(h)
        a = self.A_MAX * torch.tanh(o[..., 0])
        th = self.TH_MAX * torch.tanh(o[..., 1])
        return a, th


def closure_features(field_t, plan, eqwm_only=False):
    """Build the dimensionless matching-height feature block for one snapshot.

    ``field_t`` is (3, 48, 96, 48) physical velocity on DEV.  Returns
    (feats, e1, e2, utau_eq, faceid) where (e1, e2) is the orthonormal tangent
    frame in which the correction angle is applied.
    """
    idx_w, idx_m1, idx_m2, nrm, fid, ctx_op = plan
    flat = field_t.reshape(3, -1)
    n = nrm                                                   # (Nc, 3)

    def tangential(idx):
        u = flat[:, idx].T                                    # (Nc, 3)
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

    # local spatial context: log ratio of |u_t1| to its 3x3x3 neighbourhood mean,
    # a pressure-gradient / acceleration surrogate available to a coarse solver.
    ctx = ctx_op(idx_m1, m1)

    cos12 = (ut1 * ut2).sum(-1) / (m1 * m2)
    sin12 = (torch.cross(e1, ut2, dim=-1) * n).sum(-1) / m2

    feats = torch.stack([
        torch.log1p(y1 * utau / NU),          # matching-point y+
        m2 / m1,                              # profile shape
        un1 / m1,                             # wall-normal transpiration
        un2 / m1,
        cos12, sin12,                         # skew between the two heights
        torch.log(m1.clamp_min(1e-8)),        # local speed scale
        ctx,                                  # streamwise acceleration proxy
    ], -1).float()
    if eqwm_only:
        return feats, e1, e2, utau, fid
    return feats, e1, e2, utau, fid


def make_context_op(idx_m1_np):
    """Return a callable computing log(|u_t1| / boxmean(|u_t1|)) via a 3x3x3 mean
    over the scattered matching-height cells (count-normalised, so the operator is
    exact on an irregular support)."""
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


def closure_traction(model, field_t, plan, eqwm=False):
    """Predicted signed wall traction, as (3, Ncell_pair) in physical units."""
    feats, e1, e2, utau, fid = closure_features(field_t, plan)
    if eqwm:
        a = torch.zeros_like(utau)
        th = torch.zeros_like(utau)
    else:
        a, th = model(feats, fid)
    mag = (utau ** 2) * torch.exp(a)
    direction = torch.cos(th)[:, None] * e1 + torch.sin(th)[:, None] * e2
    return (-mag[:, None] * direction).T.contiguous()          # (3, Npair)


# --------------------------------------------------------------------------
# Generator (architecture identical to the frozen node-005 producer)
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
# Statistics
# --------------------------------------------------------------------------
def integral_tau(x, maxlag=200):
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
# Conditioning
# --------------------------------------------------------------------------
class Conditioner:
    """Builds (cond, cmask) for every arm.

    Traction arms scatter a (3, Npair) physical traction onto the wall cells,
    summing the pair contributions of the 96 multi-face cells exactly as the
    native map does.  Family B supplies the oracle velocity band on the same
    physical-wall support.
    """

    def __init__(self, family, tau_cells, tau_closure_cells, tau_eqwm_cells,
                 pair_idx, tau_mask_t, band_phys_t, tau_sd_t,
                 tau_mean_cells, shuffle_perm_t):
        self.family = family
        self.tau = tau_cells                     # (N, 3, Npair) oracle native
        self.tau_c = tau_closure_cells           # (N, 3, Npair) closure predicted
        self.tau_e = tau_eqwm_cells              # (N, 3, Npair) equilibrium only
        self.idx = pair_idx                      # (Npair,) flat volume index
        self.tau_mask_t = tau_mask_t
        self.band_phys_t = band_phys_t
        self.tau_sd_t = tau_sd_t
        self.tau_mean_cells = tau_mean_cells
        self.perm = shuffle_perm_t

    def _scatter(self, cells):
        b = cells.shape[0]
        out = torch.zeros((b, 3, 48 * 96 * 48), device=DEV, dtype=torch.float32)
        out.scatter_add_(2, self.idx.view(1, 1, -1).expand(b, 3, -1),
                         cells / self.tau_sd_t.view(1, 3, 1))
        return out.view(b, 3, 48, 96, 48)

    SOURCE = {
        "tau_native": "tau", "det_tau": "tau", "tau_closure": "tau_c",
        "tau_eqwm": "tau_e", "tau_signflip": "tau", "tau_shuffle": "tau",
        "tau_fartime": "tau",
    }

    def build(self, arm, ids, donor_ids, field_norm):
        b = field_norm.shape[0]
        if arm in ("absent", "absent_B", "det_absent"):
            return (torch.zeros_like(field_norm),
                    torch.zeros((b, 1, 48, 96, 48), device=DEV))
        if self.family == "B":
            m = self.band_phys_t.unsqueeze(0).expand(b, -1, -1, -1, -1)
            return field_norm * m, m
        if arm == "tau_trainmean":
            cells = self.tau_mean_cells.unsqueeze(0).expand(b, -1, -1)
        else:
            src = getattr(self, self.SOURCE[arm])
            if arm == "tau_fartime":
                cells = src[donor_ids]
            elif arm == "tau_signflip":
                cells = -src[ids]
            elif arm == "tau_shuffle":
                cells = src[ids][:, :, self.perm]
            else:
                cells = src[ids]
        m = self.tau_mask_t.unsqueeze(0).expand(b, -1, -1, -1, -1)
        return self._scatter(cells.contiguous()), m


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
# Family K sees a DECLARED conditioning mixture so that both primary traction
# arms and the absent arm are in-distribution at evaluation time.
K_MIXTURE = (("tau_native", 0.40), ("tau_closure", 0.40), ("absent", 0.20))


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
    names = [k for k, _ in K_MIXTURE]
    probs = np.array([p for _, p in K_MIXTURE], float)
    probs = probs / probs.sum()
    for it in range(A.steps_train):
        ids = rng.choice(train_idx, size=B, replace=True)
        arr = np.stack([np.asarray(mm[i], np.float32) for i in ids])
        x0 = torch.from_numpy(arr).to(DEV)
        x0 = ((x0 - muv) / sdv) * ft
        idt = torch.from_numpy(np.asarray(ids)).to(DEV).long()
        # draw one conditioning class per batch member from the frozen mixture
        pick = rng.choice(len(names), size=B, p=probs)
        cond = torch.zeros_like(x0)
        cm = torch.zeros((B, 1, 48, 96, 48), device=DEV)
        for ci, nm in enumerate(names):
            sel = np.flatnonzero(pick == ci)
            if not len(sel):
                continue
            st = torch.from_numpy(sel).to(DEV).long()
            c, m = cond_builder.build(nm, idt[st], idt[st], x0[st])
            cond[st] = c
            cm[st] = m
        fb = ft.expand(B, -1, -1, -1, -1)
        with torch.autocast(device_type=DEV.type, dtype=torch.bfloat16,
                            enabled=DEV.type == "cuda"):
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
            "base": A.base, "losses": losses, "mixture": list(K_MIXTURE),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "wall_seconds": round(time.time() - t0, 1)}
    ema.eval()
    if not A.smoke:
        torch.save({"complete": True, "ema": ema.state_dict(), "train_meta": meta}, ckpt)
    return ema, meta


# --------------------------------------------------------------------------
# Sampling -- members retained so distributional scores are computable
# --------------------------------------------------------------------------
@torch.no_grad()
def posterior_members(model, family, truth, ids, donor_ids, mu, sd, fluid,
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
        out = ((pm * sdv + muv) * ft)
        return out.unsqueeze(1)                     # (b, 1, 3, ...)

    cond = cond.repeat_interleave(members, 0)
    cm = cm.repeat_interleave(members, 0)
    fb = ft.expand(b * members, -1, -1, -1, -1)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((b * members, 3, 48, 96, 48), generator=g, device=DEV) * fb
    if family == "B":
        zobs = x.clone()
        for j in range(steps, 0, -1):
            t, tn = j / steps, (j - 1) / steps
            tv = torch.full((b * members,), t, device=DEV)
            v = model(x, tv, cond, cm, fb)
            x = (x + (tn - t) * v) * fb
            x = x * (1 - cm) + ((1 - tn) * cond + tn * zobs) * cm
    else:
        for j in range(steps, 0, -1):
            t, tn = j / steps, (j - 1) / steps
            tv = torch.full((b * members,), t, device=DEV)
            v = model(x, tv, cond, cm, fb)
            x = (x + (tn - t) * v) * fb
    x = x.reshape(b, members, 3, 48, 96, 48)
    return (x * sdv.unsqueeze(1) + muv.unsqueeze(1)) * ft.unsqueeze(1)


# --------------------------------------------------------------------------
# Distributional scores (computed on the fly; members are never written to disk)
# --------------------------------------------------------------------------
@torch.no_grad()
def distributional_scores(members_phys, truth_t, region_t, mean_field_t, sd_t):
    """members_phys (b, m, 3, ...) and truth_t (b, 3, ...) in PHYSICAL units.

    Returns per-target CRPS, per-cell 3-vector energy score, and the rank of the
    truth among the members (for the rank histogram), all restricted to
    ``region_t`` and expressed in the same channel-normalised units as R^2.
    """
    b, m = members_phys.shape[:2]
    nrm = sd_t.view(1, 1, 3, 1, 1, 1)
    mem = (members_phys - mean_field_t.unsqueeze(0).unsqueeze(0)) / nrm
    tru = (truth_t - mean_field_t.unsqueeze(0)) / sd_t.view(1, 3, 1, 1, 1)
    sel = region_t.reshape(-1)
    mem = mem.reshape(b, m, 3, -1)[:, :, :, sel]
    tru = tru.reshape(b, 3, -1)[:, :, sel]

    ad = (mem - tru.unsqueeze(1)).abs().mean(dim=(1, 2, 3))
    pair = (mem.unsqueeze(1) - mem.unsqueeze(2)).abs().mean(dim=(3, 4))
    crps = ad - 0.5 * pair.mean(dim=(1, 2))

    de = (mem - tru.unsqueeze(1)).pow(2).sum(2).sqrt().mean(dim=(1, 2))
    dp = (mem.unsqueeze(1) - mem.unsqueeze(2)).pow(2).sum(3).sqrt().mean(dim=(3,))
    escore = de - 0.5 * dp.mean(dim=(1, 2))

    rank = (mem < tru.unsqueeze(1)).sum(1)                     # (b, 3, Ncell)
    hist = torch.stack([(rank == r).float().mean(dim=(1, 2)) for r in range(m + 1)], -1)
    return (crps.float().cpu().numpy(), escore.float().cpu().numpy(),
            hist.float().cpu().numpy())


@torch.no_grad()
def reconstructed_wall_force(members_phys, faces_t):
    """Streamwise viscous wall-on-fluid force of the posterior-mean field.

    An engineering-facing consequence: the drag the reconstructed near-wall field
    would report.  Each (cell, face) pair is counted once.
    """
    pm = members_phys.mean(1)
    fx = torch.zeros(pm.shape[0], device=pm.device)
    for m_t, n_t in faces_t:
        u = pm.reshape(pm.shape[0], 3, -1)[:, :, m_t]
        un = (n_t.view(1, 3, 1) * u).sum(1, keepdim=True)
        ut = u - n_t.view(1, 3, 1) * un
        fx += (-NU / D_ANCHOR) * ut[:, 0].sum(1) * CELL_AREA
    return fx.float().cpu().numpy()


# --------------------------------------------------------------------------
FAMILY_ARMS = {
    "K": ("tau_closure", "absent", "tau_native", "tau_eqwm", "tau_fartime",
          "tau_trainmean", "tau_shuffle", "tau_signflip"),
    "T": ("tau_closure", "tau_native", "absent"),
}
# Decision-bearing arms carry every training seed.  The declared out-of-training
# perturbation probes and the information-matched null carry the first seed only.
MULTISEED_ARMS = ("tau_closure", "absent", "tau_native", "tau_eqwm", "tau_fartime")


def main():
    t_start = time.time()
    data_path = Path(A.case) / A.data_name
    mm = np.load(data_path, mmap_mode="r")
    n = len(mm)
    fluid, band, dist, xg, yg, zg = geometry()
    faces = wall_faces(fluid)
    tau_mask, nface = traction_masks(faces)
    plan_raw = face_sampling_plan(faces)
    idx_w, idx_m1, idx_m2, nrm_np, fid_np = plan_raw
    npair = len(idx_w)

    Y = np.broadcast_to(yg[None, :, None], (48, 96, 48))
    band_phys = band & (~(Y > 2.0))

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

    # --- evaluation units: strictly uncontacted first, then node-005's exact set ---
    prior = np.load(PRIOR_COMPONENTS, allow_pickle=True)
    prior_idx = np.asarray(prior["eval_idx"])
    strict_idx = np.setdiff1d(test_idx, prior_idx)
    if A.smoke:                    # smoke NEVER touches a held-out frame
        strict_idx = train_idx[:6]
        prior_idx = train_idx[6:12]
    print(f"[units] CONFIRM_STRICT={len(strict_idx)} (never scored by node 005) "
          f"CONFIRM_ALL={len(prior_idx)} (node-005 units, for non-regression)", flush=True)

    mu, sd, mean_field = prepare_stats(mm, train_idx, fluid)

    # --- oracle native traction on every (cell, face) pair, once ---
    tau_pairs = np.zeros((n, 3, npair), np.float32)
    fx_all = np.zeros(n, np.float64)
    off = 0
    face_slices = []
    for f in faces:
        cnt = int(f["mask"].sum())
        face_slices.append((f["name"], off, off + cnt, f["n"]))
        off += cnt
    for i in range(n):
        fld = np.asarray(mm[i], np.float32)
        flat = fld.reshape(3, -1)
        for (_, s, e, nvec) in face_slices:
            u = flat[:, idx_w[s:e]]
            un = (nvec[:, None] * u).sum(0)
            ut = u - nvec[:, None] * un[None, :]
            tau_pairs[i, :, s:e] = (-NU / D_ANCHOR) * ut
        fx_all[i] = float(tau_pairs[i, 0].sum()) * CELL_AREA
    print(f"[traction] pairs={npair} unique_cells={int(tau_mask.sum())} "
          f"multiface={int((nface > 1).sum())}", flush=True)

    # The conditioning normalisation is computed on the UNIQUE-CELL summed traction,
    # byte-identically to the frozen node-005 producer, so that re-evaluating its
    # frozen Family-T checkpoints here reproduces its numbers exactly rather than
    # approximately.  (Normalising over (cell, face) pairs would double-weight the
    # 96 multi-face cells and silently break the non-regression identity.)
    uniq_flat = np.flatnonzero(tau_mask.reshape(-1))
    pair_to_uniq = np.searchsorted(uniq_flat, idx_w)
    tau_cells_uniq = np.zeros((len(train_idx), 3, len(uniq_flat)), np.float32)
    for (_, s, e, _) in face_slices:          # no duplicate cell within one face
        tau_cells_uniq[:, :, pair_to_uniq[s:e]] += tau_pairs[train_idx][:, :, s:e]
    tau_sd = np.sqrt(np.maximum(
        np.square(tau_cells_uniq.astype(np.float64)).mean((0, 2)), 1e-16)).astype(np.float32)
    del tau_cells_uniq
    tau_mean_pairs = tau_pairs[train_idx].mean(0).astype(np.float32)

    # --- physical gates (per-face force counted once; node-005 defect repaired) ---
    probe = np.asarray(mm[train_idx[0]], np.float32)
    fx_probe_perface = integrated_fx_per_face(probe, faces)
    dots, worst_cell = {}, {}
    for (nm, s, e, nvec) in face_slices:
        u = probe.reshape(3, -1)[:, idx_w[s:e]]
        un = (nvec[:, None] * u).sum(0)
        ut = u - nvec[:, None] * un[None, :]
        tt = tau_pairs[train_idx[0], :, s:e]
        cw = (tt * ut).sum(0)
        dots[nm] = float(cw.sum())
        worst_cell[nm] = float(cw.max())
    fx_mean = float(fx_all[train_idx].mean())
    ratios = [fx_mean / v for v in NATIVE_FX_VISCOUS_PHYSWALL]
    gates = {
        "per_face_pair_accounting": True,
        "multi_face_double_count_repaired": True,
        "tau_dot_ut_negative_every_cell": bool(all(v <= 0 for v in worst_cell.values())),
        "tau_dot_ut_worst_cell_per_face": worst_cell,
        "tau_dot_ut_face_sum": dots,
        "fx_viscous_mean_pairwise": fx_mean,
        "fx_probe_perface_helper": fx_probe_perface,
        "fx_viscous_native_quadrature_physwall": list(NATIVE_FX_VISCOUS_PHYSWALL),
        "ratio_fd_to_native_quadrature": ratios,
        "wall_on_fluid_x_force_negative": bool(fx_mean < 0),
    }
    print(f"[gates] {json.dumps(gates)}", flush=True)
    if not gates["tau_dot_ut_negative_every_cell"]:
        raise RuntimeError("VOID: signed-traction convention gate failed")

    # ----------------------------------------------------------------------
    # L_A: fit the closure on the TRAINING WINDOW ONLY, then freeze
    # ----------------------------------------------------------------------
    ctx_op = make_context_op(idx_m1)
    plan = (torch.from_numpy(idx_w).to(DEV).long(),
            torch.from_numpy(idx_m1).to(DEV).long(),
            torch.from_numpy(idx_m2).to(DEV).long(),
            torch.from_numpy(nrm_np).to(DEV),
            torch.from_numpy(fid_np).to(DEV).long(),
            ctx_op)
    n_faces = len(faces)
    closure = WallClosure(8, n_faces).to(DEV)
    tau_sd_t = torch.from_numpy(tau_sd).to(DEV)

    cl_fit = train_idx[:600]
    cl_val = train_idx[600:]          # validation INSIDE the training span
    if A.smoke:
        cl_fit, cl_val = train_idx[:8], train_idx[8:12]

    if CLOSURE_CKPT.exists() and not A.smoke and torch.load(
            CLOSURE_CKPT, map_location="cpu").get("complete", False):
        q = torch.load(CLOSURE_CKPT, map_location=DEV)
        closure.load_state_dict(q["model"])
        closure_meta = q["meta"]
        print("[resume] closure L_A", flush=True)
    else:
        opt = torch.optim.AdamW(closure.parameters(), lr=3e-3, weight_decay=1e-5)
        rng = np.random.default_rng(913)
        t0 = time.time()
        hist = []
        for it in range(A.closure_steps):
            i = int(rng.choice(cl_fit))
            fld = torch.from_numpy(np.asarray(mm[i], np.float32)).to(DEV)
            pred = closure_traction(closure, fld, plan)
            tgt = torch.from_numpy(tau_pairs[i]).to(DEV)
            loss = (((pred - tgt) / tau_sd_t[:, None]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
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
                        ve.append(float((((closure_traction(closure, fj, plan, eqwm=True) - tj)
                                          / tau_sd_t[:, None]) ** 2).mean()))
                hist.append([it + 1, float(loss), float(np.mean(vs)), float(np.mean(ve))])
                print(f"[closure] {it+1}/{A.closure_steps} train={loss:.5f} "
                      f"val={np.mean(vs):.5f} val_eqwm={np.mean(ve):.5f} "
                      f"wall={time.time()-t0:.0f}s", flush=True)
        closure_meta = {"steps": A.closure_steps, "history": hist,
                        "fit_snapshots": [int(cl_fit[0]), int(cl_fit[-1])],
                        "val_snapshots": [int(cl_val[0]), int(cl_val[-1])],
                        "n_parameters": sum(p.numel() for p in closure.parameters()),
                        "wall_seconds": round(time.time() - t0, 1)}
        if not A.smoke:
            torch.save({"complete": True, "model": closure.state_dict(),
                        "meta": closure_meta}, CLOSURE_CKPT)
    closure.eval()

    # --- closure a-priori quality, reported on both windows ---
    @torch.no_grad()
    def closure_quality(idxs):
        num_c = num_e = den = 0.0
        cos_c = []
        for i in idxs:
            fld = torch.from_numpy(np.asarray(mm[i], np.float32)).to(DEV)
            tgt = torch.from_numpy(tau_pairs[i]).to(DEV)
            pc = closure_traction(closure, fld, plan)
            pe = closure_traction(closure, fld, plan, eqwm=True)
            w = tau_sd_t[:, None]
            num_c += float((((pc - tgt) / w) ** 2).sum())
            num_e += float((((pe - tgt) / w) ** 2).sum())
            den += float(((tgt / w) ** 2).sum())
            cos_c.append(float(F.cosine_similarity(pc, tgt, dim=0).mean()))
        return {"R2_tau_closure": 1 - num_c / den, "R2_tau_eqwm": 1 - num_e / den,
                "mean_direction_cosine": float(np.mean(cos_c))}

    closure_apriori = {
        "train_fit_window": closure_quality(cl_fit[::20]),
        "train_val_window": closure_quality(cl_val[::2]),
        "confirm_strict": closure_quality(strict_idx[::2]),
    }
    print(f"[closure a-priori] {json.dumps(closure_apriori)}", flush=True)

    # --- predicted traction for every snapshot, from matching-height state only ---
    tau_cl_pairs = np.zeros_like(tau_pairs)
    tau_eq_pairs = np.zeros_like(tau_pairs)
    with torch.no_grad():
        for i in range(n):
            fld = torch.from_numpy(np.asarray(mm[i], np.float32)).to(DEV)
            tau_cl_pairs[i] = closure_traction(closure, fld, plan).cpu().numpy()
            tau_eq_pairs[i] = closure_traction(closure, fld, plan, eqwm=True).cpu().numpy()
    print("[closure] predicted traction materialised for all snapshots", flush=True)

    # ----------------------------------------------------------------------
    # Scoring regions.  `full_srcex` additionally removes every cell the closure
    # READS, so the decision rule never scores a cell that carried information in.
    # ----------------------------------------------------------------------
    src_mask = np.zeros(48 * 96 * 48, bool)
    src_mask[idx_m1] = True
    src_mask[idx_m2] = True
    src_mask = src_mask.reshape(48, 96, 48)
    dup = np.zeros(96, bool)
    for r in DUP_ROWS:
        dup[r] = True
    dup[ZERO_ROW] = True
    dup3 = np.broadcast_to(dup[None, :, None], (48, 96, 48))
    regions = {
        # node-005 definitions, retained verbatim for the non-regression table
        "full_support_excluded": fluid & (~band),
        "near_support_excluded_d_le_0p5h": fluid & (~band) & (dist <= .5),
        "outer_d_gt_0p5h": fluid & (dist > .5),
        "uniq_raster_support_excluded": fluid & (~band) & (~dup3),
        # closure-source-excluded regions; `full_srcex` governs the decision rule
        "full_srcex": fluid & (~band) & (~src_mask),
        "near_srcex": fluid & (~band) & (~src_mask) & (dist <= .5),
        "outer_srcex": fluid & (dist > .5) & (~src_mask),
    }
    print({k: int(v.sum()) for k, v in regions.items()}, flush=True)

    tau_pairs_t = torch.from_numpy(tau_pairs).to(DEV)
    tau_cl_t = torch.from_numpy(tau_cl_pairs).to(DEV)
    tau_eq_t = torch.from_numpy(tau_eq_pairs).to(DEV)
    pair_idx_t = torch.from_numpy(idx_w).to(DEV).long()
    tau_mask_t = torch.tensor(tau_mask[None], dtype=torch.float32, device=DEV)
    band_phys_t = torch.tensor(band_phys[None], dtype=torch.float32, device=DEV)
    tau_mean_t = torch.from_numpy(tau_mean_pairs).to(DEV)
    mean_field_t = torch.from_numpy(mean_field).to(DEV)
    sd_t = torch.from_numpy(sd).to(DEV)
    faces_t = [(torch.from_numpy(np.flatnonzero(f["mask"].reshape(-1))).to(DEV).long(),
                torch.from_numpy(f["n"]).to(DEV)) for f in faces]
    region_t = {k: torch.from_numpy(v).to(DEV) for k, v in regions.items()}

    rng_perm = np.random.default_rng(20260731)
    shuffle_perm_t = torch.from_numpy(rng_perm.permutation(npair)).to(DEV).long()

    UNITS = {"CONFIRM_STRICT": np.asarray(strict_idx),
             "CONFIRM_ALL": np.asarray(prior_idx)}

    truth_cache, sst, tbar = {}, {}, {}
    for uname, uidx in UNITS.items():
        tr = np.stack([np.asarray(mm[i], np.float32) for i in uidx])
        tf = (tr - mean_field[None]) / sd[None, :, None, None, None]
        tbar[uname] = {k: float(tf[:, :, m].mean()) for k, m in regions.items()}
        sst[uname] = {k: np.square(tf[:, :, m] - tbar[uname][k]).sum((1, 2))
                      for k, m in regions.items()}
        truth_cache[uname] = tr
        # engineering reference: the target's own reconstructed-support wall force
        gates[f"fx_target_mean_{uname}"] = float(fx_all[uidx].mean())

    seeds = [8801 + s for s in range(A.seeds)]
    comps = {u: {} for u in UNITS}
    dist_scores = {u: {} for u in UNITS}
    fx_scores = {u: {} for u in UNITS}
    ens_sens = {}
    train_meta = {}
    reps = {}

    def make_cb(family):
        return Conditioner(family, tau_pairs_t, tau_cl_t, tau_eq_t, pair_idx_t,
                           tau_mask_t, band_phys_t, tau_sd_t, tau_mean_t,
                           shuffle_perm_t)

    for family in ("K", "T"):
        fam_seeds = seeds if family == "K" else seeds[:1]
        for si, seed in enumerate(fam_seeds):
            cb = make_cb(family)
            if family == "K":
                ckpt = RESULTS / f"{TAG}_K_s{seed}.pt"
                model, meta = train_model(family, seed, mm, train_idx, mu, sd, fluid,
                                          cb, ckpt)
                train_meta[f"K_s{seed}"] = meta
            else:
                # FROZEN node-005 Family T checkpoints, reused byte-for-byte.
                ck = RESULTS / f"e2_direct_traction_T_s{7701 + si}.pt"
                if A.smoke or not ck.exists():
                    print(f"[warn] frozen T checkpoint missing: {ck.name}", flush=True)
                    continue
                q = torch.load(ck, map_location=DEV)
                model = FlowUNet3D(A.base).to(DEV)
                model.load_state_dict(q["ema"])
                model.eval()
                train_meta[f"T_frozen_s{7701+si}"] = {
                    "reused_checkpoint": ck.name, "sha256": sha256(ck),
                    "retrained": False}

            bs = 1 if A.smoke else 4
            for arm in FAMILY_ARMS[family]:
                if arm not in MULTISEED_ARMS and seed != fam_seeds[0]:
                    continue
                key = f"{family}:{arm}"
                for uname, uidx in UNITS.items():
                    # Declared out-of-training probes are scored on the strictly
                    # uncontacted unit only; they bear no decision-rule clause.
                    if arm not in MULTISEED_ARMS and uname != "CONFIRM_STRICT":
                        continue
                    donor = np.roll(uidx, max(1, len(uidx) // 2))
                    acc = {k: [] for k in regions}
                    dcr, des, dhi, dfx = [], [], [], []
                    for j in range(0, len(uidx), bs):
                        ids = uidx[j:j + bs]
                        dids = donor[j:j + bs]
                        truth = truth_cache[uname][j:j + bs]
                        mem = posterior_members(model, family, truth, ids, dids, mu, sd,
                                                fluid, cb, arm, A.members,
                                                A.sample_steps, seed=9100 + j)
                        pm = mem.mean(1).cpu().numpy()
                        pf = (pm - mean_field[None]) / sd[None, :, None, None, None]
                        tt = (truth - mean_field[None]) / sd[None, :, None, None, None]
                        for k, m in regions.items():
                            acc[k].extend(np.square(pf[:, :, m] - tt[:, :, m]).sum((1, 2)).tolist())
                        tr_t = torch.from_numpy(truth).to(DEV)
                        if mem.shape[1] > 1:
                            c, e, h = distributional_scores(mem, tr_t,
                                                            region_t["full_srcex"],
                                                            mean_field_t, sd_t)
                            dcr.extend(c.tolist()); des.extend(e.tolist()); dhi.append(h)
                        dfx.extend(reconstructed_wall_force(mem, faces_t).tolist())
                        if j == 0 and seed == fam_seeds[0] and uname == "CONFIRM_STRICT":
                            reps[key] = pm[0]
                            if mem.shape[1] > 1 and arm in ("tau_closure", "absent",
                                                            "tau_native"):
                                ens = {}
                                for mtry in (1, 2, 4, A.members):
                                    if mtry > mem.shape[1]:
                                        continue
                                    p2 = mem[:, :mtry].mean(1).cpu().numpy()
                                    p2 = (p2 - mean_field[None]) / sd[None, :, None, None, None]
                                    m0 = regions["full_srcex"]
                                    ens[str(mtry)] = float(
                                        np.square(p2[:, :, m0] - tt[:, :, m0]).sum())
                                ens_sens[key] = ens
                    comps[uname].setdefault(key, {})[str(seed)] = {
                        k: np.asarray(v) for k, v in acc.items()}
                    if dcr:
                        dist_scores[uname].setdefault(key, {})[str(seed)] = {
                            "crps": np.asarray(dcr), "energy": np.asarray(des),
                            "rank_hist": np.concatenate(dhi, 0).mean(0)}
                    fx_scores[uname].setdefault(key, {})[str(seed)] = np.asarray(dfx)
                    r2q = 1 - np.asarray(acc["full_srcex"]).sum() / (
                        sst[uname]["full_srcex"].sum() + 1e-12)
                    print(f"[eval] {key} seed={seed} unit={uname} "
                          f"R2_srcex={r2q:+.5f} wall={time.time()-t_start:.0f}s", flush=True)

    # ----------------------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------------------
    result = {"units": {}, "closure_apriori": closure_apriori,
              "closure_meta": closure_meta, "gates": gates,
              "train_meta": train_meta, "mixture": list(K_MIXTURE),
              "matching_offsets_cells": [M_OFF1, M_OFF2],
              "matching_heights_over_delta": [M_OFF1 + 0.5, M_OFF2 + 0.5],
              "ensemble_sensitivity_sse": ens_sens,
              "region_sizes": {k: int(v.sum()) for k, v in regions.items()}}

    boots_store = {}
    for uname, uidx in UNITS.items():
        stride = max(1, int(round(np.median(np.diff(uidx))))) if len(uidx) > 1 else 1
        b_eval = max(1, int(round(block / stride)))
        b_cons = max(1, int(round(1.2551 * block / stride)))
        neff = max(1.0, len(uidx) / max(1.0, block / stride))
        bix = block_indices(len(uidx), b_eval, A.boot, np.random.default_rng(44))
        bix_c = block_indices(len(uidx), b_cons, A.boot, np.random.default_rng(45))
        S = sst[uname]

        def boot_r2(se, k, ix):
            return 1 - se[ix].sum(1) / (S[k][ix].sum(1) + 1e-12)

        U = {"n_eval": int(len(uidx)), "eval_indices": [int(v) for v in uidx],
             "stride": int(stride), "block": int(b_eval),
             "block_conservative": int(b_cons), "n_effective": float(neff),
             "arms": {}, "deltas": {}, "distributional": {}, "wall_force": {}}
        boots, boots_c = {}, {}
        for key, per_seed in comps[uname].items():
            U["arms"][key] = {}
            boots[key], boots_c[key] = {}, {}
            for k in regions:
                pts, stack = [], []
                for s, d in per_seed.items():
                    stack.append(d[k])
                    pts.append(float(1 - d[k].sum() / (S[k].sum() + 1e-12)))
                se_mean = np.mean(np.stack(stack), 0)
                br, brc = boot_r2(se_mean, k, bix), boot_r2(se_mean, k, bix_c)
                boots[key][k], boots_c[key][k] = br, brc
                U["arms"][key][k] = {
                    "R2_fluct_balanced": float(1 - se_mean.sum() / (S[k].sum() + 1e-12)),
                    "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
                    "ci95_conservative_block": [float(np.percentile(brc, 2.5)),
                                                float(np.percentile(brc, 97.5))],
                    "per_seed": pts,
                    "seed_sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else None,
                }

        def add_delta(name, a, b):
            if a not in comps[uname] or b not in comps[uname]:
                return
            common = sorted(set(comps[uname][a]) & set(comps[uname][b]))
            U["deltas"][name] = {"seeds_used": common}
            for k in regions:
                sa = np.mean(np.stack([comps[uname][a][s][k] for s in common]), 0)
                sb = np.mean(np.stack([comps[uname][b][s][k] for s in common]), 0)
                d = boot_r2(sa, k, bix) - boot_r2(sb, k, bix)
                dc = boot_r2(sa, k, bix_c) - boot_r2(sb, k, bix_c)
                # crossed seed x block: resample blocks AND seeds together
                crossed = []
                for s in common:
                    ca = boot_r2(comps[uname][a][s][k], k, bix_c)
                    cb2 = boot_r2(comps[uname][b][s][k], k, bix_c)
                    crossed.append(ca - cb2)
                crossed = np.concatenate(crossed)
                pa = float(1 - sa.sum() / (S[k].sum() + 1e-12))
                pb = float(1 - sb.sum() / (S[k].sum() + 1e-12))
                per_seed_d = [float((1 - comps[uname][a][s][k].sum() / (S[k].sum() + 1e-12))
                                    - (1 - comps[uname][b][s][k].sum() / (S[k].sum() + 1e-12)))
                              for s in common]
                U["deltas"][name][k] = {
                    "delta": pa - pb,
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "ci95_conservative_block": [float(np.percentile(dc, 2.5)),
                                                float(np.percentile(dc, 97.5))],
                    "ci95_crossed_seed_block": [float(np.percentile(crossed, 2.5)),
                                                float(np.percentile(crossed, 97.5))],
                    "per_seed": per_seed_d,
                    "all_seeds_same_sign": bool(len(per_seed_d) > 0 and (
                        all(v > 0 for v in per_seed_d) or all(v < 0 for v in per_seed_d))),
                }

        add_delta("closure_minus_absent", "K:tau_closure", "K:absent")
        add_delta("native_minus_absent", "K:tau_native", "K:absent")
        add_delta("eqwm_minus_absent", "K:tau_eqwm", "K:absent")
        add_delta("closure_minus_fartime", "K:tau_closure", "K:tau_fartime")
        add_delta("closure_minus_eqwm", "K:tau_closure", "K:tau_eqwm")
        add_delta("closure_minus_native", "K:tau_closure", "K:tau_native")
        add_delta("fartime_minus_absent", "K:tau_fartime", "K:absent")
        add_delta("trainmean_minus_absent", "K:tau_trainmean", "K:absent")
        add_delta("shuffle_minus_absent", "K:tau_shuffle", "K:absent")
        add_delta("signflip_minus_absent", "K:tau_signflip", "K:absent")
        add_delta("T_closure_minus_absent", "T:tau_closure", "T:absent")
        add_delta("T_native_minus_absent", "T:tau_native", "T:absent")

        for key, per_seed in dist_scores[uname].items():
            cr = np.mean(np.stack([v["crps"] for v in per_seed.values()]), 0)
            es = np.mean(np.stack([v["energy"] for v in per_seed.values()]), 0)
            rh = np.mean(np.stack([v["rank_hist"] for v in per_seed.values()]), 0)
            unif = 1.0 / len(rh)
            U["distributional"][key] = {
                "crps_mean": float(cr.mean()),
                "crps_ci95_block": [float(np.percentile(cr[bix_c].mean(1), 2.5)),
                                    float(np.percentile(cr[bix_c].mean(1), 97.5))],
                "energy_score_mean": float(es.mean()),
                "energy_ci95_block": [float(np.percentile(es[bix_c].mean(1), 2.5)),
                                      float(np.percentile(es[bix_c].mean(1), 97.5))],
                "rank_histogram": [float(v) for v in rh],
                "reliability_deviation_L1": float(np.abs(rh - unif).sum()),
            }
        for key, per_seed in fx_scores[uname].items():
            fxm = np.mean(np.stack(list(per_seed.values())), 0)
            tgt = fx_all[uidx]
            U["wall_force"][key] = {
                "fx_recon_mean": float(fxm.mean()),
                "fx_target_mean": float(tgt.mean()),
                "relative_bias": float((fxm.mean() - tgt.mean()) / abs(tgt.mean())),
                "relative_rmse": float(np.sqrt(np.mean((fxm - tgt) ** 2)) / abs(tgt.mean())),
                "correlation": float(np.corrcoef(fxm, tgt)[0, 1]) if len(fxm) > 2 else None,
            }
        boots_store[uname] = (boots, boots_c)
        result["units"][uname] = U

    # --- non-regression against node 005 on its own units, same region defs ---
    nonreg = {}
    P = {k: prior[k] for k in prior.files if k.startswith(("sse_", "sst_"))}
    if not A.smoke and "CONFIRM_ALL" in comps and len(prior_idx) == 240:
        for k in ("full_support_excluded", "near_support_excluded_d_le_0p5h",
                  "outer_d_gt_0p5h", "uniq_raster_support_excluded"):
            st_old = P[f"sst_{k}"]
            def old_r2(arm, seed):
                key = f"sse_{arm}_{seed}_{k}"
                return float(1 - P[key].sum() / (st_old.sum() + 1e-12)) if key in P else None
            old_nat = np.mean([v for v in (old_r2("tau_native", s) for s in (7701, 7702, 7703))
                               if v is not None])
            old_abs = np.mean([v for v in (old_r2("absent", s) for s in (7701, 7702, 7703))
                               if v is not None])
            newU = result["units"]["CONFIRM_ALL"]["arms"]
            nonreg[k] = {
                "old_R2_tau_native_familyT": float(old_nat),
                "old_R2_absent_familyT": float(old_abs),
                "old_delta_native_minus_absent": float(old_nat - old_abs),
                "new_R2_tau_native_familyK": newU.get("K:tau_native", {}).get(k, {}).get("R2_fluct_balanced"),
                "new_R2_absent_familyK": newU.get("K:absent", {}).get(k, {}).get("R2_fluct_balanced"),
                "new_delta_native_minus_absent_familyK":
                    result["units"]["CONFIRM_ALL"]["deltas"].get("native_minus_absent", {}).get(k, {}).get("delta"),
                "frozen_T_reeval_R2_tau_native": newU.get("T:tau_native", {}).get(k, {}).get("R2_fluct_balanced"),
                "frozen_T_reeval_R2_absent": newU.get("T:absent", {}).get(k, {}).get("R2_fluct_balanced"),
            }
    result["non_regression"] = nonreg

    result["provenance"] = {
        "producer": Path(__file__).name,
        "producer_sha256": sha256(Path(__file__)),
        "data": str(data_path), "data_sha256": sha256(data_path),
        "prior_components_sha256": sha256(PRIOR_COMPONENTS),
        "closure_checkpoint_sha256": sha256(CLOSURE_CKPT) if CLOSURE_CKPT.exists() else None,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
        "split": {"n": int(n), "train": int(len(train_idx)), "gap": int(gap),
                  "test": int(len(test_idx)), "tau_integral": float(tau_int)},
        "wall_seconds": round(time.time() - t_start, 1),
        "gpu_hours": round((time.time() - t_start) / 3600.0, 3),
        "smoke": bool(A.smoke),
    }

    OUT.write_text(json.dumps(result, indent=1))
    save = {}
    for uname in UNITS:
        for key, per_seed in comps[uname].items():
            for s, d in per_seed.items():
                for k, v in d.items():
                    save[f"{uname}|sse|{key}|{s}|{k}"] = v
        for k, v in sst[uname].items():
            save[f"{uname}|sst|{k}"] = v
        save[f"{uname}|eval_idx"] = np.asarray(UNITS[uname])
        for key, per_seed in dist_scores[uname].items():
            for s, d in per_seed.items():
                save[f"{uname}|crps|{key}|{s}"] = d["crps"]
                save[f"{uname}|energy|{key}|{s}"] = d["energy"]
                save[f"{uname}|rank|{key}|{s}"] = d["rank_hist"]
        for key, per_seed in fx_scores[uname].items():
            for s, v in per_seed.items():
                save[f"{uname}|fx|{key}|{s}"] = v
    save["fx_all"] = fx_all
    save["mean_field"] = mean_field
    save["sd"] = sd
    save["tau_sd"] = tau_sd
    for key, v in reps.items():
        save[f"rep|{key}"] = v
    # a-priori closure scatter for the figure: one confirm-strict snapshot
    with torch.no_grad():
        i0 = int(strict_idx[0])
        f0 = torch.from_numpy(np.asarray(mm[i0], np.float32)).to(DEV)
        save["apriori_tau_true"] = tau_pairs[i0]
        save["apriori_tau_closure"] = closure_traction(closure, f0, plan).cpu().numpy()
        save["apriori_tau_eqwm"] = closure_traction(closure, f0, plan, eqwm=True).cpu().numpy()
        save["apriori_snapshot"] = np.asarray([i0])
    np.savez_compressed(COMP, **save)
    print(f"[write] {OUT.name} {COMP.name}", flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
