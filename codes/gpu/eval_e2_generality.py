#!/usr/bin/env python3
"""
eval_e2_generality.py -- does the source-valid closure->generator interface (E2)
GENERALISE across wall regimes and across generative families?

WHY THIS PRODUCER EXISTS
------------------------
Node 006 established, on ONE record (aligned wall-mounted cube, Re_h = 5000) with
ONE generative family (rectified flow / flow matching), that a physics-grounded
wall closure reading only matching-height outer state emits a signed traction that
measurably improves a conditional generative reconstruction of the surrounding
field.  All three panel seats accepted that calculation and then rejected the level
for, among other things, exactly one scientific reason: it is one record, one
regime, one family, and the frozen Level-3 venue-shape contract
(`research/success_criteria.json`) requires the propagation claim in at least two
qualitatively different wall regimes and at least two generative families.

This producer answers that, and nothing else.  It changes NO design decision of the
node-006 protocol.  It re-executes it, unchanged, in three new cells:

  cell C1  cube  x  DIFFUSION   -- SAME record, SAME frozen closure checkpoint
                                   (byte-identical, hash-gated), SAME evaluation
                                   unit, SAME arms; only the generative family is
                                   replaced by a variance-preserving denoising
                                   diffusion model.  Isolates FAMILY.
  cell C2  hills x  FLOW MATCH  -- separating periodic-hill flow at Re_h = 2800:
                                   a curved wall with mean separation and
                                   reattachment, i.e. a qualitatively different
                                   wall regime from the sharp-edged aligned cube.
                                   Closure refitted on the hill training window
                                   only.  Isolates REGIME.
  cell C3  hills x  DIFFUSION   -- both changed at once.

The two generative families are the two the manuscript already uses everywhere
(flow matching and denoising diffusion): different forward process, different
training target, different sampler.

STATISTICAL DEFECTS REPAIRED HERE (panel majors, node 006)
----------------------------------------------------------
* traction skill is reported BOTH as conventional mean-centred R^2 and as the
  zero-reference skill node 006 published, on EVERY evaluated frame (node 006
  silently scored `strict_idx[::2]`);
* `tau_rms` is named as an RMS, never as a standard deviation;
* the seed x time-block interval is a genuine HIERARCHICAL bootstrap (seeds are
  resampled with replacement, a seed-mean replicate is formed, and time blocks are
  resampled within it) -- not a concatenation of per-seed distributions;
* the energy score uses the off-diagonal M(M-1) estimator of the Methods equation;
* CRPS, energy score and the wall-force endpoint all carry PAIRED, dependence-aware
  moving-block intervals of the closure-minus-absence contrast;
* ensemble-size sensitivity is computed over the FULL evaluation unit and every
  seed, not the first batch of the first seed.

EVIDENCE BOUNDARY (binding, stated before any outcome)
------------------------------------------------------
* a-priori (offline) wall-model protocol.  Nothing here advances a momentum
  equation: this is NOT solver-coupled WMLES and must never be called deployment.
* `tau_native` is an ORACLE reference arm read from the target's own wall-adjacent
  cell.  It is not a closure result and it is not an empirical ceiling.
* every cell the closure READS and every cell whose traction is SUPPLIED is
  excluded from the region that carries the decision rule.
* the hill record is 2-D.  The tangent space at a 2-D wall is one-dimensional, so
  the closure's direction correction does not exist there by dimension; only the
  bounded magnitude correction does.  Stated, not hidden.

Run only through cloud/gpu_run.sh --target foshan.
"""
import argparse, hashlib, json, math, os, time, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
DATA = ROOT / "codes" / "data"
RESULTS.mkdir(parents=True, exist_ok=True)

P = argparse.ArgumentParser()
P.add_argument("--cube-case", default="/root/autodl-tmp/cube_les")
P.add_argument("--cube-data", default="cube_ds2_float16.complete.npy")
P.add_argument("--hill-data", default="case3_grid_64x192_full.npz")
P.add_argument("--base", type=int, default=32)
P.add_argument("--base2d", type=int, default=32)
P.add_argument("--steps-train", type=int, default=20000)
P.add_argument("--steps-train-2d", type=int, default=20000)
P.add_argument("--closure-steps", type=int, default=12000)
P.add_argument("--batch", type=int, default=4)
P.add_argument("--batch2d", type=int, default=16)
P.add_argument("--sample-steps", type=int, default=32)
P.add_argument("--members", type=int, default=8)
P.add_argument("--boot", type=int, default=4000)
P.add_argument("--seeds", type=int, default=2)
P.add_argument("--hill-eval", type=int, default=0)   # 0 = the whole inherited unit
P.add_argument("--smoke", action="store_true")
P.add_argument("--cells", default="C1,C2,C3")
P.add_argument("--tag", default="e2_generality")
A = P.parse_args()

if A.smoke:
    A.steps_train, A.steps_train_2d, A.closure_steps = 60, 60, 200
    A.sample_steps, A.members, A.boot, A.seeds = 4, 2, 200, 1
    A.base, A.base2d, A.hill_eval = 16, 16, 6

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True

CELLS = tuple(c.strip() for c in A.cells.split(",") if c.strip())

# ---- cube constants: byte-identical to eval_e2_closure_composition.py --------
NU_CUBE = 2.0e-4
DELTA_CUBE = 1.0 / 24.0
D_ANCHOR_CUBE = 0.5 * DELTA_CUBE
CELL_AREA_CUBE = DELTA_CUBE * DELTA_CUBE
KAPPA, C_REICH = 0.41, 7.8
M_OFF1, M_OFF2 = 2, 4
DUP_ROWS = (41, 43, 51, 54, 55, 57, 58, 61, 65, 67, 68, 70, 71, 72, 73, 75, 76, 77,
            78, 79, 82, 83, 84, 85, 88, 89, 90, 91, 93, 94)
ZERO_ROW = 95

# ---- hill record constants --------------------------------------------------
RE_H_HILL = 2800.0            # CoNFiLD Case3 periodic hills, Reynolds number on h

# node-006 frozen artefacts that this producer REUSES and must never rewrite
FROZEN_CLOSURE_CUBE = RESULTS / "e2_closure_composition_closure_LA.pt"
FROZEN_CUBE_COMPONENTS = RESULTS / "e2_closure_composition_components.npz"
FROZEN_CUBE_RESULTS = RESULTS / "e2_closure_composition_results.json"
NODE005_COMPONENTS = RESULTS / "e2_direct_traction_components.npz"

TAG = A.tag
OUT = RESULTS / f"{TAG}_results.json"
COMP = RESULTS / f"{TAG}_components.npz"
HILL_CLOSURE_CKPT = RESULTS / f"{TAG}_closure_LA_hill.pt"

K_MIXTURE = (("tau_native", 0.40), ("tau_closure", 0.40), ("absent", 0.20))
ARMS = ("tau_closure", "absent", "tau_native", "tau_eqwm", "tau_fartime")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


# ==========================================================================
# Shared physics: Reichardt inversion + the closure model class
# ==========================================================================
def reichardt_uplus(yp):
    return ((1.0 / KAPPA) * torch.log1p(KAPPA * yp)
            + C_REICH * (1.0 - torch.exp(-yp / 11.0) - (yp / 11.0) * torch.exp(-yp / 3.0)))


def utau_equilibrium(umag, y, nu, iters=60):
    lo = torch.full_like(umag, 1e-10)
    hi = torch.full_like(umag, 10.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pred = mid * reichardt_uplus(mid * y / nu)
        too_small = pred < umag
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


class WallClosure(nn.Module):
    """tau_hat = -(u_tau_eq)^2 exp(a) R_theta(t_hat).

    Identical class to the frozen node-006 closure.  ``n_out`` is 2 in three
    dimensions (bounded magnitude + bounded in-plane direction correction) and 1
    in two dimensions, where the tangent space is one-dimensional and the
    direction correction does not exist.  The output layer is zero-initialised, so
    training STARTS at the classical equilibrium wall model and can only depart
    from it by earning it.  |theta| <= pi/3 < pi/2 makes tau . u_t < 0 --- the wall
    retards the fluid --- an architectural property, not a diagnostic.
    """

    A_MAX = 1.5
    TH_MAX = math.pi / 3.0

    def __init__(self, n_feat, n_face, n_out=2, hidden=64):
        super().__init__()
        self.n_out = n_out
        self.emb = nn.Embedding(n_face, 8)
        self.net = nn.Sequential(
            nn.Linear(n_feat + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_out),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats, faceid):
        o = self.net(torch.cat([feats, self.emb(faceid)], -1))
        a = self.A_MAX * torch.tanh(o[..., 0])
        th = (self.TH_MAX * torch.tanh(o[..., 1]) if self.n_out > 1
              else torch.zeros_like(a))
        return a, th


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


def block_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]


def conservative_block(idx, tau):
    """Moving-block length in UNIT-INDEX space, with a degeneracy guard.

    Two corrections, both arithmetic and both independent of any outcome:

    * STRIDE.  An evaluation unit need not be contiguous in record time.  The
      cube unit is an interleaved subset with a median spacing of three record
      frames, so a decorrelation time of `tau` record frames is `tau/stride`
      *unit* indices.  The frozen node-006 producer applies exactly this
      correction (it reports `block=33`, `block_conservative=41` for the
      103-frame cube unit, not 98/123).
    * DEGENERACY.  A circular moving-block bootstrap with `block >= n` draws
      `ceil(n/block) = 1` block of length `>= n`, truncated to `n` -- i.e. a
      cyclic permutation containing every index exactly once.  Every replicate is
      then identical and the interval collapses to zero width.  Requiring at
      least three blocks keeps the estimator non-degenerate while staying
      conservative.
    """
    idx = np.asarray(idx)
    n = len(idx)
    stride = float(np.median(np.diff(idx))) if n > 1 else 1.0
    stride = max(1.0, stride)
    b = max(1, int(math.ceil(1.2551 * tau / stride)))
    # cap so that at least three blocks are drawn per replicate
    return int(max(1, min(b, max(1, (n - 1) // 2)))), stride


# ==========================================================================
# Generators.  ONE architecture, TWO generative families.
#   family "F": rectified flow / flow matching  (x_t = (1-t)x0 + t z, target z-x0)
#   family "G": variance-preserving denoising diffusion, epsilon-prediction,
#               cosine schedule, ancestral (stochastic) sampler
# ==========================================================================
def noise_embed(t, dim=128):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    def __init__(self, ci, co, conv, norm_groups=8, ce=128):
        super().__init__()
        self.n1 = nn.GroupNorm(min(norm_groups, ci), ci)
        self.c1 = conv(ci, co, 3, padding=1)
        self.e = nn.Linear(ce, co)
        self.n2 = nn.GroupNorm(min(norm_groups, co), co)
        self.c2 = conv(co, co, 3, padding=1)
        self.skip = conv(ci, co, 1) if ci != co else nn.Identity()
        self.nd = 3 if conv is nn.Conv3d else 2

    def forward(self, x, e):
        ex = self.e(e)
        ex = ex[:, :, None, None, None] if self.nd == 3 else ex[:, :, None, None]
        h = self.c1(F.silu(self.n1(x))) + ex
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Same topology in 2-D and 3-D; in-channels = 3*C + 1 (state, cond, cmask, fluid)."""

    def __init__(self, base, nd, nc):
        super().__init__()
        conv = nn.Conv3d if nd == 3 else nn.Conv2d
        tconv = nn.ConvTranspose3d if nd == 3 else nn.ConvTranspose2d
        self.nd, self.nc = nd, nc
        self.temb = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.inc = conv(2 * nc + 2, base, 3, padding=1)
        self.b0 = Block(base, base, conv)
        self.d1 = conv(base, 2 * base, 4, 2, 1)
        self.b1 = Block(2 * base, 2 * base, conv)
        self.d2 = conv(2 * base, 4 * base, 4, 2, 1)
        self.m1 = Block(4 * base, 4 * base, conv)
        self.m2 = Block(4 * base, 4 * base, conv)
        self.u1 = tconv(4 * base, 2 * base, 4, 2, 1)
        self.ub1 = Block(4 * base, 2 * base, conv)
        self.u0 = tconv(2 * base, base, 4, 2, 1)
        self.ub0 = Block(2 * base, base, conv)
        self.out = conv(base, nc, 3, padding=1)

    def forward(self, x, t, cond, cmask, fluid):
        e = self.temb(noise_embed(t))
        h0 = self.b0(self.inc(torch.cat([x, cond, cmask, fluid], 1)), e)
        h1 = self.b1(self.d1(h0), e)
        m = self.m2(self.m1(self.d2(h1), e), e)
        u1 = self.ub1(torch.cat([self.u1(m), h1], 1), e)
        u0 = self.ub0(torch.cat([self.u0(u1), h0], 1), e)
        return self.out(F.silu(u0)) * fluid


def vp_alpha_bar(t):
    """Cosine (Nichol-Dhariwal) VP schedule, continuous in t in [0, 1]."""
    s = 0.008
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    return (f / f0).clamp(1e-5, 1.0)


class Trainer:
    """Family-specific loss and sampler.  Everything else is shared."""

    def __init__(self, family):
        assert family in ("F", "G")
        self.family = family

    def loss(self, model, x0, cond, cm, fb, ft, gen_dev_rng):
        b = x0.shape[0]
        z = torch.randn_like(x0) * ft
        tt = torch.rand(b, device=x0.device)
        shape = [b] + [1] * (x0.dim() - 1)
        tv = tt.view(shape)
        if self.family == "F":
            xt = (1 - tv) * x0 + tv * z
            target = (z - x0) * ft
        else:
            ab = vp_alpha_bar(tt).view(shape)
            xt = (ab.sqrt() * x0 + (1 - ab).sqrt() * z) * ft
            target = z * ft
        pred = model(xt, tt, cond, cm, fb)
        return (((pred - target) ** 2) * ft).sum() / (ft.sum() * b * x0.shape[1])

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
        # variance-preserving ancestral sampler
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for j in range(steps):
            t, tn = ts[j], ts[j + 1]
            ab = vp_alpha_bar(t.view(1))
            abn = vp_alpha_bar(tn.view(1))
            eps = model(x, t.expand(b), cond, cm, fb)
            x0h = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)) * fb
            x0h = x0h.clamp(-6.0, 6.0)
            if j < steps - 1:
                beta = (1 - ab / abn).clamp(1e-8, 0.999)
                sig = (beta * (1 - abn) / (1 - ab).clamp_min(1e-8)).sqrt()
                noise = torch.randn(shape, generator=generator, device=device) * fb
                x = (abn.sqrt() * x0h + (1 - abn - sig ** 2).clamp_min(0).sqrt() * eps
                     + sig * noise) * fb
            else:
                x = x0h
        return x


# ==========================================================================
# Scoring helpers (shared by both regimes)
# ==========================================================================
@torch.no_grad()
def distributional_scores(members, truth, sel, nd):
    """CRPS and energy score in channel-normalised units, restricted to ``sel``.

    ``members`` (b, m, C, ...) and ``truth`` (b, C, ...) are ALREADY normalised.
    The energy score uses the off-diagonal M(M-1) estimator of the Methods
    equation (node 006 used the diagonal-inclusive M^2 form).
    """
    b, m, C = members.shape[:3]
    mem = members.reshape(b, m, C, -1)[:, :, :, sel]
    tru = truth.reshape(b, C, -1)[:, :, sel]

    ad = (mem - tru.unsqueeze(1)).abs().mean(dim=(1, 2, 3))
    pair = (mem.unsqueeze(1) - mem.unsqueeze(2)).abs().mean(dim=(3, 4))    # (b,m,m)
    if m > 1:
        off = (pair.sum(dim=(1, 2)) - torch.diagonal(pair, dim1=1, dim2=2).sum(1)) / (m * (m - 1))
    else:
        off = torch.zeros(b, device=mem.device)
    crps = ad - 0.5 * off

    de = (mem - tru.unsqueeze(1)).pow(2).sum(2).sqrt().mean(dim=(1, 2))
    dp = (mem.unsqueeze(1) - mem.unsqueeze(2)).pow(2).sum(3).sqrt().mean(dim=3)  # (b,m,m)
    if m > 1:
        dpo = (dp.sum(dim=(1, 2)) - torch.diagonal(dp, dim1=1, dim2=2).sum(1)) / (m * (m - 1))
    else:
        dpo = torch.zeros(b, device=mem.device)
    escore = de - 0.5 * dpo

    rank = (mem < tru.unsqueeze(1)).sum(1)
    hist = torch.stack([(rank == r).float().mean(dim=(1, 2)) for r in range(m + 1)], -1)
    return (crps.float().cpu().numpy(), escore.float().cpu().numpy(),
            hist.float().cpu().numpy())


def boot_r2(sse, sst, block, boot, seed_off=0):
    """Moving-block bootstrap of R^2 = 1 - sum(sse)/sum(sst) over frames."""
    rng = np.random.default_rng(20260801 + seed_off)
    ix = block_indices(len(sse), block, boot, rng)
    num = sse[ix].sum(1)
    den = sst[ix].sum(1)
    return 1.0 - num / den


def paired_block_ci(diff, block, boot, seed_off=0):
    """Dependence-aware interval of the MEAN of a paired per-frame difference."""
    rng = np.random.default_rng(770001 + seed_off)
    ix = block_indices(len(diff), block, boot, rng)
    d = diff[ix].mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float(diff.mean())


def hierarchical_seed_block_ci(sse_by_seed, sst, block, boot, seed_off=0):
    """GENUINE hierarchical bootstrap: resample SEEDS with replacement, average the
    per-frame SSE across the resampled seeds to form one replicate system, then
    resample time blocks within that replicate.  Node 006 concatenated per-seed
    block distributions, which is not a crossed estimator."""
    rng = np.random.default_rng(880011 + seed_off)
    S = len(sse_by_seed)
    stack = np.stack(sse_by_seed)                       # (S, n)
    out = np.empty(boot)
    ix = block_indices(stack.shape[1], block, boot, rng)
    pick = rng.integers(0, S, size=(boot, S))
    for b in range(boot):
        rep = stack[pick[b]].mean(0)
        out[b] = 1.0 - rep[ix[b]].sum() / sst[ix[b]].sum()
    return out


# ==========================================================================
# REGIME 1 -- the aligned wall-mounted cube (3-D).  Geometry byte-identical to
# eval_e2_closure_composition.py; only the generative family changes.
# ==========================================================================
def cube_geometry():
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


def cube_faces(fluid):
    ii, jj, kk = np.arange(48), np.arange(96), np.arange(48)
    I, J, K = np.meshgrid(ii, jj, kk, indexing="ij")
    ci = (I >= 12) & (I <= 35)
    cj = (J >= 0) & (J <= 23)
    ck = (K >= 12) & (K <= 35)
    faces = []

    def add(name, mask, normal):
        m = mask & fluid
        if m.sum() > 0:
            faces.append({"name": name, "mask": m, "n": np.asarray(normal, np.float32),
                          "step": tuple(int(v) for v in normal)})

    add("floor", (J == 0), (0.0, 1.0, 0.0))
    add("cube_top", (J == 24) & ci & ck, (0.0, 1.0, 0.0))
    add("cube_xlo", (I == 11) & cj & ck, (-1.0, 0.0, 0.0))
    add("cube_xhi", (I == 36) & cj & ck, (1.0, 0.0, 0.0))
    add("cube_zlo", (K == 11) & ci & cj, (0.0, 0.0, -1.0))
    add("cube_zhi", (K == 36) & ci & cj, (0.0, 0.0, 1.0))
    return faces


def cube_plan(faces):
    shape = (48, 96, 48)
    idx_w, idx_m1, idx_m2, nrm, fid = [], [], [], [], []
    for fi, f in enumerate(faces):
        ci, cj, ck = np.nonzero(f["mask"])
        sx, sy, sz = f["step"]
        for off, store in ((M_OFF1, idx_m1), (M_OFF2, idx_m2)):
            gi, gj, gk = ci + sx * off, cj + sy * off, ck + sz * off
            assert gi.min() >= 0 and gi.max() < 48
            assert gj.min() >= 0 and gj.max() < 96
            assert gk.min() >= 0 and gk.max() < 48
            store.append(np.ravel_multi_index((gi, gj, gk), shape))
        idx_w.append(np.ravel_multi_index((ci, cj, ck), shape))
        nrm.append(np.repeat(f["n"][None], len(ci), 0))
        fid.append(np.full(len(ci), fi, np.int64))
    return (np.concatenate(idx_w), np.concatenate(idx_m1), np.concatenate(idx_m2),
            np.concatenate(nrm, 0).astype(np.float32), np.concatenate(fid))


def make_context_op_3d(idx_m1_np):
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


def closure_features_3d(field_t, plan, nu, y1):
    idx_w, idx_m1, idx_m2, n, fid, ctx_op = plan
    flat = field_t.reshape(3, -1)

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
    utau = utau_equilibrium(m1, torch.full_like(m1, y1), nu)
    ctx = ctx_op(idx_m1, m1)
    cos12 = (ut1 * ut2).sum(-1) / (m1 * m2)
    sin12 = (torch.cross(e1, ut2, dim=-1) * n).sum(-1) / m2
    feats = torch.stack([
        torch.log1p(y1 * utau / nu), m2 / m1, un1 / m1, un2 / m1,
        cos12, sin12, torch.log(m1.clamp_min(1e-8)), ctx,
    ], -1).float()
    return feats, e1, e2, utau, fid


def closure_traction_3d(model, field_t, plan, nu, y1, eqwm=False):
    feats, e1, e2, utau, fid = closure_features_3d(field_t, plan, nu, y1)
    if eqwm:
        a = torch.zeros_like(utau)
        th = torch.zeros_like(utau)
    else:
        a, th = model(feats, fid)
    mag = (utau ** 2) * torch.exp(a)
    direction = torch.cos(th)[:, None] * e1 + torch.sin(th)[:, None] * e2
    return (-mag[:, None] * direction).T.contiguous()


# ==========================================================================
# REGIME 2 -- separating periodic-hill flow (2-D).  Same protocol, new regime.
# ==========================================================================
def hill_load():
    d = np.load(DATA / A.hill_data, allow_pickle=True)
    fields = d["fields"]                                  # (T, 2, H, W) standardized
    mask = np.asarray(d["mask"], np.float32)
    mu = np.asarray(d["mu"], np.float32)
    sd = np.asarray(d["sd"], np.float32)
    h = float(d["h"]); Lx = float(d["Lx"]); Ly = float(d["Ly"])
    return fields, mask, mu, sd, h, Lx, Ly


def hill_geometry(mask, h, Lx, Ly):
    """Wall faces on the LOWER curved wall of the periodic-hill raster.

    A (fluid cell, no-slip face) pair exists wherever the neighbouring cell in
    +-x (periodic) or -y is OUTSIDE the fluid.  The upper boundary of the raster
    is deliberately EXCLUDED: only the lower curved wall -- the wall that
    separates and reattaches, i.e. the physics under test -- carries traction.
    """
    H, W = mask.shape
    dy, dx = Ly / H, Lx / W
    fluid = mask > 0.5
    faces = []

    def add(name, m, normal, step, anchor):
        if m.sum() > 0:
            faces.append({"name": name, "mask": m & fluid,
                          "n": np.asarray(normal, np.float32),
                          "step": step, "anchor": anchor,
                          "len": dx if abs(normal[1]) > 0 else dy})

    below = np.zeros_like(fluid)
    below[1:, :] = fluid[:-1, :]
    add("lower_y", fluid & (~below), (0.0, 1.0), (1, 0), 0.5 * dy)   # n = +y
    xlo = np.roll(fluid, 1, axis=1)         # neighbour at i-1
    xhi = np.roll(fluid, -1, axis=1)        # neighbour at i+1
    add("step_xlo", fluid & (~xlo), (1.0, 0.0), (0, 1), 0.5 * dx)    # solid at -x, n = +x
    add("step_xhi", fluid & (~xhi), (-1.0, 0.0), (0, -1), 0.5 * dx)  # solid at +x, n = -x

    # wall distance (in physical units) by BFS on the raster from the wall faces
    big = 1e9
    dist = np.full((H, W), big)
    for f in faces:
        dist[f["mask"]] = np.minimum(dist[f["mask"]], f["anchor"])
    # multi-source Dijkstra-lite on a uniform grid: iterate relaxations
    for _ in range(max(H, W)):
        upd = dist.copy()
        upd[1:, :] = np.minimum(upd[1:, :], dist[:-1, :] + dy)
        upd[:-1, :] = np.minimum(upd[:-1, :], dist[1:, :] + dy)
        upd = np.minimum(upd, np.roll(dist, 1, axis=1) + dx)
        upd = np.minimum(upd, np.roll(dist, -1, axis=1) + dx)
        upd[~fluid] = big
        if np.allclose(upd, dist):
            dist = upd
            break
        dist = upd
    dist[~fluid] = 0.0
    band = fluid & (dist <= 2.01 * dy)
    return fluid, faces, dist, band, dy, dx


def hill_plan(faces, fluid):
    """Matching-height sampling plan.  Pairs whose two matching cells are not both
    fluid are DROPPED (the count is reported), because the closure cannot read a
    solid cell."""
    H, W = fluid.shape
    shape = (H, W)
    idx_w, idx_m1, idx_m2, nrm, fid = [], [], [], [], []
    kept = dropped = 0
    for fi, f in enumerate(faces):
        cj, ci = np.nonzero(f["mask"])                    # row, col
        sj, si = f["step"]
        gj1, gi1 = cj + sj * M_OFF1, (ci + si * M_OFF1) % W
        gj2, gi2 = cj + sj * M_OFF2, (ci + si * M_OFF2) % W
        ok = ((gj1 >= 0) & (gj1 < H) & (gj2 >= 0) & (gj2 < H))
        ok &= fluid[np.clip(gj1, 0, H - 1), gi1] & fluid[np.clip(gj2, 0, H - 1), gi2]
        dropped += int((~ok).sum())
        kept += int(ok.sum())
        cj, ci, gj1, gi1, gj2, gi2 = cj[ok], ci[ok], gj1[ok], gi1[ok], gj2[ok], gi2[ok]
        idx_w.append(np.ravel_multi_index((cj, ci), shape))
        idx_m1.append(np.ravel_multi_index((gj1, gi1), shape))
        idx_m2.append(np.ravel_multi_index((gj2, gi2), shape))
        nrm.append(np.repeat(f["n"][None], len(cj), 0))
        fid.append(np.full(len(cj), fi, np.int64))
    plan = (np.concatenate(idx_w), np.concatenate(idx_m1), np.concatenate(idx_m2),
            np.concatenate(nrm, 0).astype(np.float32), np.concatenate(fid))
    return plan, kept, dropped


def make_context_op_2d(idx_m1_np, shape):
    idx_t = torch.from_numpy(idx_m1_np).to(DEV).long()
    N = int(np.prod(shape))
    ones = torch.zeros(N, device=DEV)
    ones.scatter_add_(0, idx_t, torch.ones(len(idx_m1_np), device=DEV))
    k = torch.ones((1, 1, 3, 3), device=DEV)
    cnt = F.conv2d(ones.view(1, 1, *shape), k, padding=1).reshape(-1).clamp_min(1.0)

    def op(idx, mag):
        acc = torch.zeros(N, device=DEV)
        acc.scatter_add_(0, idx_t, mag)
        sm = F.conv2d(acc.view(1, 1, *shape), k, padding=1).reshape(-1) / cnt
        return torch.log(mag.clamp_min(1e-8) / sm[idx_t].clamp_min(1e-8))

    return op


def closure_features_2d(field_t, plan, nu, y1):
    """2-D restriction of the same feature block.  The tangent space is
    one-dimensional, so the out-of-plane skew feature does not exist; the signed
    ratio (u_t2 . e1)/|u_t2| replaces it and is the flow-reversal indicator that
    matters in a separating regime."""
    idx_w, idx_m1, idx_m2, n, fid, ctx_op = plan
    flat = field_t.reshape(2, -1)

    def tangential(idx):
        u = flat[:, idx].T
        un = (u * n).sum(-1, keepdim=True)
        return u - un * n, un.squeeze(-1)

    ut1, un1 = tangential(idx_m1)
    ut2, un2 = tangential(idx_m2)
    m1 = ut1.norm(dim=-1).clamp_min(1e-10)
    m2 = ut2.norm(dim=-1).clamp_min(1e-10)
    e1 = ut1 / m1[:, None]
    utau = utau_equilibrium(m1, torch.full_like(m1, y1), nu)
    ctx = ctx_op(idx_m1, m1)
    s12 = (ut2 * e1).sum(-1) / m2
    feats = torch.stack([
        torch.log1p(y1 * utau / nu), m2 / m1, un1 / m1, un2 / m1,
        s12, torch.log(m1.clamp_min(1e-10)), ctx,
    ], -1).float()
    return feats, e1, utau, fid


def closure_traction_2d(model, field_t, plan, nu, y1, eqwm=False):
    feats, e1, utau, fid = closure_features_2d(field_t, plan, nu, y1)
    if eqwm:
        a = torch.zeros_like(utau)
    else:
        a, _ = model(feats, fid)
    mag = (utau ** 2) * torch.exp(a)
    return (-mag[:, None] * e1).T.contiguous()


# ==========================================================================
# Generic conditioner
# ==========================================================================
class Conditioner:
    def __init__(self, tau, tau_c, tau_e, pair_idx, tau_mask_t, tau_rms_t, nd, shape, nc):
        self.tau, self.tau_c, self.tau_e = tau, tau_c, tau_e
        self.idx = pair_idx
        self.tau_mask_t = tau_mask_t
        self.tau_rms_t = tau_rms_t
        self.nd, self.shape, self.nc = nd, shape, nc

    SOURCE = {"tau_native": "tau", "tau_closure": "tau_c", "tau_eqwm": "tau_e",
              "tau_fartime": "tau"}

    def _scatter(self, cells):
        b = cells.shape[0]
        N = int(np.prod(self.shape))
        out = torch.zeros((b, self.nc, N), device=DEV, dtype=torch.float32)
        out.scatter_add_(2, self.idx.view(1, 1, -1).expand(b, self.nc, -1),
                         cells / self.tau_rms_t.view(1, self.nc, 1))
        return out.view(b, self.nc, *self.shape)

    def build(self, arm, ids, donor_ids, field_norm):
        b = field_norm.shape[0]
        if arm == "absent":
            return (torch.zeros_like(field_norm),
                    torch.zeros((b, 1, *self.shape), device=DEV))
        src = getattr(self, self.SOURCE[arm])
        cells = src[donor_ids] if arm == "tau_fartime" else src[ids]
        m = self.tau_mask_t.unsqueeze(0).expand(b, -1, *([-1] * self.nd))
        return self._scatter(cells.contiguous()), m


# ==========================================================================
# One evaluation cell: train the family, sample every arm, score everything.
# ==========================================================================
def run_cell(cell, family, regime, mm_get, n_total, train_idx, eval_idx, donor_idx,
             mu, sd, mean_field, fluid, regions, cond_builder, nd, shape, nc,
             faces_t, nu, subunits, block, seeds, steps_train, base, batch,
             out_components, out_meta):
    """mm_get(i) -> (nc, *shape) float32 PHYSICAL field."""
    trainer = Trainer(family)
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=DEV)
    muv = torch.from_numpy(mu).to(DEV).view(1, nc, *([1] * nd))
    sdv = torch.from_numpy(sd).to(DEV).view(1, nc, *([1] * nd))
    mean_field_t = torch.from_numpy(mean_field).to(DEV)
    sd_t = torch.from_numpy(sd).to(DEV)
    names = [k for k, _ in K_MIXTURE]
    probs = np.array([p for _, p in K_MIXTURE], float)
    probs /= probs.sum()

    # --- truth / SST on the evaluation unit -------------------------------
    truth = np.stack([mm_get(i) for i in eval_idx])
    tf = (truth - mean_field[None]) / sd.reshape(1, nc, *([1] * nd))
    sst, tbar = {}, {}
    for k, m in regions.items():
        tbar[k] = float(tf[:, :, m].mean())
        sst[k] = np.square(tf[:, :, m] - tbar[k]).sum((1, 2))
    truth_t = torch.from_numpy(truth).to(DEV)
    region_t = {k: torch.from_numpy(v.reshape(-1)).to(DEV) for k, v in regions.items()}

    for k, v in sst.items():
        out_components[f"{cell}|sst|{k}"] = v
    out_components[f"{cell}|eval_idx"] = np.asarray(eval_idx)
    out_components[f"{cell}|donor_idx"] = np.asarray(donor_idx)
    for sname, sidx in (subunits or {}).items():
        out_components[f"{cell}|subunit|{sname}"] = np.isin(np.asarray(eval_idx),
                                                            np.asarray(sidx))

    # the target's OWN streamwise viscous wall force on the same support -- the
    # reference every reconstructed force is compared against
    fx_tgt = torch.zeros(len(eval_idx), device=DEV)
    for (m_t, n_t, an, ln) in faces_t:
        u = truth_t.reshape(len(eval_idx), nc, -1)[:, :, m_t]
        un = (n_t.view(1, nc, 1) * u).sum(1, keepdim=True)
        ut = u - n_t.view(1, nc, 1) * un
        fx_tgt += (-nu / an) * ut[:, 0].sum(1) * ln
    out_components[f"{cell}|fx_target"] = fx_tgt.float().cpu().numpy()

    for si, seed in enumerate(seeds):
        ckpt = RESULTS / f"{TAG}_{cell}_s{seed}.pt"
        torch.manual_seed(seed)
        model = UNet(base, nd, nc).to(DEV)
        ema = UNet(base, nd, nc).to(DEV)
        ema.load_state_dict(model.state_dict())
        loaded = False
        if ckpt.exists() and not A.smoke:
            q = torch.load(ckpt, map_location=DEV)
            if q.get("complete", False):
                ema.load_state_dict(q["ema"])
                out_meta[f"{cell}|s{seed}"] = q["train_meta"]
                loaded = True
                log(f"[resume] {ckpt.name}")
        if not loaded:
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .99),
                                    weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
            rng = np.random.default_rng(seed)
            losses, t0 = [], time.time()
            model.train()
            for it in range(steps_train):
                ids = rng.choice(train_idx, size=batch, replace=True)
                x0 = torch.from_numpy(np.stack([mm_get(i) for i in ids])).to(DEV)
                x0 = ((x0 - muv) / sdv) * ft
                idt = torch.from_numpy(np.asarray(ids)).to(DEV).long()
                pick = rng.choice(len(names), size=batch, p=probs)
                cond = torch.zeros_like(x0)
                cm = torch.zeros((batch, 1, *shape), device=DEV)
                for cidx, nm in enumerate(names):
                    sel = np.flatnonzero(pick == cidx)
                    if not len(sel):
                        continue
                    st = torch.from_numpy(sel).to(DEV).long()
                    c, m = cond_builder.build(nm, idt[st], idt[st], x0[st])
                    cond[st] = c
                    cm[st] = m
                fb = ft.expand(batch, *([-1] * (nd + 1)))
                with torch.autocast(device_type=DEV.type, dtype=torch.bfloat16,
                                    enabled=DEV.type == "cuda"):
                    loss = trainer.loss(model, x0, cond, cm, fb, ft, None)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), model.parameters()):
                        pe.mul_(.9995).add_(p, alpha=5e-4)
                if (it + 1) % max(1, steps_train // 10) == 0:
                    losses.append([it + 1, float(loss.detach())])
                    log(f"[train {cell} s{seed}] {it+1}/{steps_train} "
                        f"loss={loss.item():.5f} wall={time.time()-t0:.0f}s")
            meta = {"cell": cell, "family": family, "regime": regime, "seed": seed,
                    "steps": steps_train, "batch": batch, "base": base,
                    "losses": losses, "mixture": list(K_MIXTURE),
                    "n_parameters": sum(p.numel() for p in model.parameters()),
                    "wall_seconds": round(time.time() - t0, 1)}
            out_meta[f"{cell}|s{seed}"] = meta
            if not A.smoke:
                torch.save({"complete": True, "ema": ema.state_dict(),
                            "train_meta": meta}, ckpt)
        ema.eval()

        # ---- sample every arm on the evaluation unit ----------------------
        bs = max(1, min(4 if nd == 3 else 16, len(eval_idx)))
        for arm in ARMS:
            sse = {k: [] for k in regions}
            crps_l, es_l, hist_l, fx_l = [], [], [], []
            ens_l = []
            for b0 in range(0, len(eval_idx), bs):
                ids = np.asarray(eval_idx[b0:b0 + bs])
                dids = np.asarray(donor_idx[b0:b0 + bs])
                tr = truth_t[b0:b0 + bs]
                trn = ((tr - muv) / sdv) * ft
                idt = torch.from_numpy(ids).to(DEV).long()
                dnt = torch.from_numpy(dids).to(DEV).long()
                cond, cm = cond_builder.build(arm, idt, dnt, trn)
                bb = len(ids)
                M = A.members
                condr = cond.repeat_interleave(M, 0)
                cmr = cm.repeat_interleave(M, 0)
                fb = ft.expand(bb * M, *([-1] * (nd + 1)))
                g = torch.Generator(device=DEV).manual_seed(9100 + b0 + 7919 * si)
                x = trainer.sample(ema, (bb * M, nc, *shape), condr, cmr, fb,
                                   A.sample_steps, g, DEV)
                x = x.reshape(bb, M, nc, *shape)
                phys = (x * sdv.unsqueeze(1) + muv.unsqueeze(1)) * ft.unsqueeze(1)
                pm = phys.mean(1)
                # normalised residuals for R^2 (posterior mean, as published)
                mf = mean_field_t.unsqueeze(0)
                pn = (pm - mf) / sd_t.view(1, nc, *([1] * nd))
                tn = (tr - mf) / sd_t.view(1, nc, *([1] * nd))
                for k in regions:
                    sel = region_t[k]
                    r = (pn.reshape(bb, nc, -1)[:, :, sel]
                         - tn.reshape(bb, nc, -1)[:, :, sel])
                    sse[k].append(np.square(r.float().cpu().numpy()).sum((1, 2)))
                memn = (phys - mf.unsqueeze(0)) / sd_t.view(1, 1, nc, *([1] * nd))
                c, e, h = distributional_scores(memn, tn, region_t["full_srcex"], nd)
                crps_l.append(c); es_l.append(e); hist_l.append(h)
                # ensemble-size sensitivity on the FULL unit
                row = []
                for mm_ in (1, 2, 4, A.members):
                    if mm_ > M:
                        row.append(np.nan); continue
                    pmm = phys[:, :mm_].mean(1)
                    pnn = (pmm - mf) / sd_t.view(1, nc, *([1] * nd))
                    sel = region_t["full_srcex"]
                    rr = (pnn.reshape(bb, nc, -1)[:, :, sel]
                          - tn.reshape(bb, nc, -1)[:, :, sel])
                    row.append(float(np.square(rr.float().cpu().numpy()).sum()))
                ens_l.append(np.asarray(row))
                # wall force of the reconstructed field
                fx = torch.zeros(bb, device=DEV)
                for (m_t, n_t, an, ln) in faces_t:
                    u = pm.reshape(bb, nc, -1)[:, :, m_t]
                    un = (n_t.view(1, nc, 1) * u).sum(1, keepdim=True)
                    ut = u - n_t.view(1, nc, 1) * un
                    fx += (-nu / an) * ut[:, 0].sum(1) * ln
                fx_l.append(fx.float().cpu().numpy())
            for k in regions:
                out_components[f"{cell}|sse|{arm}|{seed}|{k}"] = np.concatenate(sse[k])
            out_components[f"{cell}|crps|{arm}|{seed}"] = np.concatenate(crps_l)
            out_components[f"{cell}|escore|{arm}|{seed}"] = np.concatenate(es_l)
            out_components[f"{cell}|rankhist|{arm}|{seed}"] = np.concatenate(hist_l)
            out_components[f"{cell}|fx|{arm}|{seed}"] = np.concatenate(fx_l)
            out_components[f"{cell}|ens|{arm}|{seed}"] = np.stack(ens_l)
            log(f"[score {cell} s{seed}] {arm} done "
                f"R2_full_srcex={1 - np.concatenate(sse['full_srcex']).sum() / sst['full_srcex'].sum():.5f}")
    return sst


# ==========================================================================
def main():
    t_start = time.time()
    prov = {"argv": vars(A), "device": str(DEV), "cells": list(CELLS),
            "torch": torch.__version__, "smoke": bool(A.smoke)}
    results = {"provenance": prov, "cells": {}, "gates": {}}
    comps, metas = {}, {}

    # ======================================================================
    # cube path -- reuses the FROZEN node-006 closure, byte-identical
    # ======================================================================
    if "C1" in CELLS:
        log("=== regime CUBE, family G (denoising diffusion) ===")
        mm = np.load(Path(A.cube_case) / A.cube_data, mmap_mode="r")
        n = len(mm)
        fluid, band, dist, xg, yg, zg = cube_geometry()
        faces = cube_faces(fluid)
        idx_w, idx_m1, idx_m2, nrm_np, fid_np = cube_plan(faces)
        npair = len(idx_w)
        tau_mask = np.zeros((48, 96, 48), bool)
        for f in faces:
            tau_mask |= f["mask"]

        energy = np.empty(n, float)
        for i in range(n):
            a = np.asarray(mm[i], np.float32)
            energy[i] = np.square(a[:, fluid]).mean()
        tau_int = integral_tau(energy)
        ntr = int(.60 * n)
        gap = max(8, int(math.ceil(tau_int)))
        train_idx = np.arange(ntr)
        test_idx = np.arange(min(n - 1, ntr + gap), n)
        prior = np.load(NODE005_COMPONENTS, allow_pickle=True)
        strict_idx = np.setdiff1d(test_idx, np.asarray(prior["eval_idx"]))
        if A.smoke:
            strict_idx = train_idx[:6]
        block_c, stride_c = conservative_block(strict_idx, tau_int)
        log(f"[cube split] n={n} train={ntr} tau={tau_int:.2f} "
            f"eval={len(strict_idx)} stride={stride_c} block={block_c}")

        mu = np.zeros(3); sdv_ = np.zeros(3); ss = np.zeros(3); cnt = np.zeros(3)
        mean_field = np.zeros((3, 48, 96, 48), np.float64)
        for i in train_idx:
            x = np.asarray(mm[i], np.float32)
            mean_field += x
            for c in range(3):
                v = x[c][fluid]
                cnt[c] += v.size; mu[c] += v.sum(dtype=np.float64)
                ss[c] += np.square(v, dtype=np.float64).sum(dtype=np.float64)
        mean_field /= len(train_idx)
        mu = mu / cnt
        sdv_ = np.sqrt(np.maximum(ss / cnt - mu * mu, 1e-8))
        mu = mu.astype(np.float32); sdv_ = sdv_.astype(np.float32)
        mean_field = mean_field.astype(np.float32)

        face_slices, off = [], 0
        for f in faces:
            c = int(f["mask"].sum())
            face_slices.append((f["name"], off, off + c, f["n"]))
            off += c
        tau_pairs = np.zeros((n, 3, npair), np.float32)
        for i in range(n):
            flat = np.asarray(mm[i], np.float32).reshape(3, -1)
            for (_, s, e, nv) in face_slices:
                u = flat[:, idx_w[s:e]]
                un = (nv[:, None] * u).sum(0)
                ut = u - nv[:, None] * un[None, :]
                tau_pairs[i, :, s:e] = (-NU_CUBE / D_ANCHOR_CUBE) * ut

        uniq_flat = np.flatnonzero(tau_mask.reshape(-1))
        p2u = np.searchsorted(uniq_flat, idx_w)
        tcu = np.zeros((len(train_idx), 3, len(uniq_flat)), np.float32)
        for (_, s, e, _) in face_slices:
            tcu[:, :, p2u[s:e]] += tau_pairs[train_idx][:, :, s:e]
        # RMS (not a standard deviation) of the unique-cell summed traction, the
        # frozen node-005/006 conditioning normalisation.
        tau_rms = np.sqrt(np.maximum(
            np.square(tcu.astype(np.float64)).mean((0, 2)), 1e-16)).astype(np.float32)
        del tcu

        # ---- FROZEN closure: load and hash-gate --------------------------
        if not FROZEN_CLOSURE_CUBE.exists():
            raise RuntimeError("VOID: frozen node-006 closure checkpoint missing")
        closure_hash = sha256(FROZEN_CLOSURE_CUBE)
        q = torch.load(FROZEN_CLOSURE_CUBE, map_location=DEV)
        closure = WallClosure(8, len(faces), n_out=2).to(DEV)
        closure.load_state_dict(q["model"])
        closure.eval()
        results["gates"]["cube_frozen_closure_sha256"] = closure_hash
        results["gates"]["cube_frozen_closure_reused_unmodified"] = True

        ctx_op = make_context_op_3d(idx_m1)
        plan = (torch.from_numpy(idx_w).to(DEV).long(),
                torch.from_numpy(idx_m1).to(DEV).long(),
                torch.from_numpy(idx_m2).to(DEV).long(),
                torch.from_numpy(nrm_np).to(DEV),
                torch.from_numpy(fid_np).to(DEV).long(), ctx_op)
        y1_cube = (M_OFF1 + 0.5) * DELTA_CUBE

        tau_cl = np.zeros_like(tau_pairs)
        tau_eq = np.zeros_like(tau_pairs)
        with torch.no_grad():
            for i in range(n):
                fld = torch.from_numpy(np.asarray(mm[i], np.float32)).to(DEV)
                tau_cl[i] = closure_traction_3d(closure, fld, plan, NU_CUBE, y1_cube).cpu().numpy()
                tau_eq[i] = closure_traction_3d(closure, fld, plan, NU_CUBE, y1_cube,
                                                eqwm=True).cpu().numpy()
        log("[cube] frozen-closure traction materialised")

        # ---- closure a-priori skill, BOTH definitions, ALL eval frames ----
        results["cells"].setdefault("C1", {})["closure_apriori"] = closure_apriori_scores(
            tau_cl, tau_eq, tau_pairs, strict_idx, tau_rms)

        src = np.zeros(48 * 96 * 48, bool)
        src[idx_m1] = True; src[idx_m2] = True
        src = src.reshape(48, 96, 48)
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
            "full_srcex": fluid & (~band) & (~src),
            "near_srcex": fluid & (~band) & (~src) & (dist <= .5),
            "outer_srcex": fluid & (dist > .5) & (~src),
        }
        cb = Conditioner(torch.from_numpy(tau_pairs).to(DEV),
                         torch.from_numpy(tau_cl).to(DEV),
                         torch.from_numpy(tau_eq).to(DEV),
                         torch.from_numpy(idx_w).to(DEV).long(),
                         torch.tensor(tau_mask[None], dtype=torch.float32, device=DEV),
                         torch.from_numpy(tau_rms).to(DEV), 3, (48, 96, 48), 3)
        faces_t = [(torch.from_numpy(np.flatnonzero(f["mask"].reshape(-1))).to(DEV).long(),
                    torch.from_numpy(f["n"]).to(DEV), D_ANCHOR_CUBE, CELL_AREA_CUBE)
                   for f in faces]
        donor = np.roll(np.asarray(strict_idx), len(strict_idx) // 2)

        run_cell("C1", "G", "cube", lambda i: np.asarray(mm[i], np.float32), n,
                 train_idx, np.asarray(strict_idx), donor, mu, sdv_, mean_field,
                 fluid, regions, cb, 3, (48, 96, 48), 3, faces_t, NU_CUBE,
                 {"MATCHED_NODE006_UNIT": np.asarray(strict_idx)}, block_c,
                 [9801 + s for s in range(A.seeds)], A.steps_train, A.base,
                 A.batch, comps, metas)
        results["cells"]["C1"].update({
            "regime": "cube_aligned_Re_h_5000_3D", "family": "G_denoising_diffusion",
            "n_eval": int(len(strict_idx)), "block": int(block_c),
            "tau_integral": float(tau_int),
            "n_effective": float(len(strict_idx) / max(1.0, tau_int)),
            "seeds": [9801 + s for s in range(A.seeds)],
            "closure": "node-006 frozen L_A, byte-identical",
            "tau_rms_note": "conditioning normalisation is an RMS of the unique-cell summed traction, not a standard deviation",
        })
        del tau_pairs, tau_cl, tau_eq
        torch.cuda.empty_cache() if DEV.type == "cuda" else None

    # ======================================================================
    # hill path -- NEW regime, closure refitted on the hill training window
    # ======================================================================
    if "C2" in CELLS or "C3" in CELLS:
        log("=== regime HILLS (separating periodic hills, Re_h=2800) ===")
        fields, maskf, hmu, hsd, h, Lx, Ly = hill_load()
        T = len(fields)
        H, W = maskf.shape
        fluid, faces, dist, band, dy, dx = hill_geometry(maskf, h, Lx, Ly)
        plan_np, kept, dropped = hill_plan(faces, fluid)
        idx_w, idx_m1, idx_m2, nrm_np, fid_np = plan_np
        npair = len(idx_w)
        tau_mask = np.zeros((H, W), bool)
        tau_mask.reshape(-1)[idx_w] = True
        log(f"[hills] H={H} W={W} faces={len(faces)} pairs={npair} "
            f"dropped_pairs={dropped} ({dropped/max(1,kept+dropped):.4f}) "
            f"unique_wall_cells={int(tau_mask.sum())}")

        def phys(i):
            return (np.asarray(fields[i], np.float32)
                    * hsd[:, None, None] + hmu[:, None, None]) * maskf[None]

        # bulk velocity at the hill crest -> nu from Re_h
        mean_u = np.zeros((H, W), np.float64)
        for i in range(0, T, 4):
            mean_u += phys(i)[0]
        mean_u /= len(range(0, T, 4))
        crest = int(np.argmax([np.argmax(fluid[:, j] > 0) for j in range(W)]))
        col = fluid[:, crest]
        U_b = float(mean_u[col, crest].mean())
        nu_h = U_b * h / RE_H_HILL
        log(f"[hills] crest_col={crest} U_b={U_b:.6f} h={h:.6f} nu={nu_h:.3e}")

        energy = np.array([float(np.square(np.asarray(fields[i], np.float32)[:, fluid]).mean())
                           for i in range(T)])
        tau_int_h = integral_tau(energy)
        # ---- units, and the STRICTLY UNCONTACTED sub-unit -------------------
        # Every previous producer on this record inherits ONE split rule from
        # codes/gpu/train_stats_l2.py: train [0, 0.76T), discard a decorrelation
        # gap of max(80, 3*tau), score [0.76T + gap, T).  Two consequences follow
        # mechanically and are verified in CONTACT_LEDGER.json:
        #   * frames [0.76T, 0.76T+gap) were never trained on AND never scored by
        #     ANY prior producer -- every one of them threw the gap away.  That
        #     window is the strictly uncontacted unit U_STRICT.
        #   * frames [0.76T+gap, T) were scored by earlier producers, for other
        #     estimands and other models.  They are retained as the declared
        #     previously-contacted replication unit U_PRIOR.
        # This experiment shifts its OWN training window back by one gap so that
        # U_STRICT also sits a full decorrelation gap behind its training data.
        gap_h = max(80, 3 * int(math.ceil(tau_int_h)))
        legacy_ntr = int(.76 * T)
        ntr_h = legacy_ntr - gap_h
        train_idx_h = np.arange(ntr_h)
        u_strict = np.arange(legacy_ntr, min(T, legacy_ntr + gap_h))
        u_prior = np.arange(min(T, legacy_ntr + gap_h), T)
        eval_h = np.concatenate([u_strict, u_prior])
        if A.hill_eval > 0:
            eval_h = eval_h[:A.hill_eval]
            u_strict = u_strict[u_strict <= eval_h[-1]]
            u_prior = u_prior[u_prior <= eval_h[-1]]
        if A.smoke:
            train_idx_h = np.arange(64)
            eval_h = np.arange(6)
            u_strict, u_prior = eval_h[:3], eval_h[3:]
        block_h, stride_h = conservative_block(eval_h, tau_int_h)
        log(f"[hills split] T={T} train=[0,{ntr_h}) tau={tau_int_h:.2f} gap={gap_h} "
            f"U_STRICT=[{u_strict[0]}..{u_strict[-1]}] n={len(u_strict)} "
            f"U_PRIOR=[{u_prior[0]}..{u_prior[-1]}] n={len(u_prior)} block={block_h}")

        # per-pair wall-anchor distance and face length, straight from the plan
        anchors = np.zeros(npair, np.float32)
        lens = np.zeros(npair, np.float32)
        for fi, f in enumerate(faces):
            anchors[fid_np == fi] = f["anchor"]
            lens[fid_np == fi] = f["len"]
        tau_pairs = np.zeros((T, 2, npair), np.float32)
        an_t = torch.from_numpy(anchors).to(DEV)
        for i in range(T):
            flat = phys(i).reshape(2, -1)
            u = flat[:, idx_w]
            un = (nrm_np.T * u).sum(0)
            ut = u - nrm_np.T * un[None, :]
            tau_pairs[i] = (-nu_h / anchors[None, :]) * ut

        uniq_flat = np.flatnonzero(tau_mask.reshape(-1))
        p2u = np.searchsorted(uniq_flat, idx_w)
        tcu = np.zeros((len(train_idx_h), 2, len(uniq_flat)), np.float32)
        for fi in range(len(faces)):          # no duplicate cell WITHIN one face
            sel = np.flatnonzero(fid_np == fi)
            if len(sel):
                tcu[:, :, p2u[sel]] += tau_pairs[train_idx_h][:, :, sel]
        tau_rms_h = np.sqrt(np.maximum(
            np.square(tcu.astype(np.float64)).mean((0, 2)), 1e-24)).astype(np.float32)
        del tcu

        # sign gate: the wall must retard the fluid at every pair
        probe = phys(train_idx_h[0]).reshape(2, -1)
        up = probe[:, idx_w]
        unp = (nrm_np.T * up).sum(0)
        utp = up - nrm_np.T * unp[None, :]
        dotp = (tau_pairs[train_idx_h[0]] * utp).sum(0)
        gate_sign = bool(dotp.max() <= 1e-20)
        fx_native = float((tau_pairs[train_idx_h][:, 0] * lens[None, :]).sum(1).mean())
        results["gates"]["hills"] = {
            "tau_dot_ut_max": float(dotp.max()), "sign_gate": gate_sign,
            "fx_viscous_wall_on_fluid_mean": fx_native,
            "wall_on_fluid_x_force_negative": bool(fx_native < 0),
            "U_bulk_crest": U_b, "nu": nu_h, "Re_h": RE_H_HILL,
            "dropped_matching_pairs": int(dropped),
            "matching_availability": float(kept / max(1, kept + dropped)),
            "y_plus_first_cell_mean": None,
        }
        if not gate_sign:
            raise RuntimeError("VOID: hill signed-traction convention gate failed")

        # ---- L_A on hills: fit on the training window ONLY, then freeze ----
        ctx_op = make_context_op_2d(idx_m1, (H, W))
        plan = (torch.from_numpy(idx_w).to(DEV).long(),
                torch.from_numpy(idx_m1).to(DEV).long(),
                torch.from_numpy(idx_m2).to(DEV).long(),
                torch.from_numpy(nrm_np).to(DEV),
                torch.from_numpy(fid_np).to(DEV).long(), ctx_op)
        y1_h = (M_OFF1 + 0.5) * dy
        torch.manual_seed(4242)                 # closure seeded BEFORE construction
        closure_h = WallClosure(7, len(faces), n_out=1).to(DEV)
        tau_rms_h_t = torch.from_numpy(tau_rms_h).to(DEV)
        cl_fit = train_idx_h[:int(0.90 * len(train_idx_h))]
        cl_val = train_idx_h[int(0.90 * len(train_idx_h)):]
        if A.smoke:
            cl_fit, cl_val = train_idx_h[:32], train_idx_h[32:48]

        best = {"loss": float("inf"), "state": None, "it": -1}
        if HILL_CLOSURE_CKPT.exists() and not A.smoke and torch.load(
                HILL_CLOSURE_CKPT, map_location="cpu").get("complete", False):
            qq = torch.load(HILL_CLOSURE_CKPT, map_location=DEV)
            closure_h.load_state_dict(qq["model"])
            closure_meta_h = qq["meta"]
            log("[resume] hill closure L_A")
        else:
            opt = torch.optim.AdamW(closure_h.parameters(), lr=3e-3, weight_decay=1e-5)
            rng = np.random.default_rng(913)
            t0, hist = time.time(), []
            for it in range(A.closure_steps):
                i = int(rng.choice(cl_fit))
                fld = torch.from_numpy(phys(i)).to(DEV)
                pred = closure_traction_2d(closure_h, fld, plan, nu_h, y1_h)
                tgt = torch.from_numpy(tau_pairs[i]).to(DEV)
                loss = (((pred - tgt) / tau_rms_h_t[:, None]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(closure_h.parameters(), 1.0)
                opt.step()
                if (it + 1) % max(1, A.closure_steps // 10) == 0:
                    with torch.no_grad():
                        vs, ve = [], []
                        for j in cl_val[::4]:
                            fj = torch.from_numpy(phys(j)).to(DEV)
                            tj = torch.from_numpy(tau_pairs[j]).to(DEV)
                            vs.append(float((((closure_traction_2d(closure_h, fj, plan, nu_h, y1_h) - tj)
                                              / tau_rms_h_t[:, None]) ** 2).mean()))
                            ve.append(float((((closure_traction_2d(closure_h, fj, plan, nu_h, y1_h, eqwm=True) - tj)
                                              / tau_rms_h_t[:, None]) ** 2).mean()))
                    vm = float(np.mean(vs))
                    hist.append([it + 1, float(loss.detach()), vm, float(np.mean(ve))])
                    # GENUINE validation-based model selection (node 006 only monitored)
                    if vm < best["loss"]:
                        best = {"loss": vm, "it": it + 1,
                                "state": {k: v.detach().clone() for k, v in
                                          closure_h.state_dict().items()}}
                    log(f"[hill closure] {it+1}/{A.closure_steps} train={loss:.5f} "
                        f"val={vm:.5f} val_eqwm={np.mean(ve):.5f} best@{best['it']}")
            if best["state"] is not None:
                closure_h.load_state_dict(best["state"])
            closure_meta_h = {"steps": A.closure_steps, "history": hist,
                              "selected_step": best["it"], "selected_val": best["loss"],
                              "fit_snapshots": [int(cl_fit[0]), int(cl_fit[-1])],
                              "val_snapshots": [int(cl_val[0]), int(cl_val[-1])],
                              "seeded_before_construction": 4242,
                              "n_parameters": sum(p.numel() for p in closure_h.parameters()),
                              "wall_seconds": round(time.time() - t0, 1)}
            if not A.smoke:
                torch.save({"complete": True, "model": closure_h.state_dict(),
                            "meta": closure_meta_h}, HILL_CLOSURE_CKPT)
        closure_h.eval()

        tau_cl_h = np.zeros_like(tau_pairs)
        tau_eq_h = np.zeros_like(tau_pairs)
        with torch.no_grad():
            for i in range(T):
                fld = torch.from_numpy(phys(i)).to(DEV)
                tau_cl_h[i] = closure_traction_2d(closure_h, fld, plan, nu_h, y1_h).cpu().numpy()
                tau_eq_h[i] = closure_traction_2d(closure_h, fld, plan, nu_h, y1_h,
                                                  eqwm=True).cpu().numpy()
            # y+ diagnostic at the first fluid cell
            fld = torch.from_numpy(phys(train_idx_h[0])).to(DEV)
            feats, e1, utau, fid = closure_features_2d(fld, plan, nu_h, y1_h)
            results["gates"]["hills"]["y_plus_first_cell_mean"] = float(
                (an_t * utau / nu_h).mean())
            results["gates"]["hills"]["y_plus_matching1_mean"] = float(
                (y1_h * utau / nu_h).mean())
        log("[hills] closure traction materialised")

        src = np.zeros(H * W, bool)
        src[idx_m1] = True; src[idx_m2] = True
        src = src.reshape(H, W)
        regions_h = {
            "full_support_excluded": fluid & (~band),
            "near_support_excluded_d_le_0p5h": fluid & (~band) & (dist <= .5 * h),
            "outer_d_gt_0p5h": fluid & (dist > .5 * h),
            "full_srcex": fluid & (~band) & (~src),
            "near_srcex": fluid & (~band) & (~src) & (dist <= .5 * h),
            "outer_srcex": fluid & (dist > .5 * h) & (~src),
        }
        log({k: int(v.sum()) for k, v in regions_h.items()})

        hmean = np.zeros((2, H, W), np.float64)
        for i in train_idx_h:
            hmean += phys(i)
        hmean = (hmean / len(train_idx_h)).astype(np.float32)
        hmu_f = np.zeros(2, np.float32); hsd_f = np.zeros(2, np.float32)
        acc_s = np.zeros(2); acc_ss = np.zeros(2); acc_n = np.zeros(2)
        for i in train_idx_h[::4]:
            x = phys(i)
            for c in range(2):
                v = x[c][fluid]
                acc_n[c] += v.size; acc_s[c] += v.sum(); acc_ss[c] += np.square(v).sum()
        hmu_f = (acc_s / acc_n).astype(np.float32)
        hsd_f = np.sqrt(np.maximum(acc_ss / acc_n - (acc_s / acc_n) ** 2, 1e-12)).astype(np.float32)

        cb_h = Conditioner(torch.from_numpy(tau_pairs).to(DEV),
                           torch.from_numpy(tau_cl_h).to(DEV),
                           torch.from_numpy(tau_eq_h).to(DEV),
                           torch.from_numpy(idx_w).to(DEV).long(),
                           torch.tensor(tau_mask[None], dtype=torch.float32, device=DEV),
                           tau_rms_h_t, 2, (H, W), 2)
        faces_t_h = []
        for fi, f in enumerate(faces):
            sel = np.flatnonzero(fid_np == fi)
            if not len(sel):
                continue
            faces_t_h.append((torch.from_numpy(idx_w[sel]).to(DEV).long(),
                              torch.from_numpy(f["n"]).to(DEV), f["anchor"], f["len"]))
        donor_h = np.roll(np.asarray(eval_h), len(eval_h) // 2)

        results["cells"]["hills_common"] = {
            "regime": "periodic_hills_Re_h_2800_2D_separating",
            "closure_meta": closure_meta_h,
            "closure_apriori": closure_apriori_scores(tau_cl_h, tau_eq_h, tau_pairs,
                                                      eval_h, tau_rms_h),
            "closure_apriori_val": closure_apriori_scores(tau_cl_h, tau_eq_h, tau_pairs,
                                                          cl_val, tau_rms_h),
            "n_eval": int(len(eval_h)), "block": int(block_h),
            "tau_integral": float(tau_int_h),
            "n_effective": float(len(eval_h) / max(1.0, tau_int_h)),
            "eval_window": [int(eval_h[0]), int(eval_h[-1])],
            "train_window": [0, int(ntr_h) - 1],
            "gap_frames": int(eval_h[0] - ntr_h),
            "nu": nu_h, "h": h, "dy": dy, "dx": dx,
        }
        for cell, fam in (("C2", "F"), ("C3", "G")):
            if cell not in CELLS:
                continue
            log(f"=== cell {cell}: hills x family {fam} ===")
            run_cell(cell, fam, "hills", phys, T, train_idx_h, np.asarray(eval_h),
                     donor_h, hmu_f, hsd_f, hmean, fluid, regions_h, cb_h, 2, (H, W), 2,
                     faces_t_h, nu_h,
                     {"U_STRICT": u_strict, "U_PRIOR": u_prior}, block_h,
                     [(9811 if fam == "F" else 9821) + s for s in range(A.seeds)],
                     A.steps_train_2d, A.base2d, A.batch2d, comps, metas)
            results["cells"][cell] = {
                "regime": "periodic_hills_Re_h_2800_2D_separating",
                "family": ("F_flow_matching" if fam == "F" else "G_denoising_diffusion"),
                "n_eval": int(len(eval_h)), "block": int(block_h),
                "tau_integral": float(tau_int_h),
                "n_effective": float(len(eval_h) / max(1.0, tau_int_h)),
                "seeds": [(9811 if fam == "F" else 9821) + s for s in range(A.seeds)],
            }

    # ======================================================================
    # Analysis: deltas, hierarchical intervals, paired contrasts
    # ======================================================================
    results["analysis"] = analyse(comps, results)
    results["train_meta"] = metas
    results["wall_seconds"] = round(time.time() - t_start, 1)
    results["gpu_hours"] = round((time.time() - t_start) / 3600.0, 3)

    np.savez_compressed(COMP, **comps)
    OUT.write_text(json.dumps(results, indent=1, default=float))
    (RESULTS / f"{TAG}_results.sha256").write_text(
        f"{sha256(OUT)}  {OUT.name}\n{sha256(COMP)}  {COMP.name}\n")
    log(f"[write] {OUT.name} {COMP.name}")
    log("=== done ===")


def closure_apriori_scores(tau_cl, tau_eq, tau_native, idxs, tau_rms):
    """Traction skill on EVERY frame of ``idxs``, in BOTH definitions.

    * ``R2_centred``  : conventional coefficient of determination about the mean
                        traction of the evaluated frames.
    * ``skill_zero_ref``: 1 - MSE / E[tau^2], the zero-traction-reference skill
                        node 006 published as "R^2".  Reported under its correct
                        name so the two can never be confused again.
    """
    idxs = np.asarray(idxs)
    w = tau_rms.reshape(1, -1, 1)
    tn = tau_native[idxs] / w
    tc = tau_cl[idxs] / w
    te = tau_eq[idxs] / w
    mean_t = tn.mean(axis=(0, 2), keepdims=True)
    den_c = float(np.square(tn - mean_t).sum())
    den_0 = float(np.square(tn).sum())
    out = {
        "n_frames_scored": int(len(idxs)),
        "R2_centred_closure": 1.0 - float(np.square(tc - tn).sum()) / den_c,
        "R2_centred_eqwm": 1.0 - float(np.square(te - tn).sum()) / den_c,
        "skill_zero_ref_closure": 1.0 - float(np.square(tc - tn).sum()) / den_0,
        "skill_zero_ref_eqwm": 1.0 - float(np.square(te - tn).sum()) / den_0,
    }
    a = tau_cl[idxs]; b = tau_native[idxs]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out["mean_direction_cosine"] = float(np.nanmean(num / np.maximum(den, 1e-30)))
    return out


def subunit_deltas(cell, comps, smask, regions, seeds, block, sname):
    """Registered contrasts restricted to one evaluation sub-unit, with
    dependence-aware block and hierarchical seed x block intervals."""
    res = {}
    blk = int(block)
    for reg in regions:
        sst = comps[f"{cell}|sst|{reg}"][smask]
        arm_sse = {}
        for arm in ARMS:
            ss = [comps[f"{cell}|sse|{arm}|{s}|{reg}"][smask] for s in seeds
                  if f"{cell}|sse|{arm}|{s}|{reg}" in comps]
            if ss:
                arm_sse[arm] = ss
        entry = {"n": int(smask.sum()), "block_used": blk, "arms": {}, "deltas": {}}
        for arm, ss in arm_sse.items():
            entry["arms"][arm] = {
                "R2_seedmean": float(1.0 - np.mean(ss, axis=0).sum() / sst.sum()),
                "R2_per_seed": [float(1.0 - s.sum() / sst.sum()) for s in ss]}
        for a, b, tag in (("tau_closure", "absent", "closure_minus_absent"),
                          ("tau_native", "absent", "native_minus_absent"),
                          ("tau_eqwm", "absent", "eqwm_minus_absent"),
                          ("tau_fartime", "absent", "fartime_minus_absent"),
                          ("tau_closure", "tau_fartime", "closure_minus_fartime"),
                          ("tau_closure", "tau_eqwm", "closure_minus_eqwm")):
            if a not in arm_sse or b not in arm_sse:
                continue
            sa, sb = np.stack(arm_sse[a]), np.stack(arm_sse[b])
            ma, mb = sa.mean(0), sb.mean(0)
            point = float((1 - ma.sum() / sst.sum()) - (1 - mb.sum() / sst.sum()))
            rng = np.random.default_rng(zlib.crc32(f"{cell}|{sname}|{reg}|{tag}".encode()))
            ix = block_indices(len(sst), blk, A.boot, rng)
            d = ((1 - ma[ix].sum(1) / sst[ix].sum(1))
                 - (1 - mb[ix].sum(1) / sst[ix].sum(1)))
            S = sa.shape[0]
            pick = rng.integers(0, S, size=(A.boot, S))
            hb = np.empty(A.boot)
            for k in range(A.boot):
                ra, rb = sa[pick[k]].mean(0), sb[pick[k]].mean(0)
                hb[k] = ((1 - ra[ix[k]].sum() / sst[ix[k]].sum())
                         - (1 - rb[ix[k]].sum() / sst[ix[k]].sum()))
            per_seed = [float((1 - x.sum() / sst.sum()) - (1 - y.sum() / sst.sum()))
                        for x, y in zip(arm_sse[a], arm_sse[b])]
            entry["deltas"][tag] = {
                "point": point,
                "ci_block": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "ci_hier_seed_block": [float(np.percentile(hb, 2.5)),
                                       float(np.percentile(hb, 97.5))],
                "per_seed": per_seed,
                "same_sign_all_seeds": bool(len(per_seed) > 0 and per_seed[0] != 0 and
                                            all(np.sign(v) == np.sign(per_seed[0])
                                                for v in per_seed)),
            }
        res[reg] = entry
    return res


def analyse(comps, results):
    """Every registered contrast, with dependence-aware and hierarchical intervals."""
    out = {}
    cells = [c for c in ("C1", "C2", "C3") if f"{c}|eval_idx" in comps]
    for cell in cells:
        meta = results["cells"].get(cell, {})
        block = int(meta.get("block", 1))
        seeds = meta.get("seeds", [])
        regions = sorted({k.split("|")[-1] for k in comps
                          if k.startswith(f"{cell}|sse|")})
        n_eval = len(comps[f"{cell}|eval_idx"])
        subs = {k.split("|")[-1]: comps[k] for k in comps
                if k.startswith(f"{cell}|subunit|")}
        subs["ALL"] = np.ones(n_eval, bool)
        cres = {"regions": {}, "seeds": seeds, "block": block,
                "subunits": {k: int(v.sum()) for k, v in subs.items()},
                "by_subunit": {}}
        eidx = comps[f"{cell}|eval_idx"]
        tau_c = float(meta.get("tau_integral", 1.0))
        for sname, smask in subs.items():
            if smask.sum() < 12:
                continue
            blk_s, stride_s = conservative_block(eidx[smask], tau_c)
            cres["subunit_blocks"] = cres.get("subunit_blocks", {})
            cres["subunit_blocks"][sname] = {"block": blk_s, "stride": stride_s,
                                             "n": int(smask.sum())}
            cres["by_subunit"][sname] = subunit_deltas(
                cell, comps, smask, regions, seeds, blk_s, sname)
        block = cres.get("subunit_blocks", {}).get("ALL", {}).get("block", block)
        cres["block_used_whole_unit"] = int(block)
        for reg in regions:
            sst = comps[f"{cell}|sst|{reg}"]
            arm_r2, arm_sse = {}, {}
            for arm in ARMS:
                ss = [comps[f"{cell}|sse|{arm}|{s}|{reg}"] for s in seeds
                      if f"{cell}|sse|{arm}|{s}|{reg}" in comps]
                if not ss:
                    continue
                arm_sse[arm] = ss
                mean_sse = np.mean(ss, axis=0)
                arm_r2[arm] = {
                    "R2_seedmean": float(1.0 - mean_sse.sum() / sst.sum()),
                    "R2_per_seed": [float(1.0 - s.sum() / sst.sum()) for s in ss],
                }
            entry = {"arms": arm_r2, "deltas": {}}
            def delta(a, b, tag):
                if a not in arm_sse or b not in arm_sse:
                    return
                ma = np.mean(arm_sse[a], axis=0)
                mb = np.mean(arm_sse[b], axis=0)
                point = float((1 - ma.sum() / sst.sum()) - (1 - mb.sum() / sst.sum()))
                rng = np.random.default_rng(zlib.crc32(f"{cell}|{reg}|{tag}".encode()))
                ix = block_indices(len(sst), block, A.boot, rng)
                d = ((1 - ma[ix].sum(1) / sst[ix].sum(1))
                     - (1 - mb[ix].sum(1) / sst[ix].sum(1)))
                per_seed = [float((1 - x.sum() / sst.sum()) - (1 - y.sum() / sst.sum()))
                            for x, y in zip(arm_sse[a], arm_sse[b])]
                # hierarchical seed x block interval
                S = len(arm_sse[a])
                pick = rng.integers(0, S, size=(A.boot, S))
                sa = np.stack(arm_sse[a]); sb = np.stack(arm_sse[b])
                hb = np.empty(A.boot)
                for k in range(A.boot):
                    ra = sa[pick[k]].mean(0); rb = sb[pick[k]].mean(0)
                    hb[k] = ((1 - ra[ix[k]].sum() / sst[ix[k]].sum())
                             - (1 - rb[ix[k]].sum() / sst[ix[k]].sum()))
                entry["deltas"][tag] = {
                    "point": point,
                    "ci_block": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "ci_hier_seed_block": [float(np.percentile(hb, 2.5)),
                                           float(np.percentile(hb, 97.5))],
                    "per_seed": per_seed,
                    "same_sign_all_seeds": bool(all(np.sign(v) == np.sign(per_seed[0])
                                                    for v in per_seed) and per_seed[0] != 0),
                }
            delta("tau_closure", "absent", "closure_minus_absent")
            delta("tau_native", "absent", "native_minus_absent")
            delta("tau_eqwm", "absent", "eqwm_minus_absent")
            delta("tau_fartime", "absent", "fartime_minus_absent")
            delta("tau_closure", "tau_fartime", "closure_minus_fartime")
            delta("tau_closure", "tau_eqwm", "closure_minus_eqwm")
            cres["regions"][reg] = entry

        # ---- paired distributional / force endpoints (full_srcex) ----------
        pe = {}
        for name in ("crps", "escore", "fx"):
            for arm in ARMS:
                ks = [f"{cell}|{name}|{arm}|{s}" for s in seeds
                      if f"{cell}|{name}|{arm}|{s}" in comps]
                if ks:
                    pe.setdefault(name, {})[arm] = np.mean([comps[k] for k in ks], axis=0)
        paired = {}
        for name in ("crps", "escore"):
            if name not in pe or "tau_closure" not in pe[name]:
                continue
            for base_arm in ("absent", "tau_fartime", "tau_eqwm"):
                if base_arm not in pe[name]:
                    continue
                d = pe[name]["tau_closure"] - pe[name][base_arm]
                ci, m = paired_block_ci(d, block, A.boot)
                paired[f"{name}_closure_minus_{base_arm}"] = {
                    "mean": m, "ci_block": ci,
                    "levels": {a: float(np.mean(v)) for a, v in pe[name].items()},
                }
        if "fx" in pe and f"{cell}|fx_target" in comps:
            ref = comps[f"{cell}|fx_target"]          # the TARGET's own wall force
            fxs = dict(pe["fx"])
            paired["fx_levels"] = {a: float(np.mean(v)) for a, v in fxs.items()}
            paired["fx_target_mean"] = float(np.mean(ref))
            scale = float(np.sqrt(np.mean(np.square(ref))))
            for a in ARMS:
                if a not in fxs:
                    continue
                paired[f"fx_relRMSE_{a}"] = float(
                    np.sqrt(np.mean(np.square(fxs[a] - ref))) / max(scale, 1e-30))
                if len(ref) > 2 and np.std(fxs[a]) > 0 and np.std(ref) > 0:
                    paired[f"fx_corr_{a}"] = float(np.corrcoef(fxs[a], ref)[0, 1])
            if "tau_closure" in fxs and "absent" in fxs:
                d = np.abs(fxs["tau_closure"] - ref) - np.abs(fxs["absent"] - ref)
                ci, m = paired_block_ci(d, block, A.boot)
                paired["fx_abserr_closure_minus_absent"] = {"mean": m, "ci_block": ci}
        # ---- rank reliability + ensemble sensitivity ----------------------
        rel = {}
        for arm in ARMS:
            ks = [f"{cell}|rankhist|{arm}|{s}" for s in seeds
                  if f"{cell}|rankhist|{arm}|{s}" in comps]
            if not ks:
                continue
            hh = np.mean([comps[k].mean(0) for k in ks], axis=0)
            rel[arm] = float(np.abs(hh - 1.0 / len(hh)).sum())
        ens = {}
        for arm in ARMS:
            ks = [f"{cell}|ens|{arm}|{s}" for s in seeds
                  if f"{cell}|ens|{arm}|{s}" in comps]
            if ks:
                ens[arm] = np.nansum(np.mean([comps[k] for k in ks], axis=0), axis=0).tolist()
        cres["paired_endpoints"] = paired
        cres["rank_reliability_L1"] = rel
        cres["ensemble_sse_1_2_4_8"] = ens
        out[cell] = cres
    return out


if __name__ == "__main__":
    main()
