#!/usr/bin/env python3
"""node_011 decisive experiment: the SINGLE-SLOT, QUALITY-AGNOSTIC, FLUX-CONSISTENT
wall-traction interface, with the fidelity->gain dose-response as the registered
attribution instrument.

THE THREE DEFECTS THIS PRODUCER REPAIRS (all three node_010 referees, independently)
------------------------------------------------------------------------------------
1.  ARM-EXPOSURE IMBALANCE.  node_010 trained the generator on a 7-arm mixture in
    which `closure` received 0.24 exposure but `learned_dataonly` 0.12; the
    "budget-matched" attribution claim was false at the generator level.
    REPAIR: the generator is trained on EXACTLY ONE traction-carrying condition --
    the record's own exact wall traction, corrupted by a variance-preserving
    fidelity ladder that spans the whole quality continuum (r ~ U[0,1]).  No
    predictor's output is ever seen in training.  At inference EVERY arm enters
    through the same slot with identical (zero) generator-side training exposure.
    Budgets are matched BY CONSTRUCTION, not by probability bookkeeping.

2.  STATE SIDE CHANNEL.  node_010's closure output beat the exact traction in the
    log band (+0.178 vs +0.074) while its own traction fidelity was WORSE than the
    learned control's (0.070 vs 0.084): the generator had learned to decode
    closure-specific residual structure, so the gain was matching-height state
    routing, not wall-information fidelity.
    REPAIR: because the decoder is fitted only on exact-traction content, it has no
    closure-specific decoding to exploit.  The registered attribution test is the
    measured DOSE-RESPONSE CURVE gain(fidelity) traced by the exact+noise ladder:
    every estimator arm (closure, equilibrium, learned data-only) must land ON that
    curve at its own measured fidelity.  Landing significantly ABOVE the curve is
    the registered side-channel detector and FAILS the experiment.

3.  NO BOUNDARY-FLUX CONSISTENCY.  node_010 copied traction values into rows as a
    passive input; nothing tied the generated field's wall flux to the supplied
    traction (C7 failed).
    REPAIR: hard signed-Neumann enforcement.  At every sampling step the support
    rows (y+ <= 3.2, strictly inside the viscous sublayer) are projected onto the
    exact viscous-sublayer solution implied by the supplied traction,
    u_i(y_j) = (du_i/dy)|_w * y_j, noise-consistently along the whole sampling
    path (RePaint/DDNM-style).  The generated field's wall gradient EQUALS the
    supplied traction by construction; the registered engineering endpoint
    therefore moves OUTSIDE the support: tracking of the instantaneous
    spanwise-averaged tangential momentum flux <u'v'> in a scored wall-distance
    band that the conditioning never touches.

AXIS-IDENTITY CORRECTION (referee 2 of node_010, confirmed by measurement)
--------------------------------------------------------------------------
The 100-point in-plane tangential axis is SPANWISE (z), not streamwise: the
near-wall (y+ = 15) u autocorrelation along it has the classic streak negative
lobe at lag 6--10 cells (min -0.355 at lag 10), impossible for the
streak-elongated streamwise direction.  The permutation control is therefore
`shuffle_z` and all prose refers to spanwise line averages.

FINAL EVALUATION UNIT (never scored by ANY retained producer)
-------------------------------------------------------------
Native frames [0, 459] of CoNFiLD Case2.  The machine contact ledger
(codes/results/case2_contact_ledger.json) verifies that every retained producer
scored only frames >= 500 (node_010 DEV [500,559], TEST {746..1196}) or >= 744
(eval_l3_channel_absolute).  Frames [0,459] were used only as TRAINING data by
now-discarded systems; no evaluation outcome of any kind was ever computed there.
The new system trains on [746,1199] and never reads [0,459] before the final
scoring pass.  Rehearsal (development) scoring uses the GAP window [560,740].

INFERENCE (registered primary)
------------------------------
Three contiguous physical-time blocks (>= 150 native frames each, above the
conservative 1.2551 * tau = 116-frame Politis-White scale of the FIELD series --
the contrast series decorrelates faster, tau_delta ~ 53 native frames in
node_010, so the block is conservative), one-sided t on the 3 block means with a
critical value CALIBRATED BY SIMULATION at the exact design
(codes/probes/calibrate_block_inference.py).  Plus: all block means positive,
both walls positive, both seeds positive.  Moving-block bootstrap is reported as
secondary.  No population-95% language is attached to any interval whose
coverage was not demonstrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument("--tag", default="e2_slot_channel")
P.add_argument("--phase", choices=("rehearsal", "final"), required=True)
P.add_argument("--data", default="/root/autodl-tmp/case2_channel")
P.add_argument("--steps-train", type=int, default=9000)
P.add_argument("--closure-steps", type=int, default=6000)
P.add_argument("--batch", type=int, default=16)
P.add_argument("--members", type=int, default=8)
P.add_argument("--sampler-steps", type=int, default=32)
P.add_argument("--boot", type=int, default=4000)
P.add_argument("--seeds", type=int, default=2)
P.add_argument("--matched-r", type=float, default=None,
               help="FINAL phase: frozen v2 correlation for the matched arm")
P.add_argument("--matched-ell", type=float, default=None,
               help="FINAL phase: frozen v2 spanwise noise correlation length (cells)")
P.add_argument("--smoke", action="store_true")
A = P.parse_args()

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

if A.smoke:
    A.steps_train, A.closure_steps = 60, 80
    A.members, A.boot, A.seeds, A.sampler_steps = 2, 200, 1, 4

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ============================== FROZEN CONTRACT ===============================
# Reversed-time split in native frame indices of the 1200-snapshot record.
TRAIN_W = (746, 1199)        # 454 frames -- node_010's burned TEST window, now training data
REH_W = (560, 740)           # rehearsal (development) scoring: the never-scored GAP
FINAL_W = (0, 450)           # FINAL unit: never scored by any retained producer
STRIDE = 10                  # scored frames: rehearsal 19, final 46
TAU_LEGACY = 92.5            # whole-record integral time frozen by node_010, kept as FLOOR
BLOCK_CONST = 1.2551         # Politis-White constant, as in every predecessor
N_BLOCKS_PRIMARY = 3         # contiguous physical-time blocks for the primary t
LAG_REHEARSAL = 370          # wrong-time donor lag (>= 4 tau); donors land in TRAIN
LAG_FINAL = 746              # = 8.1 tau; donor j = i + 746 in TRAIN for every final frame

# --- interface geometry (identical to node_010 for comparability) ------------
N_C = 4                      # conditioning/enforcement support: rows 0..3, y+ <= 3.2
J_M1, J_M2 = 32, 65          # closure matching heights, y+ ~ 29.9 and 60.2
READ_GUARD = 1
K_WALL = 3                   # rows in the LS wall-gradient fit (inside the support)

KAPPA, C_REICH = 0.41, 7.8
SHUFFLE_SEED = 20260803

# --- training-time corruption (the quality-agnostic continuum) ---------------
P_ABSENT = 0.20              # mask probability; otherwise the exact traction is
R_RANGE = (0.0, 1.0)         # supplied at correlation r ~ U[0,1], variance-preserving
P_CORRELATED = 0.5           # half the corruptions use spanwise-correlated noise
ELL_RANGE = (2.0, 25.0)      # correlation length of that noise, cells

# --- evaluation arms ----------------------------------------------------------
LADDER_R = (0.90, 0.75, 0.625, 0.55)   # target centred R2 = 2r-1: 0.80/0.50/0.25/0.10
FM_SEEDS = (8821, 8822)
DIFF_SEEDS = (9921, 9922)
DIFF_PARAM, DIFF_WIDTH = "v", 128      # inherited frozen node_010 DEV selection (v_cosine_w128)

TAG = A.tag + ("_smoke" if A.smoke else "")


def log(*a):
    print(*a, flush=True)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================== wall closure ==================================
def reichardt_uplus(yp):
    return ((1.0 / KAPPA) * torch.log1p(KAPPA * yp)
            + C_REICH * (1.0 - torch.exp(-yp / 11.0) - (yp / 11.0) * torch.exp(-yp / 3.0)))


def utau_equilibrium(umag, y, nu, iters=60):
    lo = torch.full_like(umag, 1e-9)
    hi = torch.full_like(umag, 2.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_small = mid * reichardt_uplus(mid * y / nu) < umag
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


class ConvTrunk(nn.Module):
    """Shared spanwise-convolutional trunk (circular padding: z is periodic in a
    plane channel).  Receptive field ~ +-14 cells, enough to see one streak
    wavelength (~20 cells, measured)."""

    def __init__(self, cin, hidden=64):
        super().__init__()
        self.c1 = nn.Conv1d(cin, hidden, 5, padding=2, padding_mode="circular")
        self.c2 = nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2, padding_mode="circular")
        self.c3 = nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4, padding_mode="circular")

    def forward(self, x):
        h = F.silu(self.c1(x))
        h = F.silu(self.c2(h)) + h
        h = F.silu(self.c3(h)) + h
        return h


class ConvPhysicsClosure(nn.Module):
    """Physics-grounded wall closure with spanwise context: Reichardt equilibrium
    scale (never reads any cell below the matching heights) times a BOUNDED
    learned correction |a|,|th| <= 0.7.  Structurally the node_006/node_010
    closure with a convolutional (footprint-aware) correction."""

    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.trunk = ConvTrunk(n_feat, hidden)
        self.head = nn.Conv1d(hidden, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats):                      # feats (B, n_feat, nx)
        o = self.head(self.trunk(feats))
        return 0.7 * torch.tanh(o[:, 0]), 0.7 * torch.tanh(o[:, 1])


class ConvDataOnly(nn.Module):
    """Capacity-, input-, optimiser- and step-matched learned control: same trunk,
    same inputs, but NO equilibrium scale and NO wall law -- regresses the
    traction directly."""

    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.trunk = ConvTrunk(n_feat, hidden)
        self.head = nn.Conv1d(hidden, 2, 1)

    def forward(self, feats):
        return self.head(self.trunk(feats)).permute(0, 2, 1)   # (B, nx, 2)


class Wall:
    """Exact wall traction and matching-height features.  Nothing here reads a
    scored cell."""

    def __init__(self, nu, dy, n_x, mu, sd):
        self.nu, self.dy, self.n_x = nu, dy, n_x
        y = (torch.arange(K_WALL, device=DEV, dtype=torch.float32) + 0.5) * dy
        self.wy = (y / (y ** 2).sum()).view(1, K_WALL, 1)
        self.y_m1 = (J_M1 + 0.5) * dy
        self.y_m2 = (J_M2 + 0.5) * dy
        self.mu, self.sd = mu, sd                  # (3,) train-window stats

    def exact_traction(self, f):
        """tau_w = -rho*nu*(du/dy, dw/dy)|_w, wall-on-fluid, rho=1.  u is the
        (out-of-plane) streamwise component, w the in-plane spanwise one."""
        g_u = (f[0, :K_WALL] * self.wy[0]).sum(0)
        g_w = (f[2, :K_WALL] * self.wy[0]).sum(0)
        return -self.nu * torch.stack([g_u, g_w], -1)          # (n_x, 2)

    def features(self, f):
        """7 pointwise physics features + 6 standardized matching-height velocity
        lines -> (13, nx).  Identical information family to node_010, now with
        spanwise context available to the trunk."""
        u1, v1, w1 = f[0, J_M1], f[1, J_M1], f[2, J_M1]
        u2, v2, w2 = f[0, J_M2], f[1, J_M2], f[2, J_M2]
        t1 = torch.stack([u1, w1], -1)
        t2 = torch.stack([u2, w2], -1)
        m1 = t1.norm(dim=-1).clamp_min(1e-9)
        m2 = t2.norm(dim=-1).clamp_min(1e-9)
        e1 = t1 / m1[:, None]
        e2 = torch.stack([-e1[:, 1], e1[:, 0]], -1)
        utau = utau_equilibrium(m1, torch.full_like(m1, self.y_m1), self.nu)
        cos12 = (t1 * t2).sum(-1) / (m1 * m2)
        sin12 = (e2 * t2).sum(-1) / m2
        pw = torch.stack([
            torch.log1p(self.y_m1 * utau / self.nu), m2 / m1,
            v1 / m1, v2 / m1, cos12, sin12, torch.log(m1)], 0).float()
        lines = torch.stack([(u1 - self.mu[0]) / self.sd[0], (v1 - self.mu[1]) / self.sd[1],
                             (w1 - self.mu[2]) / self.sd[2], (u2 - self.mu[0]) / self.sd[0],
                             (v2 - self.mu[1]) / self.sd[1], (w2 - self.mu[2]) / self.sd[2]], 0)
        return torch.cat([pw, lines], 0), e1, e2, utau, m1     # (13, nx)

    def closure_traction(self, model, f, eqwm=False):
        feats, e1, e2, utau, _ = self.features(f)
        if eqwm:
            a = torch.zeros_like(utau)
            th = torch.zeros_like(utau)
        else:
            a, th = model(feats[None])
            a, th = a[0], th[0]
        mag = (utau ** 2) * torch.exp(a)
        return -mag[:, None] * (torch.cos(th)[:, None] * e1 + torch.sin(th)[:, None] * e2)

    def dataonly_traction(self, model, f, scale):
        feats, *_ = self.features(f)
        return model(feats[None])[0] * scale


# ============================== generator =====================================
def noise_embed(t, dim=128):
    half = dim // 2
    fr = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * fr[None] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class Block2D(nn.Module):
    def __init__(self, ci, co, ce=128):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.e = nn.Linear(ce, co)
        self.n1 = nn.GroupNorm(8, co)
        self.n2 = nn.GroupNorm(8, co)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.e(e)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.sk(x)


class UNet2D(nn.Module):
    def __init__(self, cin, cout, width=128, ce=128):
        super().__init__()
        w = width
        self.emb = nn.Sequential(nn.Linear(ce, ce), nn.SiLU(), nn.Linear(ce, ce))
        self.d1 = Block2D(cin, w, ce)
        self.d2 = Block2D(w, w * 2, ce)
        self.d3 = Block2D(w * 2, w * 2, ce)
        self.mid = Block2D(w * 2, w * 2, ce)
        self.u3 = Block2D(w * 4, w * 2, ce)
        self.u2 = Block2D(w * 4, w, ce)
        self.u1 = Block2D(w * 2, w, ce)
        self.out = nn.Conv2d(w, cout, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond):
        e = self.emb(noise_embed(t, 128))
        h = torch.cat([x, cond], 1)
        h1 = self.d1(h, e)
        h2 = self.d2(F.avg_pool2d(h1, 2), e)
        h3 = self.d3(F.avg_pool2d(h2, 2), e)
        m = self.mid(h3, e)
        u3 = self.u3(torch.cat([m, h3], 1), e)
        u3 = F.interpolate(u3, size=h2.shape[-2:], mode="nearest")
        u2 = self.u2(torch.cat([u3, h2], 1), e)
        u2 = F.interpolate(u2, size=h1.shape[-2:], mode="nearest")
        u1 = self.u1(torch.cat([u2, h1], 1), e)
        return self.out(u1)


def vp_alpha_bar(t):
    return torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


class Generator:
    """Rectified flow (flow matching) and VP diffusion, identical backbone, with
    hard signed-Neumann enforcement of the supplied traction on the support rows
    along the entire sampling path (RePaint/DDNM-style noise-consistent
    replacement).  bc=None (absent arm) disables enforcement."""

    def __init__(self, family, cin, cout, width, param="v"):
        self.family, self.param = family, param
        self.net = UNet2D(cin, cout, width).to(DEV)
        self.ema = UNet2D(cin, cout, width).to(DEV)
        self.ema.load_state_dict(self.net.state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def loss(self, x0, cond):
        b = x0.shape[0]
        t = torch.rand(b, device=DEV)
        n = torch.randn_like(x0)
        if self.family == "flow_matching":
            xt = (1 - t[:, None, None, None]) * n + t[:, None, None, None] * x0
            tgt = x0 - n
        else:
            ab = vp_alpha_bar(t)[:, None, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * n
            tgt = n if self.param == "eps" else ab.sqrt() * n - (1 - ab).sqrt() * x0
        return ((self.net(xt, t, cond) - tgt) ** 2).mean()

    def update_ema(self, decay=0.999):
        with torch.no_grad():
            for a, b in zip(self.ema.parameters(), self.net.parameters()):
                a.mul_(decay).add_(b, alpha=1 - decay)
            for a, b in zip(self.ema.buffers(), self.net.buffers()):
                a.copy_(b)

    @torch.no_grad()
    def sample(self, cond, steps, gen, bc=None, gfix=None):
        b = cond.shape[0]
        sh = (b, 3) + tuple(cond.shape[-2:])
        x = torch.randn(sh, device=DEV, generator=gen)
        if bc is not None:
            eps_fix = torch.randn((b, 3, N_C, cond.shape[-1]), device=DEV, generator=gfix)
        if self.family == "flow_matching":
            for k in range(steps):
                t = torch.full((b,), k / steps, device=DEV)
                x = x + self.ema(x, t, cond) / steps
                if bc is not None:
                    tn = (k + 1) / steps
                    x[:, :, :N_C, :] = (1 - tn) * eps_fix + tn * bc
            return x
        ts = torch.linspace(1.0, 0.0, steps + 1, device=DEV)
        for k in range(steps):
            t = ts[k].expand(b)
            ab = vp_alpha_bar(t)[:, None, None, None]
            ab_n = vp_alpha_bar(ts[k + 1].expand(b))[:, None, None, None]
            o = self.ema(x, t, cond)
            eps = o if self.param == "eps" else ab.sqrt() * o + (1 - ab).sqrt() * x
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)
            x0 = x0.clamp(-6, 6)
            x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps
            if bc is not None:
                x[:, :, :N_C, :] = ab_n.sqrt() * bc + (1 - ab_n).sqrt() * eps_fix
        if bc is not None:
            # exact terminal projection: vp_alpha_bar(0) = 0.99984, not 1, so the
            # last replacement above leaves ~1% residual noise on the support rows
            # (measured 1.08e-02 in the smoke run); DDNM-style hard final step
            x[:, :, :N_C, :] = bc
        return x


# ============================== metrics =======================================
def r2_fluct(pred, tgt, mask, mean_field):
    p = (pred - mean_field)[..., mask]
    t = (tgt - mean_field)[..., mask]
    return float(1.0 - ((p - t) ** 2).sum() / (t ** 2).sum())


def crps_ensemble(ens, tgt, mask):
    e = ens[..., mask]
    t = tgt[..., mask]
    m = e.shape[0]
    t1 = (e - t[None]).abs().mean(0)
    d = (e[:, None] - e[None, :]).abs().sum((0, 1)) / (2.0 * m * (m - 1))
    return float((t1 - d).mean())


def energy_score(ens, tgt, mask):
    e = ens[..., mask].reshape(ens.shape[0], -1)
    t = tgt[..., mask].reshape(-1)
    m = e.shape[0]
    t1 = (e - t[None]).norm(dim=1).mean()
    d = torch.cdist(e, e).sum() / (2.0 * m * (m - 1))
    return float(t1 - d)


def rank_hist(ens, tgt, mask, rng, nsub=4000):
    e = ens[..., mask].reshape(ens.shape[0], -1)
    t = tgt[..., mask].reshape(-1)
    idx = rng.choice(t.numel(), min(nsub, t.numel()), replace=False)
    r = (e[:, idx] < t[idx][None]).sum(0).cpu().numpy()
    return np.bincount(r, minlength=e.shape[0] + 1)


def tau_of_series(d, maxlag=None):
    d = np.asarray(d, np.float64)
    d = d - d.mean()
    den = float((d * d).mean())
    if den <= 0:
        return 1.0
    tau = 1.0
    for lag in range(1, maxlag or max(2, len(d) // 2)):
        r = float((d[:-lag] * d[lag:]).mean() / den)
        if r <= 0:
            break
        tau += 2.0 * r
    return tau


def block3_stats(d, n_blocks=N_BLOCKS_PRIMARY):
    """Registered primary inference: contiguous near-equal partition into
    n_blocks physical-time blocks, one-sided t on the block means.  For df=2 the
    Student-t CDF is closed-form: F(t) = 1/2 + t / (2*sqrt(2 + t^2))."""
    d = np.asarray(d, np.float64)
    n = len(d)
    cuts = [round(i * n / n_blocks) for i in range(n_blocks + 1)]
    means = np.array([d[cuts[i]:cuts[i + 1]].mean() for i in range(n_blocks)])
    m = means.mean()
    s = means.std(ddof=1)
    tstat = float(m / (s / math.sqrt(n_blocks))) if s > 0 else float("inf")
    if n_blocks == 3:
        p_one_sided = float(1.0 - (0.5 + tstat / (2.0 * math.sqrt(2.0 + tstat ** 2))))
    else:
        p_one_sided = None
    return {"block_means": [float(v) for v in means], "mean": float(m),
            "sd_block_means": float(s), "t": tstat, "p_one_sided_df2": p_one_sided,
            "all_blocks_positive": bool((means > 0).all()),
            "block_sizes": [cuts[i + 1] - cuts[i] for i in range(n_blocks)]}


def time_block_boot(d, block, B, rng):
    n = len(d)
    if block > n:
        return None
    nb = int(math.ceil(n / block))
    starts = np.arange(n - block + 1)
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(len(starts), nb)
        out[b] = np.concatenate([d[starts[p]:starts[p] + block] for p in pick]).mean()
    return out


# ============================== main ==========================================
def main():
    t_start = time.time()
    data_dir = pathlib.Path(A.data)
    meta = json.loads((data_dir / "case2_channel_meta.json").read_text())
    nu = meta["scaling"]["nu"]
    utau = meta["scaling"]["u_tau"]
    dy = meta["geometry"]["dy"]
    delta = meta["geometry"]["delta"]
    arr_path = data_dir / "case2_channel_f16.npy"
    vol = np.load(arr_path, mmap_mode="r")
    T, NW, ny, nx, _ = vol.shape
    log(f"[data] {arr_path} {vol.shape} nu={nu:.4e} u_tau={utau:.5f} "
        f"Re_tau={meta['scaling']['Re_tau']:.1f}")
    log("[axis] the 100-point tangential axis is SPANWISE z "
        "(streak negative lobe at lag 6-10, measured, node_011 correction)")

    VOL = torch.from_numpy(np.asarray(vol, np.float16)).to(DEV)
    VOL = VOL.permute(0, 1, 4, 2, 3).contiguous()          # (T, 2, 3, ny, nx)

    y = (np.arange(ny) + 0.5) * dy
    y_plus = y * utau / nu
    y_d = y / delta
    y_t = torch.from_numpy(y.astype(np.float32)).to(DEV)

    # ---- scored mask ---------------------------------------------------------
    read = np.zeros(ny, bool)
    for j in (J_M1, J_M2):
        read[max(0, j - READ_GUARD): j + READ_GUARD + 1] = True
    read[:K_WALL] = True
    scorable_rows = (np.arange(ny) >= N_C) & (~read)
    base_regions = {
        "buffer_yp_lt30": scorable_rows & (y_plus <= 30.0),
        "log_yp30_100": scorable_rows & (y_plus > 30.0) & (y_plus <= 100.0),
        "outer_yp_gt100": scorable_rows & (y_plus > 100.0),
        "whole_scorable": scorable_rows,
    }
    regions = {k: torch.from_numpy(np.broadcast_to(v[:, None], (ny, nx)).copy()).to(DEV)
               for k, v in base_regions.items()}
    log(f"[support] rows 0..{N_C-1} (y+ <= {y_plus[N_C-1]:.2f}); read rows "
        f"{J_M1},{J_M2} (+-{READ_GUARD}); scored rows {int(scorable_rows.sum())}/{ny}")

    # STATIC construction assert: the enforcement/support rows and the scored
    # rows are disjoint BY INDEX before anything runs.
    assert not (scorable_rows[:N_C]).any(), "support rows leak into the scored mask"

    geom_ch = torch.from_numpy(
        np.broadcast_to((y_d[:, None]).astype(np.float32), (1, ny, nx)).copy()).to(DEV)
    flag_np = np.zeros((1, ny, nx), np.float32)
    flag_np[:, :N_C, :] = 1.0
    flag_ch = torch.from_numpy(flag_np).to(DEV)

    # ---- normalisation: NEW TRAIN window only --------------------------------
    tr = np.arange(TRAIN_W[0], TRAIN_W[1] + 1)
    sub = VOL[torch.from_numpy(tr).to(DEV)].float()
    mu = sub.mean((0, 1, 3, 4))
    sd = sub.std((0, 1, 3, 4)) + 1e-8
    mu_t = mu[:, None, None]
    sd_t = sd[:, None, None]
    mean_field = sub.mean((0, 1)).contiguous()
    del sub
    torch.cuda.empty_cache()
    log(f"[norm] TRAIN {TRAIN_W}: mu={[round(v,5) for v in mu.tolist()]} "
        f"sd={[round(v,5) for v in sd.tolist()]}")

    wall = Wall(nu, dy, nx, mu, sd)

    def get(i, w):
        return VOL[int(i), int(w)].float()

    def normed(f):
        return (f - mu_t) / sd_t

    def r2(pred, tgt, mask):
        return r2_fluct(pred, tgt, mask, mean_field)

    # ---- precompute exact tractions for the WHOLE record (cheap, wall rows only)
    with torch.no_grad():
        TAU_ALL = torch.stack([torch.stack([wall.exact_traction(get(i, w))
                                            for w in range(NW)]) for i in range(T)])
    tau_mu = TAU_ALL[torch.from_numpy(tr).to(DEV)].mean((0, 1, 2))       # (2,)
    tau_sd = TAU_ALL[torch.from_numpy(tr).to(DEV)].std((0, 1, 2)) + 1e-12
    s_data = float(TAU_ALL[torch.from_numpy(tr).to(DEV)].norm(dim=-1).mean())
    log(f"[scale] TRAIN mean|tau|={s_data:.6e} mu={tau_mu.tolist()} sd={tau_sd.tolist()}")

    # ---- integral time measured on TRAIN (the frozen legacy value is a floor) --
    flux_series = TAU_ALL[torch.from_numpy(tr).to(DEV), :, :, 0].mean((1, 2)).cpu().numpy()
    tau_train = tau_of_series(flux_series)
    tau_used = max(TAU_LEGACY, tau_train)
    block_native = int(math.ceil(BLOCK_CONST * tau_used))
    block_strided = int(math.ceil(block_native / STRIDE))
    log(f"[tau] train-measured {tau_train:.1f} native frames; frozen floor {TAU_LEGACY}; "
        f"conservative block {block_native} native = {block_strided} strided")

    # ---- corruption machinery (variance-preserving fidelity ladder) -----------
    def corr_noise(shape_bwn2, ell, gen=None):
        """Unit-variance spanwise-correlated noise, circular Gaussian filter."""
        b = torch.randn(shape_bwn2, device=DEV, generator=gen)
        if ell <= 0.5:
            return b
        k = int(max(3, min(nx - 1, round(6 * ell)))) | 1
        xs = torch.arange(k, device=DEV, dtype=torch.float32) - k // 2
        g = torch.exp(-0.5 * (xs / ell) ** 2)
        g = (g / g.sum()).view(1, 1, k)
        v = b.permute(0, 2, 1).reshape(-1, 1, nx)             # (B*2, 1, nx)
        v = F.pad(v, (k // 2, k // 2), mode="circular")
        v = F.conv1d(v, g).reshape(shape_bwn2[0], 2, nx).permute(0, 2, 1)
        return v / v.std(dim=1, keepdim=True).clamp_min(1e-12)

    def corrupt(tau, r, ell, white, gen=None):
        """Variance-preserving corruption: corr(out, tau) = r, marginal scale kept.
        tau (B, nx, 2)."""
        if white:
            xi = torch.randn(tau.shape, device=DEV, generator=gen)
        else:
            xi = corr_noise(tau.shape, ell, gen)
        return (tau_mu + r * (tau - tau_mu)
                + math.sqrt(max(0.0, 1.0 - r * r)) * tau_sd * xi)

    # ---- conditioning build ----------------------------------------------------
    C_COND = 2 + 1 + 1        # tau(2) + geometry(1) + presence flag(1)
    inv_s = 1.0 / max(s_data, 1e-12)

    def lift(tau):
        out = torch.zeros((tau.shape[0], 2, ny, nx), device=DEV)
        out[:, :, :N_C, :] = tau.permute(0, 2, 1)[:, :, None, :]
        return out

    def cond_from_tau(tau, present):
        """tau (B, nx, 2) or None -> (B, C_COND, ny, nx)."""
        B = tau.shape[0] if tau is not None else present.shape[0]
        w = lift(tau) * inv_s if tau is not None else torch.zeros((B, 2, ny, nx), device=DEV)
        fl = flag_ch[None].expand(B, -1, -1, -1) * present[:, None, None, None]
        gm = geom_ch[None].expand(B, -1, -1, -1)
        return torch.cat([w, gm, fl], 1)

    def bc_from_tau(tau):
        """Viscous-sublayer solution implied by tau on rows 0..N_C-1, normalized.
        du/dy|_w = -tau_x/nu (wall-on-fluid sign), u_j = (du/dy)*y_j, v_j = 0."""
        B = tau.shape[0]
        g = -tau / nu                                          # (B, nx, 2)
        prof = torch.zeros((B, 3, N_C, nx), device=DEV)
        yj = y_t[:N_C].view(1, N_C, 1)
        prof[:, 0] = g[:, :, 0][:, None, :] * yj
        prof[:, 2] = g[:, :, 1][:, None, :] * yj
        return (prof - mu_t[None, :, :1, :]) / sd_t[None, :, :1, :]

    # ---- generator training: quality-agnostic single slot ----------------------
    def train_one(family, seed, width, param, steps, label):
        torch.manual_seed(seed)
        gen = Generator(family, 3 + C_COND, 3, width, param)
        opt = torch.optim.Adam(gen.net.parameters(), lr=2e-4)
        r = np.random.default_rng(seed)
        gt = torch.Generator(device=DEV).manual_seed(seed + 5150)
        t0 = time.time()
        for it in range(steps):
            ii = torch.from_numpy(r.choice(tr, A.batch)).to(DEV)
            ww = torch.from_numpy(r.integers(0, NW, A.batch)).to(DEV)
            x0 = ((VOL[ii, ww].float() - mu_t) / sd_t)
            tau = TAU_ALL[ii, ww]                              # (B, nx, 2)
            pres = (torch.rand(A.batch, device=DEV, generator=gt) > P_ABSENT).float()
            rr = float(r.uniform(*R_RANGE))
            white = bool(r.random() >= P_CORRELATED)
            ell = float(r.uniform(*ELL_RANGE))
            tau_c = corrupt(tau, rr, ell, white, gt)
            tau_in = tau_c * pres[:, None, None]
            cond = cond_from_tau(tau_in, pres)
            loss = gen.loss(x0, cond)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.net.parameters(), 1.0)
            opt.step()
            gen.update_ema()
            if it % 1000 == 0:
                log(f"  [{label}] {it}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
        return gen

    # ---- closures: TRAIN window only -------------------------------------------
    ckpt_closure = RESULTS / f"{TAG}_closures.pt"
    torch.manual_seed(4243)
    closure = ConvPhysicsClosure(13).to(DEV)
    dataonly = ConvDataOnly(13).to(DEV)
    if A.phase == "final":
        st = torch.load(ckpt_closure, map_location=DEV, weights_only=True)
        closure.load_state_dict(st["closure"])
        dataonly.load_state_dict(st["dataonly"])
        log(f"[closure] loaded frozen {ckpt_closure.name} sha {sha256(ckpt_closure)[:16]}")
    else:
        opt_c = torch.optim.Adam(closure.parameters(), lr=2e-3)
        opt_d = torch.optim.Adam(dataonly.parameters(), lr=2e-3)
        rng = np.random.default_rng(4243)
        for it in range(A.closure_steps):
            i = int(rng.choice(tr))
            w = int(rng.integers(NW))
            f = get(i, w)
            tgt = wall.exact_traction(f)
            s = tgt.norm(dim=-1).mean().clamp_min(1e-12)
            lc = ((wall.closure_traction(closure, f) - tgt) / s).pow(2).mean()
            opt_c.zero_grad()
            lc.backward()
            opt_c.step()
            ld = ((wall.dataonly_traction(dataonly, f, s_data) - tgt) / s).pow(2).mean()
            opt_d.zero_grad()
            ld.backward()
            opt_d.step()
            if it % 500 == 0:
                log(f"  [closure] {it} physics {lc.item():.5f} | data-only {ld.item():.5f}")
        torch.save({"closure": closure.state_dict(), "dataonly": dataonly.state_dict()},
                   ckpt_closure)
    for m in (closure, dataonly):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # ---- evaluation windows -----------------------------------------------------
    if A.phase == "rehearsal":
        eval_idx = np.arange(REH_W[0], REH_W[1] + 1, STRIDE)
        donor_lag = LAG_REHEARSAL
    else:
        eval_idx = np.arange(FINAL_W[0], FINAL_W[1] + 1, STRIDE)
        donor_lag = LAG_FINAL
    donors = eval_idx + donor_lag
    assert donors.min() >= TRAIN_W[0] and donors.max() <= TRAIN_W[1], \
        "wrong-time donors must all lie inside TRAIN"
    assert (np.abs(donors - eval_idx) >= 4 * TAU_LEGACY - 1).all(), \
        "every wrong-time donor must be >= 4 integral times away"
    assert len(np.unique(donors)) == len(donors), "donor pile-up"
    log(f"[split] phase={A.phase} eval n={len(eval_idx)} window "
        f"[{eval_idx.min()},{eval_idx.max()}] stride {STRIDE}; donor lag {donor_lag} "
        f"(= {donor_lag/TAU_LEGACY:.1f} tau, all donors in TRAIN)")

    perm = torch.from_numpy(
        np.random.default_rng(SHUFFLE_SEED).permutation(nx)).to(DEV).long()

    # matched arm parameters
    if A.phase == "final":
        assert A.matched_r is not None and A.matched_ell is not None, \
            "FINAL phase requires the frozen v2 --matched-r/--matched-ell"
        matched_r, matched_ell = float(A.matched_r), float(A.matched_ell)
        matched_name = "matched_noise"
    else:
        matched_r, matched_ell = 0.60, 6.0     # pipeline probe only, provisional
        matched_name = "matched_probe"

    ARMS = ["absent", "exact",
            "ladder_r0900", "ladder_r0750", "ladder_r0625", "ladder_r0550",
            matched_name, "closure", "equilibrium", "learned_dataonly",
            "wrong_time", "shuffle_z"]
    DIFF_ARMS = ["absent", "exact", "closure"]
    LADDER = dict(zip(["ladder_r0900", "ladder_r0750", "ladder_r0625", "ladder_r0550"],
                      LADDER_R))

    garm = torch.Generator(device=DEV)          # deterministic per-arm noise for ladder
    ARM_ID = {a: k for k, a in enumerate(ARMS)}

    def arm_seed(arm, i, w):
        # deterministic (unlike Python's salted hash()): reproducible across runs
        return (ARM_ID[arm] * 1000003 + int(i) * 131 + int(w) * 17 + 977) % (2 ** 31)

    def arm_tau(arm, i, w):
        """Returns (tau (nx,2) or None).  Ladder noise is a fixed deterministic
        function of (arm, frame, wall) so every generator seed and member sees the
        same corrupted line."""
        f = get(i, w)
        if arm == "absent":
            return None
        if arm == "exact":
            return TAU_ALL[int(i), int(w)]
        if arm in LADDER:
            garm.manual_seed(arm_seed(arm, i, w))
            return corrupt(TAU_ALL[int(i), int(w)][None], LADDER[arm], 0.0, True,
                           garm)[0]
        if arm == matched_name:
            garm.manual_seed(arm_seed(arm, i, w))
            return corrupt(TAU_ALL[int(i), int(w)][None], matched_r, matched_ell,
                           matched_ell <= 0.5, garm)[0]
        if arm == "closure":
            return wall.closure_traction(closure, f)
        if arm == "equilibrium":
            return wall.closure_traction(closure, f, eqwm=True)
        if arm == "learned_dataonly":
            return wall.dataonly_traction(dataonly, f, s_data)
        if arm == "wrong_time":
            j = int(i) + donor_lag
            return wall.closure_traction(closure, get(j, w))
        if arm == "shuffle_z":
            return wall.closure_traction(closure, f)[perm]
        raise ValueError(arm)

    # ---- MACHINE-VERIFIED SUPPORT CHECK (before any score exists) ---------------
    srow_t = torch.from_numpy(scorable_rows).to(DEV)
    sup = {}
    for arm in ARMS:
        ta = arm_tau(arm, int(eval_idx[0]), 0)
        pres = torch.ones(1, device=DEV) if ta is not None else torch.zeros(1, device=DEV)
        c = cond_from_tau(ta[None] if ta is not None else None, pres)[0]
        chan = torch.cat([c[:2], c[3:4]], 0)     # tau channels + presence flag
        sup[arm] = float(chan[:, srow_t, :].abs().max())
    log("[support-check] max |wall-derived conditioning| on SCORED cells: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sup.items()))
    for arm in ARMS:
        assert sup[arm] == 0.0, f"arm {arm} leaks conditioning into the scored region"
    log("[support-check] PASS: zero on all scored cells for every arm; the hard "
        f"Neumann enforcement writes only rows 0..{N_C-1}, disjoint by the static assert")

    # ---- closure fidelity (a-priori centred R2 vs exact) on this window ---------
    def fidelity(idx_list):
        arms_f = ["closure", "equilibrium", "learned_dataonly", "wrong_time",
                  "shuffle_z", matched_name] + list(LADDER)
        num = {k: 0.0 for k in arms_f}
        den = 0.0
        for i in idx_list:
            for w in range(NW):
                t_ex = TAU_ALL[int(i), int(w)]
                for k in arms_f:
                    p = arm_tau(k, int(i), int(w))
                    num[k] += float(((p - t_ex) ** 2).sum())
                den += float(((t_ex - t_ex.mean(0, keepdim=True)) ** 2).sum())
        return {k: 1.0 - v / den for k, v in num.items()}

    fid = fidelity(eval_idx)
    log(f"[fidelity centred R2 vs exact, this window] "
        + ", ".join(f"{k}={v:+.4f}" for k, v in fid.items()))

    # closure error spanwise correlation length (for the v2 matched arm) ---------
    def closure_error_ell(idx_list):
        acs = []
        for i in idx_list:
            for w in range(NW):
                e = (arm_tau("closure", int(i), int(w))
                     - TAU_ALL[int(i), int(w)])[:, 0].cpu().numpy()
                e = e - e.mean()
                den = float((e * e).mean())
                if den <= 0:
                    continue
                ac = [float((e[:-l] * e[l:]).mean() / den) if l else 1.0
                      for l in range(min(30, nx // 2))]
                acs.append(ac)
        ac = np.mean(np.array(acs), 0)
        ell = 0.5
        for l in range(1, len(ac)):
            if ac[l] <= 0:
                break
            ell += ac[l]
        return float(ell), [float(v) for v in ac]

    ell_c, ac_c = closure_error_ell(eval_idx[:8])
    log(f"[closure-error] spanwise integral correlation length ~ {ell_c:.2f} cells")

    # ---- train or load generators ------------------------------------------------
    gens = {}
    fams = [("flow_matching", FM_SEEDS[:A.seeds], 128, "v"),
            ("diffusion", DIFF_SEEDS[:A.seeds], DIFF_WIDTH, DIFF_PARAM)]
    ckpt_shas = {}
    for family, seeds, width, param in fams:
        for seed in seeds:
            ck = RESULTS / f"{TAG}_{family}_s{seed}.pt"
            g = None
            if A.phase == "final" or ck.exists():
                g = Generator(family, 3 + C_COND, 3, width, param)
                g.ema.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
                log(f"[gen] loaded frozen {ck.name} sha {sha256(ck)[:16]}")
            else:
                g = train_one(family, seed, width, param, A.steps_train,
                              f"{family}/s{seed}")
                torch.save(g.ema.state_dict(), ck)
            ckpt_shas[ck.name] = sha256(ck)
            gens[(family, seed)] = g

    # ---- evaluation ---------------------------------------------------------------
    band_masks = {"log": regions["log_yp30_100"], "buffer": regions["buffer_yp_lt30"]}

    store = {}
    results_arms = {}
    for (family, seed), gen in gens.items():
        arms_here = ARMS if family == "flow_matching" else DIFF_ARMS
        for arm in arms_here:
            gsam = torch.Generator(device=DEV).manual_seed(seed * 7 + 13)
            gfix = torch.Generator(device=DEV).manual_seed(seed * 11 + 29)
            per = {k: [] for k in regions}
            per_w = {f"{k}@w{w}": [] for k in regions for w in range(NW)}
            crps, es = [], []
            flux_err = {b: [] for b in band_masks}
            flux_pred = {b: [] for b in band_masks}
            flux_true = {b: [] for b in band_masks}
            wallgrad_dev = []
            ranks = np.zeros(A.members + 1, np.int64)
            rr = np.random.default_rng(seed)
            for i in eval_idx:
                fr = {k: [] for k in regions}
                ec, ee = [], []
                fe = {b: [] for b in band_masks}
                fp = {b: [] for b in band_masks}
                ftv = {b: [] for b in band_masks}
                for w in range(NW):
                    tgt = get(int(i), w)
                    ta = arm_tau(arm, int(i), w)
                    pres = (torch.ones(1, device=DEV) if ta is not None
                            else torch.zeros(1, device=DEV))
                    c = cond_from_tau(ta[None] if ta is not None else None,
                                      pres)[0][None].expand(A.members, -1, -1, -1)
                    bc = (bc_from_tau(ta[None]).expand(A.members, -1, -1, -1)
                          if ta is not None else None)
                    ens = gen.sample(c, A.sampler_steps, gsam, bc=bc, gfix=gfix) \
                        * sd_t + mu_t
                    pm = ens.mean(0)
                    for k, m in regions.items():
                        v = r2(pm, tgt, m)
                        fr[k].append(v)
                        per_w[f"{k}@w{w}"].append(v)
                    ec.append(crps_ensemble(ens, tgt, regions["whole_scorable"]))
                    ee.append(energy_score(ens, tgt, regions["whole_scorable"]))
                    ranks += rank_hist(ens, tgt, regions["whole_scorable"], rr)
                    # engineering endpoint: per-member tangential momentum flux
                    # <u'v'> in each scored band, then member-averaged
                    for b, bm in band_masks.items():
                        up = (ens[:, 0] - mean_field[0])[..., bm]
                        vp = (ens[:, 1] - mean_field[1])[..., bm]
                        mp = float((up * vp).mean(dim=1).mean())
                        ut = (tgt[0] - mean_field[0])[bm]
                        vt = (tgt[1] - mean_field[1])[bm]
                        mt = float((ut * vt).mean())
                        fe[b].append(abs(mp - mt))
                        fp[b].append(mp)
                        ftv[b].append(mt)
                    # flux-consistency: generated wall gradient vs supplied traction
                    if ta is not None:
                        g_pm = wall.exact_traction(pm)
                        wallgrad_dev.append(float((g_pm - ta).norm(dim=-1).mean()
                                                  / max(s_data, 1e-12)))
                for k in regions:
                    per[k].append(float(np.mean(fr[k])))
                crps.append(float(np.mean(ec)))
                es.append(float(np.mean(ee)))
                for b in band_masks:
                    flux_err[b].append(float(np.mean(fe[b])))
                    flux_pred[b].append(float(np.mean(fp[b])))
                    flux_true[b].append(float(np.mean(ftv[b])))
            key = f"{family}|{arm}|{seed}"
            store[key] = {k: np.asarray(v) for k, v in per.items()}
            store[key].update({k: np.asarray(v) for k, v in per_w.items()})
            store[key]["crps"] = np.asarray(crps)
            store[key]["energy"] = np.asarray(es)
            for b in band_masks:
                store[key][f"flux_abserr_{b}"] = np.asarray(flux_err[b])
                store[key][f"flux_pred_{b}"] = np.asarray(flux_pred[b])
                store[key][f"flux_true_{b}"] = np.asarray(flux_true[b])
            store[key]["ranks"] = ranks
            results_arms[key] = {
                **{k: float(np.mean(v)) for k, v in per.items()},
                **{k: float(np.mean(v)) for k, v in per_w.items()},
                "crps": float(np.mean(crps)), "energy_score": float(np.mean(es)),
                "rank_hist": ranks.tolist(),
                **{f"flux_abserr_{b}": float(np.mean(flux_err[b])) for b in band_masks},
                "wallgrad_consistency_relerr": (float(np.mean(wallgrad_dev))
                                                if wallgrad_dev else None),
            }
            log(f"[eval] {key}: "
                + " ".join(f"{k}={np.mean(v):+.5f}" for k, v in per.items())
                + f" | flux_log={np.mean(flux_err['log']):.3e}"
                + (f" | wallgrad_rel={np.mean(wallgrad_dev):.2e}" if wallgrad_dev else ""))
        torch.cuda.empty_cache()

    # ---- paired contrasts ----------------------------------------------------------
    rngb = np.random.default_rng(20260803)

    def paired(family, a, b, series_key, seeds, flip=False):
        D = None
        for s in seeds:
            ka, kb = f"{family}|{a}|{s}", f"{family}|{b}|{s}"
            if ka not in store or kb not in store:
                return None
            dd = store[ka][series_key] - store[kb][series_key]
            D = dd if D is None else D + dd
        d = (D / len(seeds)) * (-1.0 if flip else 1.0)
        tau_d = tau_of_series(d)
        block_meas = max(1, int(math.ceil(BLOCK_CONST * tau_d)))
        block = max(block_strided, block_meas)
        out = {"delta": float(d.mean()), "primary": block3_stats(d),
               "per_seed_delta": [float(((store[f'{family}|{a}|{s}'][series_key]
                                          - store[f'{family}|{b}|{s}'][series_key])
                                         * (-1.0 if flip else 1.0)).mean())
                                  for s in seeds],
               "tau_scored_series_strided": float(tau_d),
               "block_used_bootstrap": block,
               "n_bootstrap_units": float(len(d) / block),
               "n_scored_frames": int(len(d)),
               "frac_frames_positive": float((d > 0).mean())}
        if series_key in base_regions:
            for w in range(NW):
                kw = f"{series_key}@w{w}"
                dw = np.mean([store[f"{family}|{a}|{s}"][kw]
                              - store[f"{family}|{b}|{s}"][kw] for s in seeds], axis=0)
                dw = dw * (-1.0 if flip else 1.0)
                out[f"delta_wall{w}"] = float(dw.mean())
            out["walls_agree_in_sign"] = bool(
                np.sign(out["delta_wall0"]) == np.sign(out["delta_wall1"]))
        bs = time_block_boot(d, block, A.boot, rngb)
        if bs is None or len(d) / block < 3.0:
            out["boot_ci95"] = None
            out["boot_status"] = "UNAVAILABLE (fewer than 3 units at measured scale)"
        else:
            out["boot_ci95"] = [float(np.percentile(bs, 2.5)),
                                float(np.percentile(bs, 97.5))]
            out["boot_status"] = ("SECONDARY moving-block percentile range; coverage "
                                  "at this record length is NOT demonstrated and no "
                                  "population-95% claim is attached")
        return out

    PAIRS = [("closure", "absent"), ("exact", "absent"),
             ("ladder_r0900", "absent"), ("ladder_r0750", "absent"),
             ("ladder_r0625", "absent"), ("ladder_r0550", "absent"),
             (matched_name, "absent"), ("closure", "equilibrium"),
             ("closure", "learned_dataonly"), ("closure", "wrong_time"),
             ("closure", "shuffle_z"), ("closure", "exact"),
             ("wrong_time", "absent"), ("shuffle_z", "absent"),
             ("learned_dataonly", "absent"), ("equilibrium", "absent"),
             ("closure", matched_name)]
    contrasts = {}
    for family, seeds in (("flow_matching", FM_SEEDS[:A.seeds]),
                          ("diffusion", DIFF_SEEDS[:A.seeds])):
        arms_here = ARMS if family == "flow_matching" else DIFF_ARMS
        for region in list(base_regions):
            for a, b in PAIRS:
                if a not in arms_here or b not in arms_here:
                    continue
                r = paired(family, a, b, region, seeds)
                if r:
                    contrasts[f"{family}|{a}-{b}|{region}"] = r
        # engineering + distributional endpoints (lower is better -> flip sign)
        for a, b in PAIRS:
            if a not in arms_here or b not in arms_here:
                continue
            for bnd in band_masks:
                r = paired(family, a, b, f"flux_abserr_{bnd}", seeds, flip=True)
                if r:
                    contrasts[f"{family}|{a}-{b}|flux_{bnd}"] = r
            r = paired(family, a, b, "crps", seeds, flip=True)
            if r:
                contrasts[f"{family}|{a}-{b}|crps"] = r

    # ---- dose-response curve: gain vs measured fidelity -----------------------------
    curve = []
    for arm in ["exact"] + list(LADDER):
        c = contrasts.get(f"flow_matching|{arm}-absent|whole_scorable")
        if c:
            curve.append({"arm": arm,
                          "fidelity": 1.0 if arm == "exact" else fid[arm],
                          "gain": c["delta"]})
    curve.sort(key=lambda e: e["fidelity"])

    def curve_predict(f_query):
        xs = [e["fidelity"] for e in curve]
        ys = [e["gain"] for e in curve]
        return float(np.interp(f_query, xs, ys))

    dose = {"curve": curve,
            "spearman_fidelity_gain": None,
            "closure_point": {"fidelity": fid["closure"],
                              "gain": contrasts.get(
                                  "flow_matching|closure-absent|whole_scorable",
                                  {}).get("delta"),
                              "curve_predicted_gain": curve_predict(fid["closure"])},
            "learned_dataonly_point": {"fidelity": fid["learned_dataonly"],
                                       "gain": contrasts.get(
                                           "flow_matching|learned_dataonly-absent|whole_scorable",
                                           {}).get("delta"),
                                       "curve_predicted_gain": curve_predict(
                                           fid["learned_dataonly"])},
            "equilibrium_point": {"fidelity": fid["equilibrium"],
                                  "gain": contrasts.get(
                                      "flow_matching|equilibrium-absent|whole_scorable",
                                      {}).get("delta"),
                                  "curve_predicted_gain": curve_predict(
                                      fid["equilibrium"])}}
    if len(curve) >= 3:
        xs = np.array([e["fidelity"] for e in curve])
        ys = np.array([e["gain"] for e in curve])
        rx = np.argsort(np.argsort(xs))
        ry = np.argsort(np.argsort(ys))
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        dose["spearman_fidelity_gain"] = float(
            (rx * ry).sum() / math.sqrt((rx * rx).sum() * (ry * ry).sum()))

    # ---- write ------------------------------------------------------------------------
    OUT = RESULTS / f"{TAG}_{A.phase}_results.json"
    COMP = RESULTS / f"{TAG}_{A.phase}_components.npz"
    results = {"_schema": "node_011 single-slot quality-agnostic flux-consistent "
                          "traction interface, smooth-wall channel (spanwise-y plane)",
               "_meta": {
                   "phase": A.phase, "record": str(arr_path),
                   "record_sha256": meta["outputs"]["array_sha256"],
                   "source_sha256": meta["source"]["sha256"],
                   "nu": nu, "u_tau": utau, "Re_tau": meta["scaling"]["Re_tau"],
                   "dy": dy, "delta": delta, "n_y": ny, "n_x": nx, "n_walls": NW,
                   "tangential_axis": "spanwise z (node_011 correction, measured "
                                      "streak lobe at lag 6-10)",
                   "train": TRAIN_W, "rehearsal": REH_W, "final": FINAL_W,
                   "stride": STRIDE, "eval_window": [int(eval_idx.min()),
                                                     int(eval_idx.max())],
                   "n_eval_frames": int(len(eval_idx)),
                   "tau_train_measured": float(tau_train),
                   "tau_legacy_floor": TAU_LEGACY, "tau_used": float(tau_used),
                   "block_const": BLOCK_CONST, "block_native": block_native,
                   "block_strided": block_strided,
                   "n_blocks_primary": N_BLOCKS_PRIMARY,
                   "donor_lag": donor_lag,
                   "N_C": N_C, "support_y_plus_max": float(y_plus[N_C - 1]),
                   "J_M1": J_M1, "J_M2": J_M2,
                   "y_plus_m1": float(y_plus[J_M1]), "y_plus_m2": float(y_plus[J_M2]),
                   "K_WALL": K_WALL, "read_guard": READ_GUARD,
                   "n_scored_rows": int(scorable_rows.sum()),
                   "steps_train": A.steps_train, "closure_steps": A.closure_steps,
                   "members": A.members, "sampler_steps": A.sampler_steps,
                   "seeds": A.seeds, "fm_seeds": FM_SEEDS[:A.seeds],
                   "diff_seeds": DIFF_SEEDS[:A.seeds],
                   "diffusion_config": f"{DIFF_PARAM}_cosine_w{DIFF_WIDTH} "
                                       "(inherited frozen node_010 DEV selection)",
                   "training_corruption": {"p_absent": P_ABSENT, "r_range": R_RANGE,
                                           "p_correlated": P_CORRELATED,
                                           "ell_range": ELL_RANGE,
                                           "form": "variance-preserving: mu + r*(tau-mu)"
                                                   " + sqrt(1-r^2)*sd*xi"},
                   "ladder_r": LADDER_R,
                   "matched_arm": {"name": matched_name, "r": matched_r,
                                   "ell": matched_ell},
                   "mixture_note": "SINGLE traction modality in training; every "
                                   "evaluated estimator has identical (zero) "
                                   "generator-side training exposure",
                   "hard_neumann": f"support rows 0..{N_C-1} replaced along the whole "
                                   "sampling path by the viscous-sublayer solution of "
                                   "the supplied traction (noise-consistent)",
                   "script_sha256": sha256(pathlib.Path(__file__)),
                   "checkpoint_sha256": ckpt_shas,
                   "closure_ckpt_sha256": sha256(ckpt_closure),
                   "support_leak_check": sup,
                   "s_data": s_data,
                   "tau_mu": [float(v) for v in tau_mu.tolist()],
                   "tau_sd": [float(v) for v in tau_sd.tolist()],
                   "inference_unit": "physical time; walls averaged within each frame; "
                                     "primary = 3 contiguous block means, one-sided t",
               },
               "fidelity_centred_r2": fid,
               "closure_error_spanwise_ell_cells": ell_c,
               "closure_error_autocorr": ac_c,
               "arms": results_arms,
               "contrasts": contrasts,
               "dose_response": dose}
    np.savez_compressed(COMP, **{f"{k}|{kk}": vv for k, v in store.items()
                                 for kk, vv in v.items()},
                        eval_idx=eval_idx, y_plus=y_plus, scorable_rows=scorable_rows)
    results["_meta"]["gpu_hours"] = (time.time() - t_start) / 3600.0
    results["_meta"]["components_sha256"] = sha256(COMP)
    OUT.write_text(json.dumps(results, indent=2))
    (RESULTS / f"{TAG}_{A.phase}_results.sha256").write_text(
        sha256(OUT) + "  " + OUT.name + "\n")
    log(f"[done] {OUT} ({results['_meta']['gpu_hours']:.3f} GPU-h)")
    log("=== done ===")


if __name__ == "__main__":
    main()
