#!/usr/bin/env python3
"""Decisive test of the closure-to-generator interface (link L_B of the E2 chain).

Protocol frozen in
``development/nodes/node_004/PREREGISTRATION_E2_INTERFACE.md`` before this script
was launched.

What this does
--------------
The published cube experiment supplies the generator with the *complete* held-out
LES near-wall band (three velocity components on two wall-adjacent cell layers).
A wall closure can never supply that.  A wall closure supplies exactly one signed
two-component local-frame wall-on-fluid traction per wall point, which a declared
lift must expand onto the generator's conditioning support.

This script keeps the generator, the split, the sampler noise, the targets and the
scoring masks byte-identical to the published run and changes only *what is written
into the conditioning band*:

  correct                   oracle full LES band                     (reproduction guard)
  no_wall                   absent band                              (reproduction guard)
  wrong_wall                far-time donor band                      (reproduction guard)
  correct_physwall          oracle band on the physical-wall support (matched ceiling)
  tau_lift_oracle           frozen lift of the target's own traction (PRIMARY: interface ceiling)
  tau_lift_fartime          frozen lift of the far-time traction     (adverse control)
  tau_lift_trainmean        frozen lift of the mean wall load        (information-matched null)
  tau_lift_model_predicted  frozen lift of traction inverted from the
                            no_wall posterior mean's own anchor      (no target wall information)

The four lift/physwall arms condition on the physical-wall band only (floor + cube).
The computational lid is excluded from that support because the lid-adjacent raster row
is identically zero at every retained time, so no equilibrium wall observation exists
there; ``correct_physwall`` is the ceiling matched to exactly that support.  Scoring
masks are unchanged from the published run, so every arm is scored on the identical
unsupplied volume.

Evidence boundary.  ``tau_lift_oracle`` uses target-derived traction and is therefore
an UPPER BOUND on what any wall closure could transmit through this interface.  It is
not a closure-performance measurement, and nothing here is solver-coupled WMLES.
No training, no fine-tuning, no new LES: the checkpoint is loaded EMA-complete.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "codes" / "results"

P = argparse.ArgumentParser()
P.add_argument("--case", default="/root/autodl-tmp/cube_les",
               help="directory holding the frozen cube record")
P.add_argument("--data-name", default="cube_ds2_float16.complete.npy")
P.add_argument("--ckpt", default=str(RESULTS / "cube3d_coupling_adequate.pt"))
P.add_argument("--members", type=int, default=8)
P.add_argument("--sample-steps", type=int, default=32)
P.add_argument("--boot", type=int, default=4000)
P.add_argument("--nmax", type=int, default=160)
P.add_argument("--smoke", action="store_true")
P.add_argument("--tag", default="e2_traction_interface")
A = P.parse_args()

if A.smoke:
    A.members, A.sample_steps, A.boot, A.nmax = 2, 3, 100, 8

# Import the frozen published producer for geometry / model / bootstrap, without
# letting it parse this script's argv or run anything.
sys.path.insert(0, str(HERE))
_saved_argv = sys.argv
sys.argv = ["eval_cube_3d_coupling.py"]
import eval_cube_3d_coupling as B  # noqa: E402
sys.argv = _saved_argv

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NU = 2.0e-4                      # cube_prod.par, yplus_preflight.json
DELTA = 1.0 / 24.0               # uniform cell size of the 48x96x48 model grid
D_ANCHOR = 0.5 * DELTA           # wall distance of the first fluid cell
KAPPA, C_REICH = 0.41, 7.8       # codes/closure/wall_closure.py:58, verbatim

OUT = RESULTS / f"{A.tag}_results.json"
COMP = RESULTS / f"{A.tag}_components.npz"

ARMS = ("correct", "no_wall", "wrong_wall", "correct_physwall", "tau_lift_oracle",
        "tau_lift_fartime", "tau_lift_trainmean", "tau_lift_model_predicted")
# Rasterisation audit, verified time-invariant over 9 snapshots spanning the record:
# nearest-GLL interpolation returns identical values for these Cartesian y-rows and
# their lower neighbour, and row 95 (the lid-adjacent cell) is identically zero.
DUP_ROWS = (41, 43, 51, 54, 55, 57, 58, 61, 65, 67, 68, 70, 71, 72, 73, 75, 76, 77,
            78, 79, 82, 83, 84, 85, 88, 89, 90, 91, 93, 94)
ZERO_ROW = 95
LID_SURFACE = "top"
PUBLISHED_M1 = {  # cube3d_coupling_adequate_results.json, reproduction guard
    "correct": {"full_support_excluded": 0.08948777489375381,
                "near_support_excluded_d_le_0p5h": 0.24156180692557772,
                "outer_d_gt_0p5h": -0.06622817328211594},
    "no_wall": {"full_support_excluded": -0.08435861067324524,
                "near_support_excluded_d_le_0p5h": -0.0904228213542122,
                "outer_d_gt_0p5h": -0.07818844374701106},
    "wrong_wall": {"full_support_excluded": -0.3235607952594748,
                   "near_support_excluded_d_le_0p5h": -0.49552407618890637,
                   "outer_d_gt_0p5h": -0.14758666289192401},
}
GUARD_TOL = 5e-3


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- wall model
def reichardt_uplus(yp: np.ndarray) -> np.ndarray:
    """codes/closure/wall_closure.py::reichardt_uplus, verbatim."""
    yp = np.asarray(yp, dtype=np.float64)
    return (np.log1p(KAPPA * yp) / KAPPA
            + C_REICH * (1.0 - np.exp(-yp / 11.0) - (yp / 11.0) * np.exp(-yp / 3.0)))


def invert_reichardt(u_a: np.ndarray, d_a: float, nu: float, iters: int = 70) -> np.ndarray:
    """Equilibrium wall-model inversion: solve u_a = u_tau * f_R(d_a u_tau / nu)."""
    lo = np.zeros_like(u_a, dtype=np.float64)
    hi = np.full_like(u_a, 10.0, dtype=np.float64)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = mid * reichardt_uplus(d_a * mid / nu) - u_a
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------- surface bookkeeping
# (name, axis, plane coordinate, fluid-side sign, in-plane extents on the other two axes)
SURFACES = [
    ("floor",    1, 0.0, +1, {0: (0.0, 2.0), 2: (0.0, 2.0)}),
    ("top",      1, 4.0, -1, {0: (0.0, 2.0), 2: (0.0, 2.0)}),
    ("cube_top", 1, 1.0, +1, {0: (0.5, 1.5), 2: (0.5, 1.5)}),
    ("cube_xlo", 0, 0.5, -1, {1: (0.0, 1.0), 2: (0.5, 1.5)}),
    ("cube_xhi", 0, 1.5, +1, {1: (0.0, 1.0), 2: (0.5, 1.5)}),
    ("cube_zlo", 2, 0.5, -1, {0: (0.5, 1.5), 1: (0.0, 1.0)}),
    ("cube_zhi", 2, 1.5, +1, {0: (0.5, 1.5), 1: (0.0, 1.0)}),
]


def build_interface(fluid, band):
    """Assign every band cell to one wall surface, with its anchor cell and normal.

    Ownership = smallest perpendicular distance among surfaces whose in-plane extent,
    dilated by two cells to capture edge-diagonal cells, contains the cell.  Ties are
    broken by the fixed SURFACE order above.  Deterministic; no fitting.
    """
    nx, ny, nz = 48, 96, 48
    xs = (np.arange(nx) + 0.5) * 2.0 / nx
    ys = (np.arange(ny) + 0.5) * 4.0 / ny
    zs = (np.arange(nz) + 0.5) * 2.0 / nz
    coords = [xs, ys, zs]
    IX, IY, IZ = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pos = [X, Y, Z]
    idx = [IX, IY, IZ]

    bmask = band
    bi = np.stack([IX[bmask], IY[bmask], IZ[bmask]], 1)          # (Nb,3) indices
    bp = np.stack([X[bmask], Y[bmask], Z[bmask]], 1)             # (Nb,3) positions
    nb = bi.shape[0]

    best_d = np.full(nb, np.inf)
    owner = np.full(nb, -1, np.int64)
    dil = 2.0 * DELTA
    for si, (name, ax, plane, sgn, ext) in enumerate(SURFACES):
        dp = sgn * (bp[:, ax] - plane)                            # >0 on the fluid side
        ok = dp > 0
        for oax, (lo, hi) in ext.items():
            ok &= (bp[:, oax] >= lo - dil) & (bp[:, oax] <= hi + dil)
        take = ok & (dp < best_d - 1e-12)
        best_d[take] = dp[take]
        owner[take] = si
    if (owner < 0).any():
        raise RuntimeError(f"{int((owner < 0).sum())} band cells unassigned to a wall surface")

    # Anchor index along the surface's own axis.
    anchor_axis_idx = np.empty(nb, np.int64)
    normals = np.zeros((nb, 3), np.float64)
    for si, (name, ax, plane, sgn, ext) in enumerate(SURFACES):
        sel = owner == si
        base = int(round(plane / DELTA))
        anchor_axis_idx[sel] = base if sgn > 0 else base - 1
        normals[sel, ax] = float(sgn)

    aidx = bi.copy()
    for si, (name, ax, plane, sgn, ext) in enumerate(SURFACES):
        sel = owner == si
        aidx[sel, ax] = anchor_axis_idx[sel]
    anchor_lin = np.ravel_multi_index((aidx[:, 0], aidx[:, 1], aidx[:, 2]), (nx, ny, nz))
    band_lin = np.ravel_multi_index((bi[:, 0], bi[:, 1], bi[:, 2]), (nx, ny, nz))

    flat_fluid = fluid.reshape(-1)
    if not flat_fluid[anchor_lin].all():
        raise RuntimeError("an anchor cell fell inside the solid")
    d_perp = best_d
    if not np.allclose(np.unique(np.round(d_perp / DELTA, 6)), [0.5, 1.5], atol=1e-6):
        log(f"[interface] perpendicular distances present: "
            f"{sorted(set(np.round(d_perp / DELTA, 4).tolist()))}")
    names = [s[0] for s in SURFACES]
    is_phys = owner != names.index(LID_SURFACE)
    phys_band = np.zeros(nx * ny * nz, bool)
    phys_band[band_lin[is_phys]] = True
    return {
        "band_idx": bi, "band_lin": band_lin, "anchor_lin": anchor_lin,
        "owner": owner, "normal": normals, "d_perp": d_perp,
        "surface_names": names, "is_phys": is_phys,
        "phys_band": phys_band.reshape(nx, ny, nz),
    }


def traction_from_field(V: np.ndarray, itf: dict):
    """Frozen estimator: signed two-component local-frame wall traction at each band
    cell's owning surface anchor.  V is physical velocity, shape (3, 48, 96, 48)."""
    flat = V.reshape(3, -1)
    va = flat[:, itf["anchor_lin"]].astype(np.float64)             # (3,Nb)
    n = itf["normal"].T                                            # (3,Nb)
    vn = (va * n).sum(0)
    vt = va - vn[None] * n
    ua = np.linalg.norm(vt, axis=0)
    that = np.divide(vt, np.maximum(ua, 1e-12)[None], where=True)
    utau = invert_reichardt(ua, D_ANCHOR, NU)
    tau_vec = (utau ** 2)[None] * that                             # wall-on-fluid traction
    return utau, that, tau_vec, ua


def lift_to_band(utau: np.ndarray, that: np.ndarray, itf: dict) -> np.ndarray:
    """Frozen surface-to-volume lift on the physical-wall support (floor + cube).

    The computational lid is excluded: its lid-adjacent raster row is identically
    zero at every retained time, so no equilibrium wall observation exists there.
    Every arm that uses this lift therefore conditions on exactly the same
    physical-wall support.
    """
    sel = itf["is_phys"]
    yp = itf["d_perp"][sel] * utau[sel] / NU
    mag = utau[sel] * reichardt_uplus(yp)
    v = that[:, sel] * mag[None]                                   # (3,Nphys)
    out = np.zeros((3, 48 * 96 * 48), np.float64)
    out[:, itf["band_lin"][sel]] = v
    return out.reshape(3, 48, 96, 48).astype(np.float32)


# --------------------------------------------------------------------------- sampling
@torch.no_grad()
def posterior_mean_obs(model, obs_phys, mu, sd, fluid, band, members, steps, seed,
                       absent=False):
    """Posterior mean given an explicit physical-velocity band payload.

    With absent=True the conditioning mask is zero everywhere, exactly reproducing the
    published ``no_wall`` branch.  Otherwise the band mask is applied.  Sampler,
    bridge and seeding are byte-identical to eval_cube_3d_coupling.posterior_mean.
    """
    b = obs_phys.shape[0]
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=DEV)
    bt = torch.tensor(band[None, None], dtype=torch.float32, device=DEV)
    muv = torch.tensor(mu[None, :, None, None, None], device=DEV)
    sdv = torch.tensor(sd[None, :, None, None, None], device=DEV)
    if absent:
        obs = torch.zeros((b, 3, 48, 96, 48), device=DEV)
        cm = torch.zeros_like(bt).expand(b, -1, -1, -1, -1)
    else:
        o = torch.from_numpy(np.ascontiguousarray(obs_phys)).to(DEV)
        obs = ((o - muv) / sdv) * ft
        cm = bt.expand(b, -1, -1, -1, -1)
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


def main() -> None:
    t0 = time.time()
    case = Path(A.case)
    data_path = case / A.data_name
    if not data_path.exists():
        raise SystemExit(f"frozen record not found: {data_path}")
    log(f"[data] {data_path}")
    data_digest = sha256(data_path)
    log(f"[data] sha256={data_digest}")

    mm = np.load(data_path, mmap_mode="r")
    if mm.shape != (1101, 3, 48, 96, 48):
        raise SystemExit(f"unexpected record shape {mm.shape}")
    fluid, band, dist, xg, yg, zg = B.geometry()

    # ---- split: byte-identical rule to the published producer
    energy = np.empty(len(mm), float)
    for i in range(len(mm)):
        a = np.asarray(mm[i], np.float32)
        energy[i] = np.square(a[:, fluid]).mean()
    tau = B.integral_tau(energy)
    ntr = int(.60 * len(mm))
    gap = max(8, int(math.ceil(tau)))
    train_idx = np.arange(ntr)
    test_idx_all = np.arange(min(len(mm) - 1, ntr + gap), len(mm))
    block = max(1, int(math.ceil(tau)))
    log(f"[split] tau={tau:.3f} n_train={ntr} gap={gap} n_test={len(test_idx_all)} block={block}")

    mu, sd, mean_field = B.prepare_stats(mm, train_idx, fluid)
    log(f"[stats] mu={mu} sd={sd}")

    itf = build_interface(fluid, band)
    nb = itf["band_lin"].size
    phys_band = itf["phys_band"]
    log(f"[interface] band cells={nb} physical-wall band cells={int(phys_band.sum())} "
        f"surfaces={itf['surface_names']}")
    counts = {n: int((itf['owner'] == i).sum()) for i, n in enumerate(itf["surface_names"])}
    log(f"[interface] ownership={counts}")

    # ---- rasterisation audit of the frozen record (deterministic, time-invariant)
    raster_rows = np.ones(96, bool)
    for j in DUP_ROWS:
        raster_rows[j] = False
    raster_rows[ZERO_ROW] = False
    unique_row_mask = np.zeros((48, 96, 48), bool)
    unique_row_mask[:, raster_rows, :] = True
    checks = []
    for t in (0, 137, 274, 411, 548, 685, 822, 959, 1100):
        Vt = np.asarray(mm[t], np.float32)
        dup_ok = all(np.abs(Vt[:, :, j, :][:, fluid[:, j, :] & fluid[:, j - 1, :]]
                            - Vt[:, :, j - 1, :][:, fluid[:, j, :] & fluid[:, j - 1, :]]).max() == 0
                     for j in DUP_ROWS)
        zero_ok = float(np.abs(Vt[:, :, ZERO_ROW, :]).max()) == 0.0
        checks.append(bool(dup_ok and zero_ok))
    raster_audit = {
        "duplicate_y_rows": list(DUP_ROWS),
        "n_duplicate_row_pairs": len(DUP_ROWS),
        "identically_zero_row": ZERO_ROW,
        "verified_time_invariant_snapshots": [0, 137, 274, 411, 548, 685, 822, 959, 1100],
        "all_snapshot_checks_pass": bool(all(checks)),
        "lid_band_cells": int(nb - phys_band.sum()),
        "lid_band_fraction_of_supplied_band": float((nb - phys_band.sum()) / nb),
    }
    log(f"[raster-audit] {raster_audit}")

    # frozen mean-wall-load traction (information-matched null)
    utau_m, that_m, tau_m, ua_m = traction_from_field(mean_field, itf)
    lift_mean = lift_to_band(utau_m, that_m, itf)

    # ---- frozen generator
    model = B.FlowUNet3D(base=48).to(DEV)
    q = torch.load(A.ckpt, map_location=DEV)
    if not q.get("complete", False):
        raise SystemExit("checkpoint is not a completed EMA export")
    model.load_state_dict(q["ema"])
    model.eval()
    ckpt_digest = sha256(A.ckpt)
    log(f"[model] {A.ckpt} sha256={ckpt_digest} params={sum(p.numel() for p in model.parameters())}")

    # ---- evaluation targets (identical thinning + donor rule)
    nmax = A.nmax
    if len(test_idx_all) > nmax:
        sel = np.linspace(0, len(test_idx_all) - 1, nmax).round().astype(int)
        test_idx = np.asarray(test_idx_all)[sel]
    else:
        test_idx = np.asarray(test_idx_all)
    donor_idx = np.roll(test_idx, len(test_idx) // 2)

    regions = {
        "full_support_excluded": fluid & (~band),
        "near_support_excluded_d_le_0p5h": fluid & (~band) & (dist <= .5),
        "outer_d_gt_0p5h": fluid & (dist > .5),
        # New, preregistered from the rasterisation audit above (no arm output seen):
        # the published regions include Cartesian rows that the nearest-GLL raster
        # duplicates.  This region keeps only raster-unique rows.
        "raster_unique_support_excluded": fluid & (~band) & unique_row_mask,
    }
    truth_all = np.stack([np.asarray(mm[i], np.float32) for i in test_idx])
    tf = (truth_all - mean_field[None]) / sd[None, :, None, None, None]
    tbar = {k: float(tf[:, :, m].mean()) for k, m in regions.items()}
    sst = {k: np.square(tf[:, :, m] - tbar[k]).sum((1, 2)) for k, m in regions.items()}

    comps = {arm: {k: [] for k in regions} for arm in ARMS}
    band_fidelity = {"lift_oracle": [], "lift_trainmean": [], "lift_model": []}
    tau_stats = {"utau_oracle_mean": [], "utau_model_mean": [],
                 "tau_corr_model_vs_oracle": [], "utau_corr_model_vs_oracle": []}
    representative = {}
    batch = 4 if not A.smoke else 2
    seeds = []
    bandmask = phys_band          # the support the lift actually writes

    for j in range(0, len(test_idx), batch):
        ids = test_idx[j:j + batch]
        dids = donor_idx[j:j + batch]
        truth = np.stack([np.asarray(mm[i], np.float32) for i in ids])
        donor = np.stack([np.asarray(mm[i], np.float32) for i in dids])
        seed = 9100 + j
        seeds.extend([seed] * len(ids))
        nb_ = len(ids)

        # per-target frozen lifts from the target and the donor
        lift_oracle = np.zeros_like(truth)
        lift_fartime = np.zeros_like(truth)
        for b_ in range(nb_):
            ut, th, tv, _ = traction_from_field(truth[b_], itf)
            lift_oracle[b_] = lift_to_band(ut, th, itf)
            tau_stats["utau_oracle_mean"].append(float(ut.mean()))
            ut2, th2, _, _ = traction_from_field(donor[b_], itf)
            lift_fartime[b_] = lift_to_band(ut2, th2, itf)
        lift_trainmean = np.repeat(lift_mean[None], nb_, 0)

        # (payload, conditioning support) per arm
        plan = {
            "correct": (truth, band),
            "no_wall": (truth, band),                       # absent=True: mask is zeroed
            "wrong_wall": (donor, band),
            "correct_physwall": (truth, phys_band),
            "tau_lift_oracle": (lift_oracle, phys_band),
            "tau_lift_fartime": (lift_fartime, phys_band),
            "tau_lift_trainmean": (lift_trainmean, phys_band),
        }

        # no_wall first: its posterior mean feeds the model-predicted traction arm
        pm_cache = {}
        for arm in ("no_wall", "correct", "wrong_wall", "correct_physwall",
                    "tau_lift_oracle", "tau_lift_fartime", "tau_lift_trainmean"):
            pl, cmask = plan[arm]
            pm_cache[arm] = posterior_mean_obs(model, pl, mu, sd, fluid, cmask,
                                               A.members, A.sample_steps, seed,
                                               absent=(arm == "no_wall"))

        lift_model = np.zeros_like(truth)
        for b_ in range(nb_):
            ut3, th3, _, _ = traction_from_field(pm_cache["no_wall"][b_], itf)
            lift_model[b_] = lift_to_band(ut3, th3, itf)
            ut_o, _, _, _ = traction_from_field(truth[b_], itf)
            sel = itf["is_phys"]
            tau_stats["utau_model_mean"].append(float(ut3[sel].mean()))
            if ut_o[sel].std() > 0 and ut3[sel].std() > 0:
                tau_stats["utau_corr_model_vs_oracle"].append(
                    float(np.corrcoef(ut_o[sel], ut3[sel])[0, 1]))
        pm_cache["tau_lift_model_predicted"] = posterior_mean_obs(
            model, lift_model, mu, sd, fluid, phys_band, A.members, A.sample_steps, seed)

        # a-priori band fidelity of each lift against the true band, on the physical-wall
        # support that the lift actually writes (standardised fluctuation R^2)
        tb = (truth - mean_field[None]) / sd[None, :, None, None, None]
        for key, arr in (("lift_oracle", lift_oracle), ("lift_trainmean", lift_trainmean),
                         ("lift_model", lift_model)):
            lb = (arr - mean_field[None]) / sd[None, :, None, None, None]
            num = np.square(lb[:, :, bandmask] - tb[:, :, bandmask]).sum((1, 2))
            den = np.square(tb[:, :, bandmask] - tb[:, :, bandmask].mean()).sum((1, 2))
            band_fidelity[key].extend((1.0 - num / np.maximum(den, 1e-12)).tolist())

        for arm in ARMS:
            pm = pm_cache[arm]
            pf = (pm - mean_field[None]) / sd[None, :, None, None, None]
            tt = (truth - mean_field[None]) / sd[None, :, None, None, None]
            for k, m in regions.items():
                comps[arm][k].extend(np.square(pf[:, :, m] - tt[:, :, m]).sum((1, 2)).tolist())
            if j == 0:
                representative[arm] = pm[0]
        log(f"[eval] batch {j//batch + 1}/{math.ceil(len(test_idx)/batch)} "
            f"targets={len(ids)} elapsed={time.time()-t0:.0f}s")

    # ---- statistics: identical bootstrap protocol
    rng = np.random.default_rng(44)
    stride = max(1, int(round(np.median(np.diff(test_idx))))) if len(test_idx) > 1 else 1
    b_eval = max(1, int(round(block / stride)))
    bix = B.block_indices(len(test_idx), b_eval, A.boot, rng)
    result = {"arms": {}, "deltas": {}, "n_eval": len(test_idx), "eval_block": b_eval}
    boots = {}
    for arm in ARMS:
        result["arms"][arm] = {}
        boots[arm] = {}
        for k in regions:
            se = np.asarray(comps[arm][k])
            st = np.asarray(sst[k])
            point = float(1 - se.sum() / (st.sum() + 1e-12))
            br = 1 - se[bix].sum(1) / (st[bix].sum(1) + 1e-12)
            boots[arm][k] = br
            result["arms"][arm][k] = {
                "R2_fluct_balanced": point,
                "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]}

    def add_delta(a, b_):
        name = f"{a}_minus_{b_}"
        result["deltas"][name] = {}
        for k in regions:
            d = boots[a][k] - boots[b_][k]
            point = (result["arms"][a][k]["R2_fluct_balanced"]
                     - result["arms"][b_][k]["R2_fluct_balanced"])
            result["deltas"][name][k] = {
                "point": float(point),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "ci_positive": bool(np.percentile(d, 2.5) > 0),
                "ci_negative": bool(np.percentile(d, 97.5) < 0)}

    for other in ("no_wall", "wrong_wall"):
        add_delta("correct", other)
    for arm in ("correct_physwall", "tau_lift_oracle", "tau_lift_fartime",
                "tau_lift_trainmean", "tau_lift_model_predicted"):
        add_delta(arm, "no_wall")
    add_delta("tau_lift_oracle", "tau_lift_trainmean")
    add_delta("tau_lift_oracle", "tau_lift_fartime")
    add_delta("correct_physwall", "tau_lift_oracle")
    add_delta("correct", "correct_physwall")

    # transmission ratio: matched support (physical-wall oracle band) in the denominator
    result["transmission_ratio"] = {}
    for ref in ("correct_physwall", "correct"):
        result["transmission_ratio"][ref] = {}
        for k in regions:
            num_p = result["deltas"]["tau_lift_oracle_minus_no_wall"][k]["point"]
            den_p = result["deltas"][f"{ref}_minus_no_wall"][k]["point"]
            num_b = boots["tau_lift_oracle"][k] - boots["no_wall"][k]
            den_b = boots[ref][k] - boots["no_wall"][k]
            safe = np.abs(den_b) > 1e-9
            rb = num_b[safe] / den_b[safe]
            result["transmission_ratio"][ref][k] = {
                "point": float(num_p / den_p) if abs(den_p) > 1e-12 else None,
                "ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                "note": "ratio of paired gains; unstable when the denominator interval spans zero",
                "denominator_ci_excludes_zero":
                    bool(np.percentile(den_b, 2.5) > 0 or np.percentile(den_b, 97.5) < 0)}

    # ---- reproduction guard
    guard = {"tolerance": GUARD_TOL, "checks": [], "all_within_tolerance": True}
    for arm, ref in PUBLISHED_M1.items():
        for k, v in ref.items():
            got = result["arms"][arm][k]["R2_fluct_balanced"]
            ok = abs(got - v) <= GUARD_TOL
            guard["checks"].append({"arm": arm, "region": k, "published": v,
                                    "reproduced": got, "abs_diff": abs(got - v), "pass": ok})
            guard["all_within_tolerance"] &= bool(ok)
    result["reproduction_guard"] = guard

    result["a_priori_interface"] = {
        "band_fluctuation_R2_of_lift_vs_true_band": {
            k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                "min": float(np.min(v)), "max": float(np.max(v))}
            for k, v in band_fidelity.items() if len(v)},
        "utau_oracle_band_mean": float(np.mean(tau_stats["utau_oracle_mean"])),
        "utau_model_predicted_band_mean": float(np.mean(tau_stats["utau_model_mean"])),
        "utau_corr_model_vs_oracle_mean":
            float(np.mean(tau_stats["utau_corr_model_vs_oracle"]))
            if tau_stats["utau_corr_model_vs_oracle"] else None,
        "band_cells": int(nb),
        "physical_wall_band_cells": int(phys_band.sum()),
        "surface_ownership": counts,
        "fidelity_support": "physical-wall band (floor + cube), lid excluded",
    }
    result["raster_audit"] = raster_audit
    result["region_sizes"] = {k: int(m.sum()) for k, m in regions.items()}

    payload = {"test_idx": test_idx.astype(np.int64),
               "donor_idx": donor_idx.astype(np.int64),
               "sampler_seed": np.asarray(seeds, np.int64)}
    for k in regions:
        payload[f"sst__{k}"] = np.asarray(sst[k], np.float64)
        for arm in ARMS:
            payload[f"sse__{arm}__{k}"] = np.asarray(comps[arm][k], np.float64)
    for k, v in band_fidelity.items():
        payload[f"band_fidelity__{k}"] = np.asarray(v, np.float64)
    payload["representative_truth"] = truth_all[0]
    for arm in ARMS:
        payload[f"representative__{arm}"] = representative[arm]
    np.savez_compressed(COMP, **payload)

    out = {
        "_meta": {
            "script": Path(__file__).name,
            "script_sha256": sha256(Path(__file__)),
            "parent_script": "eval_cube_3d_coupling.py",
            "parent_script_sha256": sha256(Path(B.__file__)),
            "preregistration":
                "development/nodes/node_004/PREREGISTRATION_E2_INTERFACE.md",
            "device": str(DEV),
            "evidence_level":
                "offline interface test: frozen generator, frozen equilibrium lift, "
                "oracle (target-derived) wall traction. NOT closure performance, "
                "NOT solver-coupled WMLES.",
            "data_memmap_sha256": data_digest,
            "checkpoint_sha256": ckpt_digest,
            "training_performed": False,
            "nu": NU, "kappa": KAPPA, "C_reichardt": C_REICH,
            "anchor_wall_distance_h": D_ANCHOR,
            "cell_size_h": DELTA,
            "members": A.members, "sample_steps": A.sample_steps, "boot": A.boot,
            "n_train": int(ntr), "split_gap_snapshots": int(gap),
            "tau_integral_snapshots": float(tau),
            "bootstrap_block_snapshots": int(block),
            "common_sampler_noise_across_arms": True,
            "arms": list(ARMS),
            "components": COMP.name,
            "wall_seconds_total": round(time.time() - t0, 1),
        },
        "evaluation": result,
    }
    out["_meta"]["components_sha256"] = sha256(COMP)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    digest = sha256(OUT)
    OUT.with_suffix(".sha256").write_text(f"{digest}  {OUT.name}\n")

    log("\n===== E2 INTERFACE RESULT =====")
    for arm in ARMS:
        r = result["arms"][arm]
        log(f"  {arm:26s} full={r['full_support_excluded']['R2_fluct_balanced']:+.5f} "
            f"near={r['near_support_excluded_d_le_0p5h']['R2_fluct_balanced']:+.5f} "
            f"outer={r['outer_d_gt_0p5h']['R2_fluct_balanced']:+.5f} "
            f"uniq={r['raster_unique_support_excluded']['R2_fluct_balanced']:+.5f}")
    for name, d in result["deltas"].items():
        f = d["full_support_excluded"]
        log(f"  D {name:44s} full={f['point']:+.5f} CI[{f['ci95'][0]:+.5f},{f['ci95'][1]:+.5f}]")
    log(f"  transmission ratio (matched, full) = "
        f"{result['transmission_ratio']['correct_physwall']['full_support_excluded']}")
    log(f"  reproduction guard all_within_tolerance="
        f"{result['reproduction_guard']['all_within_tolerance']}")
    log(f"  a-priori band fidelity = {out['evaluation']['a_priori_interface']['band_fluctuation_R2_of_lift_vs_true_band']}")
    log(f"=== done === [out] {OUT} sha256={digest}")


if __name__ == "__main__":
    main()
