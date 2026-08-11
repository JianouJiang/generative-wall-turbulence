#!/usr/bin/env python3
"""Causal-Wiener observation floor + generative-residual benchmark (F1 stage).

Level-1 attempt-3 mechanism: wall data are OBSERVATIONS of the hidden turbulent
state.  The surface-to-volume posterior is factorised as an exactly computable
training-covariance causal Wiener estimator (conditional mean + low-rank
Gaussian conditional covariance) plus a full-resolution periodic rectified-flow
model of the cross-fitted residual.  Every learned claim is a paired,
common-random-number exceedance over the model's own linear observation floor.

Data = the charter-pinned complete aligned-cube record (1101,3,48,96,48) with
the admissible chronological split (first 60% train, one integral-time gap,
chronological remainder test).  Evidence class = held-out LES oracle near-wall
band observations; NOT closure-conditioned, NOT solver-coupled, NOT native
pressure/shear (that is the Orig lane).

Phases (each terminal, incremental JSON, wall-clock guarded):
  A  linear causal Wiener floor: per-arm identical-exposure fits, six
     intervention arms, Gaussian factor-model posterior, fair (U-statistic)
     energy score + CRPS, wall-distance-resolved skill, block bootstrap.
  C  residual rectified flow (seed argument; F1 runs seed 1234) trained on
     cross-fitted out-of-fold residuals; common noise across arms.
  B  matched deterministic parity regressor (same backbone, level=0, MSE).
  D  physical statistics: mean/Reynolds-stress profiles, x-spectra, one-point
     PDFs/coverage, raster divergence.  Drag/pressure need the native record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from causal_wiener_transfer import (  # noqa: E402
    CausalWienerModel,
    cross_fitted_predictions,
    fit_causal_wiener,
    select_hyperparameters,
)
from periodic_generative_residual import (  # noqa: E402
    PeriodicGenerativeResidual,
    rectified_flow_residual_loss,
    sample_rectified_flow,
    zero_residual_on_support,
)

P = argparse.ArgumentParser()
P.add_argument("--data", default="/root/autodl-tmp/cube_les/cube_ds2_float16.complete.npy")
P.add_argument("--smoke", action="store_true")
P.add_argument("--seed", type=int, default=1234, help="training seed for phases B/C")
P.add_argument("--rf-steps", type=int, default=8000)
P.add_argument("--det-steps", type=int, default=5000)
P.add_argument("--width", type=int, default=32)
P.add_argument("--members-gauss", type=int, default=32)
P.add_argument("--members-rf", type=int, default=8)
P.add_argument("--sample-steps", type=int, default=12)
P.add_argument("--n-eval", type=int, default=160)
P.add_argument("--boot", type=int, default=2000)
P.add_argument("--deadline-s", type=float, default=6000.0)
A = P.parse_args()

EXPECTED_SHA = "8bac93f1537eab6667d692282b76c7bccd28f28965d35ea97668bcc2567bc45a"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
TAG = "cube_wiener_floor_bench" + ("_smoke" if A.smoke else "")
OUT = RESULTS / f"{TAG}_results.json"
COMP = RESULTS / f"{TAG}_components.npz"
CKPT_RF = RESULTS / f"{TAG}_rf_seed{A.seed}.pt"
CKPT_DET = RESULTS / f"{TAG}_det_seed{A.seed}.pt"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
T_START = time.time()

if A.smoke:
    A.rf_steps, A.det_steps = 60, 40
    A.members_gauss, A.members_rf, A.sample_steps = 8, 2, 4
    A.n_eval, A.boot = 12, 100

# Frozen admissible split (identical to the terminal adequate pilot).
N_TOTAL, N_TRAIN, GAP = 1101, 660, 98
# Frozen hyperparameter grid; selection on the last VAL_FRAMES of train only.
VAL_FRAMES = 110
GRID = (
    [{"horizon": 1, "rank_wall": 64, "rank_field": 96, "ridge": 3e-2}]
    if A.smoke
    else [
        {"horizon": h, "rank_wall": 192, "rank_field": 384, "ridge": r}
        for h in (1, 4)
        for r in (3e-3, 3e-2)
    ]
)
SEED_GLOBAL = 20260726


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.time()-T_START:7.1f}s] {msg}", flush=True)


def write_json(payload: dict) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(tmp, OUT)


def integral_tau(x: np.ndarray, maxlag: int = 400) -> dict:
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return {"tau_integral": 1.0, "first_zero": 1, "efold": 1}
    ac = [1.0]
    for k in range(1, min(maxlag, len(x) // 2)):
        ac.append(float(np.corrcoef(x[:-k], x[k:])[0, 1]))
    ac = np.nan_to_num(np.asarray(ac), nan=0.0)
    neg = np.where(ac < 0)[0]
    m = int(neg[0]) if len(neg) else len(ac)
    below = np.where(ac < 1 / math.e)[0]
    return {
        "tau_integral": float(max(1.0, 1.0 + 2.0 * ac[1:m].sum())),
        "first_zero": int(m),
        "efold": int(below[0]) if len(below) else len(ac),
    }


def geometry():
    nx, ny, nz = 48, 96, 48
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) * 4.0 / ny
    z = (np.arange(nz) + 0.5) * 2.0 / nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    solid = (X >= 0.5) & (X <= 1.5) & (Y <= 1.0) & (Z >= 0.5) & (Z <= 1.5)
    fluid = ~solid
    dx = np.maximum.reduce([0.5 - X, np.zeros_like(X), X - 1.5])
    dy = np.maximum.reduce([0.0 - Y, np.zeros_like(Y), Y - 1.0])
    dz = np.maximum.reduce([0.5 - Z, np.zeros_like(Z), Z - 1.5])
    dcube = np.sqrt(dx * dx + dy * dy + dz * dz)
    dist = np.minimum.reduce([Y, 4.0 - Y, dcube])
    band = fluid & (dist <= 2.01 * (4.0 / ny))
    return fluid, band, dist


def fair_pairwise_mean(sorted_vals: torch.Tensor) -> torch.Tensor:
    """Mean pairwise absolute difference (fair, off-diagonal) via order stats.

    sorted_vals: [..., M] ascending.  sum_{m<l}(v_l - v_m) = sum_i v_(i)(2i-M+1).
    """
    m = sorted_vals.shape[-1]
    w = torch.arange(m, device=sorted_vals.device, dtype=sorted_vals.dtype) * 2 - (m - 1)
    return (sorted_vals * w).sum(-1) * (2.0 / (m * (m - 1)))


def fair_scores(ens: torch.Tensor, truth: torch.Tensor) -> dict:
    """Fair energy score + fair mean-CRPS for one frame over one region.

    ens: [M, F] members, truth: [F].  Norms are Euclidean over the region.
    """
    m = ens.shape[0]
    d_true = torch.linalg.vector_norm(ens - truth[None], dim=1)
    if m > 1:
        idx = torch.triu_indices(m, m, offset=1, device=ens.device)
        d_pair = torch.linalg.vector_norm(ens[idx[0]] - ens[idx[1]], dim=1)
        es_fair = d_true.mean() - 0.5 * d_pair.mean()
        es_biased = d_true.mean() - 0.5 * d_pair.sum() / (m * m)
        srt, _ = torch.sort(ens, dim=0)
        crps = (ens - truth[None]).abs().mean() - 0.5 * fair_pairwise_mean(srt.T).mean()
    else:
        es_fair = es_biased = d_true.mean()
        crps = (ens - truth[None]).abs().mean()
    return {
        "es_fair": float(es_fair),
        "es_biased": float(es_biased),
        "crps_fair": float(crps),
    }


def selfcheck_fair_scores() -> None:
    g = torch.Generator().manual_seed(0)
    ens = torch.randn(5, 40, generator=g)
    truth = torch.randn(40, generator=g)
    out = fair_scores(ens, truth)
    m = 5
    dt = torch.stack([torch.linalg.vector_norm(ens[i] - truth) for i in range(m)]).mean()
    dp = torch.tensor(
        [
            torch.linalg.vector_norm(ens[i] - ens[j])
            for i in range(m)
            for j in range(m)
            if i != j
        ]
    ).mean()
    ref_es = float(dt - 0.5 * dp)
    ref_crps = float(
        (ens - truth[None]).abs().mean()
        - 0.5
        * torch.stack(
            [(ens[i] - ens[j]).abs() for i in range(m) for j in range(m) if i != j]
        ).mean()
    )
    assert abs(out["es_fair"] - ref_es) < 1e-4, (out["es_fair"], ref_es)
    assert abs(out["crps_fair"] - ref_crps) < 1e-4, (out["crps_fair"], ref_crps)


def block_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]


def boot_r2(sse: np.ndarray, sst: np.ndarray, draws: np.ndarray) -> tuple[float, list]:
    point = 1.0 - sse.sum() / max(sst.sum(), 1e-30)
    vals = 1.0 - sse[draws].sum(1) / np.maximum(sst[draws].sum(1), 1e-30)
    return float(point), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_mean(vals_per_t: np.ndarray, draws: np.ndarray) -> tuple[float, list]:
    point = float(vals_per_t.mean())
    vals = vals_per_t[draws].mean(1)
    return point, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ----------------------------------------------------------------------------
# Load + preflight
# ----------------------------------------------------------------------------
selfcheck_fair_scores()
data_path = Path(A.data)
log(f"hashing {data_path}")
got = sha256(data_path)
if got != EXPECTED_SHA:
    raise SystemExit(f"FATAL data sha256 {got} != charter {EXPECTED_SHA}")
raw = np.load(data_path, mmap_mode="r")
assert raw.shape == (N_TOTAL, 3, 48, 96, 48) and raw.dtype == np.float16, raw.shape
fluid, band, dist = geometry()
solid = ~fluid
sol_max = float(np.abs(np.asarray(raw[0], np.float32)[:, solid]).max())
assert sol_max == 0.0, f"solid cells not zero: {sol_max}"
log(f"record ok; loading to RAM ({raw.nbytes/2**30:.2f} GiB)")
rec = np.asarray(raw, np.float16)

train_idx = np.arange(0, N_TRAIN)
test_idx = np.arange(N_TRAIN + GAP, N_TOTAL)
fit_idx = train_idx[:-VAL_FRAMES]
val_idx = train_idx[-VAL_FRAMES:]
sel = np.linspace(0, len(test_idx) - 1, min(A.n_eval, len(test_idx))).round().astype(int)
eval_idx = test_idx[sel]
donor_pos = np.roll(np.arange(len(eval_idx)), len(eval_idx) // 2)

meta = {
    "producer": Path(__file__).name,
    "producer_sha256": sha256(Path(__file__)),
    "data_path": str(data_path),
    "data_sha256": got,
    "evidence_level": (
        "held-out LES oracle near-wall-band observations on the frozen complete "
        "cube record; methodology/control benchmark; not closure-conditioned, "
        "not solver-coupled, not native pressure/shear"
    ),
    "split": {
        "rule": "first 60% train; one integral-correlation-time gap (98); chronological remainder test",
        "n_train": int(N_TRAIN),
        "gap": int(GAP),
        "n_test_available": int(len(test_idx)),
        "n_eval": int(len(eval_idx)),
        "val_frames_inside_train": int(VAL_FRAMES),
    },
    "seed_training": int(A.seed),
    "grid": GRID,
    "members_gauss": int(A.members_gauss),
    "members_rf": int(A.members_rf),
    "sample_steps": int(A.sample_steps),
    "device": str(DEV),
    "gpu_name": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
    "gpu_uuid": None,
    "smoke": bool(A.smoke),
}
try:
    meta["gpu_uuid"] = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    )
except Exception as exc:  # pragma: no cover
    meta["gpu_uuid"] = f"unavailable: {exc}"

payload = {"_meta": meta, "phases_complete": {}}
write_json(payload)

# ----------------------------------------------------------------------------
# Normalisation (train-only) + feature extraction
# ----------------------------------------------------------------------------
log("train statistics")
mu = np.zeros(3, np.float64)
sd = np.zeros(3, np.float64)
mean_field = np.zeros((3, 48, 96, 48), np.float64)
for i in train_idx:
    x = np.asarray(rec[i], np.float32)
    mean_field += x
mean_field /= len(train_idx)
for c in range(3):
    v = rec[train_idx, c][:, fluid].astype(np.float64)
    mu[c] = v.mean()
    sd[c] = max(v.std(), 1e-8)
mean_norm = ((mean_field - mu[:, None, None, None]) / sd[:, None, None, None]).astype(np.float32)

band_flat = band.reshape(-1)
fluid_flat = fluid.reshape(-1)
n_band, n_fluid = int(band.sum()), int(fluid.sum())
# Equal-support interior shell: contiguous distance shell, cell count closest to band.
cand = []
edges = np.arange(0.35, 0.85, 0.01)
for lo in edges:
    m = fluid & (dist > lo) & (dist <= lo + 2.0 * (4.0 / 96))
    cand.append((abs(int(m.sum()) - n_band), float(lo), m))
cand.sort(key=lambda t: (t[0], t[1]))
shell = cand[0][2]
meta["interior_shell"] = {
    "d_lo": cand[0][1],
    "thickness_h": 2.0 * 4.0 / 96,
    "cells": int(shell.sum()),
    "band_cells": n_band,
}
shell_flat = shell.reshape(-1)


def features(idx: np.ndarray, mask_flat: np.ndarray) -> np.ndarray:
    """Fluctuation features: (x - train mean FIELD)/sd on the masked cells.

    Centering by the train-mean field restores the parent balanced-fluctuation
    convention and gives exact zero-load semantics: a no-wall observation (all
    zeros) corresponds to the wall at its train-mean state and the linear
    prediction collapses to the train-mean field (R2_fluct = 0 baseline).
    """
    base = mean_norm.reshape(3, -1)[:, mask_flat].ravel()
    out = np.empty((1, len(idx), 3 * int(mask_flat.sum())), np.float64)
    for j, i in enumerate(idx):
        x = (np.asarray(rec[i], np.float32) - mu[:, None, None, None].astype(np.float32)) / sd[
            :, None, None, None
        ].astype(np.float32)
        out[0, j] = x.reshape(3, -1)[:, mask_flat].ravel() - base
    return out


log("extracting features (band, shell, field)")
wall_all = features(np.arange(N_TOTAL), band_flat)
shell_all = features(np.arange(N_TOTAL), shell_flat)
field_all = features(np.arange(N_TOTAL), fluid_flat)

# Autocorrelation battery (charter item 1): wall, outer, energy, drag-proxy.
log("autocorrelation battery")
floor_first = fluid & (dist <= (4.0 / 96)) & (np.mgrid[0:48, 0:96, 0:48][1] == 0)
sig = {
    "band_mean_u": wall_all[0, :, : n_band].mean(1),
    "band_mean_energy": np.square(wall_all[0]).mean(1),
    "volume_tke": np.square(field_all[0]).mean(1),
    "outer_plane_u_y2h": np.stack(
        [
            ((np.asarray(rec[i, 0], np.float32) - mu[0]) / sd[0])[:, 48, :].mean()
            for i in range(N_TOTAL)
        ]
    ),
    "floor_shear_proxy_u1": np.stack(
        [np.asarray(rec[i, 0], np.float32)[floor_first].mean() / (0.5 * 4.0 / 96) for i in range(N_TOTAL)]
    ),
}
acf = {k: integral_tau(v) for k, v in sig.items()}
tau_max_snap = max(v["tau_integral"] for v in acf.values())
payload["autocorrelation"] = {
    "signals": acf,
    "tau_max_snapshots": float(tau_max_snap),
    "snapshot_dt_tu": 0.3,
    "tau_max_tu": float(tau_max_snap * 0.3),
    "effective_independent_total": float(N_TOTAL / tau_max_snap),
    "effective_independent_eval": float(len(test_idx) / tau_max_snap),
    "note": "used for Lane-1 cadence justification, far-time offset and bootstrap blocks",
}
stride = (test_idx[-1] - test_idx[0]) / max(1, len(eval_idx) - 1)
block = max(2, int(np.ceil(tau_max_snap / max(stride, 1e-9))))
rng_boot = np.random.default_rng(SEED_GLOBAL)
draws = block_indices(len(eval_idx), block, A.boot, rng_boot)
payload["_meta"]["bootstrap"] = {"block_eval_samples": int(block), "draws": int(A.boot)}
write_json(payload)

# ----------------------------------------------------------------------------
# Phase A: causal Wiener floor
# ----------------------------------------------------------------------------
log("Phase A: hyperparameter selection (validation inside train)")
H_MAX = max(g["horizon"] for g in GRID)
fit_idx_h = fit_idx[fit_idx >= H_MAX - 1]
model_w, records_w = select_hyperparameters(
    wall_all, field_all, fit_idx_h, val_idx, GRID, seed=SEED_GLOBAL
)
sel_spec = next(r for r in records_w if r["selected"])
log(f"selected {sel_spec}")
# Refit on the FULL train block with the selected spec (still no test contact).
model_w = fit_causal_wiener(
    wall_all,
    field_all,
    train_idx[train_idx >= model_w.horizon - 1],
    horizon=model_w.horizon,
    rank_wall=model_w.rank_wall,
    rank_field=model_w.rank_field,
    ridge=model_w.ridge,
    seed=SEED_GLOBAL,
)
model_shell, records_shell = select_hyperparameters(
    shell_all, field_all, fit_idx_h, val_idx, GRID, seed=SEED_GLOBAL + 999
)
sel_shell = next(r for r in records_shell if r["selected"])
model_shell = fit_causal_wiener(
    shell_all,
    field_all,
    train_idx[train_idx >= model_shell.horizon - 1],
    horizon=model_shell.horizon,
    rank_wall=model_shell.rank_wall,
    rank_field=model_shell.rank_field,
    ridge=model_shell.ridge,
    seed=SEED_GLOBAL + 999,
)

H = model_w.horizon


def histories(source: np.ndarray, idx: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([source[0, i - np.arange(horizon)] for i in idx])


hist_correct = histories(wall_all, eval_idx, H)
rng_arm = np.random.default_rng(SEED_GLOBAL + 5)
perm = rng_arm.permutation(3 * n_band)
wall_mu = wall_all[0, train_idx].mean(0)
wall_sig = wall_all[0, train_idx].std(0)
arms_hist = {
    "correct": hist_correct,
    "no_wall": np.broadcast_to(model_w.wall_mean[None, None], hist_correct.shape).copy(),
    "wrong_time": histories(wall_all, eval_idx[donor_pos], H),
    "spatial_permutation": hist_correct[:, :, perm],
    "random_value": wall_mu[None, None]
    + wall_sig[None, None] * rng_arm.standard_normal(hist_correct.shape),
}
hist_shell = histories(shell_all, eval_idx, model_shell.horizon)

# Gaussian factor model on train: within-basis residual covariance + diagonal
# complement variance (per fluid feature), all train-only, one pack per model.
log("Phase A: train residual covariances")


def gaussian_pack(mdl: CausalWienerModel, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = train_idx[train_idx >= mdl.horizon - 1]
    pred = mdl.predict(src, times)[0]
    truth = field_all[0, times]
    coeff = (truth - pred) @ mdl.field_basis
    sig = coeff.T @ coeff / max(1, len(coeff) - 1)
    chol = np.linalg.cholesky(sig + 1e-10 * np.eye(len(sig)))
    cvar = np.zeros(field_all.shape[2], np.float64)
    for j in range(len(truth)):
        resid = truth[j] - pred[j]
        resid_out = resid - (resid @ mdl.field_basis) @ mdl.field_basis.T
        cvar += np.square(resid_out)
    return chol, np.maximum(cvar / len(truth), 0.0)


L_chol, comp_var = gaussian_pack(model_w, wall_all)
L_shell, comp_shell = gaussian_pack(model_shell, shell_all)

truth_eval = field_all[0, eval_idx].astype(np.float32)
# Features are already centred on the train-mean field, so the "mean field" in
# feature space is exactly zero (zero-load semantics).
mean_norm_feat = np.zeros(3 * n_fluid, np.float32)
regions_feat = {}
for key, mask in {
    "full_support_excluded": fluid & (~band),
    "near_support_excluded_d_le_0p5h": fluid & (~band) & (dist <= 0.5),
    "outer_d_gt_0p5h": fluid & (dist > 0.5),
}.items():
    regions_feat[key] = np.repeat(mask.reshape(-1)[fluid_flat][None], 3, 0).ravel()
# wall-distance-resolved bins (support-excluded)
dist_feat = np.tile(dist.reshape(-1)[fluid_flat], 3)
band_feat = np.tile(band.reshape(-1)[fluid_flat], 3)
bin_edges = np.array([0.083, 0.15, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0])
sst_frames = {}
for k, m in regions_feat.items():
    tbar = truth_eval[:, m].mean()
    sst_frames[k] = np.array(
        [np.square(truth_eval[j, m] - tbar).sum() for j in range(len(eval_idx))]
    )


def torch_pack(chol: np.ndarray, cvar: np.ndarray, basis: np.ndarray):
    b = torch.from_numpy(basis.astype(np.float32))
    lc = torch.from_numpy(chol.astype(np.float32))
    cs = torch.from_numpy(np.sqrt(cvar).astype(np.float32))
    if DEV.type == "cuda":
        b, lc, cs = b.to(DEV), lc.to(DEV), cs.to(DEV)
    return b, lc, cs


packs = {
    "wall": torch_pack(L_chol, comp_var, model_w.field_basis),
    "shell": torch_pack(L_shell, comp_shell, model_shell.field_basis),
}

phaseA = {"selection": records_w, "selection_shell": records_shell, "arms": {}}
per_frame_es = {}
for arm, hist in list(arms_hist.items()) + [("equal_support_interior", hist_shell)]:
    mdl = model_shell if arm == "equal_support_interior" else model_w
    pred = mdl.predict_histories(hist).astype(np.float32)
    entry = {}
    sse_frames = {}
    for key, m in regions_feat.items():
        sse = np.array(
            [np.square(pred[j, m] - truth_eval[j, m]).sum() for j in range(len(eval_idx))]
        )
        sse_frames[key] = sse
        point, ci = boot_r2(sse, sst_frames[key], draws)
        entry[key] = {"R2_fluct_balanced": point, "ci95": ci}
    # distance-resolved profile
    prof = []
    for b0, b1 in zip(bin_edges[:-1], bin_edges[1:]):
        m = (~band_feat) & (dist_feat > b0) & (dist_feat <= b1)
        if m.sum() == 0:
            continue
        sse = np.square(pred[:, m] - truth_eval[:, m]).sum()
        sst = np.square(truth_eval[:, m] - truth_eval[:, m].mean()).sum()
        prof.append({"d_lo": float(b0), "d_hi": float(b1), "R2": float(1 - sse / max(sst, 1e-30))})
    entry["distance_resolved_R2"] = prof
    # Gaussian posterior fair scores + coverage on outer + full regions
    es_frames = {k: np.zeros(len(eval_idx)) for k in ("outer_d_gt_0p5h", "full_support_excluded")}
    crps_frames = {k: np.zeros(len(eval_idx)) for k in es_frames}
    cover_hits = {k: 0.0 for k in es_frames}
    cover_tot = {k: 0 for k in es_frames}
    M = A.members_gauss
    basis_t, Lc_t, comp_sd_t = packs["shell" if arm == "equal_support_interior" else "wall"]
    for j in range(len(eval_idx)):
        # Common random numbers across arms: noise depends on frame only.
        gen_frame = torch.Generator(device="cpu").manual_seed(SEED_GLOBAL + 1000 + j)
        xi = torch.randn(M, mdl.rank_field, generator=gen_frame)
        eta = torch.randn(M, pred.shape[1], generator=gen_frame)
        if DEV.type == "cuda":
            xi, eta = xi.to(DEV), eta.to(DEV)
        base = torch.from_numpy(pred[j]).to(DEV) if DEV.type == "cuda" else torch.from_numpy(pred[j])
        ens = base[None] + (xi @ Lc_t.T) @ basis_t.T + eta * comp_sd_t
        tt = torch.from_numpy(truth_eval[j]).to(ens.device)
        for key in es_frames:
            m = torch.from_numpy(regions_feat[key]).to(ens.device)
            sc = fair_scores(ens[:, m], tt[m])
            es_frames[key][j] = sc["es_fair"]
            crps_frames[key][j] = sc["crps_fair"]
            lo = torch.quantile(ens[:, m], 0.05, dim=0)
            hi = torch.quantile(ens[:, m], 0.95, dim=0)
            cover_hits[key] += float(((tt[m] >= lo) & (tt[m] <= hi)).float().mean())
            cover_tot[key] += 1
    for key in es_frames:
        p, ci = boot_mean(es_frames[key], draws)
        q, ci2 = boot_mean(crps_frames[key], draws)
        entry.setdefault("gaussian_posterior", {})[key] = {
            "energy_score_fair": p,
            "energy_score_fair_ci95": ci,
            "crps_fair": q,
            "crps_fair_ci95": ci2,
            "coverage_90": cover_hits[key] / max(1, cover_tot[key]),
        }
    per_frame_es[arm] = es_frames
    phaseA["arms"][arm] = entry
    log(f"Phase A arm {arm}: full R2={entry['full_support_excluded']['R2_fluct_balanced']:+.4f}")

# paired deltas (common bootstrap draws), including the equal-support control
deltas = {}
for arm in per_frame_es:
    if arm == "correct":
        continue
    d = {}
    for key in ("outer_d_gt_0p5h", "full_support_excluded"):
        diff = per_frame_es[arm][key] - per_frame_es["correct"][key]
        p, ci = boot_mean(diff, draws)
        d[key] = {"es_fair_arm_minus_correct": p, "ci95": ci, "ci_positive": ci[0] > 0}
    deltas[f"correct_vs_{arm}"] = d
phaseA["paired_energy_score_deltas"] = deltas
payload["phase_A_wiener_floor"] = phaseA
payload["phases_complete"]["A"] = True
write_json(payload)
log("Phase A complete")

# ----------------------------------------------------------------------------
# Phase C: residual rectified flow (cross-fitted targets), seed A.seed
# ----------------------------------------------------------------------------
log("Phase C: cross-fitted residual targets")
oof_pred, oof_truth, oof_audit = cross_fitted_predictions(
    wall_all,
    field_all,
    train_idx[train_idx >= H - 1],
    horizon=H,
    rank_wall=model_w.rank_wall,
    rank_field=model_w.rank_field,
    ridge=model_w.ridge,
    folds=2 if A.smoke else 4,
    buffer=GAP,
    seed=SEED_GLOBAL,
)
oof_times = train_idx[train_idx >= H - 1]


def vol_from_feat(feat: np.ndarray) -> np.ndarray:
    out = np.zeros((3, 48 * 96 * 48), np.float32)
    out[:, fluid_flat] = feat.reshape(3, -1)
    return out.reshape(3, 48, 96, 48)


def to_torch_vol(a: np.ndarray) -> torch.Tensor:
    # [3,x,y,z] -> [3,y,x,z] for the periodic residual network.
    return torch.from_numpy(np.ascontiguousarray(np.transpose(a, (0, 2, 1, 3))))


fluid_t = to_torch_vol(np.repeat(fluid[None], 1, 0).astype(np.float32))[:1][None].to(DEV)
band_t = to_torch_vol(np.repeat(band[None], 1, 0).astype(np.float32))[:1][None].to(DEV)
dist_t = to_torch_vol(np.repeat(dist[None], 1, 0).astype(np.float32))[:1][None].to(DEV)

wiener_train = np.zeros((len(oof_times), 3, 48, 96, 48), np.float16)
resid_train = np.zeros_like(wiener_train)
for j in range(len(oof_times)):
    wv = vol_from_feat(oof_pred[0, j].astype(np.float32))
    tv = vol_from_feat(oof_truth[0, j].astype(np.float32))
    wiener_train[j] = wv.astype(np.float16)
    resid_train[j] = (tv - wv).astype(np.float16)

COND_CH = 9  # wiener(3) + band obs(3) + band mask + dist + fluid
torch.manual_seed(A.seed)
model_rf = PeriodicGenerativeResidual(3, COND_CH, 3, width=A.width, depth=4).to(DEV)
n_par = sum(p.numel() for p in model_rf.parameters())
log(f"Phase C: model {n_par} params; training {A.rf_steps} steps")


def cond_tensor(wiener_vol: torch.Tensor, band_obs_vol: torch.Tensor, keep: torch.Tensor):
    b = wiener_vol.shape[0]
    cm = band_t.expand(b, 1, -1, -1, -1) * keep
    return torch.cat(
        [
            wiener_vol,
            band_obs_vol * cm,
            cm,
            dist_t.expand(b, 1, -1, -1, -1),
            fluid_t.expand(b, 1, -1, -1, -1),
        ],
        1,
    ), cm


def train_residual(model, steps, objective, ckpt, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    ema = PeriodicGenerativeResidual(3, COND_CH, 3, width=A.width, depth=4).to(DEV)
    ema.load_state_dict(model.state_dict())
    rng = np.random.default_rng(A.seed)
    losses = []
    t0 = time.time()
    done_steps = 0
    for it in range(steps):
        if time.time() - T_START > A.deadline_s:
            log(f"[{tag}] wall-clock guard: stopping at {it} steps")
            break
        ids = rng.integers(0, len(oof_times), size=2)
        wv = torch.stack([to_torch_vol(wiener_train[i].astype(np.float32)) for i in ids]).to(DEV)
        rv = torch.stack([to_torch_vol(resid_train[i].astype(np.float32)) for i in ids]).to(DEV)
        frame_ids = oof_times[ids]
        obs = torch.stack(
            [
                to_torch_vol(
                    vol_from_feat(field_all[0, i].astype(np.float32))
                )
                for i in frame_ids
            ]
        ).to(DEV)
        keep = (torch.rand(2, device=DEV) > 0.25).float()[:, None, None, None, None]
        # no-wall in-distribution: when dropped, wiener input collapses to the mean.
        mean_t = to_torch_vol(vol_from_feat(mean_norm_feat))[None].to(DEV)
        wv = keep * wv + (1 - keep) * mean_t.expand_as(wv)
        rv_eff = keep * rv + (1 - keep) * (
            rv + (torch.stack([to_torch_vol(wiener_train[i].astype(np.float32)) for i in ids]).to(DEV) - mean_t)
        )
        cond, cm = cond_tensor(wv, obs, keep)
        # fp32 + TF32: torch.fft does not support bfloat16 autocast.
        loss = objective(model, rv_eff, cond, cm, fluid_t.expand(2, -1, -1, -1, -1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.9995).add_(p, alpha=0.0005)
        done_steps = it + 1
        if done_steps % max(1, steps // 20) == 0:
            losses.append([done_steps, float(loss.detach())])
            log(f"[{tag}] {done_steps}/{steps} loss={float(loss):.5f}")
    ema.eval()
    torch.save({"ema": ema.state_dict(), "steps": done_steps, "losses": losses, "seed": A.seed}, ckpt)
    return ema, done_steps, losses


def rf_objective(model, resid, cond, cm, ft):
    return rectified_flow_residual_loss(model, resid, cond, cm, ft)


ema_rf, rf_done, rf_losses = train_residual(model_rf, A.rf_steps, rf_objective, CKPT_RF, "rf")

log("Phase C: evaluation with common noise across arms")
arm_preds_pm = {}
es_rf = {a: {k: np.zeros(len(eval_idx)) for k in ("outer_d_gt_0p5h", "full_support_excluded")} for a in arms_hist}
crps_rf = {a: {k: np.zeros(len(eval_idx)) for k in ("outer_d_gt_0p5h", "full_support_excluded")} for a in arms_hist}
cover_rf = {a: {k: [0.0, 0] for k in ("outer_d_gt_0p5h", "full_support_excluded")} for a in arms_hist}
sse_rf = {a: {k: np.zeros(len(eval_idx)) for k in regions_feat} for a in arms_hist}
member_store = {}
eval_guard_tripped = False
with torch.no_grad():
    for arm, hist in arms_hist.items():
        wpred = (model_w.predict_histories(hist).astype(np.float32))
        for j in range(len(eval_idx)):
            if time.time() - T_START > A.deadline_s + 1800:
                log("Phase C eval guard tripped")
                eval_guard_tripped = True
                break
            wv = to_torch_vol(vol_from_feat(wpred[j]))[None].to(DEV)
            obs_feat = hist[j, 0]
            obs_vol = np.zeros((3, 48 * 96 * 48), np.float32)
            obs_vol[:, band_flat] = obs_feat.reshape(3, -1) if arm != "no_wall" else 0.0
            obs_v = to_torch_vol(obs_vol.reshape(3, 48, 96, 48))[None].to(DEV)
            keep = torch.zeros(1, 1, 1, 1, 1, device=DEV) if arm == "no_wall" else torch.ones(
                1, 1, 1, 1, 1, device=DEV
            )
            mean_t = to_torch_vol(vol_from_feat(mean_norm_feat))[None].to(DEV)
            wv_in = keep * wv + (1 - keep) * mean_t
            cond, cm = cond_tensor(wv_in, obs_v, keep)
            Mrf = A.members_rf
            g = torch.Generator(device=DEV).manual_seed(9100 + j)
            condM = cond.expand(Mrf, -1, -1, -1, -1)
            res = sample_rectified_flow(
                ema_rf,
                condM,
                (Mrf, 3, 96, 48, 48),
                cm.expand(Mrf, -1, -1, -1, -1),
                fluid_t.expand(Mrf, -1, -1, -1, -1),
                steps=A.sample_steps,
                generator=g,
            )
            fields = (wv_in + res.float()).permute(0, 1, 3, 2, 4)  # back to [M,3,x,y,z]
            feats = fields.reshape(Mrf, 3, -1)[:, :, torch.from_numpy(fluid_flat).to(DEV)].reshape(
                Mrf, -1
            )
            tt = torch.from_numpy(truth_eval[j]).to(DEV)
            pm = feats.mean(0)
            for key, m in regions_feat.items():
                mt = torch.from_numpy(m).to(DEV)
                sse_rf[arm][key][j] = float(((pm[mt] - tt[mt]) ** 2).sum())
            for key in ("outer_d_gt_0p5h", "full_support_excluded"):
                mt = torch.from_numpy(regions_feat[key]).to(DEV)
                sc = fair_scores(feats[:, mt], tt[mt])
                es_rf[arm][key][j] = sc["es_fair"]
                crps_rf[arm][key][j] = sc["crps_fair"]
                lo = torch.quantile(feats[:, mt], 0.05, dim=0)
                hi = torch.quantile(feats[:, mt], 0.95, dim=0)
                cover_rf[arm][key][0] += float(((tt[mt] >= lo) & (tt[mt] <= hi)).float().mean())
                cover_rf[arm][key][1] += 1
            if j == 0:
                member_store[arm] = feats[:2].float().cpu().numpy()
        log(f"Phase C arm {arm} done")

phaseC = {
    "train": {
        "steps_requested": int(A.rf_steps),
        "steps_done": int(rf_done),
        "n_parameters": int(n_par),
        "losses": rf_losses,
        "cross_fit_audit": {k: v for k, v in oof_audit.items() if k != "fold_records"},
        "objective": "rectified_flow_residual_loss on cross-fitted OOF residuals",
        "seed": int(A.seed),
    },
    "arms": {},
}
for arm in arms_hist:
    entry = {}
    for key in regions_feat:
        p, ci = boot_r2(sse_rf[arm][key], sst_frames[key], draws)
        entry[key] = {"R2_fluct_balanced_pm": p, "ci95": ci}
    for key in ("outer_d_gt_0p5h", "full_support_excluded"):
        p, ci = boot_mean(es_rf[arm][key], draws)
        q, ci2 = boot_mean(crps_rf[arm][key], draws)
        entry.setdefault("posterior", {})[key] = {
            "energy_score_fair": p,
            "energy_score_fair_ci95": ci,
            "crps_fair": q,
            "crps_fair_ci95": ci2,
            "coverage_90": cover_rf[arm][key][0] / max(1, cover_rf[arm][key][1]),
        }
    # paired vs the Gaussian floor (same frames, same draws)
    for key in ("outer_d_gt_0p5h", "full_support_excluded"):
        diff = per_frame_es[arm][key] - es_rf[arm][key]
        p, ci = boot_mean(diff, draws)
        entry.setdefault("vs_gaussian_floor", {})[key] = {
            "floor_minus_rf_es_fair": p,
            "ci95": ci,
            "rf_better_ci_positive": ci[0] > 0,
        }
    phaseC["arms"][arm] = entry
phaseC["eval_guard_tripped"] = bool(eval_guard_tripped)
payload["phase_C_residual_rf"] = phaseC
payload["phases_complete"]["C"] = not eval_guard_tripped
write_json(payload)
log("Phase C complete")

# ----------------------------------------------------------------------------
# Phase D: physical statistics (correct arm + truth), cheap
# ----------------------------------------------------------------------------
log("Phase D: physical statistics")


def profiles(frames_feat: np.ndarray) -> dict:
    vol = np.zeros((len(frames_feat), 3, 48 * 96 * 48), np.float32)
    vol[:, :, fluid_flat] = frames_feat.reshape(len(frames_feat), 3, -1)
    vol = vol.reshape(-1, 3, 48, 96, 48)
    cnt = fluid.sum(axis=(0, 2))  # fluid cells per y level (sum over x and z)
    mean_u = vol[:, 0].sum(axis=(0, 1, 3)) / (len(frames_feat) * np.maximum(cnt, 1))
    fl = vol - vol.mean(0, keepdims=True)
    stresses = {}
    for name, (a, b) in {"uu": (0, 0), "vv": (1, 1), "ww": (2, 2), "uv": (0, 1)}.items():
        s = (fl[:, a] * fl[:, b]).sum(axis=(0, 1, 3)) / (
            len(frames_feat) * np.maximum(cnt, 1)
        )
        stresses[name] = s.tolist()
    return {"mean_u_y": mean_u.tolist(), "stresses_y": stresses}


truth_prof = profiles(truth_eval)
pm_correct = np.zeros_like(truth_eval)
vol_pm = np.zeros((3, 48 * 96 * 48), np.float32)
phaseD = {"truth_profiles": truth_prof}
try:
    pm_feats = []
    hist = arms_hist["correct"]
    wpred = model_w.predict_histories(hist).astype(np.float32)
    phaseD["wiener_correct_profiles"] = profiles(wpred)
    # x-spectra of u at three heights, truth vs wiener mean vs stored rf members
    def xspec(volume: np.ndarray, yidx: int) -> list:
        # volume [3,x,y,z]
        line = volume[0, :, yidx, :]
        spec = np.abs(np.fft.rfft(line, axis=0)) ** 2
        return spec.mean(1).tolist()

    heights = {"y0p75h": 18, "y1p5h": 36, "y2p5h": 60}
    spect = {}
    for hname, yi in heights.items():
        spect[hname] = {
            "truth_mean": np.mean(
                [xspec(vol_from_feat(truth_eval[j]), yi) for j in range(0, len(eval_idx), 8)], 0
            ).tolist(),
            "wiener_mean": np.mean(
                [xspec(vol_from_feat(wpred[j]), yi) for j in range(0, len(eval_idx), 8)], 0
            ).tolist(),
        }
        if "correct" in member_store:
            spect[hname]["rf_member"] = xspec(
                vol_from_feat(member_store["correct"][0]), yi
            )
    phaseD["x_spectra_u"] = spect
    # raster divergence
    def divergence(volume: np.ndarray) -> float:
        u, v, w = volume
        du = np.gradient(u, 2.0 / 48, axis=0)
        dv = np.gradient(v, 4.0 / 96, axis=1)
        dw = np.gradient(w, 2.0 / 48, axis=2)
        d = du + dv + dw
        return float(np.abs(d[fluid & (dist > 0.15)]).mean())

    phaseD["divergence_raster_mean_abs"] = {
        "truth": float(np.mean([divergence(vol_from_feat(truth_eval[j])) for j in range(0, len(eval_idx), 16)])),
        "wiener_correct": float(
            np.mean([divergence(vol_from_feat(wpred[j])) for j in range(0, len(eval_idx), 16)])
        ),
    }
    if "correct" in member_store:
        phaseD["divergence_raster_mean_abs"]["rf_member"] = divergence(
            vol_from_feat(member_store["correct"][0])
        )
    phaseD["note_drag_pressure"] = (
        "drag/pressure-drop metrics require the native-record Orig lane; the frozen "
        "record stores velocity only"
    )
except Exception as exc:  # pragma: no cover
    phaseD["error"] = repr(exc)
payload["phase_D_physical_stats"] = phaseD
payload["phases_complete"]["D"] = True
write_json(payload)

# ----------------------------------------------------------------------------
# Phase B: deterministic parity regressor (same backbone, level=0)
# ----------------------------------------------------------------------------
if time.time() - T_START < A.deadline_s:
    log("Phase B: deterministic parity regressor")
    torch.manual_seed(A.seed + 1)
    model_det = PeriodicGenerativeResidual(3, COND_CH, 3, width=A.width, depth=4).to(DEV)

    def det_objective(model, resid, cond, cm, ft):
        zero_state = torch.zeros_like(resid)
        level = torch.zeros(resid.shape[0], device=resid.device)
        pred = model(zero_state, cond, level)
        pred = zero_residual_on_support(pred, cm, ft)
        target = zero_residual_on_support(resid, cm, ft)
        scored = ft * (1.0 - cm)
        return ((pred - target) ** 2 * scored).sum() / scored.sum().clamp_min(1.0)

    ema_det, det_done, det_losses = train_residual(
        model_det, A.det_steps, det_objective, CKPT_DET, "det"
    )
    sse_det = {a: {k: np.zeros(len(eval_idx)) for k in regions_feat} for a in ("correct", "no_wall")}
    with torch.no_grad():
        for arm in ("correct", "no_wall"):
            hist = arms_hist[arm]
            wpred = model_w.predict_histories(hist).astype(np.float32)
            for j in range(len(eval_idx)):
                wv = to_torch_vol(vol_from_feat(wpred[j]))[None].to(DEV)
                obs_feat = hist[j, 0]
                obs_vol = np.zeros((3, 48 * 96 * 48), np.float32)
                obs_vol[:, band_flat] = obs_feat.reshape(3, -1) if arm != "no_wall" else 0.0
                obs_v = to_torch_vol(obs_vol.reshape(3, 48, 96, 48))[None].to(DEV)
                keep = torch.zeros(1, 1, 1, 1, 1, device=DEV) if arm == "no_wall" else torch.ones(
                    1, 1, 1, 1, 1, device=DEV
                )
                mean_t = to_torch_vol(vol_from_feat(mean_norm_feat))[None].to(DEV)
                wv_in = keep * wv + (1 - keep) * mean_t
                cond, cm = cond_tensor(wv_in, obs_v, keep)
                res = ema_det(
                    torch.zeros(1, 3, 96, 48, 48, device=DEV),
                    cond,
                    torch.zeros(1, device=DEV),
                )
                res = zero_residual_on_support(res, cm, fluid_t)
                field = (wv_in + res.float()).permute(0, 1, 3, 2, 4)
                feat = field.reshape(1, 3, -1)[:, :, torch.from_numpy(fluid_flat).to(DEV)].reshape(-1)
                tt = torch.from_numpy(truth_eval[j]).to(DEV)
                for key, m in regions_feat.items():
                    mt = torch.from_numpy(m).to(DEV)
                    sse_det[arm][key][j] = float(((feat[mt] - tt[mt]) ** 2).sum())
    phaseB = {"train": {"steps_requested": int(A.det_steps), "steps_done": int(det_done)}, "arms": {}}
    for arm in sse_det:
        entry = {}
        for key in regions_feat:
            p, ci = boot_r2(sse_det[arm][key], sst_frames[key], draws)
            entry[key] = {"R2_fluct_balanced": p, "ci95": ci}
        phaseB["arms"][arm] = entry
    payload["phase_B_deterministic_parity"] = phaseB
    payload["phases_complete"]["B"] = True
else:
    payload["phase_B_deterministic_parity"] = {"skipped": "wall-clock guard"}
    payload["phases_complete"]["B"] = False

np.savez_compressed(
    COMP,
    eval_idx=eval_idx,
    draws_shape=np.array(draws.shape),
    sst_full=sst_frames["full_support_excluded"],
    sse_wiener_correct_full=np.array(
        [
            np.square(
                model_w.predict_histories(arms_hist["correct"]).astype(np.float32)[j][
                    regions_feat["full_support_excluded"]
                ]
                - truth_eval[j][regions_feat["full_support_excluded"]]
            ).sum()
            for j in range(len(eval_idx))
        ]
    ),
    es_correct_outer_gauss=per_frame_es["correct"]["outer_d_gt_0p5h"],
    es_correct_outer_rf=es_rf["correct"]["outer_d_gt_0p5h"],
)
payload["_meta"]["components"] = COMP.name
payload["_meta"]["components_sha256"] = sha256(COMP)
payload["_meta"]["wall_seconds_total"] = round(time.time() - T_START, 1)
write_json(payload)
log(f"wrote {OUT}")
print("=== WIENER_FLOOR_BENCH_DONE ===", flush=True)
