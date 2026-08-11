#!/usr/bin/env python3
"""Capacity/optimization repair of the preserved genuine-3-D cube experiment.

This script deliberately reuses the terminal Coceal LES and the frozen split from
``eval_cube_3d_coupling.py``.  It changes only posterior capacity, training budget
and sampler resolution, and it preserves per-time sufficient statistics so the
reported intervention and block intervals can be recomputed independently.

Evidence remains held-out LES oracle-band propagation.  This script neither
creates additional independent LES times nor claims closure/solver coupling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


P = argparse.ArgumentParser()
P.add_argument("--smoke", action="store_true")
P.add_argument("--base", type=int, default=48)
P.add_argument("--steps-train", type=int, default=90000)
P.add_argument("--sample-steps", type=int, default=32)
P.add_argument("--members", type=int, default=8)
P.add_argument("--boot", type=int, default=4000)
P.add_argument("--stack", default="controlled_raw/cube_les/stack")
P.add_argument("--case", default="controlled_raw/cube_les")
A = P.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The parent is a preserved, hash-pinned producer.  Import it with an empty
# argv, then specialize it without editing the original terminal script.
saved_argv = sys.argv
sys.argv = ["eval_cube_3d_coupling.py"]
import eval_cube_3d_coupling as B  # noqa: E402
sys.argv = saved_argv

tag = "cube3d_coupling_adequate_smoke" if A.smoke else "cube3d_coupling_adequate"
# The preserved parent smoke path intentionally truncates to 12 fields, which
# cannot satisfy its own eight-field exclusion gap plus four-field test guard.
# A wrapper smoke therefore keeps the full frozen split but reduces training,
# sampler steps, members and bootstrap draws below.
B.A.smoke = False
B.A.steps_train = 8 if A.smoke else int(A.steps_train)
B.A.sample_steps = 3 if A.smoke else int(A.sample_steps)
B.A.members = 2 if A.smoke else int(A.members)
B.A.boot = 100 if A.smoke else int(A.boot)
B.STACK = Path(A.stack)
B.CASE = Path(A.case)
B.OUT = B.RESULTS / f"{tag}_results.json"
B.FIG = B.RESULTS / f"fig_{tag}.png"
B.CKPT = B.RESULTS / f"{tag}.pt"
COMP = B.RESULTS / f"{tag}_components.npz"

ParentFlowUNet3D = B.FlowUNet3D


class AdequateFlowUNet3D(ParentFlowUNet3D):
    def __init__(self, base: int = A.base):
        super().__init__(base=int(A.base))


B.FlowUNet3D = AdequateFlowUNet3D


def evaluate_with_components(model, mm, test_idx, mu, sd, mean_field, fluid, band, dist, block):
    """Parent evaluation with byte-saved per-time SSE/SST and common seeds."""
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
    truth_all = np.stack([np.asarray(mm[i], np.float32) for i in test_idx])
    tf = (truth_all - mean_field[None]) / sd[None, :, None, None, None]
    tbar = {k: float(tf[:, :, m].mean()) for k, m in regions.items()}
    sst = {k: np.square(tf[:, :, m] - tbar[k]).sum((1, 2)) for k, m in regions.items()}
    arms = ("correct", "no_wall", "wrong_wall")
    comps = {arm: {k: [] for k in regions} for arm in arms}
    representative = {}
    batch = 1 if A.smoke else 4
    seeds = []
    for arm in arms:
        for j in range(0, len(test_idx), batch):
            ids = test_idx[j:j + batch]
            dids = donor_idx[j:j + batch]
            truth = np.stack([np.asarray(mm[i], np.float32) for i in ids])
            donor = np.stack([np.asarray(mm[i], np.float32) for i in dids])
            seed = 9100 + j
            pm = B.posterior_mean(model, truth, donor, mu, sd, fluid, band, arm,
                                  B.A.members, B.A.sample_steps, seed=seed)
            pf = (pm - mean_field[None]) / sd[None, :, None, None, None]
            tt = (truth - mean_field[None]) / sd[None, :, None, None, None]
            for key, mask in regions.items():
                comps[arm][key].extend(
                    np.square(pf[:, :, mask] - tt[:, :, mask]).sum((1, 2)).tolist())
            if j == 0:
                representative[arm] = pm[0]
            if arm == "correct":
                seeds.extend([seed] * len(ids))
        print(f"[eval] arm={arm} n={len(test_idx)} common-noise=true", flush=True)

    payload = {
        "test_idx": test_idx.astype(np.int64),
        "donor_idx": donor_idx.astype(np.int64),
        "sampler_seed": np.asarray(seeds, np.int64),
    }
    for key in regions:
        payload[f"sst__{key}"] = np.asarray(sst[key], np.float64)
        for arm in arms:
            payload[f"sse__{arm}__{key}"] = np.asarray(comps[arm][key], np.float64)
    np.savez_compressed(COMP, **payload)

    rng = np.random.default_rng(44)
    stride = max(1, int(round(np.median(np.diff(test_idx))))) if len(test_idx) > 1 else 1
    b_eval = max(1, int(round(block / stride)))
    bix = B.block_indices(len(test_idx), b_eval, B.A.boot, rng)
    result = {"arms": {}, "deltas": {}, "n_eval": len(test_idx), "eval_block": b_eval}
    boots = {}
    for arm in arms:
        result["arms"][arm] = {}
        boots[arm] = {}
        for key in regions:
            se = np.asarray(comps[arm][key])
            st = np.asarray(sst[key])
            point = float(1 - se.sum() / (st.sum() + 1e-12))
            br = 1 - se[bix].sum(1) / (st[bix].sum(1) + 1e-12)
            boots[arm][key] = br
            result["arms"][arm][key] = {
                "R2_fluct_balanced": point,
                "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
            }
    for other in ("no_wall", "wrong_wall"):
        dname = f"correct_minus_{other}"
        result["deltas"][dname] = {}
        for key in regions:
            delta = boots["correct"][key] - boots[other][key]
            point = (result["arms"]["correct"][key]["R2_fluct_balanced"]
                     - result["arms"][other][key]["R2_fluct_balanced"])
            result["deltas"][dname][key] = {
                "point": float(point),
                "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
                "ci_positive": bool(np.percentile(delta, 2.5) > 0),
            }
    return result, representative, truth_all[0], test_idx


B.evaluate = evaluate_with_components


def main() -> None:
    B.main()
    result = json.loads(B.OUT.read_text())
    result["_meta"].update({
        "producer_script": Path(__file__).name,
        "producer_script_sha256": sha256(Path(__file__)),
        "parent_script": "eval_cube_3d_coupling.py",
        "parent_script_sha256": sha256(Path(B.__file__)),
        "capacity_repair": {
            "base_width": int(A.base),
            "train_updates": int(B.A.steps_train),
            "sampler_steps": int(B.A.sample_steps),
            "posterior_members": int(B.A.members),
            "les_bytes_frozen": True,
            "claim_boundary": "does not add independent LES times",
        },
        "components": COMP.name,
        "components_sha256": sha256(COMP),
        "common_sampler_noise_across_arms": True,
    })
    B.OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    digest = sha256(B.OUT)
    B.OUT.with_suffix(".sha256").write_text(f"{digest}  {B.OUT.name}\n")
    print(f"=== adequate done ===\n[out] {B.OUT} sha256={digest}\n[components] {COMP} sha256={sha256(COMP)}",
          flush=True)


if __name__ == "__main__":
    main()
