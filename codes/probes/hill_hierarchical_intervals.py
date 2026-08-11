#!/usr/bin/env python3
"""
hill_hierarchical_intervals.py -- one named estimator for every periodic-hill interval.

WHY THIS EXISTS
===============
Methods (`sec:statistics`, band-lift paragraph) declares that the band-lift
intervals are *hierarchical* bootstraps: "seeds resampled with replacement, then
a moving-block resample in time with block ceil(1.2551 tau)".  The audit that
actually produced the released hill intervals
(`audit_hill_positive_control.py`, lines 74-77) instead averages the per-frame
sum of squared errors over seeds FIRST and then resamples time only.  That
estimator conditions on the realised sampler draws: it propagates temporal
dependence but not across-seed sampling variability, so it is narrower than the
interval the Methods describes.

Every hill number in the manuscript is therefore quoted from an estimator that is
not the declared one.  This probe implements the DECLARED estimator and reports
it as primary, alongside the seed-conditional estimator that was used before, so
the two are visible side by side and no interval silently changes provenance.

The seed level is resampled with replacement from the three realised seeds.
Three is a small number of seeds and the resulting interval is correspondingly
conservative; that is the point.  It is reported for favourable and adverse arms
alike, and the estimator is selected here by what Methods declares, not by which
interval is narrower.

WHAT IT COMPUTES
================
For every cell, region and arm, against the `absent` arm on identical resamples:

    delta R2_fluct  = R2(arm) - R2(absent)        Eq. (2) recomputed per replicate
    delta CRPS      = CRPS(arm) - CRPS(absent)    proper, lower is better
    delta ES        = ES(arm)   - ES(absent)      proper, lower is better

and, as a DIAGNOSTIC and not a gate, the rank-histogram extreme-bin mass with a
hierarchical interval, so that ensemble dispersion carries uncertainty wherever
it is quoted.

CALIBRATION IS A DIAGNOSTIC HERE, NOT A GATE
============================================
An earlier draft made ensemble calibration a standing adequacy criterion.  The
paper's own stochastic-sampler experiment refutes the premise of that criterion:
repairing the dispersion of the flow-matching cell did not restore its positive
control, it failed more clearly.  A criterion whose motivating mechanism has
been prospectively refuted cannot be used to discount a cell.  The statistic is
therefore reported uniformly, with uncertainty, and interpretation of any
contrast rests on the positive control alone.

The reported rule is executable and applied identically everywhere:
    calibration-adequate (for this diagnostic)  <=>  the hierarchical 95%
    interval of the extreme-bin mass covers the uniform value 2/9.
With ~3 effective temporal blocks this test has little power, so failing to
reject uniformity is NOT evidence of calibration; only a rejection is
informative.  That asymmetry is stated wherever the number is quoted.

Reads retained arrays only: trains nothing, samples nothing, and contacts no
new data.

Usage:  python3 hill_hierarchical_intervals.py [tag ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"

B_BOOT = 4000
SEED = 20260804          # same seed as the released audit, so the time level is comparable

# Two conditioning representations were run on this record and both are audited
# here on identical terms.  `tau_*` supplies the surface traction directly, as the
# quantity the generator was trained to read; `band_*` first expands that traction
# into a near-wall velocity band through the closure's Reichardt profile.  The
# oracle positive control of each representation is its own truth arm:
# `tau_native` (the record's exact DNS traction) and `band_oracle` (the record's
# exact DNS band velocities), respectively.
ARM_SETS = {
    "tau": ("tau_native", "tau_closure", "tau_eqwm", "tau_fartime", "tau_random"),
    "band": ("band_oracle", "band_closure", "band_eqwm", "band_fartime", "band_random"),
}
ORACLE_ARM = {"tau": "tau_native", "band": "band_oracle"}
METHOD_ARM = {"tau": "tau_closure", "band": "band_closure"}
REGIONS = ("full_srcex", "near_srcex", "outer_srcex")
FAMILY = {"H1": "F_flow_matching",
          "H2": "G_denoising_diffusion",
          "H1S": "F_flow_matching_stochastic_sampler",
          "H1S0": "F_flow_matching_deterministic_port_control"}
DEFAULT_TAGS = ("e2_grouped_hills",            # direct traction, 64x192
                "e2_grouped_hills_band",       # band lift, 64x192 (release)
                "e2_grouped_hills_band_r64",   # band lift, 64x192 port control
                "e2_grouped_hills_band_r96",   # band lift, 96x288
                "e2_hill_sde_h1s")             # band lift, stochastic sampler


def block_index_sets(n, block, rng, B):
    """Circular moving-block resamples of the physical-time axis."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    return [np.concatenate([np.arange(s, s + block) % n
                            for s in starts[b]])[:n] for b in range(B)]


def audit_tag(tag, rng):
    comp = np.load(RESULTS / f"{tag}_components.npz", allow_pickle=True)
    meta = json.loads((RESULTS / f"{tag}_results.json").read_text())["_meta"]
    block = int(meta["block"])

    cells = sorted({k.split("|")[0] for k in comp.files if "|test|" in k})

    # discover which conditioning representation this tag ran
    present = {k.split("|")[4] for k in comp.files if "|sse|" in k}
    rep = "tau" if any(a.startswith("tau_") for a in present) else "band"
    arms = tuple(a for a in ARM_SETS[rep] if a in present)
    rec_tag = {"block": block, "representation": rep,
               "oracle_arm": ORACLE_ARM[rep], "method_arm": METHOD_ARM[rep],
               "cells": {}}
    print(f"### {tag}: representation={rep}  arms={arms}")

    for cell in cells:
        seeds = sorted({k.split("|")[2] for k in comp.files
                        if k.startswith(f"{cell}|test|")})
        ns = len(seeds)
        rec = {"family": FAMILY.get(cell, cell), "seeds": seeds,
               "n_seeds": ns, "regions": {}, "proper": {}, "calibration": {}}
        print(f"===== {tag}  {cell}  {FAMILY.get(cell, cell)}  seeds={seeds}")

        # One shared draw of (seed multiset, time blocks) per replicate, reused by
        # every arm and every region so that arm contrasts are paired.  Independent
        # draws per arm would inflate the contrast variance.
        n = len(comp[f"{cell}|test|{seeds[0]}|sst|{REGIONS[0]}"])
        idxs = block_index_sets(n, block, rng, B_BOOT)
        seed_draws = rng.integers(0, ns, size=(B_BOOT, ns))

        def stack(kind, arm=None, reg=None):
            """(n_seeds, n_frames) per-frame statistic."""
            if kind == "sse":
                ks = [f"{cell}|test|{sd}|sse|{arm}|{reg}" for sd in seeds]
            elif kind == "sst":
                ks = [f"{cell}|test|{sd}|sst|{reg}" for sd in seeds]
            else:
                ks = [f"{cell}|test|{sd}|{kind}|{arm}" for sd in seeds]
            if not all(k in comp.files for k in ks):
                return None
            return np.stack([comp[k] for k in ks])

        # ---------------- fluctuation R2 -------------------------------------
        for reg in REGIONS:
            sst = stack("sst", reg=reg)                       # (ns, n); seed-invariant
            sse = {a: stack("sse", a, reg) for a in ("absent",) + arms}
            if sse["absent"] is None:
                continue
            reg_rec = {}
            ab_full = sse["absent"].mean(0)
            sst_full = sst.mean(0)

            for arm in arms:
                if sse[arm] is None:
                    continue
                pt = ((1 - sse[arm].mean(0).sum() / sst_full.sum())
                      - (1 - ab_full.sum() / sst_full.sum()))

                # PRIMARY: declared hierarchical seed -> time-block estimator
                bs = np.empty(B_BOOT)
                for b in range(B_BOOT):
                    sd_ix, ix = seed_draws[b], idxs[b]
                    st = sst[sd_ix].mean(0)[ix].sum()
                    a_ = sse[arm][sd_ix].mean(0)[ix].sum()
                    z_ = sse["absent"][sd_ix].mean(0)[ix].sum()
                    bs[b] = (1 - a_ / st) - (1 - z_ / st)
                lo, hi = np.percentile(bs, [2.5, 97.5])

                # SECONDARY: seed-conditional estimator used by the released audit
                bs2 = np.empty(B_BOOT)
                for b in range(B_BOOT):
                    ix = idxs[b]
                    st = sst_full[ix].sum()
                    bs2[b] = ((1 - sse[arm].mean(0)[ix].sum() / st)
                              - (1 - ab_full[ix].sum() / st))
                lo2, hi2 = np.percentile(bs2, [2.5, 97.5])

                # across-seed spread of the point estimate, reported plainly
                per_seed = [float((1 - sse[arm][i].sum() / sst[i].sum())
                                  - (1 - sse["absent"][i].sum() / sst[i].sum()))
                            for i in range(ns)]

                reg_rec[arm] = {
                    "delta_vs_absent": float(pt),
                    "ci95_hierarchical": [float(lo), float(hi)],
                    "excludes_zero_hierarchical": bool(lo > 0 or hi < 0),
                    "ci95_seed_conditional": [float(lo2), float(hi2)],
                    "excludes_zero_seed_conditional": bool(lo2 > 0 or hi2 < 0),
                    "per_seed_delta": per_seed,
                }
                flag = "  <-- excludes zero" if (lo > 0 or hi < 0) else ""
                narrowed = "" if (lo > 0 or hi < 0) == (lo2 > 0 or hi2 < 0) \
                    else "   [VERDICT DIFFERS FROM SEED-CONDITIONAL]"
                print(f"   {reg:12s} {arm:13s} {pt:+.5f} "
                      f"hier[{lo:+.5f},{hi:+.5f}]{flag}"
                      f"  seedcond[{lo2:+.5f},{hi2:+.5f}]{narrowed}")
            reg_rec["absent_absolute_R2"] = float(1 - ab_full.sum() / sst_full.sum())

            # -------- paired method-arm-vs-control-arm contrasts --------------
            # "Does the method beat absence?" and "is the method distinguishable
            # from a control of identical support?" are different questions.  The
            # second cannot be read off two intervals against absence: those share
            # the absent arm and are correlated.  Each contrast below is therefore
            # recomputed arm-minus-arm inside every replicate, on the shared
            # (seed, time-block) draw.
            meth = METHOD_ARM[rep]
            if sse.get(meth) is not None:
                pair = {}
                for other in arms:
                    if other == meth:
                        continue
                    ptp = ((1 - sse[meth].mean(0).sum() / sst_full.sum())
                           - (1 - sse[other].mean(0).sum() / sst_full.sum()))
                    bsp = np.empty(B_BOOT)
                    for b in range(B_BOOT):
                        sd_ix, ix = seed_draws[b], idxs[b]
                        st = sst[sd_ix].mean(0)[ix].sum()
                        m_ = sse[meth][sd_ix].mean(0)[ix].sum()
                        o_ = sse[other][sd_ix].mean(0)[ix].sum()
                        bsp[b] = (1 - m_ / st) - (1 - o_ / st)
                    lop, hip = np.percentile(bsp, [2.5, 97.5])
                    pair[other] = {"delta": float(ptp),
                                   "ci95_hierarchical": [float(lop), float(hip)],
                                   "excludes_zero_hierarchical": bool(lop > 0 or hip < 0)}
                    print(f"   {reg:12s} {meth}-vs-{other:13s} {ptp:+.5f} "
                          f"hier[{lop:+.5f},{hip:+.5f}]"
                          f"{'  <-- excludes zero' if (lop > 0 or hip < 0) else ''}")
                reg_rec["_paired_method_contrasts"] = pair
            rec["regions"][reg] = reg_rec

        # ---------------- proper scores --------------------------------------
        for metric in ("crps", "escore"):
            ab = stack(metric, "absent")
            if ab is None:
                continue
            mrec = {}
            for arm in arms:
                ar = stack(metric, arm)
                if ar is None:
                    continue
                d_full = (ar - ab).mean(0)
                bs = np.empty(B_BOOT)
                for b in range(B_BOOT):
                    sd_ix, ix = seed_draws[b], idxs[b]
                    bs[b] = (ar[sd_ix] - ab[sd_ix]).mean(0)[ix].mean()
                lo, hi = np.percentile(bs, [2.5, 97.5])
                mrec[arm] = {"delta_vs_absent": float(d_full.mean()),
                             "ci95_hierarchical": [float(lo), float(hi)],
                             "excludes_zero_hierarchical": bool(lo > 0 or hi < 0)}
            rec["proper"][metric] = mrec

        # ---------------- calibration DIAGNOSTIC (not a gate) ----------------
        for arm in ("absent",) + arms:
            rh = stack("rankhist", arm)
            if rh is None:
                continue
            nbins = rh.shape[-1]
            uniform = 2.0 / nbins
            per_frame = rh[:, :, 0] + rh[:, :, -1]          # (ns, n) extreme mass
            pt = float(per_frame.mean())
            bs = np.empty(B_BOOT)
            for b in range(B_BOOT):
                sd_ix, ix = seed_draws[b], idxs[b]
                bs[b] = per_frame[sd_ix].mean(0)[ix].mean()
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rec["calibration"][arm] = {
                "extreme_bin_mass": pt,
                "ci95_hierarchical": [float(lo), float(hi)],
                "uniform_reference": uniform,
                # executable rule, applied identically to every cell in the paper
                "interval_covers_uniform": bool(lo <= uniform <= hi),
                "direction": ("under-dispersed" if pt > uniform else "over-dispersed"),
            }
        if rec["calibration"]:
            for arm in ("absent", ORACLE_ARM[rep]):
                if arm in rec["calibration"]:
                    c = rec["calibration"][arm]
                    print(f"   calibration  {arm:13s} extreme mass {c['extreme_bin_mass']:.4f} "
                          f"[{c['ci95_hierarchical'][0]:.4f},{c['ci95_hierarchical'][1]:.4f}] "
                          f"vs uniform {c['uniform_reference']:.4f} "
                          f"({'covers' if c['interval_covers_uniform'] else 'REJECTS'} uniformity, "
                          f"{c['direction']})")
        rec_tag["cells"][cell] = rec
    return rec_tag


def main():
    tags = sys.argv[1:] or list(DEFAULT_TAGS)
    rng = np.random.default_rng(SEED)
    out = {"_meta": {
        "probe": "hill_hierarchical_intervals.py",
        "B": B_BOOT, "seed": SEED,
        "primary_estimator": "hierarchical: seeds resampled with replacement, "
                             "then circular moving-block resample in physical time "
                             "(the estimator declared in Methods)",
        "secondary_estimator": "seed-conditional: per-frame SSE averaged over the "
                               "realised seeds, then moving-block resample in time "
                               "(the estimator used by the released audit)",
        "calibration_status": "DIAGNOSTIC, not a gate; see module docstring",
        "reads": "retained per-frame components only; no training, no sampling",
    }, "tags": {}}

    for tag in tags:
        out["tags"][tag] = audit_tag(tag, rng)

    dst = RESULTS / "hill_hierarchical_intervals.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dst}")

    # -------- compact summary of what changed under the declared estimator ----
    print("\n===== verdict changes: declared (hierarchical) vs released (seed-conditional) =====")
    nchange = 0
    for tag, t in out["tags"].items():
        for cell, c in t["cells"].items():
            for reg, r in c["regions"].items():
                for arm, a in r.items():
                    if not isinstance(a, dict):
                        continue
                    if a["excludes_zero_hierarchical"] != a["excludes_zero_seed_conditional"]:
                        nchange += 1
                        print(f"  {tag} {cell} {reg} {arm}: "
                              f"seed-conditional={'SIG' if a['excludes_zero_seed_conditional'] else 'ns'}"
                              f" -> hierarchical={'SIG' if a['excludes_zero_hierarchical'] else 'ns'}"
                              f"  ({a['delta_vs_absent']:+.5f})")
    if nchange == 0:
        print("  none: every arm keeps its verdict under the declared estimator")
    print(f"\n{nchange} verdict change(s)")


if __name__ == "__main__":
    main()
