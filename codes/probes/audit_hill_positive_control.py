#!/usr/bin/env python3
"""
audit_hill_positive_control.py -- what the 64x192 grouped hill run actually shows.

The manuscript states, without qualification, that the periodic-hill unit "fails
its own positive control (even true DNS near-wall information does not improve
reconstruction there)".  That sentence is written as a property of the RECORD.
The retained per-frame components say it is a property of ONE OF THE TWO
GENERATIVE FAMILIES.

This probe re-derives the positive control from
`e2_grouped_hills_band_components.npz` with the same moving-block bootstrap and
the same block length the run itself recorded, and reports each family
separately.  It reads only retained arrays; it trains nothing and samples
nothing, so it cannot be accused of re-running until a favourable draw.

Reported for each family, each region:
    <arm> vs absent  =  R2_fluct(arm) - R2_fluct(absent),  95% block-bootstrap CI

The standing rule is repair-2's: an interface conclusion may be drawn from a unit
only if the ORACLE arm improves on absent with an interval excluding zero.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
# tag may be overridden so the same audit can be run against the resolution
# re-test (`..._r64`, `..._r96`) on identical terms
TAG = sys.argv[1] if len(sys.argv) > 1 else "e2_grouped_hills_band"
B_BOOT = 4000
SEED = 20260804

ARMS = ("band_oracle", "band_closure", "band_eqwm", "band_fartime", "band_random")
REGIONS = ("full_srcex", "near_srcex", "outer_srcex")
FAMILY = {"H1": "F_flow_matching", "H2": "G_denoising_diffusion",
          "H1S": "F_flow_matching_stochastic_sampler",
          "H1S0": "F_flow_matching_deterministic_port_control"}


def main():
    comp = np.load(RESULTS / f"{TAG}_components.npz", allow_pickle=True)
    meta = json.loads((RESULTS / f"{TAG}_results.json").read_text())["_meta"]
    block = int(meta["block"])
    rng = np.random.default_rng(SEED)
    out = {"_meta": {"source": f"{TAG}_components.npz", "block": block,
                     "B": B_BOOT, "seed": SEED,
                     "note": "seed-averaged SSE, moving-block bootstrap over "
                             "the test unit's physical times"},
           "cells": {}}

    # Cells are discovered from the components file rather than hard-coded, so
    # the same audit runs unchanged on the released tag, the resolution re-tests
    # and the stochastic-sampler cells.
    cells = sorted({k.split("|")[0] for k in comp.files if "|test|" in k})
    for cell in cells:
        seeds = sorted({k.split("|")[2] for k in comp.files
                        if k.startswith(f"{cell}|test|")})
        rec = {"family": FAMILY.get(cell, cell), "seeds": seeds, "regions": {}}
        print(f"===== {cell}  {FAMILY.get(cell, cell)}  seeds={seeds}")
        for reg in REGIONS:
            sst = comp[f"{cell}|test|{seeds[0]}|sst|{reg}"]
            n = len(sst)
            nb = int(np.ceil(n / block))
            # one shared set of bootstrap resamples for every arm in this region,
            # so the arm contrasts are paired rather than independently noisy
            starts = rng.integers(0, n, size=(B_BOOT, nb))
            idxs = [np.concatenate([np.arange(s, s + block) % n
                                    for s in starts[b]])[:n] for b in range(B_BOOT)]

            def r2(arm):
                s = np.mean([comp[f"{cell}|test|{sd}|sse|{arm}|{reg}"]
                             for sd in seeds], axis=0)
                return s

            sse_ab = r2("absent")
            reg_rec = {}
            for arm in ARMS:
                sse_ar = r2(arm)
                pt = ((1 - sse_ar.sum() / sst.sum())
                      - (1 - sse_ab.sum() / sst.sum()))
                bs = np.empty(B_BOOT)
                for b, ix in enumerate(idxs):
                    st = sst[ix].sum()
                    bs[b] = ((1 - sse_ar[ix].sum() / st)
                             - (1 - sse_ab[ix].sum() / st))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                excl = bool(lo > 0 or hi < 0)
                reg_rec[arm] = {"delta_vs_absent": float(pt),
                                "ci95": [float(lo), float(hi)],
                                "excludes_zero": excl}
                flag = "  <-- excludes zero" if excl else ""
                print(f"   {reg:12s} {arm:13s} {pt:+.4f} "
                      f"[{lo:+.4f},{hi:+.4f}]{flag}")
            reg_rec["absent_absolute_R2"] = float(1 - sse_ab.sum() / sst.sum())
            rec["regions"][reg] = reg_rec
        # ---- ensemble calibration -------------------------------------------
        # A contrast in R2 between conditioning arms is only interpretable if the
        # ensemble it is computed from is calibrated.  The rank histogram of the
        # truth within the 8-member ensemble is uniform (mass 2/9 = 0.222 in the
        # two extreme bins) for a calibrated ensemble; mass concentrated in the
        # extremes means the ensemble is too narrow, so the model commits to
        # structure it has no support for and is penalised for the commitment
        # rather than for the information it was given.
        cal = {}
        for arm in ("absent", "band_oracle"):
            k = [f"{cell}|test|{sd}|rankhist|{arm}" for sd in seeds]
            if not all(kk in comp.files for kk in k):
                continue
            hh = np.sum([comp[kk].sum(0) for kk in k], axis=0)
            hh = hh / hh.sum()
            cal[arm] = {"hist": [float(x) for x in hh],
                        "extreme_mass": float(hh[0] + hh[-1]),
                        "uniform_reference": 2.0 / len(hh)}
        rec["calibration"] = cal
        if "absent" in cal and "band_oracle" in cal:
            print(f"   calibration  extreme-bin mass  absent="
                  f"{cal['absent']['extreme_mass']:.3f}  "
                  f"oracle={cal['band_oracle']['extreme_mass']:.3f}  "
                  f"(uniform={cal['absent']['uniform_reference']:.3f})")

        # ---- proper scoring rules -------------------------------------------
        # R2 alone cannot separate "the conditioning carries no information" from
        # "the conditioning makes an already-overconfident ensemble worse".  CRPS
        # and the energy score are proper and penalise miscalibration directly,
        # so agreement across all three is what makes a positive control
        # trustworthy.  Lower is better; the contrast is arm - absent.
        proper = {}
        for metric in ("crps", "escore"):
            mrec = {}
            ab = np.mean([comp[f"{cell}|test|{sd}|{metric}|absent"]
                          for sd in seeds], axis=0)
            n = len(ab)
            nb = int(np.ceil(n / block))
            starts = rng.integers(0, n, size=(B_BOOT, nb))
            idxs = [np.concatenate([np.arange(s, s + block) % n
                                    for s in starts[b]])[:n] for b in range(B_BOOT)]
            for arm in ARMS:
                k = [f"{cell}|test|{sd}|{metric}|{arm}" for sd in seeds]
                if not all(kk in comp.files for kk in k):
                    continue
                ar = np.mean([comp[kk] for kk in k], axis=0)
                diff = ar - ab
                bs = np.array([diff[ix].mean() for ix in idxs])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                mrec[arm] = {"delta_vs_absent": float(diff.mean()),
                             "ci95": [float(lo), float(hi)],
                             "favours": ("arm" if hi < 0 else
                                         "absent" if lo > 0 else "neither")}
                print(f"   {metric:7s} {arm:13s} {diff.mean():+.5f} "
                      f"[{lo:+.5f},{hi:+.5f}]  favours {mrec[arm]['favours']}")
            proper[metric] = mrec
        rec["proper_scores"] = proper

        rec["positive_control_passes"] = bool(
            rec["regions"]["full_srcex"]["band_oracle"]["excludes_zero"]
            and rec["regions"]["full_srcex"]["band_oracle"]["delta_vs_absent"] > 0)
        print(f"   POSITIVE CONTROL ({cell}): "
              f"{'PASS' if rec['positive_control_passes'] else 'FAIL/INCONCLUSIVE'}")
        out["cells"][cell] = rec

    p = RESULTS / f"hill_positive_control_audit_{TAG}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[write] {p.name}")


if __name__ == "__main__":
    main()
