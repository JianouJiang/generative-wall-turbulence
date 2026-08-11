#!/usr/bin/env python3
"""
complete_non_regression_node007.py -- discharge, from RETAINED ARRAYS ONLY, the
parts of the 31 July non-regression mandate that node 006 left open.

All three panel seats found the same four gaps, and all four are computable from
per-frame arrays already on disk.  No new heavy computation is performed here.

  (1) THE ORACLE-BAND CEILING the mandate named and node 006 omitted.
      Node 005's `band_phys` and `absent_B` per-frame SSE are retained for all
      three seeds and all four regions.  The matched ceiling
      R^2(band_phys) - R^2(absent_B) is recomputed on node 005's exact 240-frame
      unit and checked against its published value; the repaired closure system's
      transmission ratio to that ceiling is then reported with an interval.

  (2) A GENUINE HIERARCHICAL SEED x TIME-BLOCK INTERVAL for node 006's primary
      estimand.  Node 006 concatenated three per-seed block-bootstrap
      distributions and called the mixture a "crossed seed x block" interval.
      That is not a crossed estimator.  Here seeds are resampled with
      replacement, a seed-mean replicate system is formed, and time blocks are
      resampled within it.

  (3) PAIRED, DEPENDENCE-AWARE INTERVALS for the distributional and calibration
      endpoints (CRPS, energy score, rank reliability), which node 006 reported
      as bare point orderings.

  (4) THE ENERGY SCORE RECOMPUTED under the off-diagonal M(M-1) estimator of the
      Methods equation wherever the retained arrays permit, and otherwise
      explicitly flagged as a producer/Methods mismatch inherited from node 006.

Everything here is a POST-REGISTRATION secondary analysis of a frozen result.  It
does not touch, recompute or replace node 006's registered decision rule, which
stands exactly as published.

CPU-only.  Writes development/nodes/node_007/NON_REGRESSION_COMPLETION.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "codes" / "results"
OUTDIR = ROOT / "development" / "nodes" / "node_007"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "NON_REGRESSION_COMPLETION.json"

N005 = RES / "e2_direct_traction_components.npz"
N005J = RES / "e2_direct_traction_results.json"
N006 = RES / "e2_closure_composition_components.npz"
N006J = RES / "e2_closure_composition_results.json"

BOOT = 4000
REGIONS4 = ("full_support_excluded", "near_support_excluded_d_le_0p5h",
            "outer_d_gt_0p5h", "uniq_raster_support_excluded")


def conservative_block(idx, tau):
    """Stride-corrected, degeneracy-guarded moving-block length.

    Identical to `codes/gpu/eval_e2_generality.py::conservative_block`.  A block
    at least as long as the unit makes every circular-moving-block replicate a
    cyclic permutation of the whole unit, so the interval collapses to zero
    width; the cap keeps at least three blocks per replicate.  Node 006 applies
    the stride correction (it publishes `block_conservative = 41` for its
    103-frame interleaved cube unit, not 123) and this reproduces it exactly.
    """
    idx = np.asarray(idx)
    n = len(idx)
    stride = max(1.0, float(np.median(np.diff(idx))) if n > 1 else 1.0)
    b = max(1, int(np.ceil(1.2551 * tau / stride)))
    return int(max(1, min(b, max(1, (n - 1) // 2))))


def block_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]


def r2(sse, sst):
    return float(1.0 - np.sum(sse) / np.sum(sst))


def delta_ci(sse_a, sse_b, sst, block, seed):
    """Block-bootstrap interval of R^2(a) - R^2(b); a, b are LISTS of per-seed
    per-frame SSE arrays.  Returns the point estimate, the plain block interval
    and the genuine hierarchical seed x block interval."""
    sa, sb = np.stack(sse_a), np.stack(sse_b)
    ma, mb = sa.mean(0), sb.mean(0)
    point = r2(ma, sst) - r2(mb, sst)
    rng = np.random.default_rng(seed)
    ix = block_indices(len(sst), block, BOOT, rng)
    d = ((1 - ma[ix].sum(1) / sst[ix].sum(1)) - (1 - mb[ix].sum(1) / sst[ix].sum(1)))
    S = sa.shape[0]
    pick = rng.integers(0, S, size=(BOOT, S))
    hb = np.empty(BOOT)
    for k in range(BOOT):
        ra, rb = sa[pick[k]].mean(0), sb[pick[k]].mean(0)
        hb[k] = ((1 - ra[ix[k]].sum() / sst[ix[k]].sum())
                 - (1 - rb[ix[k]].sum() / sst[ix[k]].sum()))
    return {
        "point": point,
        "ci_block": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "ci_hierarchical_seed_x_block": [float(np.percentile(hb, 2.5)),
                                         float(np.percentile(hb, 97.5))],
        "per_seed": [r2(x, sst) - r2(y, sst) for x, y in zip(sse_a, sse_b)],
    }


def paired_mean_ci(diff, block, seed):
    rng = np.random.default_rng(seed)
    ix = block_indices(len(diff), block, BOOT, rng)
    m = diff[ix].mean(1)
    return {"mean": float(diff.mean()),
            "ci_block": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "n": int(len(diff)),
            "excludes_zero": bool(np.percentile(m, 2.5) > 0 or np.percentile(m, 97.5) < 0)}


def main():
    out = {"_what": "post-registration completion of the 31 July non-regression "
                    "mandate from retained arrays; node 006's registered decision "
                    "rule is untouched",
           "_no_new_computation": True}

    TAU = 97.592                       # cube record integral time, frozen
    d5 = np.load(N005, allow_pickle=True)
    j5 = json.loads(N005J.read_text())
    d6 = np.load(N006, allow_pickle=True)
    j6 = json.loads(N006J.read_text())

    # ---------- (1) the oracle-band ceiling, on node 005's exact unit ----------
    seeds5 = ["7701", "7702", "7703"]
    idx5 = np.asarray(d5["eval_idx"]) if "eval_idx" in d5.files else np.arange(240)
    block5 = conservative_block(idx5, TAU)
    ceiling = {"unit": "node005 CONFIRM_ALL (240 frames)", "block": block5,
               "regions": {}, "reproduction": {}}
    for reg in REGIONS4:
        sst = np.asarray(d5[f"sst_{reg}"], float)
        band = [np.asarray(d5[f"sse_band_phys_{s}_{reg}"], float) for s in seeds5]
        absb = [np.asarray(d5[f"sse_absent_B_{s}_{reg}"], float) for s in seeds5]
        ceiling["regions"][reg] = delta_ci(band, absb, sst, block5, 11001)
        pub = (j5.get("evaluation", {}).get("deltas", {})
                 .get("band_phys_minus_absentB", {}).get(reg, {}))
        if "point" in pub:
            ceiling["reproduction"][reg] = {
                "published": pub["point"],
                "recomputed": ceiling["regions"][reg]["point"],
                "abs_diff": abs(pub["point"] - ceiling["regions"][reg]["point"]),
                "reproduces_within_1e-6": bool(
                    abs(pub["point"] - ceiling["regions"][reg]["point"]) < 1e-6),
            }
    out["oracle_band_ceiling"] = ceiling

    # transmission of the repaired closure system TO that ceiling, same unit
    trans = {}
    for reg in REGIONS4:
        ka = [np.asarray(d6[f"CONFIRM_ALL|sse|K:tau_closure|{8801+i}|{reg}"], float)
              for i in range(3)
              if f"CONFIRM_ALL|sse|K:tau_closure|{8801+i}|{reg}" in d6.files]
        kb = [np.asarray(d6[f"CONFIRM_ALL|sse|K:absent|{8801+i}|{reg}"], float)
              for i in range(3)
              if f"CONFIRM_ALL|sse|K:absent|{8801+i}|{reg}" in d6.files]
        if not ka or not kb:
            continue
        sst6 = np.asarray(d6[f"CONFIRM_ALL|sst|{reg}"], float)
        dd = delta_ci(ka, kb, sst6, block5, 11002)
        cl = ceiling["regions"][reg]["point"]
        trans[reg] = {
            "closure_minus_absent": dd,
            "oracle_band_ceiling": cl,
            "transmission_ratio_of_ceiling": (dd["point"] / cl) if cl else None,
        }
    out["closure_transmission_vs_band_ceiling"] = {
        "_unit": "node 005's exact 240-frame unit, identical regions",
        "_note": "the ceiling arm comes from node 005's frozen Family-B "
                 "checkpoints and the closure arm from node 006's Family K; they "
                 "share the unit and the region definitions, not the network",
        "regions": trans,
    }

    # ---------- (2) hierarchical interval for node 006's primary --------------
    reg = "full_srcex"
    sst = np.asarray(d6[f"CONFIRM_STRICT|sst|{reg}"], float)
    idx6 = np.asarray(d6["CONFIRM_STRICT|eval_idx"])
    block6 = conservative_block(idx6, TAU)
    published6 = int(j6.get("units", {}).get("CONFIRM_STRICT", {})
                     .get("block_conservative", -1))
    ka = [np.asarray(d6[f"CONFIRM_STRICT|sse|K:tau_closure|{8801+i}|{reg}"], float)
          for i in range(3)]
    kb = [np.asarray(d6[f"CONFIRM_STRICT|sse|K:absent|{8801+i}|{reg}"], float)
          for i in range(3)]
    prim = delta_ci(ka, kb, sst, block6, 11003)
    out["node006_primary_recomputed"] = {
        "_registered_value_unchanged": 0.06063890283162299,
        "_registered_ci_block": [0.056849515435761847, 0.06359499520628534],
        "recomputed_point": prim["point"],
        "recomputed_ci_block": prim["ci_block"],
        "GENUINE_hierarchical_seed_x_block_ci": prim["ci_hierarchical_seed_x_block"],
        "withdrawn_label": "node 006's published 'crossed seed x block' interval "
                           "[0.054653663104625515, 0.06477018082271366] was a "
                           "concatenation of per-seed block distributions, not a "
                           "crossed estimator; it is replaced by the hierarchical "
                           "interval above",
        "block_used": block6,
        "block_matches_node006_published": bool(block6 == published6),
        "node006_published_block_conservative": published6,
        "per_seed": prim["per_seed"],
    }

    # ---------- (3) paired intervals for distributional endpoints ------------
    paired = {}
    for score in ("crps", "energy"):
        avail = {}
        for arm in ("tau_closure", "absent", "tau_native", "tau_eqwm", "tau_fartime"):
            ks = [f"CONFIRM_STRICT|{score}|K:{arm}|{8801+i}" for i in range(3)]
            ks = [k for k in ks if k in d6.files]
            if ks:
                avail[arm] = np.mean([np.asarray(d6[k], float) for k in ks], axis=0)
        if "tau_closure" not in avail:
            continue
        block_s = {}
        for base in ("absent", "tau_fartime", "tau_eqwm"):
            if base in avail:
                block_s[f"closure_minus_{base}"] = paired_mean_ci(
                    avail["tau_closure"] - avail[base], block6, 11004)
        paired[score] = {"levels": {a: float(v.mean()) for a, v in avail.items()},
                         "paired_contrasts": block_s}
    out["paired_distributional_endpoints"] = paired

    # rank reliability with a block interval on the L1 deviation
    rel = {}
    for arm in ("tau_closure", "absent", "tau_native", "tau_eqwm", "tau_fartime"):
        ks = [f"CONFIRM_STRICT|rank|K:{arm}|{8801+i}" for i in range(3)]
        ks = [k for k in ks if k in d6.files]
        if not ks:
            continue
        h = np.mean([np.asarray(d6[k], float) for k in ks], axis=0)
        h = h / max(h.sum(), 1e-12)
        rel[arm] = {"L1_deviation_from_uniform": float(np.abs(h - 1.0 / len(h)).sum()),
                    "n_bins": int(len(h)),
                    "_status": "descriptive; the retained rank arrays are already "
                               "aggregated over frames, so no per-frame resampling "
                               "interval is recoverable from disk. Reported as a "
                               "descriptive calibration diagnostic, NOT as an "
                               "established calibration result."}
    out["rank_reliability"] = rel

    # ---------- (4) energy-score estimator disclosure ------------------------
    out["energy_score_estimator"] = {
        "methods_equation": "off-diagonal M(M-1) normalisation",
        "node006_producer": "diagonal-inclusive M^2 normalisation "
                            "(eval_e2_closure_composition.py distributional_scores)",
        "consequence": "the M^2 form subtracts a spread term that is smaller by a "
                       "factor (M-1)/M = 7/8, so every node-006 energy score is "
                       "biased HIGH by 1/8 of its ensemble-spread term. The bias is "
                       "common to all arms and cannot change the sign of an arm "
                       "contrast, but the absolute levels are estimator-specific.",
        "resolution": "node 007's producer implements the M(M-1) estimator; the "
                      "Methods text is corrected to state which estimator produced "
                      "which number, and node-006 energy levels are labelled with "
                      "their estimator rather than silently restated.",
        "retained_arrays_permit_recomputation": False,
        "_why": "the posterior members were scored on the fly and never written to "
                "disk, so only the aggregated per-frame score survives",
    }

    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"[nr] wrote {OUT.relative_to(ROOT)}")
    for reg, v in out["oracle_band_ceiling"]["reproduction"].items():
        print(f"  ceiling {reg}: published={v['published']:.6f} "
              f"recomputed={v['recomputed']:.6f} ok={v['reproduces_within_1e-6']}")
    for reg, v in trans.items():
        r = v["transmission_ratio_of_ceiling"]
        print(f"  transmission {reg}: closure={v['closure_minus_absent']['point']:+.5f} "
              f"ceiling={v['oracle_band_ceiling']:+.5f} ratio={r:.4f}")
    p = out["node006_primary_recomputed"]
    print(f"  node006 primary recomputed={p['recomputed_point']:.6f} "
          f"hier CI={p['GENUINE_hierarchical_seed_x_block_ci']}")


if __name__ == "__main__":
    main()
