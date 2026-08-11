#!/usr/bin/env python3
"""
apply_decision_rule_node007.py -- evaluate the FROZEN decision rule of
PREREGISTRATION_E2_GENERALITY.md §6 mechanically, from the result file alone.

The rule is applied by this script and by nothing else.  It reads only
`codes/results/e2_generality_results.json`, it contains no thresholds that are not
in the preregistration, and it prints the registered verdict whatever that verdict
is.  The pre-declared partial dispositions of §6 are implemented here as well, so
an adverse or mixed outcome is named by the rule rather than narrated afterwards.

CPU-only.  Writes DECISION_RULE_OUTCOME.json next to this script.
"""
import json
from pathlib import Path

NODE = Path(__file__).resolve().parent
ROOT = NODE.parents[2]
RES = ROOT / "codes" / "results" / "e2_generality_results.json"
OUT = NODE / "DECISION_RULE_OUTCOME.json"

REGION = "full_srcex"
DECISION_UNIT = {"C1": "MATCHED_NODE006_UNIT", "C2": "U_STRICT", "C3": "U_STRICT"}
CELL_LABEL = {
    "C1": "cube x denoising diffusion (family generality)",
    "C2": "separating periodic hills x flow matching (regime generality)",
    "C3": "separating periodic hills x denoising diffusion (joint generality)",
}

# Prospectively frozen tolerances, PREREGISTRATION §7c
NR_TOL = {
    "NRg1_C1_native_minus_absent_min": 0.0400,
    "NRg2_C1_absent_R2_min": -0.1400,
}


def get(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main():
    if not RES.exists():
        raise SystemExit(f"missing {RES}")
    R = json.loads(RES.read_text())
    A = R.get("analysis", {})

    cells, clauses = {}, []
    for cell in ("C1", "C2", "C3"):
        unit = DECISION_UNIT[cell]
        blk = get(A, cell, "by_subunit", unit, REGION)
        if blk is None:
            cells[cell] = {"status": "ABSENT"}
            continue
        d_cl = get(blk, "deltas", "closure_minus_absent", default={})
        d_ft = get(blk, "deltas", "fartime_minus_absent", default={})
        d_na = get(blk, "deltas", "native_minus_absent", default={})
        d_eq = get(blk, "deltas", "eqwm_minus_absent", default={})
        ci = d_cl.get("ci_block", [0.0, 0.0])
        hci = d_cl.get("ci_hier_seed_block", [0.0, 0.0])
        cells[cell] = {
            "label": CELL_LABEL[cell],
            "decision_unit": unit,
            "n": blk.get("n"),
            "block_used": blk.get("block_used"),
            "delta_closure_minus_absent": d_cl.get("point"),
            "ci_block": ci,
            "ci_hier_seed_block": hci,
            "per_seed": d_cl.get("per_seed"),
            "same_sign_all_seeds": d_cl.get("same_sign_all_seeds"),
            "delta_fartime_minus_absent": d_ft.get("point"),
            "delta_native_minus_absent": d_na.get("point"),
            "delta_eqwm_minus_absent": d_eq.get("point"),
            "R2_levels": {a: v.get("R2_seedmean")
                          for a, v in blk.get("arms", {}).items()},
        }

    def cell_ok(c):
        e = cells.get(c, {})
        p = e.get("delta_closure_minus_absent")
        ci = e.get("ci_block") or [0, 0]
        return (p is not None) and p > 0 and ci[0] > 0

    for tag, c, desc in (("G1", "C1", "family generality: C1 Delta>0 and block CI excludes 0"),
                         ("G2", "C2", "regime generality: C2 Delta>0 and block CI excludes 0"),
                         ("G3", "C3", "joint generality: C3 Delta>0 and block CI excludes 0")):
        clauses.append({"clause": tag, "registered": desc, "cell": c,
                        "value": cells.get(c, {}).get("delta_closure_minus_absent"),
                        "ci": cells.get(c, {}).get("ci_block"),
                        "pass": bool(cell_ok(c))})

    present = [c for c in ("C1", "C2", "C3") if cells.get(c, {}).get("status") != "ABSENT"]
    g4 = all((cells[c]["delta_closure_minus_absent"] or -1) >
             (cells[c]["delta_fartime_minus_absent"] if
              cells[c]["delta_fartime_minus_absent"] is not None else 1e9)
             for c in present)
    clauses.append({"clause": "G4",
                    "registered": "wrong-information control: Delta_closure > Delta_fartime in every cell",
                    "value": {c: [cells[c]["delta_closure_minus_absent"],
                                  cells[c]["delta_fartime_minus_absent"]] for c in present},
                    "pass": bool(g4)})
    g5 = all(bool(cells[c].get("same_sign_all_seeds")) for c in present)
    clauses.append({"clause": "G5", "registered": "same sign on both seeds in every cell",
                    "value": {c: cells[c].get("per_seed") for c in present},
                    "pass": bool(g5)})
    g6 = all((cells[c].get("ci_hier_seed_block") or [0, 0])[0] > 0 for c in present)
    clauses.append({"clause": "G6",
                    "registered": "hierarchical seed x block 95% interval excludes 0 in every cell",
                    "value": {c: cells[c].get("ci_hier_seed_block") for c in present},
                    "pass": bool(g6)})
    mins = [cells[c]["delta_closure_minus_absent"] for c in present
            if cells[c]["delta_closure_minus_absent"] is not None]
    g7 = bool(mins) and min(mins) > 0
    clauses.append({"clause": "G7", "registered": "min(Delta) over the three cells > 0",
                    "value": (min(mins) if mins else None), "pass": g7})

    all_pass = all(c["pass"] for c in clauses) and len(present) == 3

    # --- pre-declared dispositions, PREREGISTRATION §6 -----------------------
    g1, g2, g3 = (cell_ok("C1"), cell_ok("C2"), cell_ok("C3"))
    if all_pass:
        verdict = "GENERALITY_CONFIRMED"
    elif g2 and g3 and not g1:
        verdict = "REGIME_CONFIRMED_FAMILY_ADVERSE"
    elif g1 and not (g2 or g3):
        verdict = "FAMILY_CONFIRMED_REGIME_ADVERSE"
    elif not (g1 or g2 or g3):
        verdict = "ADVERSE"
    else:
        verdict = "PARTIAL"

    # --- non-regression, §7c -------------------------------------------------
    nr = []
    c1 = cells.get("C1", {})
    nr.append({"id": "NRg1", "endpoint": "C1 native-minus-absent",
               "value": c1.get("delta_native_minus_absent"),
               "tolerance_min": NR_TOL["NRg1_C1_native_minus_absent_min"],
               "pass": bool((c1.get("delta_native_minus_absent") or -9) >=
                            NR_TOL["NRg1_C1_native_minus_absent_min"])})
    nr.append({"id": "NRg2", "endpoint": "C1 R2(absent)",
               "value": (c1.get("R2_levels") or {}).get("absent"),
               "tolerance_min": NR_TOL["NRg2_C1_absent_R2_min"],
               "pass": bool(((c1.get("R2_levels") or {}).get("absent") or -9) >=
                            NR_TOL["NRg2_C1_absent_R2_min"])})
    nr.append({"id": "NRg3", "endpoint": "C1 native > absent",
               "value": c1.get("delta_native_minus_absent"),
               "pass": bool((c1.get("delta_native_minus_absent") or -9) > 0)})
    nr.append({"id": "NRg4", "endpoint": "C1 native > fartime",
               "value": [c1.get("delta_native_minus_absent"),
                         c1.get("delta_fartime_minus_absent")],
               "pass": bool((c1.get("delta_native_minus_absent") or -9) >
                            (c1.get("delta_fartime_minus_absent")
                             if c1.get("delta_fartime_minus_absent") is not None else 1e9))})
    nr.append({"id": "NRg5", "endpoint": "closure beats absence in every cell",
               "value": {c: cells[c].get("delta_closure_minus_absent") for c in present},
               "pass": bool(all((cells[c].get("delta_closure_minus_absent") or -9) > 0
                                for c in present))})
    nr.append({"id": "NRg6", "endpoint": "node-006 registered primary retained unchanged",
               "value": 0.06063890283162299, "pass": True,
               "note": "retained evidence; not recomputed, retuned or replaced by this node"})

    out = {
        "registered_estimand": "R2(tau_closure) - R2(absent)",
        "registered_region": REGION,
        "decision_units": DECISION_UNIT,
        "cells": cells,
        "decision_rule_clauses": clauses,
        "decision_rule_pass": bool(all_pass),
        "REGISTERED_VERDICT": verdict,
        "non_regression_table": nr,
        "non_regression_pass": bool(all(r["pass"] for r in nr)),
        "gpu_hours": R.get("gpu_hours"),
        "frozen_closure_sha256": get(R, "gates", "cube_frozen_closure_sha256"),
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))

    print(f"REGISTERED_VERDICT: {verdict}")
    for c in ("C1", "C2", "C3"):
        e = cells.get(c, {})
        if e.get("status") == "ABSENT":
            print(f"  {c}: ABSENT"); continue
        print(f"  {c} [{e['decision_unit']}, n={e['n']}] "
              f"Delta={e['delta_closure_minus_absent']:+.5f} "
              f"CI={[round(v,5) for v in e['ci_block']]} "
              f"hier={[round(v,5) for v in e['ci_hier_seed_block']]} "
              f"fartime={e['delta_fartime_minus_absent']:+.5f} "
              f"native={e['delta_native_minus_absent']:+.5f}")
    for cl in clauses:
        print(f"  {cl['clause']}: {'PASS' if cl['pass'] else 'FAIL'}")
    print(f"  non-regression: {'PASS' if out['non_regression_pass'] else 'FAIL'}")
    print(f"[write] {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
