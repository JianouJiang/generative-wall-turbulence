#!/usr/bin/env python3
"""Apply the frozen node-006 decision rule mechanically to the harvested result.

The verdict is COMPUTED from the result file, clause by clause, exactly as
registered in PREREGISTRATION_E2_CLOSURE.md Sec. 7.  Nothing here selects a
region, a unit or a comparison: all three are fixed by the preregistration
(`full_srcex`, `CONFIRM_STRICT`, closure-vs-absent).

It also evaluates the frozen non-regression table of Sec. 8 and refuses to
report a pass unless both the decision rule and the regression gates hold.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NODE = Path(__file__).resolve().parent
RES = ROOT / "codes" / "results" / "e2_closure_composition_results.json"
PRIOR = ROOT / "codes" / "results" / "e2_direct_traction_results.json"

REGION = "full_srcex"          # registered primary region
UNIT = "CONFIRM_STRICT"        # registered primary unit

# Frozen non-regression tolerances, PREREGISTRATION Sec. 8b.  Signs are
# "repaired value must be >= threshold".
NR = [
    ("NR1", "native_minus_absent", "full_support_excluded", 0.0500, +0.07415),
    ("NR2", "native_minus_absent", "near_support_excluded_d_le_0p5h", 0.0950, +0.14024),
    ("NR3", "native_minus_absent", "outer_d_gt_0p5h", -0.0020, +0.00649),
    ("NR4", "native_minus_absent", "uniq_raster_support_excluded", 0.0650, +0.09504),
]
NR_LEVELS = [
    ("NR5", "K:tau_native", "full_support_excluded", -0.0350, -0.01432),
    ("NR6", "K:absent", "full_support_excluded", -0.1150, -0.08847),
]
IDENTITY_TOL = 0.005           # Sec. 8a exact-identity gate on the frozen Family T


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    if not RES.exists():
        print("RESULT FILE ABSENT — decision rule not applicable")
        return 2
    res = json.loads(RES.read_text())
    U = res["units"][UNIT]
    d = U["deltas"]

    clauses, verdict = [], True

    def clause(n, text, ok, value):
        nonlocal verdict
        verdict = verdict and bool(ok)
        clauses.append({"clause": n, "registered": text, "value": value, "pass": bool(ok)})

    ca = d["closure_minus_absent"][REGION]
    clause(1, "Delta_closure > 0", ca["delta"] > 0, ca["delta"])
    lo, hi = ca["ci95_conservative_block"]
    clause(2, "conservative-block 95% interval excludes zero", lo > 0 or hi < 0, [lo, hi])
    clause(3, "same sign on all three training seeds",
           ca["all_seeds_same_sign"] and len(ca["per_seed"]) == 3, ca["per_seed"])
    xlo, xhi = ca["ci95_crossed_seed_block"]
    clause(4, "crossed seed x block 95% interval excludes zero",
           xlo > 0 or xhi < 0, [xlo, xhi])
    cf = d["closure_minus_fartime"][REGION]
    flo, fhi = cf["ci95_conservative_block"]
    clause(5, "Delta_closure > Delta_fartime with interval excluding zero",
           cf["delta"] > 0 and (flo > 0 or fhi < 0), {"delta": cf["delta"], "ci": [flo, fhi]})

    # ---- frozen non-regression table -------------------------------------
    A = res["units"]["CONFIRM_ALL"]
    reg, reg_ok = [], True
    for tag, key, region, thresh, old in NR:
        v = A["deltas"].get(key, {}).get(region, {}).get("delta")
        ok = v is not None and v >= thresh
        reg_ok = reg_ok and ok
        reg.append({"id": tag, "endpoint": f"{key}/{region}", "old_node005": old,
                    "repaired": v, "tolerance_min": thresh, "pass": bool(ok)})
    for tag, arm, region, thresh, old in NR_LEVELS:
        v = A["arms"].get(arm, {}).get(region, {}).get("R2_fluct_balanced")
        ok = v is not None and v >= thresh
        reg_ok = reg_ok and ok
        reg.append({"id": tag, "endpoint": f"{arm}/{region}", "old_node005": old,
                    "repaired": v, "tolerance_min": thresh, "pass": bool(ok)})
    # NR7/NR8 orderings
    nat = A["deltas"].get("native_minus_absent", {})
    far = A["deltas"].get("fartime_minus_absent", {})
    v7 = (nat.get("full_support_excluded", {}).get("delta"),
          far.get("full_support_excluded", {}).get("delta"))
    ok7 = None not in v7 and v7[0] > v7[1]
    reg_ok = reg_ok and ok7
    reg.append({"id": "NR7", "endpoint": "native > fartime (complete)",
                "repaired": v7, "pass": bool(ok7)})
    regions4 = [r for _, _, r, _, _ in NR]
    v8 = {r: nat.get(r, {}).get("delta") for r in regions4}
    ok8 = all(v is not None and v > 0 for v in v8.values())
    reg_ok = reg_ok and ok8
    reg.append({"id": "NR8", "endpoint": "native > absent in all four regions",
                "repaired": v8, "pass": bool(ok8)})

    # ---- Sec. 8a identity gate on the reused frozen Family T --------------
    ident, ident_ok = [], True
    if PRIOR.exists():
        prior = json.loads(PRIOR.read_text())["evaluation"]["arms"]
        for arm_new, arm_old in (("T:tau_native", "tau_native"), ("T:absent", "absent")):
            for region in regions4:
                new_v = A["arms"].get(arm_new, {}).get(region, {}).get("R2_fluct_balanced")
                per = prior.get(arm_old, {}).get(region, {}).get("per_seed")
                old_v = per[0] if per else None       # seed 7701, the re-evaluated seed
                if new_v is None or old_v is None:
                    continue
                ok = abs(new_v - old_v) <= IDENTITY_TOL
                ident_ok = ident_ok and ok
                ident.append({"arm": arm_new, "region": region, "node005_seed7701": old_v,
                              "reevaluated": new_v, "abs_diff": abs(new_v - old_v),
                              "tolerance": IDENTITY_TOL, "pass": bool(ok)})

    overall = "POSITIVE" if (verdict and reg_ok and ident_ok) else "NOT POSITIVE"
    out = {
        "registered_estimand": "R2(tau_closure) - R2(absent)",
        "registered_region": REGION,
        "registered_unit": UNIT,
        "n_eval": U["n_eval"], "n_effective": U["n_effective"],
        "decision_rule_clauses": clauses,
        "decision_rule_pass": bool(verdict),
        "non_regression_table": reg,
        "non_regression_pass": bool(reg_ok),
        "frozen_familyT_identity_gate": ident,
        "identity_gate_pass": bool(ident_ok),
        "REGISTERED_VERDICT": overall,
        "result_sha256": sha256(RES),
        "closure_apriori": res.get("closure_apriori"),
        "gpu_hours": res.get("provenance", {}).get("gpu_hours"),
    }
    (NODE / "DECISION_RULE_OUTCOME.json").write_text(json.dumps(out, indent=1))
    for c in clauses:
        print(f"  clause {c['clause']}: {'PASS' if c['pass'] else 'FAIL'}  {c['value']}")
    for r in reg:
        print(f"  {r['id']}: {'PASS' if r['pass'] else 'FAIL'}  {r['endpoint']} -> {r['repaired']}")
    for i in ident:
        print(f"  identity {i['arm']}/{i['region']}: {'PASS' if i['pass'] else 'FAIL'} "
              f"diff={i['abs_diff']:.6f}")
    print(f"REGISTERED VERDICT: {overall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
