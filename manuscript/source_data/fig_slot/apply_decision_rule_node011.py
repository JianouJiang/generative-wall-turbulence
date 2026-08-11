#!/usr/bin/env python3
"""node_011 mechanical decision applicator.

Reads the FINAL-phase result JSON and the frozen calibration artifact, evaluates
the registered clauses C0-C9 and secondaries S1-S2 of
PREREGISTRATION_SLOT_INTERFACE.md exactly as written, and emits
DECISION_RULE_OUTCOME_NODE011.json.  No thresholds live anywhere else; the v2
amendment freezes this file's hash before the FINAL window is contacted.

Usage: python3 apply_decision_rule_node011.py [--results PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
NODE = pathlib.Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--results",
                default=str(ROOT / "codes/results/e2_slot_channel_final_results.json"))
ap.add_argument("--out", default=str(NODE / "DECISION_RULE_OUTCOME_NODE011.json"))
args = ap.parse_args()

R = json.loads(pathlib.Path(args.results).read_text())
CAL = json.loads((ROOT / "codes/results/block_inference_calibration.json").read_text())
T_STAR = float(CAL["t_star_frozen"])            # 4.5, simulation-calibrated

# v2-frozen parameters (must equal the values in the v2 amendment)
V2 = json.loads((NODE / "V2_FROZEN_PARAMS.json").read_text())
M5 = float(V2["c5_margin_m5"])
C9_BAND = V2["c9_band"]                          # "log" or "buffer"

FAM = "flow_matching"
REGION = "whole_scorable"
fid = R["fidelity_centred_r2"]
con = R["contrasts"]
dose = R["dose_response"]
arms = R["arms"]


def c(name):
    return con[f"{FAM}|{name}|{REGION}"]


def tpass(entry):
    p = entry["primary"]
    return bool(p["t"] >= T_STAR)


out = {"_frozen_inputs": {
    "results": args.results,
    "results_sha256": hashlib.sha256(
        pathlib.Path(args.results).read_bytes()).hexdigest(),
    "t_star": T_STAR, "m5": M5, "c9_band": C9_BAND,
    "v2_params": V2},
    "clauses": {}}

# C0 construction: support leak zero for every arm + wall-gradient consistency
sup = R["_meta"]["support_leak_check"]
wg = {k: v.get("wallgrad_consistency_relerr") for k, v in arms.items()
      if v.get("wallgrad_consistency_relerr") is not None}
out["clauses"]["C0"] = {
    "support_leak_max": max(sup.values()),
    "wallgrad_max_relerr": max(wg.values()),
    "pass": max(sup.values()) == 0.0 and max(wg.values()) < 1e-3}

# C1 power/ceiling gate
e = c("exact-absent")
out["clauses"]["C1"] = {"delta": e["delta"], "t": e["primary"]["t"],
                        "walls_agree": e["walls_agree_in_sign"],
                        "pass": tpass(e) and e["walls_agree_in_sign"]
                        and e["delta"] > 0}

# C2 primary usefulness
k = c("closure-absent")
out["clauses"]["C2"] = {
    "delta": k["delta"], "t": k["primary"]["t"],
    "block_means": k["primary"]["block_means"],
    "per_seed": k["per_seed_delta"],
    "delta_wall0": k.get("delta_wall0"), "delta_wall1": k.get("delta_wall1"),
    "pass": (tpass(k) and k["primary"]["all_blocks_positive"]
             and k.get("delta_wall0", 0) > 0 and k.get("delta_wall1", 0) > 0
             and all(v > 0 for v in k["per_seed_delta"]))}

# C3 absolute skill of the closure arm
clo_abs = [v[REGION] for kk, v in arms.items()
           if kk.startswith(f"{FAM}|closure|")]
out["clauses"]["C3"] = {"closure_absolute_by_seed": clo_abs,
                        "absent_absolute_by_seed": [
                            v[REGION] for kk, v in arms.items()
                            if kk.startswith(f"{FAM}|absent|")],
                        "pass": sum(clo_abs) / len(clo_abs) > 0}

# C4 ceiling ordering
ce = c("closure-exact")
out["clauses"]["C4"] = {"closure_minus_exact": ce["delta"],
                        "pass": ce["delta"] <= 0.01}

# C5 on-curve attribution (AMENDMENT A3, frozen at v2 before FINAL contact):
# the no-side-channel reference is the LARGER of the white-noise curve
# prediction and the structure-matched noise arm's gain -- rehearsal showed a
# measured error-structure premium (+0.012..0.019) that the white curve alone
# under-predicts for every correlated-error arm.
cp = dose["closure_point"]
matched = R["_meta"]["matched_arm"]["name"]
mn = con.get(f"{FAM}|{matched}-absent|{REGION}")
resid_white = cp["gain"] - cp["curve_predicted_gain"]
ref = max(cp["curve_predicted_gain"], mn["delta"]) if mn else cp["curve_predicted_gain"]
resid = cp["gain"] - ref
out["clauses"]["C5"] = {"closure_fidelity": cp["fidelity"],
                        "closure_gain": cp["gain"],
                        "curve_predicted": cp["curve_predicted_gain"],
                        "matched_noise_gain": mn["delta"] if mn else None,
                        "residual_above_white_curve": resid_white,
                        "residual_above_reference": resid, "margin_m5": M5,
                        "pass": resid <= M5,
                        "reported_learned_dataonly": dose["learned_dataonly_point"],
                        "reported_equilibrium": dose["equilibrium_point"]}

# C6 dose monotonicity
out["clauses"]["C6"] = {"spearman": dose["spearman_fidelity_gain"],
                        "curve": dose["curve"],
                        "pass": dose["spearman_fidelity_gain"] is not None
                        and dose["spearman_fidelity_gain"] >= 0.7}

# C7 wrong-information controls
wt = c("closure-wrong_time")
sz = c("closure-shuffle_z")
wta = c("wrong_time-absent")
out["clauses"]["C7"] = {"closure_minus_wrong_time": wt["delta"],
                        "t_wt": wt["primary"]["t"],
                        "closure_minus_shuffle_z": sz["delta"],
                        "t_sz": sz["primary"]["t"],
                        "wrong_time_minus_absent": wta["delta"],
                        "pass": tpass(wt) and wt["delta"] > 0 and tpass(sz)
                        and sz["delta"] > 0 and wta["delta"] <= 0.02}

# C8 beats the classical law (learned_dataonly reported, NOT gated)
eq = c("closure-equilibrium")
ld = c("closure-learned_dataonly")
out["clauses"]["C8"] = {"closure_minus_equilibrium": eq["delta"],
                        "t": eq["primary"]["t"],
                        "pass": tpass(eq) and eq["delta"] > 0,
                        "reported_closure_minus_learned_dataonly": ld["delta"],
                        "reported_t": ld["primary"]["t"]}

# C9 engineering endpoint (band frozen at v2); flux contrasts were sign-flipped
# in the producer so positive delta = closure improves (lower abs error)
fx = con[f"{FAM}|closure-absent|flux_{C9_BAND}"]
fxe = con[f"{FAM}|exact-absent|flux_{C9_BAND}"]
out["clauses"]["C9"] = {"band": C9_BAND,
                        "closure_improvement": fx["delta"],
                        "t": fx["primary"]["t"],
                        "exact_improvement": fxe["delta"],
                        "pass": tpass(fx) and fx["delta"] > 0
                        and fxe["delta"] > 0}

# S1 distributional (secondary, reported)
cr = con.get(f"{FAM}|closure-absent|crps")
out["clauses"]["S1"] = {"crps_improvement": cr["delta"] if cr else None,
                        "t": cr["primary"]["t"] if cr else None,
                        "directional_pass": bool(cr and cr["delta"] > 0)}

# S2 diffusion sign robustness (secondary, reported)
dca = con.get(f"diffusion|closure-absent|{REGION}")
out["clauses"]["S2"] = {"diffusion_closure_minus_absent":
                        dca["delta"] if dca else None,
                        "per_seed": dca["per_seed_delta"] if dca else None,
                        "directional_pass": bool(
                            dca and all(v > 0 for v in dca["per_seed_delta"]))}

CL = out["clauses"]
if not CL["C0"]["pass"]:
    label = "CONSTRUCTION_FAILURE"
elif not CL["C1"]["pass"]:
    label = "UNIT_BLIND"
elif not (CL["C4"]["pass"] and CL["C5"]["pass"]):
    label = "SIDE_CHANNEL_DETECTED"
elif not CL["C2"]["pass"]:
    label = "CLOSURE_INSUFFICIENT"
elif all(CL[f"C{i}"]["pass"] for i in range(10) if f"C{i}" in CL):
    label = "INTERFACE_TRANSMITS_FIDELITY"
else:
    label = "TRANSMITS_NO_ENGINEERING" if not CL["C9"]["pass"] else \
        "PARTIAL_" + "_".join(f"C{i}" for i in range(10)
                              if f"C{i}" in CL and not CL[f"C{i}"]["pass"])
out["registered_label"] = label
out["clause_pass_vector"] = {kk: v.get("pass") for kk, v in CL.items()}

pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
print(json.dumps(out["clause_pass_vector"], indent=1))
print("REGISTERED LABEL:", label)
