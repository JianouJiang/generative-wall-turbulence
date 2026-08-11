#!/usr/bin/env python3
"""Harvest gate for a *future* contract-faithful level-2 closure reconstruction.

The preserved six-frame CPU artifact is deliberately expected to fail this gate.  Passing
requires causal coarse pressure/history inputs, a target-independent family branch and an
adequately powered CUDA posterior run; absence of a target DNS ensemble mean is necessary but
not sufficient.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
path = os.path.join(ROOT, "results", "runtime_closure_conditioning_results.json")
checks = []
def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

chk("result_exists", os.path.exists(path), path)
if not os.path.exists(path):
    print("ALL_PASS: NO"); sys.exit(1)
d = json.load(open(path)); m = d.get("_meta", {}); gates = d.get("_gates", {})
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
print(f"[info] sha256={digest}")
chk("evidence_level_2", m.get("evidence_level") == 2, m.get("claim", ""))
chk("not_solver_coupled", "NOT solver-coupled" in m.get("claim", "") and gates.get("solver_coupled_claim") is False)
chk("runtime_input_no_target_mean", gates.get("runtime_features_no_target_mean") is True and
    "no target DNS time/ensemble mean" in m.get("runtime_input", ""), m.get("runtime_input", ""))
chk("causal_pressure_history_contract",
    gates.get("native_pressure_or_verified_unsteady_balance") is True
    and gates.get("temporal_estimand_registered") is True,
    "single-snapshot steady-Euler proxy is inadmissible")
chk("target_independent_family_branch",
    gates.get("target_independent_family_branch") is True,
    "frozen target-member X2 embedding is inadmissible")
chk("not_smoke_artifact", m.get("smoke_only") is False,
    f"smoke_only={m.get('smoke_only', 'unmarked')}")
chk("no_extraction_failures", gates.get("extract_failures") == 0)
chk("cuda_full_test", m.get("dev") == "cuda" and int(m.get("n_test", 0)) >= 800,
    f"dev={m.get('dev')} n_test={m.get('n_test')}")
chk("effective_sample_size", float(m.get("n_eff", 0.0)) >= 20.0,
    f"n_eff={m.get('n_eff')}")
chk("adequate_ensemble", int(m.get("L", 0)) >= 8 and int(m.get("steps", 0)) >= 32,
    f"L={m.get('L')} steps={m.get('steps')}")
fams = d.get("families", {})
chk("both_families_complete", set(fams) == {"diffusion", "flow_matching"} and gates.get("both_families_complete") is True)
expected = {"correct", "passthrough", "closure", "eqwm", "wrong", "random", "no_wall"}
for fam, rec in fams.items():
    arms = rec.get("arms", {})
    chk(f"{fam}_all_arms", set(arms) == expected, ",".join(arms))
    chk(f"{fam}_all_finite", rec.get("gates", {}).get("all_samples_finite") is True and
        all(a.get("finite_fraction") == 1.0 for a in arms.values()))
    for estimand in ("bootstrap_total", "bootstrap_fluct"):
        b = rec[estimand]
        for base in ("passthrough", "eqwm", "wrong", "random", "no_wall"):
            q = b[f"closure_minus_{base}"]
            chk(f"{fam}_{estimand}_closure_minus_{base}_gate_consistent",
                q["ci_positive"] == (q["ci95"][0] > 0),
                f"delta={q['mean']:+.4f} CI={q['ci95']}")
    bt = rec["bootstrap_total"]
    chk(f"{fam}_stored_no_wall_gate_consistent",
        rec["gates"]["closure_beats_no_wall_total"] == bt["closure_minus_no_wall"]["ci_positive"])
    chk(f"{fam}_stored_eqwm_gate_consistent",
        rec["gates"]["closure_beats_eqwm_total"] == bt["closure_minus_eqwm"]["ci_positive"])
    print(f"[outcome:{fam}] closure-no_wall={bt['closure_minus_no_wall']['mean']:+.4f} "
          f"CI={bt['closure_minus_no_wall']['ci95']} closure-EQWM={bt['closure_minus_eqwm']['mean']:+.4f} "
          f"CI={bt['closure_minus_eqwm']['ci95']}")

wall = d.get("wall_models_alone", {})
chk("wall_models_alone_present", set(wall) == {"closure", "eqwm", "passthrough"})
for name in ("closure", "eqwm"):
    q = wall.get(name, {})
    chk(f"{name}_wall_stress_finite", q.get("finite_fraction") == 1.0,
        f"corr_inst={q.get('corr_tau_instantaneous')} relRMSE={q.get('relRMSE_tau')}")
    chk(f"{name}_loads_and_separation", "load_relative_error" in q and "separation_pred" in q)

nfail = sum(not ok for _, ok, _ in checks)
print(f"checks: {len(checks)-nfail}/{len(checks)} PASS; {nfail} FAIL")
print("ALL_PASS: YES" if nfail == 0 else "ALL_PASS: NO")
sys.exit(0 if nfail == 0 else 1)
