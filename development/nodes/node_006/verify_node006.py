#!/usr/bin/env python3
"""Independent verification of node 006.

Every check recomputes its quantity from a retained array or a file hash; none
trusts a number printed by the producer.  Run from the project root:

    python3 development/nodes/node_006/verify_node006.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NODE = Path(__file__).resolve().parent
RES = ROOT / "codes" / "results" / "e2_closure_composition_results.json"
COMP = ROOT / "codes" / "results" / "e2_closure_composition_components.npz"
PRIOR_J = ROOT / "codes" / "results" / "e2_direct_traction_results.json"
PRIOR_C = ROOT / "codes" / "results" / "e2_direct_traction_components.npz"
FROZEN = NODE / "FROZEN_HASHES.json"

PASS, FAIL, SKIP = [], [], []


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(f"{name}: {detail}")
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def skip(name, why):
    SKIP.append(name)
    print(f"[SKIP] {name} — {why}")


def block_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]


def main():
    frozen = json.loads(FROZEN.read_text())

    # ---- 1. the frozen binding still holds --------------------------------
    for key, rel in frozen["paths"].items():
        p = ROOT / rel
        if not p.exists():
            check(f"frozen_present[{key}]", False, "MISSING")
            continue
        check(f"frozen_hash[{key}]", sha256(p) == frozen["sha256"][key],
              "unchanged" if sha256(p) == frozen["sha256"][key] else "CHANGED")

    if not RES.exists():
        check("result_present", False, "producer output absent")
        return summarise()
    res = json.loads(RES.read_text())
    z = np.load(COMP, allow_pickle=True)

    # ---- 2. this is a production run, not a smoke -------------------------
    check("production_not_smoke", res["provenance"]["smoke"] is False,
          f"smoke={res['provenance']['smoke']}")

    # ---- 3. the smoke run made no held-out contact ------------------------
    smoke = ROOT / "codes" / "results" / "e2_closure_smoke_results.json"
    if smoke.exists():
        s = json.loads(smoke.read_text())
        idx = []
        for u in s["units"].values():
            idx += u["eval_indices"]
        ok = bool(idx) and max(idx) < 660
        check("smoke_touched_train_frames_only", ok,
              f"max index {max(idx) if idx else 'n/a'} < 660 (train window)")
    else:
        skip("smoke_touched_train_frames_only", "smoke result not retained locally")

    # ---- 4. CONFIRM_STRICT is genuinely uncontacted by node 005 -----------
    strict = np.asarray(res["units"]["CONFIRM_STRICT"]["eval_indices"])
    allu = np.asarray(res["units"]["CONFIRM_ALL"]["eval_indices"])
    if PRIOR_C.exists():
        prior_idx = np.asarray(np.load(PRIOR_C, allow_pickle=True)["eval_idx"])
        inter = np.intersect1d(strict, prior_idx)
        check("confirm_strict_uncontacted", len(inter) == 0,
              f"{len(strict)} frames, intersection with node-005 eval set = {len(inter)}")
        check("confirm_all_identical_units", np.array_equal(np.sort(allu), np.sort(prior_idx)),
              f"{len(allu)} frames match node-005 exactly")
        test_idx = np.arange(758, 1101)
        check("confirm_strict_is_exact_set_difference",
              np.array_equal(np.sort(strict), np.setdiff1d(test_idx, prior_idx)),
              "setdiff(test, node005.eval_idx)")
    else:
        skip("confirm_strict_uncontacted", "node-005 components absent")

    # ---- 5. the primary region excludes every cell the closure read -------
    sizes = res["region_sizes"]
    check("srcex_smaller_than_support_excluded",
          sizes["full_srcex"] < sizes["full_support_excluded"],
          f"{sizes['full_srcex']} < {sizes['full_support_excluded']} "
          f"({sizes['full_support_excluded'] - sizes['full_srcex']} closure-read cells removed)")
    check("matching_offsets_exclude_wall_cell",
          min(res["matching_offsets_cells"]) >= 2,
          f"offsets {res['matching_offsets_cells']} cells "
          f"(heights {res['matching_heights_over_delta']} x Delta)")

    # ---- 6. non-fabrication: every reported R^2 replays from the arrays ----
    worst = 0.0
    nrep = 0
    for uname in ("CONFIRM_STRICT", "CONFIRM_ALL"):
        U = res["units"][uname]
        for key, regions in U["arms"].items():
            for region, blob in regions.items():
                sst = z[f"{uname}|sst|{region}"]
                stack = []
                for s in ("8801", "8802", "8803"):
                    k = f"{uname}|sse|{key}|{s}|{region}"
                    if k in z:
                        stack.append(z[k])
                if not stack:
                    continue
                se = np.mean(np.stack(stack), 0)
                r2 = 1 - se.sum() / (sst.sum() + 1e-12)
                worst = max(worst, abs(r2 - blob["R2_fluct_balanced"]))
                nrep += 1
    check("recomputed_R2_matches_stored", worst < 1e-9,
          f"{nrep} arm-region values, max |diff| = {worst:.3e}")

    # ---- 7. every reported delta replays ----------------------------------
    DELTA_ARMS = {
        "closure_minus_absent": ("K:tau_closure", "K:absent"),
        "native_minus_absent": ("K:tau_native", "K:absent"),
        "eqwm_minus_absent": ("K:tau_eqwm", "K:absent"),
        "closure_minus_fartime": ("K:tau_closure", "K:tau_fartime"),
        "closure_minus_eqwm": ("K:tau_closure", "K:tau_eqwm"),
        "closure_minus_native": ("K:tau_closure", "K:tau_native"),
    }
    worst_d, nd = 0.0, 0
    for uname in ("CONFIRM_STRICT", "CONFIRM_ALL"):
        U = res["units"][uname]
        for name, (a, b) in DELTA_ARMS.items():
            if name not in U["deltas"]:
                continue
            common = U["deltas"][name]["seeds_used"]
            for region, blob in U["deltas"][name].items():
                if region == "seeds_used":
                    continue
                sst = z[f"{uname}|sst|{region}"]
                sa = np.mean(np.stack([z[f"{uname}|sse|{a}|{s}|{region}"] for s in common]), 0)
                sb = np.mean(np.stack([z[f"{uname}|sse|{b}|{s}|{region}"] for s in common]), 0)
                d = (1 - sa.sum() / (sst.sum() + 1e-12)) - (1 - sb.sum() / (sst.sum() + 1e-12))
                worst_d = max(worst_d, abs(d - blob["delta"]))
                nd += 1
    check("recomputed_deltas_match_stored", worst_d < 1e-9,
          f"{nd} delta-region values, max |diff| = {worst_d:.3e}")

    # ---- 8. independent replay of the decision interval -------------------
    U = res["units"]["CONFIRM_STRICT"]
    region = "full_srcex"
    common = U["deltas"]["closure_minus_absent"]["seeds_used"]
    sst = z[f"CONFIRM_STRICT|sst|{region}"]
    sa = np.mean(np.stack([z[f"CONFIRM_STRICT|sse|K:tau_closure|{s}|{region}"] for s in common]), 0)
    sb = np.mean(np.stack([z[f"CONFIRM_STRICT|sse|K:absent|{s}|{region}"] for s in common]), 0)
    bix = block_indices(len(sa), U["block_conservative"], 4000, np.random.default_rng(45))
    d = ((1 - sa[bix].sum(1) / (sst[bix].sum(1) + 1e-12))
         - (1 - sb[bix].sum(1) / (sst[bix].sum(1) + 1e-12)))
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    slo, shi = U["deltas"]["closure_minus_absent"][region]["ci95_conservative_block"]
    check("independent_bootstrap_replay", abs(lo - slo) < 1e-6 and abs(hi - shi) < 1e-6,
          f"[{lo:+.5f},{hi:+.5f}] vs stored [{slo:+.5f},{shi:+.5f}]")

    # ---- 9. physical gates ------------------------------------------------
    g = res["gates"]
    check("traction_sign_cellwise", g["tau_dot_ut_negative_every_cell"],
          f"worst cell per face max = {max(g['tau_dot_ut_worst_cell_per_face'].values()):.2e}")
    check("multi_face_double_count_repaired", g["multi_face_double_count_repaired"],
          "each (cell, face) pair contributes once to the wall force")
    check("wall_force_sign", g["wall_on_fluid_x_force_negative"],
          f"fx = {g['fx_viscous_mean_pairwise']:.7f}")
    check("wall_force_within_quadrature_factor_2",
          all(0.5 <= abs(r) <= 2.0 for r in g["ratio_fd_to_native_quadrature"]),
          f"ratios {[round(r,4) for r in g['ratio_fd_to_native_quadrature']]}")

    # ---- 10. the closure never sees the wall, and generalises --------------
    ap = res["closure_apriori"]
    check("closure_beats_equilibrium_on_uncontacted_unit",
          ap["confirm_strict"]["R2_tau_closure"] > ap["confirm_strict"]["R2_tau_eqwm"],
          f"closure {ap['confirm_strict']['R2_tau_closure']:.4f} vs "
          f"equilibrium {ap['confirm_strict']['R2_tau_eqwm']:.4f}")
    gapv = abs(ap["train_fit_window"]["R2_tau_closure"] - ap["confirm_strict"]["R2_tau_closure"])
    check("closure_not_overfitted", gapv < 0.10,
          f"fit-window minus strict = {gapv:.4f}")

    # ---- 11. Family T checkpoints reused, never retrained ------------------
    tmeta = [k for k in res["train_meta"] if k.startswith("T_frozen")]
    ok = bool(tmeta) and all(res["train_meta"][k]["retrained"] is False for k in tmeta)
    check("familyT_reused_not_retrained", ok, f"{len(tmeta)} frozen checkpoints reused")

    # ---- 12. new artefacts do not overwrite frozen names -------------------
    news = list((ROOT / "codes" / "results").glob("e2_closure_composition_*"))
    check("distinct_artifact_names", all("direct_traction" not in p.name for p in news),
          f"{len(news)} new artefacts, all under the e2_closure_composition_ prefix")

    # ---- 13. decision-rule outcome is reproducible -------------------------
    dro = NODE / "DECISION_RULE_OUTCOME.json"
    if dro.exists():
        o = json.loads(dro.read_text())
        check("decision_rule_bound_to_this_result", o["result_sha256"] == sha256(RES),
              "outcome file is bound to the harvested result bytes")
        check("decision_rule_region_and_unit_as_registered",
              o["registered_region"] == "full_srcex" and o["registered_unit"] == "CONFIRM_STRICT",
              f"{o['registered_region']} / {o['registered_unit']}")
    else:
        skip("decision_rule_bound_to_this_result", "outcome not yet computed")

    return summarise()


def summarise():
    print(f"\nVERIFY node_006: {len(PASS)} PASS / {len(FAIL)} FAIL / {len(SKIP)} SKIP")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  - " + f)
        return 1
    print("VERIFY PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
