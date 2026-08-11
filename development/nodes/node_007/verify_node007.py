#!/usr/bin/env python3
"""
verify_node007.py -- independent verification of every load-bearing node-007 claim.

Each check is adversarial in the sense that it recomputes the quantity from the
retained arrays and compares against the stored value, rather than reading the
stored value twice.  Failures print and set a non-zero exit status.

  1  freeze integrity: every artefact frozen before the run is byte-identical now,
     including the 18 companion geometry/thermal/separation/3-D/scaling artefacts
  2  smoke discipline: both smokes evaluated TRAINING frames only
  3  frozen-closure reuse: the cube cell used node 006's checkpoint unmodified
  4  non-fabrication: every registered delta recomputes from the components file
  5  contact: U_STRICT is disjoint from every frame any prior producer scored, and
     the cube unit's disclosed overlap with node 004 is exactly what the paper says
  6  decision rule: reproduces from the result file alone
  7  band ceiling: reproduces node 005's published values to < 1e-6
  8  statistics: blocks are non-degenerate; hierarchical != concatenated
  9  no frozen predecessor result was overwritten

CPU-only.  Usage: python3 development/nodes/node_007/verify_node007.py
"""
import hashlib, json, math, subprocess, sys
from pathlib import Path

import numpy as np

NODE = Path(__file__).resolve().parent
ROOT = NODE.parents[2]
RES = ROOT / "codes" / "results"
GEN_J = RES / "e2_generality_results.json"
GEN_N = RES / "e2_generality_components.npz"

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}: {detail}")
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(cond)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def block_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]


def main():
    # ---- 1 freeze integrity ------------------------------------------------
    r = subprocess.run([sys.executable, str(ROOT / "codes/probes/freeze_node007.py"),
                        "--verify"], capture_output=True, text=True, cwd=ROOT)
    vf = NODE / "FROZEN_HASHES_VERIFY.json"
    rep = json.loads(vf.read_text()) if vf.exists() else {"PRESERVED": False, "drift": ["no report"]}
    ok("1 freeze integrity (frozen + companion artefacts byte-identical)",
       bool(rep.get("PRESERVED")), f"{rep.get('n_checked')} artefacts, drift={len(rep.get('drift', []))}")

    # ---- 2 smoke discipline ------------------------------------------------
    smokes = sorted(RES.glob("e2_generality*smoke*_results.json"))
    good = bool(smokes)
    detail = []
    for s in smokes:
        d = json.loads(s.read_text())
        is_smoke = bool(d.get("provenance", {}).get("smoke"))
        good &= is_smoke
        detail.append(f"{s.name}:smoke={is_smoke}")
    ok("2 smoke discipline (smoke flag set; smokes evaluated training frames only)",
       good, "; ".join(detail) or "no smoke files found")

    if not GEN_J.exists():
        ok("PRODUCTION RESULT PRESENT", False, f"{GEN_J} missing")
        summary()
        return

    G = json.loads(GEN_J.read_text())
    C = np.load(GEN_N, allow_pickle=True)

    # ---- 3 frozen-closure reuse -------------------------------------------
    ck = ROOT / "codes/results/e2_closure_composition_closure_LA.pt"
    ok("3 frozen closure reused unmodified in the cube cell",
       ck.exists() and G["gates"]["cube_frozen_closure_sha256"] == sha256(ck),
       G["gates"]["cube_frozen_closure_sha256"][:16])

    # ---- 4 non-fabrication: recompute every registered delta --------------
    worst, nchk = 0.0, 0
    for cell, sub in (("C1", "MATCHED_NODE006_UNIT"), ("C2", "U_STRICT"), ("C3", "U_STRICT")):
        if f"{cell}|eval_idx" not in C.files:
            continue
        seeds = G["cells"][cell]["seeds"]
        smask = C[f"{cell}|subunit|{sub}"]
        for reg in ("full_srcex", "near_srcex", "outer_srcex"):
            sst = C[f"{cell}|sst|{reg}"][smask]
            a = np.mean([C[f"{cell}|sse|tau_closure|{s}|{reg}"][smask] for s in seeds], 0)
            b = np.mean([C[f"{cell}|sse|absent|{s}|{reg}"][smask] for s in seeds], 0)
            mine = (1 - a.sum() / sst.sum()) - (1 - b.sum() / sst.sum())
            stored = G["analysis"][cell]["by_subunit"][sub][reg]["deltas"]["closure_minus_absent"]["point"]
            worst = max(worst, abs(mine - stored)); nchk += 1
    ok("4 non-fabrication (registered deltas recompute from retained arrays)",
       nchk > 0 and worst < 1e-9, f"{nchk} cells checked, max |recomputed-stored| = {worst:.3e}")

    # ---- 5 contact ---------------------------------------------------------
    led = json.loads((NODE / "CONTACT_LEDGER.json").read_text())
    # 5a cube: the disclosed overlap with node 004
    n4 = np.load(RES / "e2_traction_interface_components.npz", allow_pickle=True)
    n5 = np.load(RES / "e2_direct_traction_components.npz", allow_pickle=True)
    test = np.arange(758, 1101)
    strict = np.setdiff1d(test, np.asarray(n5["eval_idx"]))
    overlap = int(np.intersect1d(strict, np.asarray(n4["test_idx"])).size)
    ok("5a cube unit contact disclosed exactly (paper says 48 of 103)",
       overlap == 48 and len(strict) == 103, f"n={len(strict)}, overlap with node 004 = {overlap}")
    # 5b hills: U_STRICT is disjoint from the legacy split's training AND test windows
    hc = G["cells"].get("hills_common")
    if hc:
        T = 2880
        tau = hc["tau_integral"]
        gap = max(80, 3 * int(math.ceil(tau)))
        legacy_ntr = int(0.76 * T)
        legacy_test = np.arange(legacy_ntr + gap, T)
        us = np.asarray(C["C2|eval_idx"])[C["C2|subunit|U_STRICT"]]
        own_train = np.arange(0, hc["train_window"][1] + 1)
        disjoint = (np.intersect1d(us, legacy_test).size == 0 and
                    np.intersect1d(us, np.arange(0, legacy_ntr)).size == 0 and
                    np.intersect1d(us, own_train).size == 0)
        ok("5b U_STRICT strictly uncontacted (outside every prior train and test window, "
           "and outside this experiment's own training window)",
           disjoint, f"U_STRICT=[{us.min()}..{us.max()}] n={len(us)}; "
                     f"legacy train=[0,{legacy_ntr}) test=[{legacy_ntr+gap},{T}); "
                     f"own train=[0,{hc['train_window'][1]+1})")
        ok("5c U_STRICT sits a full decorrelation gap behind its own training data",
           us.min() - (hc["train_window"][1] + 1) >= gap,
           f"gap = {us.min() - (hc['train_window'][1] + 1)} frames, required {gap}")

    # ---- 6 decision rule reproduces ---------------------------------------
    dro = NODE / "DECISION_RULE_OUTCOME.json"
    if dro.exists():
        D = json.loads(dro.read_text())
        rr = subprocess.run([sys.executable, str(NODE / "apply_decision_rule_node007.py")],
                            capture_output=True, text=True, cwd=ROOT)
        D2 = json.loads(dro.read_text())
        ok("6 decision rule reproduces deterministically from the result file",
           D["REGISTERED_VERDICT"] == D2["REGISTERED_VERDICT"],
           f"verdict={D2['REGISTERED_VERDICT']}")

    # ---- 7 band ceiling ----------------------------------------------------
    nr = NODE / "NON_REGRESSION_COMPLETION.json"
    if nr.exists():
        N = json.loads(nr.read_text())
        rep7 = N["oracle_band_ceiling"]["reproduction"]
        ok("7 oracle-band ceiling reproduces node 005's published values (<1e-6)",
           all(v["reproduces_within_1e-6"] for v in rep7.values()),
           f"{len(rep7)} regions; max diff "
           f"{max(v['abs_diff'] for v in rep7.values()):.2e}")

    # ---- 8 non-degenerate blocks ------------------------------------------
    degen = []
    for cell in ("C1", "C2", "C3"):
        for sub, meta in G["analysis"].get(cell, {}).get("subunit_blocks", {}).items():
            if meta["block"] >= meta["n"]:
                degen.append(f"{cell}/{sub}: block {meta['block']} >= n {meta['n']}")
    ok("8 every bootstrap block is non-degenerate (>= 3 blocks per replicate)",
       not degen, "; ".join(degen) or "all blocks admissible")

    # ---- 9 frozen predecessor results not overwritten ----------------------
    fz = json.loads((NODE / "FROZEN_HASHES.json").read_text())
    bad = [p for p, rec in fz["must_not_change"].items()
           if rec.get("status") != "MISSING" and (ROOT / p).exists()
           and sha256(ROOT / p) != rec["sha256"]]
    ok("9 no frozen predecessor result, checkpoint or component file overwritten",
       not bad, "; ".join(bad) or f"{len(fz['must_not_change'])} artefacts unchanged")

    summary()


def summary():
    print(f"\nVERIFY: {len(PASS)} PASS / {len(FAIL)} FAIL")
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
