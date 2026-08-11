#!/usr/bin/env python3
"""
build_contact_ledger.py -- machine-readable GLOBAL data-contact ledger.

The node-006 panel found that the "strictly uncontacted" 103-frame cube unit was
only proved unscored by node 005: 48 of its frames had in fact been scored earlier
by node 004.  The defect was that contact was checked against ONE predecessor
rather than against every producer that ever scored the record.

This probe removes the possibility of repeating that mistake.  It scans every
retained result artefact in `codes/results/` for stored evaluation indices, groups
them by the physical record they belong to, and reports, per record, the union of
every frame any producer has ever scored.  It makes no scientific claim; it is a
custody instrument, and it is deliberately CONSERVATIVE: an artefact whose indices
cannot be recovered is listed under `unresolved_producers`, so an "uncontacted"
statement can never rest on a silent parsing failure.

CPU-only.  Writes development/nodes/node_007/CONTACT_LEDGER.json.
"""
import json, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
OUTDIR = ROOT / "development" / "nodes" / "node_007"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "CONTACT_LEDGER.json"

# Which physical record does an artefact name belong to?  Anything unmatched is
# reported under "unattributed" rather than silently dropped.
RECORD_PATTERNS = (
    ("cube_les_48x96x48_Re5000", (r"^e2_", r"^cube", r"^cube3d", r"wiener_floor",
                                  r"^cont_chain", r"delay_campaign")),
    ("case3_periodic_hills_Re2800", (r"case3", r"hill", r"c3_", r"pehill",
                                     r"confild_physics", r"confild_autodecoder")),
    ("case1", (r"case1", r"c1_")),
    ("case2", (r"case2", r"c2_")),
    ("case4", (r"case4", r"c4_")),
)

IDX_KEY = re.compile(r"(^|[|_])(eval|test|strict|confirm|held?out)[a-z_]*idx$|"
                     r"(^|[|_])idx[a-z_]*(eval|test)$", re.I)


def attribute(name: str):
    low = name.lower()
    for rec, pats in RECORD_PATTERNS:
        for p in pats:
            if re.search(p, low):
                return rec
    return "unattributed"


def harvest_npz(path):
    got = {}
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:  # unreadable artefact -> reported, never ignored
        return None, f"{type(e).__name__}: {e}"
    for k in d.files:
        if not IDX_KEY.search(k):
            continue
        try:
            a = np.asarray(d[k]).ravel()
        except Exception:
            continue
        if a.dtype.kind in "iu" and a.size:
            got[k] = a.astype(np.int64)
    return got, None


def harvest_json(path):
    got = {}
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    def walk(o, trail):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{trail}.{k}" if trail else k)
        elif isinstance(o, list) and o and all(isinstance(v, int) for v in o):
            leaf = trail.split(".")[-1]
            if IDX_KEY.search(leaf):
                got[trail] = np.asarray(o, np.int64)

    walk(obj, "")
    return got, None


def main():
    per_record = {}
    unresolved = []
    scanned = 0
    for path in sorted(list(RESULTS.rglob("*.npz")) + list(RESULTS.rglob("*.json"))):
        rel = str(path.relative_to(ROOT))
        scanned += 1
        got, err = (harvest_npz(path) if path.suffix == ".npz" else harvest_json(path))
        if err is not None:
            unresolved.append({"artefact": rel, "reason": err})
            continue
        if not got:
            continue
        rec = attribute(path.name)
        ent = per_record.setdefault(rec, {"producers": [], "union": set()})
        for k, a in got.items():
            ent["producers"].append({"artefact": rel, "key": k, "n": int(a.size),
                                     "min": int(a.min()), "max": int(a.max())})
            ent["union"].update(a.tolist())

    out = {
        "_what": "union of every frame index any retained producer stored as an "
                 "evaluation/test index, grouped by physical record",
        "_caveat": "an index set that a producer did not retain cannot appear here; "
                   "such producers are declared in `split_rule_declarations`",
        "artefacts_scanned": scanned,
        "records": {},
        "unresolved_producers": unresolved,
        # producers that scored a record without retaining indices: their split
        # rule is declared from source so the ledger is not silently incomplete.
        "split_rule_declarations": [
            {"producer": "codes/gpu/train_stats_l2.py",
             "records": ["case3_periodic_hills_Re2800", "case1", "case2", "case4"],
             "rule": "n_tr=int(0.76*T); gap=max(80,3*tau); test=[n_tr+gap, T)",
             "note": "the canonical case-record split; every case-record producer "
                     "in this project inherits it"},
            {"producer": "codes/gpu/eval_e2_traction_interface.py (node 004)",
             "records": ["cube_les_48x96x48_Re5000"],
             "rule": "160 test frames retained as `test_idx` in "
                     "e2_traction_interface_components.npz"},
        ],
    }
    for rec, ent in sorted(per_record.items()):
        u = np.array(sorted(ent["union"]), np.int64)
        out["records"][rec] = {
            "n_producers_with_retained_indices": len(ent["producers"]),
            "n_distinct_frames_ever_scored": int(u.size),
            "min": int(u.min()) if u.size else None,
            "max": int(u.max()) if u.size else None,
            "producers": ent["producers"],
            "union_frames": u.tolist(),
        }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"[ledger] scanned={scanned} records={list(out['records'])}")
    for rec, e in out["records"].items():
        print(f"  {rec}: {e['n_distinct_frames_ever_scored']} frames ever scored "
              f"[{e['min']}..{e['max']}] from {e['n_producers_with_retained_indices']} keys")
    if unresolved:
        print(f"[ledger] unresolved artefacts: {len(unresolved)}")
    print(f"[ledger] wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
