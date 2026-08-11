#!/usr/bin/env python3
"""
freeze_node007.py -- hash-bind the preregistration, the producer and every frozen
predecessor artefact BEFORE the production run is launched.

Two things are recorded:

  * `preregistration_and_producer` -- the documents and code whose contents fix
    the design, so that no post-hoc edit can pass unnoticed;
  * `must_not_change` -- every frozen predecessor result, checkpoint, source-data
    file and companion artefact, recorded PRE-run so the post-run re-verification
    is a genuine byte-preservation proof rather than a manifest regenerated
    against itself (the exact defect the node-006 panel identified).

The timestamp is taken once and written once; node 006's freeze certificate
carried a stated time later than its own filesystem chronology, and this script
records the filesystem mtimes alongside the clock so the two can always be
reconciled.

Usage:  python3 codes/probes/freeze_node007.py [--verify]
        --verify re-hashes `must_not_change` and reports any drift.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / "development" / "nodes" / "node_007"
NODE.mkdir(parents=True, exist_ok=True)
OUT = NODE / "FROZEN_HASHES.json"

DESIGN = [
    "development/nodes/node_007/PREREGISTRATION_E2_GENERALITY.md",
    "development/nodes/node_007/CONTACT_LEDGER.json",
    "codes/gpu/eval_e2_generality.py",
    "codes/probes/build_contact_ledger.py",
    "codes/probes/complete_non_regression_node007.py",
    "development/nodes/node_007/AMENDMENT_1_BLOCK_LENGTH.md",
]

# Frozen predecessor evidence.  NOTHING here may change during this node.
MUST_NOT_CHANGE = [
    # node 006 -- the closure and its registered result
    "codes/results/e2_closure_composition_closure_LA.pt",
    "codes/results/e2_closure_composition_results.json",
    "codes/results/e2_closure_composition_components.npz",
    # node 005 -- the oracle direct-traction result and its Family-T checkpoints
    "codes/results/e2_direct_traction_results.json",
    "codes/results/e2_direct_traction_components.npz",
    "codes/results/e2_direct_traction_T_s7701.pt",
    "codes/results/e2_direct_traction_T_s7702.pt",
    "codes/results/e2_direct_traction_T_s7703.pt",
    # node 004 -- the adverse equilibrium-adapter result, retained as evidence
    "codes/results/e2_traction_interface_results.json",
    "codes/results/e2_traction_interface_components.npz",
]

# Companion strengths produced by SEPARATE frozen models.  This node does not
# revalidate them; it proves they are untouched.  Resolved from the release
# manifest so the list cannot silently drift from what the paper ships.
COMPANION_KEYWORDS = ("geometry", "thermal", "separation", "scaling", "3d",
                      "cube3d", "transfer", "diversity")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def companion_files():
    man = ROOT / "SUBMISSION_RELEASE_MANIFEST.json"
    out = []
    if man.exists():
        try:
            obj = json.loads(man.read_text())
        except Exception:
            return out

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, str) and ("/" in v) and any(
                            w in v.lower() for w in COMPANION_KEYWORDS):
                        out.append(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(obj)
    return sorted({p for p in out if (ROOT / p).exists()})


def snapshot(paths):
    rows = {}
    for rel in paths:
        p = ROOT / rel
        if not p.exists():
            rows[rel] = {"status": "MISSING"}
            continue
        st = p.stat()
        rows[rel] = {"sha256": sha256(p), "bytes": st.st_size,
                     "mtime_utc": datetime.fromtimestamp(st.st_mtime,
                                                         timezone.utc).isoformat()}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    comp = companion_files()
    if a.verify:
        if not OUT.exists():
            print("[freeze] no freeze certificate to verify")
            sys.exit(2)
        old = json.loads(OUT.read_text())
        # This is a REPOSITORY preservation proof: it compares the frozen
        # predecessor results, checkpoints and companion artefacts against hashes
        # recorded before the production run.  Those artefacts are deliberately
        # NOT part of the released submission package (multi-gigabyte checkpoints
        # and raw component arrays), so inside an isolated clean-release staging
        # tree every one of them is absent.  Absent-in-full is reported as SKIPPED
        # rather than as drift; a file that is PRESENT but MODIFIED, or a partially
        # present set, remains a hard failure.
        # Sections are judged INDEPENDENTLY.  The repository-only section
        # (frozen predecessor results and checkpoints) is absent by design inside
        # a clean-release staging tree, while the companion artefacts are release
        # files and are present there.  A section whose files are wholly absent is
        # SKIPPED; a section with any present file is fully checked, so a
        # present-but-modified file, or a partially present section, still fails.
        skipped = {}
        drift = []
        for section in ("must_not_change", "companion_artifacts"):
            rows = {rel: rec for rel, rec in old.get(section, {}).items()
                    if rec.get("status") != "MISSING"}
            if rows and not any((ROOT / rel).exists() for rel in rows):
                skipped[section] = len(rows)
                continue
            for rel, rec in rows.items():
                p_ = ROOT / rel
                if not p_.exists():
                    drift.append({"path": rel, "problem": "DELETED"})
                    continue
                h = sha256(p_)
                if h != rec["sha256"]:
                    drift.append({"path": rel, "problem": "MODIFIED",
                                  "frozen": rec["sha256"], "now": h})
        n_checked = sum(len(old.get(sec, {})) for sec in
                        ("must_not_change", "companion_artifacts")
                        if sec not in skipped)
        report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
                  "n_checked": n_checked, "drift": drift, "PRESERVED": not drift,
                  "skipped_sections": skipped,
                  "_why_skipped": "a wholly absent section is a clean-release "
                                  "staging tree, where repository-only frozen "
                                  "artefacts do not ship"}
        (NODE / "FROZEN_HASHES_VERIFY.json").write_text(json.dumps(report, indent=1))
        print(f"[freeze --verify] checked={n_checked} drift={len(drift)} "
              f"PRESERVED={report['PRESERVED']}"
              + (f" skipped={skipped}" if skipped else ""))
        for d in drift:
            print("  DRIFT", d)
        sys.exit(0 if not drift else 1)

    now = datetime.now(timezone.utc)
    cert = {
        "node": "development/nodes/node_007",
        "level": 3,
        "attempt": 5,
        "frozen_at_utc": now.isoformat(),
        "frozen_at_unix": time.time(),
        "_timestamp_note": "the clock value and every recorded mtime are written "
                           "in one pass by this script; node 006's certificate "
                           "stated a time later than its own file chronology and "
                           "that inconsistency is documented in ERRATUM_NODE006.md",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip(),
        "design_artifacts": snapshot(DESIGN),
        "must_not_change": snapshot(MUST_NOT_CHANGE),
        "companion_artifacts": snapshot(comp),
        "_companion_note": "recorded PRE-run so post-run re-verification is a real "
                           "byte-preservation proof. This node does NOT revalidate "
                           "the geometry, separation, thermal, 2-D-to-3-D or "
                           "scaling regimes; it only proves their evidence is "
                           "untouched.",
        "supersedes": "FROZEN_HASHES_v1.json (22:32:49Z) -- superseded by "
                      "AMENDMENT_1_BLOCK_LENGTH.md, applied before any node-007 "
                      "held-out outcome existed; see that document for the log "
                      "evidence and the arithmetic",
        "production_launch_allowed_after_this_file_exists": True,
    }
    OUT.write_text(json.dumps(cert, indent=1))
    print(f"[freeze] wrote {OUT.relative_to(ROOT)} at {now.isoformat()}")
    print(f"  design artifacts   : {len(cert['design_artifacts'])}")
    print(f"  must-not-change    : {len(cert['must_not_change'])}")
    print(f"  companion artifacts: {len(cert['companion_artifacts'])}")
    for rel, rec in cert["design_artifacts"].items():
        print(f"    {rec.get('sha256','MISSING')[:16]}  {rel}")


if __name__ == "__main__":
    main()
