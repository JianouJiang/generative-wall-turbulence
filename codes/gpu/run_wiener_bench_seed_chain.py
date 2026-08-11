#!/usr/bin/env python3
"""Sequential seed replication of the frozen-record benchmark producer (Foshan lane).

Charter 2026-07-26 item 2: >=3 independent training initializations for every
load-bearing learned model. Seed 1234 is terminal (node_004 F1). This driver
re-runs the SAME producer `eval_cube_wiener_floor_bench.py` byte-unchanged for
additional seeds, sequentially, and renames the fixed-name result/component
files to seed-suffixed variants after each run (checkpoints are already
seed-suffixed by the producer). Training/eval configuration is byte-identical
across seeds; no result is inspected or selected by this driver.

Terminal marker: === WIENER_SEED_CHAIN_DONE ===
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "codes" / "results"
PRODUCER = HERE / "eval_cube_wiener_floor_bench.py"
TAG = "cube_wiener_floor_bench"

P = argparse.ArgumentParser()
P.add_argument("--seeds", type=int, nargs="+", default=[2234, 3234])
A = P.parse_args()

T0 = time.time()


def log(m):
    print(f"[seedchain {time.time()-T0:7.1f}s] {m}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


q = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if q.stdout.strip():
    print(f"FATAL: GPU busy with foreign job(s):\n{q.stdout}", flush=True)
    sys.exit(3)

runs = []
for seed in A.seeds:
    log(f"seed {seed}: launching producer (byte-identical config, --seed only)")
    rc = subprocess.run(
        [sys.executable, str(PRODUCER), "--seed", str(seed)], cwd=str(ROOT)
    ).returncode
    src_json = RESULTS / f"{TAG}_results.json"
    src_comp = RESULTS / f"{TAG}_components.npz"
    if rc != 0 or not src_json.exists():
        runs.append({"seed": seed, "rc": rc, "status": "FAILED"})
        log(f"seed {seed} FAILED rc={rc}; stopping chain")
        break
    dst_json = RESULTS / f"{TAG}_results_seed{seed}.json"
    dst_comp = RESULTS / f"{TAG}_components_seed{seed}.npz"
    src_json.rename(dst_json)
    if src_comp.exists():
        src_comp.rename(dst_comp)
    runs.append(
        {
            "seed": seed,
            "rc": rc,
            "status": "TERMINAL",
            "results": dst_json.name,
            "results_sha256": sha256(dst_json),
            "components": dst_comp.name if dst_comp.exists() else None,
            "components_sha256": sha256(dst_comp) if dst_comp.exists() else None,
            "ckpt_rf": f"{TAG}_rf_seed{seed}.pt",
            "ckpt_det": f"{TAG}_det_seed{seed}.pt",
        }
    )
    log(f"seed {seed} TERMINAL -> {dst_json.name}")

manifest = {
    "driver": Path(__file__).name,
    "producer_sha256": sha256(PRODUCER),
    "seeds_requested": A.seeds,
    "runs": runs,
    "wall_seconds": round(time.time() - T0, 1),
}
out = RESULTS / "wiener_seed_chain_node005.json"
out.write_text(json.dumps(manifest, indent=1))
log(f"wrote {out}")
print("=== WIENER_SEED_CHAIN_DONE ===", flush=True)
