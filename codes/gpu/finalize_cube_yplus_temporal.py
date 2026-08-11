#!/usr/bin/env python3
"""Finalize native cube y+ from the already completed LES continuation.

nekRS wrote continuation checkpoints in the case root, while the launch
wrapper inspected the preserved ``restart/`` directory.  This postprocessing
repair launches no solver.  It selects the root-field pair whose time spacing
matches the preregistered 30-flow-time checkpoint interval, runs the frozen
native GLL-face derivative postprocessor on exactly that pair, and adds the
continuation provenance needed for independent adjudication.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pymech.neksuite import readnek

import eval_cube_native_yplus as native


CASE = Path(os.environ.get("GWT_CUBE_CASE", "controlled_raw/cube_les"))
RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "cube3d_native_yplus_temporally_separated.json"
LOG = CASE / "yplus_temporal_continuation.log"
HISTORICAL_SIGNAL_INTEGRAL_TIME = 97.59202614855501 * 0.3
CURRENT_CONSERVATIVE_INTEGRAL_TIME = 122.48173024910811 * 0.3
INTERVAL = 30.0
START_TIME = 370.0004874808


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    log_text = LOG.read_text(errors="replace")
    if "finished with exit code 0" not in log_text:
        raise RuntimeError("completed continuation lacks terminal nekRS marker")
    candidates = []
    for path in sorted(CASE.glob("cube0.f?????")):
        time = float(readnek(str(path)).time)
        if time > START_TIME + 1.0:
            candidates.append((time, path))
    pairs = []
    for i, (ta, pa) in enumerate(candidates):
        for tb, pb in candidates[i + 1:]:
            separation = tb - ta
            if separation >= 0.9 * HISTORICAL_SIGNAL_INTEGRAL_TIME:
                pairs.append((abs(separation - INTERVAL), -tb, ta, tb, pa, pb))
    if not pairs:
        raise RuntimeError(
            f"no pair meets the historical signal-specific spacing: {candidates}"
        )
    _, _, ta, tb, pa, pb = min(pairs)
    selected = [pa, pb]

    # The scientific native-face calculation remains byte-identical; only its
    # explicit input field list and output pathname are rebound.
    native.FIELDS = selected
    native.OUT = OUT
    native.main()
    result = json.loads(OUT.read_text())
    result["_meta"].update({
        "producer_wrapper": Path(__file__).name,
        "producer_wrapper_sha256": sha256(Path(__file__)),
        "failed_launcher_wrapper": "run_cube_yplus_temporal_continuation.py",
        "failed_launcher_wrapper_sha256": (
            "a0a78972f354dddae6ba247f417b0859a5ea1b5f46efee84624628f50f11f3a0"
        ),
        "execution_defect_repair": (
            "postprocess root checkpoints; no LES continuation or model rerun"
        ),
        "continuation_start_time": START_TIME,
        "checkpoint_interval": INTERVAL,
        "historical_signal_specific_integral_time": HISTORICAL_SIGNAL_INTEGRAL_TIME,
        "current_conservative_integral_time": CURRENT_CONSERVATIVE_INTEGRAL_TIME,
        "current_conservative_integral_time_native_snapshots": 122.48173024910811,
        "retained_time_separation": tb - ta,
        "retained_fields_historical_signal_specific_integral_time_separated": True,
        "retained_fields_current_conservative_integral_time_separated": False,
        "temporal_independence_claim": False,
        "continuation_log": LOG.name,
        "continuation_log_sha256": sha256(LOG),
        "selected_field_paths": [str(p) for p in selected],
        "claim_boundary": (
            "native-yplus two-field audit only; fields exceed a historical "
            "signal-specific estimate but not the current conservative "
            "integral-time convention; no temporal-independence claim"
        ),
    })
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    digest = sha256(OUT)
    OUT.with_suffix(".sha256").write_text(f"{digest}  {OUT.name}\n")
    print(f"=== done ===\n[out] {OUT} sha256={digest}")
    print(f"[times] {[ta, tb]} separation={tb-ta}")


if __name__ == "__main__":
    main()
