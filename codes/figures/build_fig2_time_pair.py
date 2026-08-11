#!/usr/bin/env python3
"""Extract the real LES volumes used by Figure 2's temporal control.

The target is the producer-fixed first evaluation record.  The donor is the
exact far-time record used to form the mismatched-wall control.  The pair
artefact preserves all three velocity components.  A second compact artefact
stores the streamwise component of three evenly spaced retained records between
the endpoints for Figure 2's time-evolution strip.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = ROOT / "codes/data/cube_record/cube_ds2_float16.npy"
TIMES_PATH = ROOT / "codes/data/cube_record/cube_ds2_times.npy"
REPRESENTATIVE_PATH = ROOT / "manuscript/source_data/fig2/cube3d_representative_fields.npz"
TIMELINE_PATH = ROOT / "manuscript/source_data/fig2/cube3d_timeline.json"
OUTPUT = ROOT / "manuscript/source_data/fig2/cube3d_time_pair.npz"
SEQUENCE_OUTPUT = ROOT / "manuscript/source_data/fig2/cube3d_time_sequence.npz"

EXPECTED_RECORD_SHA256 = "8bac93f1537eab6667d692282b76c7bccd28f28965d35ea97668bcc2567bc45a"
EXPECTED_TIMES_SHA256 = "99e8e0a45cf6c361bcefc10b251ceac15bd60992d42255cf76c945eae9655482"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for path, expected in (
    (RECORD_PATH, EXPECTED_RECORD_SHA256),
    (TIMES_PATH, EXPECTED_TIMES_SHA256),
):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"source hash mismatch for {path}: {actual} != {expected}")

timeline = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
target_index = int(timeline["displayed_target"]["index"])
donor_index = int(timeline["far_time_condition_for_displayed_target"]["index"])
intermediate_indices = np.rint(
    np.linspace(target_index, donor_index, 5)[1:-1]
).astype(np.int64)
record = np.load(RECORD_PATH, mmap_mode="r")
times = np.load(TIMES_PATH)

with np.load(REPRESENTATIVE_PATH, allow_pickle=False) as representative:
    target = np.asarray(record[target_index])
    if not np.array_equal(target, representative["truth"]):
        raise RuntimeError("record target does not match the frozen representative LES target")
    np.savez_compressed(
        OUTPUT,
        target=target,
        donor=np.asarray(record[donor_index]),
        indices=np.asarray([target_index, donor_index], dtype=np.int64),
        times_Ubar_over_h=np.asarray([times[target_index], times[donor_index]], dtype=np.float64),
        relative_times_Ubar_over_h=np.asarray(
            [0.0, times[donor_index] - times[target_index]], dtype=np.float64
        ),
        fluid=np.asarray(representative["fluid"]),
        x=np.asarray(representative["x"]),
        y=np.asarray(representative["y"]),
        z=np.asarray(representative["z"]),
    )

    np.savez_compressed(
        SEQUENCE_OUTPUT,
        intermediate_u=np.asarray(record[intermediate_indices, 0]),
        indices=intermediate_indices,
        times_Ubar_over_h=np.asarray(times[intermediate_indices], dtype=np.float64),
        relative_times_Ubar_over_h=np.asarray(
            times[intermediate_indices] - times[target_index], dtype=np.float64
        ),
        fluid=np.asarray(representative["fluid"]),
        x=np.asarray(representative["x"]),
        y=np.asarray(representative["y"]),
        z=np.asarray(representative["z"]),
    )

print(f"wrote {OUTPUT}")
print(f"sha256={sha256(OUTPUT)}")
print(f"wrote {SEQUENCE_OUTPUT}")
print(f"sha256={sha256(SEQUENCE_OUTPUT)}")
