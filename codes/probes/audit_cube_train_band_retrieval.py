#!/usr/bin/env python3
"""Frozen-record nearest-training-band leakage diagnostic.

This script performs no simulation, training, or neural inference. It uses the
frozen cube split to ask whether the evaluation wall band retrieves a training
field whose unsupplied volume explains the oracle-band reconstruction result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "record": "8bac93f1537eab6667d692282b76c7bccd28f28965d35ea97668bcc2567bc45a",
    "times": "99e8e0a45cf6c361bcefc10b251ceac15bd60992d42255cf76c945eae9655482",
    "indices": "28264e996586f961b1a3cd8c369f494f62d39d4d18d73f0625fc4983e4ab3d18",
}
REGIONS = (
    "full_support_excluded",
    "near_support_excluded_d_le_0p5h",
    "outer_d_gt_0p5h",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = 48, 96, 48
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) * 4.0 / ny
    z = (np.arange(nz) + 0.5) * 2.0 / nz
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    solid = (
        (xx >= 0.5)
        & (xx <= 1.5)
        & (yy <= 1.0)
        & (zz >= 0.5)
        & (zz <= 1.5)
    )
    fluid = ~solid
    dx = np.maximum.reduce([0.5 - xx, np.zeros_like(xx), xx - 1.5])
    dy = np.maximum.reduce([0.0 - yy, np.zeros_like(yy), yy - 1.0])
    dz = np.maximum.reduce([0.5 - zz, np.zeros_like(zz), zz - 1.5])
    dcube = np.sqrt(dx * dx + dy * dy + dz * dz)
    distance = np.minimum.reduce([yy, 4.0 - yy, dcube])
    band = fluid & (distance <= 2.01 * (4.0 / ny))
    return fluid, band, distance


def training_statistics(
    record: np.ndarray, train_indices: np.ndarray, fluid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros(3, dtype=np.float64)
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    mean_field = np.zeros((3, 48, 96, 48), dtype=np.float64)
    for index in train_indices:
        field = np.asarray(record[index], dtype=np.float32)
        mean_field += field
        for channel in range(3):
            values = field[channel][fluid]
            count[channel] += values.size
            total[channel] += values.sum(dtype=np.float64)
            squares[channel] += np.square(values, dtype=np.float64).sum(
                dtype=np.float64
            )
    mean_field /= len(train_indices)
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - mean * mean, 1e-8))
    return scale.astype(np.float32), mean_field.astype(np.float32)


def block_indices(
    n_items: int, block: int, draws: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_items / block))
    starts = rng.integers(0, n_items, size=(draws, n_blocks))
    offsets = np.arange(block)[None, None, :]
    return ((starts[:, :, None] + offsets).reshape(draws, -1) % n_items)[
        :, :n_items
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record", default="codes/data/cube_record/cube_ds2_float16.npy"
    )
    parser.add_argument(
        "--times", default="codes/data/cube_record/cube_ds2_times.npy"
    )
    parser.add_argument(
        "--indices",
        default=(
            "manuscript/source_data/fig3/"
            "cube_periodic_topology_components.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="manuscript/source_data/review_audit",
    )
    args = parser.parse_args()

    record_path = ROOT / args.record
    times_path = ROOT / args.times
    indices_path = ROOT / args.indices
    for name, path in (
        ("record", record_path),
        ("times", times_path),
        ("indices", indices_path),
    ):
        require_hash(path, EXPECTED[name])

    record = np.load(record_path, mmap_mode="r")
    times = np.load(times_path)
    index_data = np.load(indices_path)
    eval_indices = np.asarray(index_data["test_idx"], dtype=np.int64)
    train_indices = np.arange(660, dtype=np.int64)
    if record.shape != (1101, 3, 48, 96, 48):
        raise RuntimeError(f"unexpected record shape: {record.shape}")
    if len(eval_indices) != 160 or np.any(eval_indices < 758):
        raise RuntimeError("frozen evaluation-index contract changed")

    fluid, band, distance = geometry()
    if int(fluid.sum()) != 207360 or int(band.sum()) != 14008:
        raise RuntimeError("geometry-mask contract changed")
    regions = {
        "full_support_excluded": fluid & (~band),
        "near_support_excluded_d_le_0p5h": fluid & (~band) & (distance <= 0.5),
        "outer_d_gt_0p5h": fluid & (distance > 0.5),
    }
    scale, mean_field = training_statistics(record, train_indices, fluid)

    # The common training mean cancels in pairwise distances. Channel scaling
    # matches the fitted model's standardisation.
    train_band = np.empty((len(train_indices), 3 * int(band.sum())), np.float32)
    for position, index in enumerate(train_indices):
        field = np.asarray(record[index], dtype=np.float32)
        train_band[position] = (field[:, band] / scale[:, None]).reshape(-1)
    eval_band = np.empty((len(eval_indices), train_band.shape[1]), np.float32)
    for position, index in enumerate(eval_indices):
        field = np.asarray(record[index], dtype=np.float32)
        eval_band[position] = (field[:, band] / scale[:, None]).reshape(-1)

    # Exact squared Euclidean distance via Gram products, with float64 norms
    # and float32 BLAS products. Argmin ties, if any, resolve to the earliest
    # training index.
    train_norm = np.square(train_band, dtype=np.float64).sum(axis=1)
    eval_norm = np.square(eval_band, dtype=np.float64).sum(axis=1)
    distances = (
        eval_norm[:, None]
        + train_norm[None, :]
        - 2.0 * (eval_band @ train_band.T).astype(np.float64)
    )
    distances = np.maximum(distances, 0.0)
    donor_position = np.argmin(distances, axis=1)
    donor_indices = train_indices[donor_position]
    nearest_distance = distances[np.arange(len(eval_indices)), donor_position]

    truth = np.stack(
        [np.asarray(record[index], dtype=np.float32) for index in eval_indices]
    )
    donors = np.stack(
        [np.asarray(record[index], dtype=np.float32) for index in donor_indices]
    )
    truth_fluctuation = (
        (truth - mean_field[None]) / scale[None, :, None, None, None]
    )
    donor_fluctuation = (
        (donors - mean_field[None]) / scale[None, :, None, None, None]
    )

    components: dict[str, np.ndarray] = {
        "train_idx": train_indices,
        "eval_idx": eval_indices,
        "donor_idx": donor_indices,
        "eval_time": times[eval_indices].astype(np.float64),
        "donor_time": times[donor_indices].astype(np.float64),
        "time_lag_snapshots": (eval_indices - donor_indices).astype(np.int64),
        "time_lag_nondimensional": (
            times[eval_indices] - times[donor_indices]
        ).astype(np.float64),
        "nearest_band_squared_distance": nearest_distance.astype(np.float64),
    }
    point: dict[str, float] = {}
    block_summary: dict[str, dict[str, object]] = {}
    draws = block_indices(len(eval_indices), 57, 4000, 20260730)
    for name, mask in regions.items():
        target_mean = float(truth_fluctuation[:, :, mask].mean())
        sst = np.square(
            truth_fluctuation[:, :, mask] - target_mean,
            dtype=np.float64,
        ).sum(axis=(1, 2), dtype=np.float64)
        sse = np.square(
            donor_fluctuation[:, :, mask] - truth_fluctuation[:, :, mask],
            dtype=np.float64,
        ).sum(axis=(1, 2), dtype=np.float64)
        components[f"sst__{name}"] = sst.astype(np.float64)
        components[f"sse__nearest_training_band__{name}"] = sse.astype(
            np.float64
        )
        value = float(1.0 - sse.sum() / sst.sum())
        bootstrap = 1.0 - sse[draws].sum(axis=1) / sst[draws].sum(axis=1)
        point[name] = value
        block_summary[name] = {
            "point": value,
            "conditional_interval": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
        }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    component_path = output_dir / "cube_train_band_retrieval_components.npz"
    np.savez_compressed(component_path, **components)
    result = {
        "schema": "gwt-cube-train-band-retrieval-v1",
        "compute_boundary": (
            "Frozen-record CPU diagnostic; no simulation, training or neural inference."
        ),
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": EXPECTED[name]}
            for name, path in (
                ("record", record_path),
                ("times", times_path),
                ("indices", indices_path),
            )
        },
        "protocol": {
            "train_indices": [0, 659],
            "evaluation_count": 160,
            "retrieval_feature": (
                "all three channel-standardised values on the 14,008-cell supplied band"
            ),
            "distance": "exact squared Euclidean",
            "prediction": "unchanged associated training full field",
            "support_excluded": True,
            "block_evaluation_samples": 57,
            "bootstrap_draws": 4000,
            "bootstrap_seed": 20260730,
        },
        "nearest_training_band": {
            "r2_fluctuation": point,
            "block57_sensitivity": block_summary,
            "lag_snapshots": {
                "minimum": int(np.min(components["time_lag_snapshots"])),
                "median": float(np.median(components["time_lag_snapshots"])),
                "maximum": int(np.max(components["time_lag_snapshots"])),
            },
            "lag_nondimensional": {
                "minimum": float(np.min(components["time_lag_nondimensional"])),
                "median": float(np.median(components["time_lag_nondimensional"])),
                "maximum": float(np.max(components["time_lag_nondimensional"])),
            },
        },
        "components": {
            "path": str(component_path.relative_to(ROOT)),
            "sha256": sha256(component_path),
        },
        "interpretation_boundary": (
            "This control tests one simple correlated-record retrieval explanation. "
            "Failure to match the learned endpoint does not prove temporal independence, "
            "population generalisation, or causal closure composition."
        ),
    }
    result_path = output_dir / "cube_train_band_retrieval_results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RETRIEVAL_AUDIT_PASS "
        f"complete={point['full_support_excluded']:+.6f} "
        f"near={point['near_support_excluded_d_le_0p5h']:+.6f} "
        f"farther={point['outer_d_gt_0p5h']:+.6f}"
    )
    print(f"result={result_path} sha256={sha256(result_path)}")
    print(f"components={component_path} sha256={sha256(component_path)}")


if __name__ == "__main__":
    main()
