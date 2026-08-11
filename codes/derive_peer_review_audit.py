#!/usr/bin/env python3
"""Derive the Level-1 peer-review statistics from immutable on-disk evidence.

This script performs no simulation, training, or neural-network inference.  It
recomputes the fair finite-ensemble energy score, all registered physical
statistics under the conservative dependence block, point-score sensitivity,
and committed comparator summary from stored sufficient statistics and
terminal JSON records.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript" / "source_data"
DESTINATION = Path(
    os.environ.get(
        "GWT_PEER_REVIEW_AUDIT_OUTPUT",
        ROOT
        / "manuscript"
        / "source_data"
        / "review_audit"
        / "derived_peer_review_statistics.json",
    )
)

INPUTS = {
    "m0_result": (
        SOURCE / "review_audit/cube3d_coupling_results.json",
        "61e7d09d297bd6fa6a5aa8d55476e1c2a6fe56ddda0e7efe93b7eab25c3ce2a2",
    ),
    "m1_result": (
        SOURCE / "review_audit/cube3d_coupling_adequate_results.json",
        "6762e9a89855c38538463c217698dab0e33c587a00d2fac1f937cf5ce35e982b",
    ),
    "m2_result": (
        SOURCE / "fig3/cube_periodic_topology_results.json",
        "0fc302421d622879e00ef16a14636fb8849ee65c0680c85491659788807f914d",
    ),
    "m2_components": (
        SOURCE / "fig3/cube_periodic_topology_components.npz",
        "28264e996586f961b1a3cd8c369f494f62d39d4d18d73f0625fc4983e4ab3d18",
    ),
    "ensemble_result": (
        SOURCE / "fig4/cube_distributional_rapid_results.json",
        "c70342f6fc4bf8e21f0883ba76de806f01eb8cfc34f9b0d52018443beba00588",
    ),
    "ensemble_components": (
        SOURCE / "fig4/cube_distributional_rapid_components.npz",
        "a7f1c74f909e1a805d4a3642db83285371ecd80e4a59cd11a873a14253010e2d",
    ),
    "wiener_seed2234": (
        SOURCE / "review_audit/cube_wiener_floor_bench_results_seed2234.json",
        "77e6ff62f926130616cd8b09b044024c02a953f1f3d463758f85e5bca753f48c",
    ),
    "wiener_seed3234": (
        SOURCE / "review_audit/cube_wiener_floor_bench_results_seed3234.json",
        "1ee765378277f3264db777574bbe8ef8e91c75a1c2a20fb6d1ca1051af704918",
    ),
    "wiener_chain": (
        SOURCE / "review_audit/wiener_seed_chain_node005.json",
        "52cac551f0d1f7f4661053a6bd81f4322c466dbb75457970189c8aa2093f2842",
    ),
    "retrieval_result": (
        SOURCE / "review_audit/cube_train_band_retrieval_results.json",
        "94fa34214b23c2ae2ed046e846ba84bbed642ce4c90618be04417c3e82d5674f",
    ),
    "retrieval_components": (
        SOURCE / "review_audit/cube_train_band_retrieval_components.npz",
        "02b412daeaf5314c237291534c7c995876c66c81794ac07f19327eea93fec0ce",
    ),
}

POINT_ARMS = ("correct", "no_wall", "wrong_wall")
ENSEMBLE_ARMS = ("correct", "no_wall", "far_time_wall")
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


def require_inputs() -> None:
    for name, (path, expected) in INPUTS.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} SHA-256 mismatch: {actual} != {expected}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def block_draws(n: int, block: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, count))
    offsets = np.arange(block)[None, None, :]
    return ((starts[:, :, None] + offsets).reshape(draws, -1) % n)[:, :n]


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, (2.5, 97.5))]


def summarize_loss(
    values: dict[str, np.ndarray], block: int, draws: int, seed: int
) -> dict:
    index = block_draws(len(next(iter(values.values()))), block, draws, seed)
    output: dict[str, object] = {"arms": {}, "improvements": {}}
    for arm, arm_values in values.items():
        bootstrap = arm_values[index].mean(axis=1)
        output["arms"][arm] = {
            "mean": float(arm_values.mean()),
            "ci95": interval(bootstrap),
        }
    for control in ("no_wall", "far_time_wall"):
        difference = values[control] - values["correct"]
        bootstrap = difference[index].mean(axis=1)
        output["improvements"][f"{control}_minus_correct"] = {
            "mean": float(difference.mean()),
            "ci95": interval(bootstrap),
            "ci_positive": bool(np.percentile(bootstrap, 2.5) > 0),
        }
    return output


def fair_energy_summary() -> dict:
    data = np.load(INPUTS["ensemble_components"][0])
    values: dict[str, np.ndarray] = {}
    for arm in ENSEMBLE_ARMS:
        truth = data[f"energy_truth_distance__{arm}"].mean(axis=1)
        pairs = data[f"energy_pair_distance__{arm}"]
        members = pairs.shape[1]
        if pairs.shape[2] != members or members != 8:
            raise RuntimeError(f"unexpected ensemble shape for {arm}: {pairs.shape}")
        # Diagonal distances are zero. The M(M-1) denominator makes this the
        # U-statistic estimator for draws from the fitted continuous sampler.
        values[arm] = truth - pairs.sum(axis=(1, 2)) / (
            2.0 * members * (members - 1)
        )
    return {
        "estimand": (
            "off-diagonal U-statistic estimate of the multivariate energy "
            "score of the fitted continuous conditional sampler"
        ),
        "members": 8,
        "normalization": "Euclidean distances divided by sqrt(D)",
        "block49_historical": summarize_loss(values, 49, 4000, 20260723),
        "block57_conservative": summarize_loss(values, 57, 4000, 20260723),
    }


def physical_statistic_summary() -> dict:
    """Reblock two valid physical families and descriptive interval hits."""

    data = np.load(INPUTS["ensemble_components"][0])
    metrics = {
        "component_spectrum_log_rmse": {
            arm: data[f"spectrum_log_rmse__{arm}"] for arm in ENSEMBLE_ARMS
        },
        "reynolds_stress_profile_nrmse": {
            arm: data[f"reynolds_profile_nrmse__{arm}"] for arm in ENSEMBLE_ARMS
        },
    }
    output: dict[str, object] = {
        "estimand": (
            "two valid farther-from-wall physical-statistic families; "
            "the eight-member interpolated 0.1/0.9 sample-quantile hit fraction "
            "is retained descriptively but withdrawn from calibration adjudication"
        ),
        "members": 8,
        "valid_family_names": [
            "component_spectrum_log_rmse",
            "reynolds_stress_profile_nrmse",
        ],
        "withdrawn_family": {
            "name": "coverage80_abs_error",
            "reason": (
                "0.8 is not a finite-ensemble-valid reference for linearly "
                "interpolated 0.1/0.9 sample quantiles with M=8; retained arrays "
                "lack ranks/member fields for a valid replacement"
            ),
            "used_for_adjudication": False,
        },
    }
    for block, label in ((49, "block49_historical"), (57, "block57_conservative")):
        block_output = {
            metric: summarize_loss(values, block, 4000, 20260723)
            for metric, values in metrics.items()
        }
        index = block_draws(160, block, 4000, 20260723)
        coverage_arms: dict[str, object] = {}
        for arm in ENSEMBLE_ARMS:
            coverage = data[f"coverage80__{arm}"]
            bootstrap = coverage[index].mean(axis=1)
            coverage_arms[arm] = {
                "mean": float(coverage.mean()),
                "ci95": interval(bootstrap),
            }
        block_output["coverage80"] = {"arms": coverage_arms}
        output[label] = block_output
    return output


def retrieval_summary() -> dict:
    """Replay the frozen nearest-training-band scores from released sufficients."""

    stored = load_json(INPUTS["retrieval_result"][0])
    data = np.load(INPUTS["retrieval_components"][0])
    index = block_draws(160, 57, 4000, 20260730)
    regions: dict[str, object] = {}
    for region in REGIONS:
        sst = data[f"sst__{region}"]
        sse = data[f"sse__nearest_training_band__{region}"]
        point = float(1.0 - sse.sum() / sst.sum())
        bootstrap = 1.0 - sse[index].sum(axis=1) / sst[index].sum(axis=1)
        regions[region] = {
            "point": point,
            "conditional_interval": interval(bootstrap),
        }
        expected = stored["nearest_training_band"]["block57_sensitivity"][region]
        if abs(point - expected["point"]) > 2e-12 or np.max(
            np.abs(np.asarray(regions[region]["conditional_interval"]) -
                   np.asarray(expected["conditional_interval"]))
        ) > 2e-12:
            raise RuntimeError(f"retrieval sufficient-statistic replay failed: {region}")
    return {
        "claim_boundary": stored["interpretation_boundary"],
        "regions": regions,
        "lag_snapshots": stored["nearest_training_band"]["lag_snapshots"],
        "components_sha256": stored["components"]["sha256"],
    }


def point_sensitivity() -> dict:
    data = np.load(INPUTS["m2_components"][0])
    output: dict[str, object] = {}
    for block, label in ((49, "block49_historical"), (57, "block57_conservative")):
        index = block_draws(160, block, 4000, 44)
        arms: dict[str, dict[str, dict[str, object]]] = {}
        bootstrap_cache: dict[str, dict[str, np.ndarray]] = {}
        for arm in POINT_ARMS:
            arms[arm] = {}
            bootstrap_cache[arm] = {}
            for region in REGIONS:
                sst = data[f"sst__{region}"]
                sse = data[f"sse__{arm}__{region}"]
                point = 1.0 - sse.sum() / (sst.sum() + 1e-12)
                bootstrap = 1.0 - sse[index].sum(axis=1) / (
                    sst[index].sum(axis=1) + 1e-12
                )
                bootstrap_cache[arm][region] = bootstrap
                arms[arm][region] = {
                    "point": float(point),
                    "ci95": interval(bootstrap),
                }
        differences: dict[str, dict[str, dict[str, object]]] = {}
        for control in ("no_wall", "wrong_wall"):
            key = f"correct_minus_{control}"
            differences[key] = {}
            for region in REGIONS:
                point = (
                    arms["correct"][region]["point"]
                    - arms[control][region]["point"]
                )
                bootstrap = (
                    bootstrap_cache["correct"][region]
                    - bootstrap_cache[control][region]
                )
                differences[key][region] = {
                    "point": float(point),
                    "ci95": interval(bootstrap),
                    "ci_positive": bool(np.percentile(bootstrap, 2.5) > 0),
                }
        output[label] = {
            "block_evaluation_samples": block,
            "arms": arms,
            "differences": differences,
        }
    return output


def regional_values(item: dict) -> dict[str, float]:
    return {
        "complete": float(item["full_support_excluded"]["R2_fluct_balanced"]),
        "near": float(
            item["near_support_excluded_d_le_0p5h"]["R2_fluct_balanced"]
        ),
        "outer": float(item["outer_d_gt_0p5h"]["R2_fluct_balanced"]),
    }


def comparator_summary() -> dict:
    first = load_json(INPUTS["wiener_seed2234"][0])
    second = load_json(INPUTS["wiener_seed3234"][0])
    phase_a = first["phase_A_wiener_floor"]["arms"]
    deterministic = []
    for result in (first, second):
        deterministic.append(
            {
                "training_seed": int(result["_meta"]["seed_training"]),
                **regional_values(
                    result["phase_B_deterministic_parity"]["arms"]["correct"]
                ),
            }
        )
    return {
        "claim_boundary": (
            "Contacted methodology controls on the same chronological allocation. "
            "Estimator and training protocols differ from M0, so the valid values "
            "bound generative-method attribution rather than establish a clean ranking. "
            "The interior-shell arm is quarantined because its observations overlap "
            "its score support."
        ),
        "linear_wiener": {
            "climatological_no_wall": regional_values(phase_a["no_wall"]),
            "near_wall_band": regional_values(phase_a["correct"]),
            "random_value": regional_values(phase_a["random_value"]),
            "spatial_permutation": regional_values(phase_a["spatial_permutation"]),
            "wrong_time": regional_values(phase_a["wrong_time"]),
        },
        "quarantined_location_comparator": {
            "label": "nominal equal-support interior shell",
            "values_retained_for_forensic_custody": regional_values(
                phase_a["equal_support_interior"]
            ),
            "reason": (
                "All 13,984 observed shell cells remain inside the arm's complete "
                "score. Of these, 9,112 are in the near score and 4,872 are in the "
                "outer score, whereas every observed near-wall-band cell is excluded "
                "for the wall arm. The arm cannot support a location comparison."
            ),
            "shell_cells": 13984,
            "complete_overlap_cells": 13984,
            "near_overlap_cells": 9112,
            "near_overlap_fraction": 9112 / 74844,
            "outer_overlap_cells": 4872,
            "outer_overlap_fraction": 4872 / 118508,
            "active_claim": False,
        },
        "deterministic_direct_correct_band": deterministic,
    }


def first_contact_summary() -> dict:
    result = load_json(INPUTS["m0_result"][0])
    evaluation = result["evaluation"]
    return {
        "claim_boundary": (
            "First arm output after the allocation freeze. Only terminal aggregate "
            "JSON is retained; per-time SSE/SST arrays are unavailable, so this "
            "audit does not claim an independent M0 endpoint replay or a "
            "122.48-snapshot block sensitivity."
        ),
        "arms": {
            arm: {
                region: float(values["R2_fluct_balanced"])
                for region, values in evaluation["arms"][arm].items()
            }
            for arm in POINT_ARMS
        },
        "differences": {
            contrast: {
                region: float(values["point"])
                for region, values in regions.items()
            }
            for contrast, regions in evaluation["deltas"].items()
        },
        "historical_resampling": {
            "block_evaluation_samples": int(evaluation["eval_block"]),
            "status": (
                "terminal summaries only; not promoted as conservative "
                "population-confidence intervals"
            ),
        },
    }


def dependence_summary() -> dict:
    result = load_json(INPUTS["wiener_seed2234"][0])
    autocorrelation = result["autocorrelation"]
    conservative = float(autocorrelation["tau_max_snapshots"])
    return {
        "primary_rule": (
            "maximum audited integral time over band mean energy, band mean "
            "streamwise velocity, first-cell shear proxy, outer-plane streamwise "
            "velocity, and volume TKE"
        ),
        "signals": autocorrelation["signals"],
        "conservative_tau_snapshots": conservative,
        "gap_snapshots": 98,
        "gap_in_conservative_integral_times": 98.0 / conservative,
        "evaluation_span_snapshots": 343,
        "evaluation_span_in_conservative_integral_times": 343.0 / conservative,
        "post_spinup_snapshots": 1101,
        "post_spinup_effective_events": 1101.0 / conservative,
        "conservative_block_evaluation_samples": 57,
    }


def local_asset_audit() -> dict:
    assets = {
        "m0_checkpoint": (
            ROOT / "codes/results/cube3d_rectified_flow.pt",
            "a78dcd506d16be8ee81077cefbf29918b76086a5c0f1ac32f2e90c52f0c733d6",
        ),
        "m2_checkpoint": (
            ROOT / "codes/results/cube_periodic_topology/cube_periodic_topology.pt",
            "6f507fd1fb97a7e52dd60a631507dbc37d2005fdd12abd856f165eed8f6135c2",
        ),
        "cube_record": (
            ROOT / "codes/data/cube_record/cube_ds2_float16.npy",
            "8bac93f1537eab6667d692282b76c7bccd28f28965d35ea97668bcc2567bc45a",
        ),
        "cube_times": (
            ROOT / "codes/data/cube_record/cube_ds2_times.npy",
            "99e8e0a45cf6c361bcefc10b251ceac15bd60992d42255cf76c945eae9655482",
        ),
    }
    output: dict[str, object] = {}
    for name, (path, expected) in assets.items():
        output[name] = {
            "repository_relative_path": str(path.relative_to(ROOT)),
            "expected_sha256": expected,
            "included_in_compact_release": False,
            "external_reviewer_access_route": "requires author confirmation",
        }
    return output


def main() -> None:
    require_inputs()
    payload = {
        "schema": "gwt-level1-peer-review-derived-audit-v1",
        "compute_boundary": (
            "Deterministic CPU derivation from committed terminal records and "
            "stored sufficient statistics; no simulation, training, or neural "
            "inference."
        ),
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": expected,
            }
            for name, (path, expected) in INPUTS.items()
        },
        "dependence": dependence_summary(),
        "first_contact_m0": first_contact_summary(),
        "m2_point_block_sensitivity": point_sensitivity(),
        "fair_energy_score": fair_energy_summary(),
        "physical_statistics": physical_statistic_summary(),
        "train_band_retrieval": retrieval_summary(),
        "committed_comparators": comparator_summary(),
        "local_reviewer_assets": local_asset_audit(),
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PEER_REVIEW_AUDIT_PASS {DESTINATION}")


if __name__ == "__main__":
    main()
