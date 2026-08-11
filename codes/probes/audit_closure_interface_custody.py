#!/usr/bin/env python3
"""Audit the frozen wall-closure interface against its actual export contract.

This is a lightweight, read-only scientific audit.  It does not train a model or
run neural inference beyond the deterministic forward pass of the frozen
4-64-64-16 trunk already stored in ``codes/closure/x2fam_weights.h``.

Two modes are supported:

* default: derive the audit from the retained Krank profiles and companion
  export record, then write a compact JSON result;
* ``--verify-retained``: recompute every reported metric and frozen-header
  forward pass from the compact JSON, so the review package remains checkable
  without silently depending on the sibling project tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "manuscript"
    / "source_data"
    / "review_audit"
    / "closure_interface_custody.json"
)
DEFAULT_PROFILE = (
    ROOT.parent
    / "familyclass_wall_model"
    / "codes"
    / "geometry_bridge_codes"
    / "repeating_structure_codes"
    / "new_data_download"
    / "geometry_driven"
    / "krank_pehill_Re10595_wall_profiles.npz"
)
DEFAULT_EXPORT_RESULT = (
    ROOT.parent
    / "familyclass_wall_model"
    / "codes"
    / "results"
    / "node004_re10595_apriori.json"
)
DEFAULT_EXPORTER = (
    ROOT.parent / "familyclass_wall_model" / "codes" / "node004_x2_export.py"
)
DEFAULT_DESCRIPTOR_BUILDER = (
    ROOT.parent
    / "familyclass_wall_model"
    / "codes"
    / "geometry_bridge_codes"
    / "beta_bridge_crosskind.py"
)
DEFAULT_STAGEA_ADAPTER = (
    ROOT.parent / "familyclass_wall_model" / "codes" / "stageA_corpus_adapter.py"
)
HEADER = ROOT / "codes" / "closure" / "x2fam_weights.h"
LOCAL_CLOSURE = ROOT / "codes" / "closure" / "wall_closure.py"

EXPECTED_SOURCE_HASHES = {
    "profile": "09889bbb54b5acff2fa56c8f47dcfc320ac3f1f4389a8a8507900ec824f41c1c",
    "export_result": "bcbf569a1225c3913a5d72249ab91b0bb699fd26d8e695386750aff252b52d1f",
    "exporter": "9222b04afe5a756ff3db0929821c46d2ab74d3be4a51c3c66bbf6834e74c47f3",
    "descriptor_builder": "15f0207db3bd03f74afbc21e53cf2f3b0cd1728cf0ccd778c109ae7f9e1c95d8",
    "stagea_adapter": "31cb150470d4e5a8a4b5fff6c85d709489e1c17eb1c9e901bbe0ddc5b4e94570",
    "header": "63e4941f682996ef2f95f8882363d73ca440199cd7b5ccee61002f1c5fa99280",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def trapz(values: np.ndarray, coordinates: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, coordinates))


def smooth_edge(values: np.ndarray, width: int = 5) -> np.ndarray:
    """Exact five-station edge-padded smoother used by the local candidate seam."""

    values = np.asarray(values, dtype=np.float64)
    pad = width // 2
    return np.convolve(
        np.pad(values, pad, mode="edge"),
        np.ones(width, dtype=np.float64) / width,
        mode="valid",
    )


def relative_rmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(
        np.sqrt(np.mean((candidate - reference) ** 2))
        / np.sqrt(np.mean(reference**2))
    )


def r2_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    return float(
        1.0
        - np.sum((reference - candidate) ** 2)
        / np.sum((reference - reference.mean()) ** 2)
    )


def header_contract() -> dict:
    text = HEADER.read_text(encoding="utf-8")

    def define(name: str) -> int:
        match = re.search(rf"#define\s+{name}\s+(\d+)", text)
        if not match:
            raise RuntimeError(f"missing {name} in {HEADER}")
        return int(match.group(1))

    emb_match = re.search(
        r"static const double\s+X2_EMB\s*\[\s*(\d+)\s*\]", text
    )
    if not emb_match:
        raise RuntimeError("missing X2_EMB in frozen header")
    member_match = re.search(r"member\s*=\s*([^\n]+)", text)
    return {
        "sha256": sha256(HEADER),
        "architecture": {
            "local_input_dimension": define("X2_NLOC"),
            "trunk_hidden_width": define("X2_H"),
            "latent_dimension": define("X2_P"),
            "frozen_embedding_dimension": int(emb_match.group(1)),
            "descriptor_knot_count": define("X2_NKNOT"),
        },
        "member_comment": member_match.group(1).strip() if member_match else None,
        "contains_branch_network_weights": "X2_BRANCH" in text,
        "contains_set_normalisation": "X2_MU_SET" in text or "X2_SD_SET" in text,
    }


def frozen_forward(features: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(LOCAL_CLOSURE.parent))
    from wall_closure import predict_Cf  # pylint: disable=import-outside-toplevel

    return np.asarray(predict_Cf(features), dtype=np.float64)


def derive(profile_path: Path, export_result_path: Path) -> dict:
    source_paths = {
        "profile": profile_path,
        "export_result": export_result_path,
        "exporter": DEFAULT_EXPORTER,
        "descriptor_builder": DEFAULT_DESCRIPTOR_BUILDER,
        "stagea_adapter": DEFAULT_STAGEA_ADAPTER,
        "header": HEADER,
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required {name} artifact is absent: {path}")
        require_hash(path, EXPECTED_SOURCE_HASHES[name])

    profile = np.load(profile_path, allow_pickle=True)
    x = np.asarray(profile["x"], dtype=np.float64)
    native_dpdx = np.asarray(profile["dp_dx"], dtype=np.float64)
    tau_w = np.asarray(profile["tau_w"], dtype=np.float64)
    if x.shape != (10,) or native_dpdx.shape != (10,) or tau_w.shape != (10,):
        raise RuntimeError("retained Krank audit expects exactly ten station profiles")

    edge_velocity = []
    delta99 = []
    matching_velocity = []
    dstar_full = []
    dstar_resolved = []
    for y_raw, u_raw in zip(profile["y"], profile["U"]):
        order = np.argsort(y_raw)
        y = np.asarray(y_raw, dtype=np.float64)[order]
        u = np.asarray(u_raw, dtype=np.float64)[order]
        ue = float(np.max(u))
        hits = np.flatnonzero(u >= 0.99 * ue)
        d99 = float(y[hits[0]]) if hits.size else float(y[-1])
        ym = 0.10 * d99
        full = y <= d99
        resolved = (y >= ym) & (y <= d99)
        if full.sum() < 3 or resolved.sum() < 3:
            raise RuntimeError("insufficient profile support for displacement thickness")
        edge_velocity.append(ue)
        delta99.append(d99)
        matching_velocity.append(float(np.interp(ym, y, u)))
        dstar_full.append(trapz(1.0 - u[full] / ue, y[full]))
        dstar_resolved.append(trapz(1.0 - u[resolved] / ue, y[resolved]))

    ue = np.asarray(edge_velocity)
    d99 = np.asarray(delta99)
    um = np.asarray(matching_velocity)
    dstar_full_array = np.asarray(dstar_full)
    dstar_resolved_array = np.asarray(dstar_resolved)

    smoothed_ue = smooth_edge(ue, width=5)
    candidate_dpdx = -smoothed_ue * np.gradient(smoothed_ue, x)
    calibration_features = np.column_stack(
        [
            um / ue,
            np.full_like(ue, 0.10),
            native_dpdx * d99 / ue**2,
            native_dpdx * dstar_resolved_array / ue**2,
        ]
    )
    candidate_features = np.column_stack(
        [
            um / ue,
            np.full_like(ue, 0.10),
            candidate_dpdx * d99 / ue**2,
            candidate_dpdx * dstar_full_array / ue**2,
        ]
    )
    cf_calibration = frozen_forward(calibration_features)
    cf_candidate = frozen_forward(candidate_features)
    cf_reference = 2.0 * tau_w / ue**2

    with export_result_path.open(encoding="utf-8") as handle:
        export_result = json.load(handle)
    station_rows = export_result["krank_10595_apriori"]["stations"]
    stored_cf_x2 = np.asarray([row["Cf_x2"] for row in station_rows])
    stored_cf_mlp = np.asarray([row["Cf_mlp"] for row in station_rows])
    stored_cf_true = np.asarray([row["Cf_true"] for row in station_rows])

    ratio = dstar_full_array / dstar_resolved_array
    metrics = {
        "pressure_surrogate_relative_rmse": relative_rmse(
            candidate_dpdx, native_dpdx
        ),
        "pressure_surrogate_pearson": float(
            np.corrcoef(candidate_dpdx, native_dpdx)[0, 1]
        ),
        "dstar_full_over_resolved_median": float(np.median(ratio)),
        "dstar_full_over_resolved_min": float(np.min(ratio)),
        "dstar_full_over_resolved_max": float(np.max(ratio)),
        "closure_output_relative_rmse": relative_rmse(
            cf_candidate, cf_calibration
        ),
        "closure_output_sign_changes": int(
            np.count_nonzero(np.sign(cf_candidate) != np.sign(cf_calibration))
        ),
        "calibration_feature_cf_r2_all": r2_score(
            cf_reference, cf_calibration
        ),
        "candidate_feature_cf_r2_all": r2_score(cf_reference, cf_candidate),
        "exported_forward_max_abs_difference": float(
            np.max(np.abs(cf_calibration - stored_cf_x2))
        ),
        "reference_cf_max_abs_difference": float(
            np.max(np.abs(cf_reference - stored_cf_true))
        ),
        "paired_local_mlp_cf_r2_all": r2_score(
            cf_reference, stored_cf_mlp
        ),
        "paired_local_mlp_cf_r2_separated": r2_score(
            cf_reference[cf_reference < 0], stored_cf_mlp[cf_reference < 0]
        ),
        "frozen_x2_cf_r2_separated": r2_score(
            cf_reference[cf_reference < 0],
            cf_calibration[cf_reference < 0],
        ),
    }
    contract = header_contract()
    if contract["architecture"] != {
        "local_input_dimension": 4,
        "trunk_hidden_width": 64,
        "latent_dimension": 16,
        "frozen_embedding_dimension": 16,
        "descriptor_knot_count": 10,
    }:
        raise RuntimeError("frozen header architecture differs from audited contract")
    if contract["contains_branch_network_weights"] or contract["contains_set_normalisation"]:
        raise RuntimeError("unexpected source-general branch material in frozen header")

    payload = {
        "schema": "gwt-closure-interface-custody-v1",
        "status": "hard_launch_block_calibration_and_branch_contract_not_source_general",
        "compute": "deterministic CPU audit of retained profiles and frozen forward only",
        "source_artifacts": {
            name: {
                "path": (
                    str(path.relative_to(ROOT))
                    if path.is_relative_to(ROOT)
                    else f"companion:{path.name}"
                ),
                "sha256": sha256(path),
            }
            for name, path in source_paths.items()
        },
        "frozen_header_contract": contract,
        "actual_closure_map": {
            "equation": "Cf_hat=<T(Phi),B(S)>+b",
            "local_features": [
                "U_m/U_e",
                "y_m/delta99",
                "delta99*dpdx/U_e^2",
                "dstar_resolved*dpdx/U_e^2",
            ],
            "set_features": [
                "U_m/U_e",
                "y_m/delta99",
                "delta99*dpdx/U_e^2",
                "dstar_resolved*dpdx/U_e^2",
                "beta_p",
            ],
            "branch_source": (
                "Krank Re=10595 target member's own ten-station, "
                "wall-stress-free set"
            ),
            "pressure_gradient_calibration": (
                "source-native differentiated wall-pressure trace"
            ),
            "displacement_thickness_calibration": (
                "integral over the resolved interval y_m<=y<=delta99"
            ),
        },
        "candidate_local_adapter": {
            "pressure_gradient": (
                "-U_e*dU_e/dx after five-station edge-padded smoothing"
            ),
            "displacement_thickness": "integral over the full wall-to-delta99 profile",
            "branch": "reuses fixed Krank embedding",
            "scientific_status": (
                "diagnostic only; not calibration-consistent or source-general"
            ),
        },
        "metrics": metrics,
        "retained_arrays": {
            "x": x.tolist(),
            "edge_velocity": ue.tolist(),
            "delta99": d99.tolist(),
            "matching_velocity": um.tolist(),
            "native_dpdx": native_dpdx.tolist(),
            "candidate_dpdx": candidate_dpdx.tolist(),
            "dstar_full": dstar_full_array.tolist(),
            "dstar_resolved": dstar_resolved_array.tolist(),
            "cf_reference": cf_reference.tolist(),
            "calibration_features": calibration_features.tolist(),
            "candidate_features": candidate_features.tolist(),
            "cf_calibration": cf_calibration.tolist(),
            "cf_candidate": cf_candidate.tolist(),
            "stored_cf_x2": stored_cf_x2.tolist(),
            "stored_cf_mlp": stored_cf_mlp.tolist(),
        },
        "adjudication": {
            "calibration_shift_load_bearing": True,
            "fixed_embedding_transplant_permitted": False,
            "source_general_branch_reconstruction_from_header_possible": False,
            "paired_mlp_information_matched_to_family_branch": False,
            "paired_mlp_disposition": (
                "adverse secondary closure-side baseline; cannot identify the "
                "value of the family-set input"
            ),
            "required_before_any_e2_launch": [
                "nominate a source and record closure-corpus/source overlap",
                "freeze calibration-consistent native pressure and resolved dstar inputs",
                "freeze a leakage-safe source for the member station set S",
                "export or otherwise custody the branch network and set normalisation",
                "include an information-matched data-only comparator",
                "keep launch_allowed=false until panel and operator authority",
            ],
        },
    }
    return payload


def verify_retained(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "gwt-closure-interface-custody-v1":
        raise RuntimeError("unexpected retained audit schema")
    require_hash(HEADER, EXPECTED_SOURCE_HASHES["header"])
    arrays = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in payload["retained_arrays"].items()
    }
    recalculated = {
        "pressure_surrogate_relative_rmse": relative_rmse(
            arrays["candidate_dpdx"], arrays["native_dpdx"]
        ),
        "pressure_surrogate_pearson": float(
            np.corrcoef(arrays["candidate_dpdx"], arrays["native_dpdx"])[0, 1]
        ),
        "dstar_full_over_resolved_median": float(
            np.median(arrays["dstar_full"] / arrays["dstar_resolved"])
        ),
        "dstar_full_over_resolved_min": float(
            np.min(arrays["dstar_full"] / arrays["dstar_resolved"])
        ),
        "dstar_full_over_resolved_max": float(
            np.max(arrays["dstar_full"] / arrays["dstar_resolved"])
        ),
        "closure_output_relative_rmse": relative_rmse(
            arrays["cf_candidate"], arrays["cf_calibration"]
        ),
        "closure_output_sign_changes": int(
            np.count_nonzero(
                np.sign(arrays["cf_candidate"])
                != np.sign(arrays["cf_calibration"])
            )
        ),
        "calibration_feature_cf_r2_all": r2_score(
            arrays["cf_reference"], arrays["cf_calibration"]
        ),
        "candidate_feature_cf_r2_all": r2_score(
            arrays["cf_reference"], arrays["cf_candidate"]
        ),
        "exported_forward_max_abs_difference": float(
            np.max(np.abs(arrays["cf_calibration"] - arrays["stored_cf_x2"]))
        ),
        "reference_cf_max_abs_difference": payload["metrics"][
            "reference_cf_max_abs_difference"
        ],
        "paired_local_mlp_cf_r2_all": r2_score(
            arrays["cf_reference"], arrays["stored_cf_mlp"]
        ),
        "paired_local_mlp_cf_r2_separated": r2_score(
            arrays["cf_reference"][arrays["cf_reference"] < 0],
            arrays["stored_cf_mlp"][arrays["cf_reference"] < 0],
        ),
        "frozen_x2_cf_r2_separated": r2_score(
            arrays["cf_reference"][arrays["cf_reference"] < 0],
            arrays["cf_calibration"][arrays["cf_reference"] < 0],
        ),
    }
    forward_calibration = frozen_forward(arrays["calibration_features"])
    forward_candidate = frozen_forward(arrays["candidate_features"])
    if not np.allclose(
        forward_calibration, arrays["cf_calibration"], rtol=0.0, atol=5e-13
    ):
        raise RuntimeError("retained calibration forward does not replay")
    if not np.allclose(
        forward_candidate, arrays["cf_candidate"], rtol=0.0, atol=5e-13
    ):
        raise RuntimeError("retained candidate forward does not replay")
    for name, value in recalculated.items():
        retained = payload["metrics"][name]
        if isinstance(value, int):
            if value != retained:
                raise RuntimeError(f"retained metric mismatch for {name}")
        elif not np.isclose(value, retained, rtol=0.0, atol=5e-13):
            raise RuntimeError(
                f"retained metric mismatch for {name}: {value} != {retained}"
            )
    if payload["frozen_header_contract"] != header_contract():
        raise RuntimeError("retained frozen-header contract does not match header")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--export-result", type=Path, default=DEFAULT_EXPORT_RESULT
    )
    parser.add_argument("--verify-retained", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.verify_retained:
        payload = verify_retained(output)
        print(
            "CLOSURE_INTERFACE_CUSTODY_PASS "
            f"status={payload['status']} n=10"
        )
        return
    payload = derive(args.profile, args.export_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "CLOSURE_INTERFACE_CUSTODY_DERIVED "
        f"output={output.relative_to(ROOT)} status={payload['status']}"
    )


if __name__ == "__main__":
    main()
