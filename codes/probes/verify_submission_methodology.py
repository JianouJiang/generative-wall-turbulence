#!/usr/bin/env python3
"""Independent, stable audit of the claim-matched submission methodology."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from pymech.neksuite import readre2


ROOT = Path(__file__).resolve().parents[2]
POINT_RESULT = ROOT / "manuscript/source_data/fig3/cube_periodic_topology_results.json"
POINT_COMPONENTS = (
    ROOT / "manuscript/source_data/fig3/cube_periodic_topology_components.npz"
)
RAPID_RESULT = (
    ROOT / "manuscript/source_data/fig4/cube_distributional_rapid_results.json"
)
RAPID_COMPONENTS = (
    ROOT / "manuscript/source_data/fig4/cube_distributional_rapid_components.npz"
)
PEER_REVIEW_AUDIT = (
    ROOT / "manuscript/source_data/review_audit/derived_peer_review_statistics.json"
)
ARMS_POINT = ("correct", "no_wall", "wrong_wall")
ARMS_RAPID = ("correct", "no_wall", "far_time_wall")
REGIONS = (
    "full_support_excluded",
    "near_support_excluded_d_le_0p5h",
    "outer_d_gt_0p5h",
)
METHODOLOGY = ROOT / "manuscript/sections/methods.tex"
PREREGISTRATION = (
    ROOT / "development/iteration_20260723T150839/nodes/node_006/preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "3f4bfdace6efb5d4162ca0b129dfb4d42077960010ead234be66edff745cc4de"
)
MESH_SHA256 = "09a68e5c763ba500c5dfe9d7281ef2d34557b1c4632707fd36a03ee4ce0b4128"
NATIVE_YPLUS_SHA256 = (
    "bddf92ed67c25250dced3569829603da4461abd9a8a001cf668bdad1c6bab54e"
)
NATIVE_YPLUS_ORIGIN_SHA256 = (
    "cb40ca68c6ad076681f236f6126149daee9a0f1f6efc917766d737beb5123bc1"
)
PRODUCTION_SHA256 = (
    "44c28d22481e67bce5202af558ede1dee19d2dd0d4dd31de09ec09799e910c73"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def isolated_source_function(path: Path, name: str):
    """Load one pure function from a producer without importing its run-time side effects."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def status_strings(value: object) -> list[str]:
    """Collect status-like strings without treating arbitrary provenance text as a claim."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if "status" in key.lower() and isinstance(child, str):
                found.append(child)
            found.extend(status_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(status_strings(child))
    return found


def block_draws(n: int, block: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, count))
    offsets = np.arange(block)[None, None, :]
    return ((starts[:, :, None] + offsets).reshape(draws, -1) % n)[:, :n]


def add_check(checks: dict, name: str, passed: bool, **details: object) -> None:
    checks[name] = {"pass": bool(passed), **details}


def add_close(
    checks: dict,
    name: str,
    actual: float,
    expected: float,
    tolerance: float = 2e-10,
) -> None:
    error = abs(float(actual) - float(expected))
    add_check(
        checks,
        name,
        error <= tolerance,
        actual=float(actual),
        expected=float(expected),
        abs_error=error,
        tolerance=tolerance,
    )


def verify_point(checks: dict) -> None:
    result = load(POINT_RESULT)
    data = np.load(POINT_COMPONENTS)
    producer = ROOT / "codes/gpu/eval_cube_periodic_topology.py"
    inherited = ROOT / "codes/gpu/eval_cube_3d_coupling_adequate.py"
    sampler = ROOT / "codes/gpu/eval_cube_3d_coupling.py"
    add_check(
        checks,
        "point_result_hash",
        sha256(POINT_RESULT)
        == "0fc302421d622879e00ef16a14636fb8849ee65c0680c85491659788807f914d",
        actual=sha256(POINT_RESULT),
    )
    add_check(
        checks,
        "point_components_hash",
        sha256(POINT_COMPONENTS)
        == "28264e996586f961b1a3cd8c369f494f62d39d4d18d73f0625fc4983e4ab3d18",
        actual=sha256(POINT_COMPONENTS),
    )
    inherited_text = inherited.read_text(encoding="utf-8")
    sampler_text = sampler.read_text(encoding="utf-8")
    add_check(
        checks,
        "point_producer_code_identity",
        result["_meta"]["producer_script_sha256"]
        == "df41b7d8524d42e0b3d33892bdcaf9bad636d1fb7961014527800b835926ed20"
        and sha256(producer)
        == "559f065d5ef3dc3e42ff85362059a037aec7f4ee434d3de3a7bff69e2cbd7ae7"
        and result["_meta"]["components_sha256"] == sha256(POINT_COMPONENTS),
        origin_producer_sha256=result["_meta"]["producer_script_sha256"],
        path_sanitized_release_producer_sha256=sha256(producer),
    )
    add_check(
        checks,
        "implemented_scalar_pooled_sst",
        "tbar = {k: float(tf[:, :, m].mean())" in inherited_text
        and "np.square(tf[:, :, m] - tbar[k])" in inherited_text,
    )
    add_check(
        checks,
        "implemented_explicit_euler_sampler",
        "v = model(x, tv, obs, cm, fluid_b)" in sampler_text
        and "x = (x + (tnext - t) * v) * fluid_b" in sampler_text,
    )
    add_check(
        checks,
        "point_index_contract",
        len(data["test_idx"]) == 160
        and np.array_equal(data["donor_idx"], np.roll(data["test_idx"], 80))
        and len(np.unique(data["sampler_seed"])) == 40
        and np.all(np.diff(data["sampler_seed"].reshape(40, 4), axis=1) == 0),
    )
    index = block_draws(160, result["evaluation"]["eval_block"], 4000, 44)
    replay: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
    for arm in ARMS_POINT:
        replay[arm] = {}
        for region in REGIONS:
            sst = data[f"sst__{region}"]
            sse = data[f"sse__{arm}__{region}"]
            point = 1.0 - sse.sum() / (sst.sum() + 1e-12)
            bootstrap = 1.0 - sse[index].sum(1) / (sst[index].sum(1) + 1e-12)
            replay[arm][region] = (float(point), bootstrap)
            stored = result["evaluation"]["arms"][arm][region]
            add_close(
                checks,
                f"point_{arm}_{region}",
                point,
                stored["R2_fluct_balanced"],
                1e-7,
            )
            for bound, percentile in enumerate((2.5, 97.5)):
                add_close(
                    checks,
                    f"point_{arm}_{region}_ci{bound}",
                    np.percentile(bootstrap, percentile),
                    stored["ci95"][bound],
                    1e-7,
                )
    for control in ("no_wall", "wrong_wall"):
        for region in REGIONS:
            point = replay["correct"][region][0] - replay[control][region][0]
            bootstrap = replay["correct"][region][1] - replay[control][region][1]
            stored = result["evaluation"]["deltas"][f"correct_minus_{control}"][region]
            add_close(checks, f"delta_{control}_{region}", point, stored["point"], 1e-7)
            for bound, percentile in enumerate((2.5, 97.5)):
                add_close(
                    checks,
                    f"delta_{control}_{region}_ci{bound}",
                    np.percentile(bootstrap, percentile),
                    stored["ci95"][bound],
                    1e-7,
                )
    outer = result["evaluation"]["arms"]["correct"]["outer_d_gt_0p5h"]
    add_check(
        checks,
        "point_claim_boundary",
        result["evaluation"]["arms"]["correct"]["full_support_excluded"][
            "R2_fluct_balanced"
        ]
        > 0
        and result["evaluation"]["arms"]["correct"][
            "near_support_excluded_d_le_0p5h"
        ]["R2_fluct_balanced"]
        > 0
        and outer["R2_fluct_balanced"] < 0
        and result["evaluation"]["deltas"]["correct_minus_no_wall"][
            "outer_d_gt_0p5h"
        ]["ci95"][0]
        > 0
        and result["evaluation"]["deltas"]["correct_minus_wrong_wall"][
            "outer_d_gt_0p5h"
        ]["ci95"][0]
        > 0,
    )
    add_check(
        checks,
        "historical_dependence_record",
        97.0 < result["_meta"]["tau_integral_snapshots"] < 98.0
        and result["evaluation"]["eval_block"] == 49
        and result["_meta"]["test_span_in_integral_times"] < 4.0
        and result["_meta"]["effective_independent_post_spinup"] < 12.0,
    )
    add_check(
        checks,
        "topology_ablation_scoped",
        result["_meta"]["single_changed_factor"] == "convolutional padding topology"
        and result["topology_gates"]["positive_outer_absolute_skill"] is False,
    )


def verify_les_release(checks: dict) -> None:
    mesh_path = ROOT / "codes/cube_les/cube.re2"
    par_path = ROOT / "codes/cube_les/cube_prod.par"
    generator_path = ROOT / "codes/cube_les/gen_cube_mesh.py"
    native_path = ROOT / "codes/results/cube3d_native_yplus_temporally_separated.json"
    production_path = ROOT / "codes/results/cube_production_complete.json"
    native = load(native_path)
    production = load(production_path)

    add_check(
        checks,
        "les_primary_artifact_hashes",
        sha256(mesh_path) == MESH_SHA256
        and sha256(native_path) == NATIVE_YPLUS_SHA256
        and sha256(production_path) == PRODUCTION_SHA256
        and production["mesh_sha256"] == MESH_SHA256
        and native["_meta"]["mesh_sha256"] == MESH_SHA256,
        mesh_sha256=sha256(mesh_path),
        native_yplus_sha256=sha256(native_path),
        production_sha256=sha256(production_path),
    )

    mesh = readre2(str(mesh_path))
    coordinates = np.concatenate(
        [
            np.stack(
                [
                    np.asarray(element.pos[0]).reshape(-1),
                    np.asarray(element.pos[1]).reshape(-1),
                    np.asarray(element.pos[2]).reshape(-1),
                ],
                axis=1,
            )
            for element in mesh.elem
        ],
        axis=0,
    )
    wall_faces = 0
    periodic_faces = 0
    unreciprocated = 0
    for element_index, element in enumerate(mesh.elem):
        for face_index, boundary in enumerate(element.bcs[0]):
            if boundary[0] == "W":
                wall_faces += 1
            elif boundary[0] == "P":
                periodic_faces += 1
                partner_element = int(boundary[3]) - 1
                partner_face = int(boundary[4]) - 1
                partner = mesh.elem[partner_element].bcs[0][partner_face]
                if not (
                    partner[0] == "P"
                    and int(partner[3]) - 1 == element_index
                    and int(partner[4]) - 1 == face_index
                ):
                    unreciprocated += 1
    add_check(
        checks,
        "les_mesh_semantics",
        mesh.nel == 14176
        and np.allclose(coordinates.min(axis=0), [0.0, 0.0, 0.0])
        and np.allclose(coordinates.max(axis=0), [2.0, 4.0, 2.0])
        and wall_faces == 2336
        and periodic_faces == 2464
        and unreciprocated == 0,
        elements=mesh.nel,
        bounds=[coordinates.min(axis=0).tolist(), coordinates.max(axis=0).tolist()],
        wall_faces=wall_faces,
        periodic_faces=periodic_faces,
        unreciprocated_periodic_faces=unreciprocated,
    )

    with tempfile.TemporaryDirectory(prefix="gwt-mesh-regeneration-") as directory:
        regenerated = subprocess.run(
            [sys.executable, str(generator_path)],
            cwd=directory,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        regenerated_mesh = Path(directory) / "cube.re2"
        regenerated_hash = sha256(regenerated_mesh) if regenerated_mesh.is_file() else None
    add_check(
        checks,
        "les_mesh_byte_regeneration",
        regenerated.returncode == 0
        and regenerated_hash == MESH_SHA256
        and "[check] OK" in regenerated.stdout,
        returncode=regenerated.returncode,
        regenerated_sha256=regenerated_hash,
        stderr=regenerated.stderr,
    )

    parameters = par_path.read_text(encoding="utf-8")
    required_parameters = (
        "polynomialOrder = 7",
        "cubaturePolynomialOrder = 9",
        "timeStepper = tombo2",
        "dt = targetCFL=2.0 + max=1.5e-3",
        "advectionSubCyclingSteps = 2",
        "checkpointInterval = 0.3",
        "constFlowRate = meanVelocity=1.0 + direction=X",
        "regularization = hpfrt + nModes=1 + scalingCoeff=10",
        "viscosity = 1/5000.0",
        "boundaryTypeMap = zeroDirichlet",
    )
    add_check(
        checks,
        "les_parameter_contract",
        sha256(par_path)
        == "99c9d69542b8ebcd72f075926d170b81bc32411b8379916d31a8b49a320a3e9d"
        and all(setting in parameters for setting in required_parameters),
        missing=[setting for setting in required_parameters if setting not in parameters],
    )

    pooled = native["pooled_two_terminal_fields"]
    maxima = {
        name: values["yplus"]["maximum"] for name, values in pooled.items()
    }
    add_check(
        checks,
        "native_yplus_temporal_audit",
        native["_meta"]["script_sha256"]
        == "6dbd7a4479728e6d42c76bfba468b43e7c97211da117dc3633663bae72dde65e"
        and native["_meta"]["producer_wrapper_sha256"]
        == "357fb5a4f7ea24fd26a42e06f381d185943aa5664465824d9200a6ace2800e6d"
        and native["_meta"]["origin_artifact_sha256_before_path_sanitization"]
        == NATIVE_YPLUS_ORIGIN_SHA256
        and native["_meta"]["origin_machine_paths_redacted"] is True
        and not Path(native["_meta"]["mesh"]).is_absolute()
        and all(
            not Path(path).is_absolute()
            for path in native["_meta"]["selected_field_paths"]
        )
        and native["_meta"][
            "retained_fields_historical_signal_specific_integral_time_separated"
        ]
        is True
        and native["_meta"][
            "retained_fields_current_conservative_integral_time_separated"
        ]
        is False
        and native["_meta"]["temporal_independence_claim"] is False
        and native["_meta"]["retained_time_separation"]
        > native["_meta"]["historical_signal_specific_integral_time"]
        and native["_meta"]["retained_time_separation"]
        < native["_meta"]["current_conservative_integral_time"]
        and abs(
            native["_meta"]["current_conservative_integral_time"]
            - 36.74451907473243
        )
        < 1e-12
        and abs(maxima["floor_between_cubes"] - 0.3163093575376479) < 1e-12
        and abs(maxima["cube_top"] - 1.336837937014454) < 1e-12
        and abs(maxima["cube_vertical_faces"] - 1.92125603598742) < 1e-12
        and max(maxima.values()) < 2.0
        and native["_meta"]["max_native_wall_speed"] == 0.0,
        field_times=native["_meta"]["times"],
        yplus_maxima=maxima,
        max_native_wall_speed=native["_meta"]["max_native_wall_speed"],
    )
    add_check(
        checks,
        "les_terminal_and_stationary_record",
        production["exit_code"] == 0
        and production["n_rasters"] == 1234
        and production["n_post_spinup"] == 1101
        and production["time_minmax_post_spinup"] == [40.20043129, 370.00048748]
        and POINT_RESULT.is_file()
        and abs(
            load(POINT_RESULT)["_meta"]["stationarity_energy_drift_middle_to_final"]
            - 0.0015475580402026837
        )
        < 1e-15,
    )


def rapid_values(data: np.lib.npyio.NpzFile, checks: dict) -> dict:
    values: dict[str, dict[str, np.ndarray]] = {
        "energy_score": {},
        "component_spectrum_log_rmse": {},
        "reynolds_stress_profile_nrmse": {},
        "coverage80_abs_error": {},
    }
    for arm in ARMS_RAPID:
        truth_term = data[f"energy_truth_distance__{arm}"].mean(axis=1)
        pair_term = data[f"energy_pair_distance__{arm}"].mean(axis=(1, 2))
        energy = truth_term - 0.5 * pair_term
        add_close(
            checks,
            f"rapid_energy_component_{arm}",
            np.max(np.abs(energy - data[f"energy_score__{arm}"])),
            0,
            2e-12,
        )
        values["energy_score"][arm] = energy

        spectrum = np.sqrt(
            np.mean(
                np.square(
                    np.log10(data[f"pred_spectrum__{arm}"] + 1e-12)
                    - np.log10(data["truth_spectrum"] + 1e-12)
                ),
                axis=(1, 2),
            )
        )
        add_close(
            checks,
            f"rapid_spectrum_component_{arm}",
            np.max(np.abs(spectrum - data[f"spectrum_log_rmse__{arm}"])),
            0,
            2e-12,
        )
        values["component_spectrum_log_rmse"][arm] = spectrum

        numerator = np.mean(
            np.square(
                data[f"pred_reynolds_profile__{arm}"] - data["truth_reynolds_profile"]
            ),
            axis=(1, 2),
        )
        denominator = np.mean(np.square(data["truth_reynolds_profile"]), axis=(1, 2))
        profile = np.sqrt(numerator / np.maximum(denominator, 1e-12))
        add_close(
            checks,
            f"rapid_profile_component_{arm}",
            np.max(np.abs(profile - data[f"reynolds_profile_nrmse__{arm}"])),
            0,
            2e-12,
        )
        values["reynolds_stress_profile_nrmse"][arm] = profile

        coverage = data[f"coverage80_hits__{arm}"] / data[f"coverage80_total__{arm}"]
        coverage_error = np.abs(coverage - 0.8)
        add_close(
            checks,
            f"rapid_coverage_component_{arm}",
            np.max(np.abs(coverage_error - data[f"coverage80_abs_error__{arm}"])),
            0,
            2e-12,
        )
        values["coverage80_abs_error"][arm] = coverage_error
    return values


def verify_rapid(checks: dict) -> None:
    result = load(RAPID_RESULT)
    data = np.load(RAPID_COMPONENTS)
    producer_path = ROOT / "codes/gpu/eval_cube_distributional_rapid.py"
    producer_text = producer_path.read_text(encoding="utf-8")
    add_check(
        checks,
        "rapid_result_hash",
        sha256(RAPID_RESULT)
        == "c70342f6fc4bf8e21f0883ba76de806f01eb8cfc34f9b0d52018443beba00588",
        actual=sha256(RAPID_RESULT),
    )
    add_check(
        checks,
        "rapid_components_hash",
        sha256(RAPID_COMPONENTS)
        == "a7f1c74f909e1a805d4a3642db83285371ecd80e4a59cd11a873a14253010e2d",
        actual=sha256(RAPID_COMPONENTS),
    )
    add_check(
        checks,
        "rapid_index_contract",
        len(data["test_idx"]) == 160
        and np.array_equal(data["donor_idx"], np.roll(data["test_idx"], 80))
        and np.array_equal(data["sampler_seed"], 910000 + np.arange(160)),
    )
    outer_y = data["outer_y_index"]
    add_check(
        checks,
        "spectral_staged_array_and_producer_axis_contract",
        np.array_equal(outer_y, np.arange(36, 84))
        and result["_meta"]["complete_outer_y_planes"] == 48
        and data["truth_spectrum"].shape == (160, 3, 12)
        and data["truth_reynolds_profile"].shape == (160, 6, 48)
        and "(1101, 3, 48, 96, 48)" in producer_text
        and "q = field_std[:, :, outer_y, :]" in producer_text
        and "np.fft.rfft(q, axis=1)" in producer_text
        and "(q.shape[1] ** 2)" in producer_text
        and "power.mean(axis=(2, 3))[:, 1:13]" in producer_text,
        model_axis_order="[time, component, x, y, z]",
        model_grid=[48, 96, 48],
        complete_wall_normal_plane_indices=outer_y.tolist(),
        truth_spectrum_shape=list(data["truth_spectrum"].shape),
        truth_reynolds_profile_shape=list(data["truth_reynolds_profile"].shape),
    )
    component_spectrum = isolated_source_function(producer_path, "component_spectrum")
    rng = np.random.default_rng(20260730)
    probe = rng.standard_normal((3, 48, 48, 48))
    actual_spectrum = component_spectrum(probe, np.arange(48))
    expected_spectrum = (
        np.abs(np.fft.rfft(probe, axis=1)) ** 2 / (48**2)
    ).mean(axis=(2, 3))[:, 1:13]
    add_check(
        checks,
        "spectral_producer_executable_semantics",
        actual_spectrum.shape == (3, 12)
        and np.max(np.abs(actual_spectrum - expected_spectrum)) <= 2e-14,
        transform_axis="x (axis 1 after component axis)",
        transform_length=48,
        power_normalization="48^2",
        averaged_axes="selected y planes and spanwise z",
        retained_modes="1--12",
        max_abs_error=float(np.max(np.abs(actual_spectrum - expected_spectrum))),
    )
    methods_text = METHODOLOGY.read_text(encoding="utf-8")
    methods_norm = re.sub(r"\s+", " ", methods_text)
    add_check(
        checks,
        "spectral_methods_match_producer",
        "48-point streamwise transforms" in methods_norm
        and "48 complete" in methods_norm
        and "modes 1--12" in methods_norm
        and "96-point real Fourier" not in methods_text
        and "$N_x^2=96^2$" not in methods_text
        and "divide squared amplitudes by $96^2$" not in methods_text,
    )
    values = rapid_values(data, checks)
    meta = result["_meta"]["bootstrap"]
    index = block_draws(
        len(data["test_idx"]),
        meta["block_evaluation_samples"],
        meta["draws"],
        meta["seed"],
    )
    stored_metrics = {
        "energy_score": result["primary_energy_score"],
        **result["secondary_physical_statistics"],
    }
    wins: dict[str, bool] = {}
    for metric, arms in values.items():
        stored = stored_metrics[metric]
        for arm in ARMS_RAPID:
            bootstrap = arms[arm][index].mean(1)
            add_close(checks, f"rapid_{metric}_{arm}", arms[arm].mean(), stored["arms"][arm]["mean_loss"])
            for bound, percentile in enumerate((2.5, 97.5)):
                add_close(
                    checks,
                    f"rapid_{metric}_{arm}_ci{bound}",
                    np.percentile(bootstrap, percentile),
                    stored["arms"][arm]["ci95"][bound],
                )
        control_wins = []
        for control in ("no_wall", "far_time_wall"):
            difference = arms[control] - arms["correct"]
            bootstrap = difference[index].mean(1)
            key = f"correct_vs_{control}"
            add_close(
                checks,
                f"rapid_{metric}_{key}",
                difference.mean(),
                stored["improvements"][key]["control_minus_correct"],
            )
            for bound, percentile in enumerate((2.5, 97.5)):
                add_close(
                    checks,
                    f"rapid_{metric}_{key}_ci{bound}",
                    np.percentile(bootstrap, percentile),
                    stored["improvements"][key]["ci95"][bound],
                )
            control_wins.append(np.percentile(bootstrap, 2.5) > 0)
        wins[metric] = all(control_wins)
    valid_physical = {
        key: wins[key]
        for key in (
            "component_spectrum_log_rmse",
            "reynolds_stress_profile_nrmse",
        )
    }
    gates = result["registered_gates"]
    add_check(
        checks,
        "rapid_historical_gate_replay_and_coverage_withdrawal",
        wins["energy_score"] is True
        and valid_physical
        == {
            key: gates["physical_family_wins"][key]
            for key in valid_physical
        }
        and sum(valid_physical.values()) == 0
        and gates["physical_family_wins"]["coverage80_abs_error"] is False
        and gates["rapid_distributional_claim_pass"] is False,
        valid_physical_wins=valid_physical,
        withdrawn_family="coverage80_abs_error",
    )
    # Retain the raw hit fraction while explicitly withdrawing calibration
    # against 0.8 for this M=8 interpolated sample-quantile estimator.
    correct_coverage = (
        data["coverage80_hits__correct"].sum()
        / data["coverage80_total__correct"].sum()
    )
    add_close(
        checks,
        "coverage_correct_raw",
        correct_coverage,
        0.5140762648935093,
    )
    add_check(
        checks,
        "coverage_hit_fraction_descriptive_only",
        abs(correct_coverage - 0.5140762648935093) < 2e-12
        and result["_meta"]["members"] == 8
        and valid_physical == {
            "component_spectrum_log_rmse": False,
            "reynolds_stress_profile_nrmse": False,
        },
        observed=float(correct_coverage),
        ensemble_members=result["_meta"]["members"],
        finite_ensemble_calibration_target=None,
    )
    add_check(
        checks,
        "rapid_claim_boundary",
        result["claim_boundary"]["terminal_outer_posterior_mean_R2"] < 0
        and result["coverage_descriptives"]["correct"]["coverage80"] < 0.55
        and result["_meta"]["effective_test_events_approx"] < 4,
    )

    physical57 = load(PEER_REVIEW_AUDIT)["physical_statistics"][
        "block57_conservative"
    ]
    physical57_ok = True
    physical57_details: dict[str, object] = {}
    index57 = block_draws(160, 57, 4000, 20260723)
    for metric in (
        "component_spectrum_log_rmse",
        "reynolds_stress_profile_nrmse",
    ):
        metric_details: dict[str, object] = {}
        for arm in ARMS_RAPID:
            arm_values = values[metric][arm]
            actual_ci = np.percentile(
                arm_values[index57].mean(axis=1), (2.5, 97.5)
            )
            stored = physical57[metric]["arms"][arm]
            physical57_ok &= abs(arm_values.mean() - stored["mean"]) < 2e-12
            physical57_ok &= np.max(np.abs(actual_ci - stored["ci95"])) < 2e-12
            metric_details[arm] = {
                "mean": float(arm_values.mean()),
                "ci95": actual_ci.tolist(),
            }
        physical57_details[metric] = metric_details
    for arm in ARMS_RAPID:
        coverage = data[f"coverage80__{arm}"]
        actual_ci = np.percentile(coverage[index57].mean(axis=1), (2.5, 97.5))
        stored = physical57["coverage80"]["arms"][arm]
        physical57_ok &= abs(coverage.mean() - stored["mean"]) < 2e-12
        physical57_ok &= np.max(np.abs(actual_ci - stored["ci95"])) < 2e-12
    add_check(
        checks,
        "physical_statistics_block57_replay",
        physical57_ok,
        details=physical57_details,
        correct_coverage=physical57["coverage80"]["arms"]["correct"],
    )

    audit = load(PEER_REVIEW_AUDIT)["fair_energy_score"]["block57_conservative"]
    fair_values: dict[str, np.ndarray] = {}
    for arm in ARMS_RAPID:
        truth = data[f"energy_truth_distance__{arm}"].mean(axis=1)
        pairs = data[f"energy_pair_distance__{arm}"]
        members = pairs.shape[1]
        fair_values[arm] = truth - pairs.sum(axis=(1, 2)) / (
            2.0 * members * (members - 1)
        )
    fair_index = block_draws(160, 57, 4000, 20260723)
    fair_ok = True
    details: dict[str, object] = {}
    for arm in ARMS_RAPID:
        bootstrap = fair_values[arm][fair_index].mean(axis=1)
        stored = audit["arms"][arm]
        actual_ci = np.percentile(bootstrap, (2.5, 97.5))
        fair_ok &= abs(fair_values[arm].mean() - stored["mean"]) < 2e-12
        fair_ok &= np.max(np.abs(actual_ci - stored["ci95"])) < 2e-12
        details[arm] = {
            "mean": float(fair_values[arm].mean()),
            "ci95": actual_ci.tolist(),
        }
    for control in ("no_wall", "far_time_wall"):
        difference = fair_values[control] - fair_values["correct"]
        bootstrap = difference[fair_index].mean(axis=1)
        stored = audit["improvements"][f"{control}_minus_correct"]
        actual_ci = np.percentile(bootstrap, (2.5, 97.5))
        fair_ok &= abs(difference.mean() - stored["mean"]) < 2e-12
        fair_ok &= np.max(np.abs(actual_ci - stored["ci95"])) < 2e-12
        fair_ok &= actual_ci[0] > 0
        details[f"{control}_minus_correct"] = {
            "mean": float(difference.mean()),
            "ci95": actual_ci.tolist(),
        }
    add_check(
        checks,
        "fair_energy_u_statistic_block57",
        fair_ok,
        values=details,
    )


def verify_sources_and_text(checks: dict) -> None:
    m0_path = ROOT / "manuscript/source_data/review_audit/cube3d_coupling_results.json"
    m1_path = ROOT / "manuscript/source_data/review_audit/cube3d_coupling_adequate_results.json"
    m0 = load(m0_path)
    m1 = load(m1_path)
    add_check(
        checks,
        "cube_sequential_contact_artifacts",
        sha256(m0_path)
        == "61e7d09d297bd6fa6a5aa8d55476e1c2a6fe56ddda0e7efe93b7eab25c3ce2a2"
        and sha256(m1_path)
        == "6762e9a89855c38538463c217698dab0e33c587a00d2fac1f937cf5ce35e982b"
        and abs(
            m0["evaluation"]["deltas"]["correct_minus_no_wall"][
                "full_support_excluded"
            ]["point"]
            - 0.17548029930792042
        )
        < 1e-12
        and abs(
            m1["evaluation"]["deltas"]["correct_minus_no_wall"][
                "outer_d_gt_0p5h"
            ]["ci95"][0]
            - (-0.002318719540220954)
        )
        < 1e-12,
    )
    retrieval_result_path = (
        ROOT
        / "manuscript/source_data/review_audit/"
        "cube_train_band_retrieval_results.json"
    )
    retrieval_components_path = (
        ROOT
        / "manuscript/source_data/review_audit/"
        "cube_train_band_retrieval_components.npz"
    )
    retrieval_result = load(retrieval_result_path)
    retrieval_components = np.load(retrieval_components_path)
    retrieval_draws = block_draws(160, 57, 4000, 20260730)
    retrieval_ok = (
        sha256(retrieval_result_path)
        == "94fa34214b23c2ae2ed046e846ba84bbed642ce4c90618be04417c3e82d5674f"
        and sha256(retrieval_components_path)
        == "02b412daeaf5314c237291534c7c995876c66c81794ac07f19327eea93fec0ce"
        and np.array_equal(retrieval_components["eval_idx"], np.load(POINT_COMPONENTS)["test_idx"])
        and retrieval_result["nearest_training_band"]["lag_snapshots"]["minimum"] == 108
        and retrieval_result["nearest_training_band"]["lag_snapshots"]["median"] == 483.0
    )
    retrieval_replay: dict[str, object] = {}
    for region in REGIONS:
        sst = retrieval_components[f"sst__{region}"]
        sse = retrieval_components[f"sse__nearest_training_band__{region}"]
        point = float(1.0 - sse.sum() / sst.sum())
        bootstrap = 1.0 - sse[retrieval_draws].sum(axis=1) / sst[
            retrieval_draws
        ].sum(axis=1)
        interval = np.percentile(bootstrap, (2.5, 97.5))
        stored = retrieval_result["nearest_training_band"]["block57_sensitivity"][
            region
        ]
        retrieval_ok &= abs(point - stored["point"]) < 2e-12
        retrieval_ok &= np.max(
            np.abs(interval - stored["conditional_interval"])
        ) < 2e-12
        retrieval_ok &= point < 0
        retrieval_replay[region] = {
            "point": point,
            "conditional_interval": interval.tolist(),
        }
    add_check(
        checks,
        "frozen_train_band_retrieval_replay",
        retrieval_ok,
        regions=retrieval_replay,
        interpretation=(
            "rejects literal nearest-record retrieval; does not establish independence"
        ),
    )
    standalone_env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="gwt-source-audit-") as directory:
        staged_source = Path(directory) / "source_data"
        standalone_env["GWT_SOURCE_DATA"] = str(staged_source)
        builder = subprocess.run(
            [sys.executable, "codes/build_submission_source_data.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=standalone_env,
        )
        add_check(
            checks,
            "source_data_staged_build",
            builder.returncode == 0 and "SOURCE_DATA_BUILD_PASS" in builder.stdout,
            stdout=builder.stdout.replace(
                str(staged_source), "<STAGED_SOURCE_DATA>"
            ),
            stderr=builder.stderr,
        )
        staged_files = sorted(
            path.relative_to(staged_source)
            for path in staged_source.rglob("*")
            if path.is_file()
        )
        add_check(
            checks,
            "source_data_staged_completeness",
            # 2026-08-04: 62 -> 63.  The corrected Fig. 11f plots the paired
            # positive-control contrasts for BOTH generative families, so the
            # audit that derives them is now a staged source artifact
            # (fig11_reachgated/hill_positive_control_audit.json, pinned in
            # FIGURE_DATA_SHA256SUMS).
            #
            # 2026-08-04 (peer-review round 2): 63 -> 71, after the manuscript
            # tree and the release checkout were consolidated into one root.
            # The consolidation surfaced staged custody artifacts that were
            # physically present under manuscript/source_data/fig_slot but had no
            # pinned hash, so the builder had been aborting on the first of them:
            #     PREREG_SNAPSHOT_v1.md, PREREG_SNAPSHOT_v1a_postA1.md,
            #     PREREG_SNAPSHOT_v2.md, apply_decision_rule_node011.py,
            #     case2_contact_ledger.json, eval_e2_slot_channel.py,
            #     e2_slot_channel_final_results.sha256, fig14_slot_interface.py
            # Five of these carry a digest identical to the one the node_011
            # preregistration custody chain (fig_slot/PREREG_HASHES.txt) froze
            # before any producer outcome existed; the .sha256 sidecar declares
            # the digest of the live results file and matches it; the staged
            # fig14 artist is a snapshot that differs from the current release
            # script and is recorded as such rather than silently re-staged.
            #
            # 2026-08-05: 71 -> 72.  Fig. 11f now draws the hierarchical
            # seed-then-time interval Methods declares and the Results text
            # quotes, instead of the audit file's narrower seed-conditional
            # ci95, so hill_hierarchical_intervals.json became a staged source
            # artifact (fig11_reachgated/, pinned in FIGURE_DATA_SHA256SUMS).
            # The count is a tripwire, not the guarantee: the guarantee is that
            # build_submission_source_data.py raises on ANY staged file without
            # a pinned hash, so all 72 are hash-pinned by construction.
            len(staged_files) == 72
            and Path("methods/cube_les_provenance.json") in staged_files
            and Path("methods/cube_production_complete.json") in staged_files
            and Path("methods/cube3d_native_yplus_temporally_separated.json")
            in staged_files
            and not any(str(path).startswith("fig5/") for path in staged_files)
            and Path("current_fig5/l2_case1_closure_results.json") in staged_files
            and Path("current_fig5/wallstress_cond_case1_bh1_results.json")
            in staged_files
            and Path("current_fig5/SOURCE_INVALID_evaluated_bridge_rendering.png")
            in staged_files
            and Path("current_fig5/INVALIDATED.json") in staged_files
            and Path("fig7/capacity_crossed_adjudication.json") in staged_files
            and Path("review_audit/derived_peer_review_statistics.json")
            in staged_files
            and Path("review_audit/legacy_metadata_supersession.json")
            in staged_files
            and Path("review_audit/publication_provenance_facts.json")
            in staged_files
            and Path("review_audit/publication_quantitative_contract.json")
            in staged_files
            and Path("review_audit/publication_semantic_contract.json")
            in staged_files
            and Path("review_audit/legacy_success_gate_disposition.json")
            in staged_files
            and Path("review_audit/cube_train_band_retrieval_results.json")
            in staged_files
            and Path("review_audit/cube_train_band_retrieval_components.npz")
            in staged_files
            and Path("review_audit/case1_source_native_audit.json") in staged_files
            and Path("review_audit/closure_interface_custody.json") in staged_files
            and Path("diagnostic_e2/runtime_closure_conditioning_results.json")
            in staged_files
            and Path("diagnostic_e2/causal_wall_history_results.json")
            in staged_files,
            files=[str(path) for path in staged_files],
        )

    audit = load(PEER_REVIEW_AUDIT)
    dependence = audit["dependence"]
    comparator = audit["committed_comparators"]
    quarantined_location = comparator["quarantined_location_comparator"]
    local_assets = audit["local_reviewer_assets"]
    add_check(
        checks,
        "peer_review_dependence_and_comparator_audit",
        abs(dependence["conservative_tau_snapshots"] - 122.48173024910811)
        < 1e-12
        and abs(
            dependence["evaluation_span_in_conservative_integral_times"]
            - 2.800417656595749
        )
        < 1e-12
        and dependence["conservative_block_evaluation_samples"] == 57
        and "equal_support_interior" not in comparator["linear_wiener"]
        and quarantined_location["active_claim"] is False
        and quarantined_location["complete_overlap_cells"] == 13984
        and quarantined_location["near_overlap_cells"] == 9112
        and quarantined_location["outer_overlap_cells"] == 4872
        and len(comparator["deterministic_direct_correct_band"]) == 2
        and all(item["included_in_compact_release"] is False for item in local_assets.values())
        and all(
            item["external_reviewer_access_route"] == "requires author confirmation"
            for item in local_assets.values()
        ),
    )

    les_provenance = load(
        ROOT / "manuscript/source_data/methods/cube_les_provenance.json"
    )
    add_check(
        checks,
        "source_data_les_provenance",
        les_provenance["schema"] == "gwt-cube-les-release-provenance-v1"
        and les_provenance["case_files"]["mesh"]["sha256"] == MESH_SHA256
        and les_provenance["mesh_semantics"]["elements"] == 14176
        and les_provenance["mesh_semantics"]["unreciprocated_periodic_faces"] == 0
        and les_provenance["native_yplus"]["sha256"] == NATIVE_YPLUS_SHA256
        and les_provenance["native_yplus"][
            "origin_sha256_before_path_sanitization"
        ]
        == NATIVE_YPLUS_ORIGIN_SHA256
        and les_provenance["native_yplus"]["origin_machine_paths_redacted"] is True
        and les_provenance["production"]["sha256"] == PRODUCTION_SHA256,
    )
    grouped = load(ROOT / "manuscript/source_data/fig6/grouped_hill_supporting.json")
    add_check(
        checks,
        "grouped_hill_honesty",
        grouped["diffusion"]["arms"]["correct"] < 0
        and grouped["flow_matching"]["bootstrap"]["d_correct_minus_no_wall"]["ci95"][1] < 0
        and grouped["split"]["planes_grouped_by_physical_time"],
    )
    capacity = load(ROOT / "manuscript/source_data/fig7/capacity_crossed_adjudication.json")
    add_check(
        checks,
        "capacity_negative_boundary",
        capacity["gates"]["scaling_claim_pass"] is False
        and capacity["gates"]["convergence_parity"] is False
        and capacity["crossed_capacity_slopes"]["correct"]["crossed_ci95"][1] < 0,
    )
    e2_result_path = (
        ROOT
        / "manuscript/source_data/diagnostic_e2/wallstress_cond_case1_bh1_results.json"
    )
    e2_input_path = (
        ROOT / "manuscript/source_data/diagnostic_e2/case1_closure_inputs.json"
    )
    runtime_e2_path = (
        ROOT
        / "manuscript/source_data/diagnostic_e2/runtime_closure_conditioning_results.json"
    )
    archived_e2_path = (
        ROOT
        / "manuscript/source_data/diagnostic_e2/causal_wall_history_results.json"
    )
    e2 = load(e2_result_path)
    runtime_e2 = load(runtime_e2_path)
    archived_e2 = load(archived_e2_path)
    case1_audit_path = (
        ROOT / "manuscript/source_data/review_audit/case1_source_native_audit.json"
    )
    case1_audit = load(case1_audit_path)
    add_check(
        checks,
        "case1_source_native_withdrawal",
        sha256(case1_audit_path)
        == "cec89f247e990462ce48b247c2659dde5aba9de0688cc831b1e98f564c883898"
        and case1_audit["status"] == "source_invalid_withdrawn"
        and case1_audit["all_checks_pass"] is True
        and case1_audit["raw_arrays"]["coords"]["shape"] == [918, 2]
        and case1_audit["raw_arrays"]["data"]["shape"] == [16001, 918, 3]
        and case1_audit["derived_custody"]["raster_obstacle_cells"] == 0,
        raw_shapes=[
            case1_audit["raw_arrays"]["coords"]["shape"],
            case1_audit["raw_arrays"]["data"]["shape"],
        ],
        finding=(
            "local Case1 geometry, height, Reynolds number, viscosity and stress "
            "are unsupported; manuscript withdraws E2"
        ),
    )
    add_check(
        checks,
        "e2_offline_boundary_and_controls",
        sha256(e2_result_path)
        == "f058a9990ebd2e02b059b6629c4aa2d3cc5c77594e6d4a27491b9e895bcb1073"
        and sha256(e2_input_path)
        == "7474a8ed7277e77ca4cd900bf7c008553e893d7e1ed77f23d2cfe69802fde3fa"
        and e2["families"]["diffusion"]["boot_total"][
            "d_closure_minus_random"
        ]["ci95"][0]
        < 0
        and e2["families"]["diffusion"]["boot_total"][
            "d_closure_minus_wrong_mean"
        ]["ci95"][1]
        < 0
        and e2["families"]["flow_matching"]["boot_total"][
            "d_closure_minus_wrong_mean"
        ]["ci95"][0]
        < 0,
    )
    add_check(
        checks,
        "e2_runtime_software_test_bounded",
        sha256(runtime_e2_path)
        == "d96b352e146f23f94bd16011337051c7993bea15b8ead7d638b96496163e6457"
        and runtime_e2["_meta"]["dev"] == "cpu"
        and runtime_e2["_meta"]["n_test"] == 6
        and abs(runtime_e2["_meta"]["n_eff"] - 1.0 / 6.0) < 1e-15
        and runtime_e2["_meta"]["L"] == 2
        and runtime_e2["_meta"]["steps"] == 6
        and runtime_e2["_gates"]["solver_coupled_claim"] is False,
    )
    add_check(
        checks,
        "e2_archived_adequacy_failure_disclosed",
        sha256(archived_e2_path)
        == "0eb97c78b80d569b067dfec58f4ee4d9213335b57395eba1d2f7fadf819cf641"
        and archived_e2["_meta"]["n_test_times"] == 370
        and abs(archived_e2["_meta"]["n_eff_clustered"] - 3.245614035087719)
        < 1e-15
        and archived_e2["_gates"]["closure_beats_no_wall_both"] is False
        and archived_e2["_gates"]["history_load_bearing_any"] is False
        and archived_e2["families"]["flow_matching"]["arms"]["correct_history"][
            "R2_total"
        ]
        < archived_e2["families"]["flow_matching"]["arms"]["no_wall"]["R2_total"]
        and all(
            family["arms"]["closure_history"]["R2_fluct"] < 0
            for family in archived_e2["families"].values()
        ),
    )

    manuscript_paths = [
        ROOT / "manuscript/main.tex",
        ROOT / "manuscript/sections/introduction.tex",
        ROOT / "manuscript/sections/results.tex",
        ROOT / "manuscript/sections/discussion.tex",
        ROOT / "manuscript/sections/methods.tex",
    ]
    manuscript = "\n".join(path.read_text(encoding="utf-8") for path in manuscript_paths)
    methodology = METHODOLOGY.read_text(encoding="utf-8")
    combined = (
        manuscript
        + "\n"
        + methodology
        + "\n"
        + (ROOT / "manuscript/supplementary.tex").read_text(encoding="utf-8")
    )
    normal = np.array([0.0, 1.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])
    projector = np.eye(3) - np.outer(normal, normal)
    viscosity = 0.2

    def wall_on_fluid_traction(shear: float) -> np.ndarray:
        gradient = np.zeros((3, 3))
        gradient[0, 1] = shear
        return -projector @ (viscosity * (gradient + gradient.T) @ normal)

    forward_traction = wall_on_fluid_traction(1.0)
    reversed_traction = wall_on_fluid_traction(-1.0)
    add_check(
        checks,
        "traction_canonical_planar_sign",
        np.allclose(forward_traction, [-viscosity, 0.0, 0.0])
        and np.allclose(reversed_traction, [viscosity, 0.0, 0.0])
        and np.dot(forward_traction, tangent) < 0
        and np.dot(reversed_traction, tangent) > 0
        and np.dot(forward_traction, normal) == 0,
        normal_convention="solid-to-fluid",
        traction_convention="wall-on-fluid",
        positive_lower_wall_shear=forward_traction.tolist(),
        reversed_lower_wall_shear=reversed_traction.tolist(),
    )
    compact_methodology = re.sub(r"\s+", "", methodology)
    traction_sync_paths = (
        "manuscript/sections/discussion.tex",
        "manuscript/supplementary.tex",
        "manuscript/source_data/README.md",
    )
    traction_sync = {
        path: "wall-on-fluid" in (ROOT / path).read_text(encoding="utf-8")
        for path in traction_sync_paths
    }
    add_check(
        checks,
        "traction_text_equation_and_package_sync",
        "point from the solid into the fluid" in methodology
        and r"=-\mathbfP_t\!\left[\nu(" in compact_methodology
        and r"\tau_1^{\,\mathrm{W\toF}}=-\nu" in compact_methodology
        and "fluid-on-wall traction" in methodology
        and "wall-on-fluid traction" in manuscript
        and all(traction_sync.values()),
        synchronized_paths=traction_sync,
    )
    add_check(
        checks,
        "traction_basis_is_geometry_anchored",
        "geometry-anchored unit tangent fixed independently of the"
        in re.sub(r"\s+", " ", methodology)
        and "so flow reversal changes the signed components"
        in re.sub(r"\s+", " ", methodology)
        and "unit resolved tangential-flow direction" not in methodology,
    )
    supplement_text = (ROOT / "manuscript/supplementary.tex").read_text(encoding="utf-8")
    add_check(
        checks,
        "wall_band_threshold_and_nominal_thickness",
        abs(2.01 * (4.0 / 96.0) - 0.08375) < 1e-15
        and abs(2.0 * (4.0 / 96.0) - 0.08333333333333333) < 1e-15
        and "2.01(4/96)h=0.08375h" in methodology
        and "2(4/96)h=0.08333h" in methodology
        and "2.01(4/96)h=0.08375h" in supplement_text
        and "2(4/96)h=0.08333h" in supplement_text,
        numerical_mask_threshold_h=2.01 * (4.0 / 96.0),
        nominal_two_cell_thickness_h=2.0 * (4.0 / 96.0),
    )
    required = (
        "0.10308",
        "0.17548",
        "0.30688",
        "0.12049",
        "0.28386",
        "-0.04678",
        "122.4817",
        "2.800",
        "57 evaluation",
        "first-contact comparison",
        "model-development sensitivity",
        "post-selection",
        "independent confirmatory test",
        "oracle wall-band propagation",
        "derived from the target time mean",
        "runtime software",
        "solver-coupled",
        "explicit Euler",
        "pools standardised components, locations and times",
        "51.4",
        "not a finite-ensemble-valid calibration test",
        "14,176",
        "TOMBO2",
        "1.92126",
        "native first-cell audit",
        "equal-support interior",
        "off-diagonal U-statistic",
        "causal runtime composition",
    )
    normalized_combined = re.sub(r"\s+", " ", combined)
    for phrase in required:
        add_check(
            checks,
            f"text_required_{hashlib.md5(phrase.encode()).hexdigest()[:8]}",
            phrase.lower() in normalized_combined.lower(),
            phrase=phrase,
        )
    forbidden_patterns = {
        "legacy_23p6": r"23\.6\\s*(?:x|\\\\times)",
        "latent_architecture": r"latent diffusion",
        "phase_selection": r"phase selection",
        "topology_law": r"topology law",
        "registered_breakthrough": r"registered distributional breakthrough",
        "backward_euler": r"backward[- ]Euler",
        "causal_question": r"causal question",
    }
    # The methodology may name prohibited interpretations only to negate them. Static
    # claim checks are therefore applied to the manuscript, where submitted claims live.
    for name, pattern in forbidden_patterns.items():
        add_check(
            checks,
            f"manuscript_forbidden_{name}",
            re.search(pattern, manuscript, flags=re.IGNORECASE) is None,
            pattern=pattern,
        )
    quarantine_claim_paths = (
        "manuscript/main.tex",
        "manuscript/supplementary.tex",
        "manuscript/cover_letter.md",
        "manuscript/nature_communications_checklist.md",
        "manuscript/source_data/README.md",
        "codes/build_submission_source_data.py",
        "codes/figures/fig5_physical_composite.py",
        "codes/release_manifest.txt",
        "SUBMISSION_RELEASE_MANIFEST.json",
    )
    quarantine_claim_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8") for path in quarantine_claim_paths
    )
    forbidden_exact = (
        "0.61667",
        "0.6166723436748704",
        "0.53176",
        "0.5317590917967943",
    )
    add_check(
        checks,
        "quarantine_exact_values_absent_from_active_claims",
        all(value not in quarantine_claim_text for value in forbidden_exact),
        scanned=list(quarantine_claim_paths),
        forbidden=list(forbidden_exact),
    )
    add_check(
        checks,
        "provisional_origin_absent_from_active_claims",
        "node002_full_logo_harness" not in quarantine_claim_text,
        scanned=list(quarantine_claim_paths),
    )
    add_check(
        checks,
        "point_registration_wording_bounded",
        "registered point endpoint" not in manuscript.lower()
        and "registered matched differences" not in manuscript.lower(),
    )
    active_coverage_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "manuscript/main.tex",
            "manuscript/supplementary.tex",
            "manuscript/source_data/README.md",
            "codes/figures/fig4_mechanism_robustness.py",
            "codes/figures/fig5_physical_composite.py",
        )
    )
    invalid_coverage_patterns = {
        "invalid_62p2_reference": r"62\.2",
        "invalid_28_over_45": r"28\s*/\s*45|28\s*\\over\s*45",
        "invalid_finite_ensemble_correction": (
            r"finite[- ]ensemble[- ](?:adjusted|corrected|calibrated)"
        ),
    }
    for name, pattern in invalid_coverage_patterns.items():
        add_check(
            checks,
            name,
            re.search(pattern, active_coverage_text, flags=re.IGNORECASE) is None,
            pattern=pattern,
        )
    editorial = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "manuscript/cover_letter.md",
            "manuscript/nature_communications_checklist.md",
        )
    )
    add_check(
        checks,
        "editorial_check_count_not_stale",
        "148-check" not in editorial
        and "148/148" not in editorial
        and "174-check" not in editorial
        and "174/174" not in editorial,
    )
    supplement = (ROOT / "manuscript/supplementary.tex").read_text(encoding="utf-8")
    includes = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}.]+)",
        manuscript + "\n" + supplement,
    )
    expected = [
        "fig1_architecture",
        "fig2_generation",
        "fig3_propagation",
        "fig6_interface",
        "fig7_direct_traction",
        "fig8_closure_composition",
        "fig11_reachgated_composition",
        "fig14_slot_interface",
    ]
    add_check(
        checks,
        "active_figure_contract",
        manuscript.count(r"\begin{figure") == 8
        and supplement.count(r"\begin{figure") == 0
        and includes == expected,
        includes=includes,
    )
    fig4_text = (ROOT / "codes/figures/fig4_mechanism_robustness.py").read_text(
        encoding="utf-8"
    )
    fig3_text = (ROOT / "codes/figures/fig3_propagation.py").read_text(
        encoding="utf-8"
    )
    fig5_text = (ROOT / "codes/figures/fig5_physical_composite.py").read_text(
        encoding="utf-8"
    )
    add_check(
        checks,
        "figure_statistics_visual_contract",
        ".bar(" not in fig4_text
        and ".bar(" not in fig3_text
        and 'audit["physical_statistics"]["block57_conservative"]' in fig5_text
        and "coverage_block" in fig5_text
        and "ax.axhline(0.8" not in fig5_text
        and fig5_text.count("ax.errorbar(") >= 2,
        figure4="dot-and-interval display; absolute-score ordinate starts at zero",
        figure3="terminal aggregates shown as points without precision-implying bars",
        figure5="block-57 intervals; no invalid finite-M calibration line",
    )
    labels = re.findall(r"\\label\{([^}]+)\}", manuscript)
    duplicate_labels = sorted(
        label for label in set(labels) if labels.count(label) > 1
    )
    add_check(
        checks,
        "latex_source_labels_unique",
        not duplicate_labels,
        duplicates=duplicate_labels,
    )


def pdf_info(path: Path) -> tuple[int, float, float]:
    completed = subprocess.run(
        ["pdfinfo", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    pages_match = re.search(r"^Pages:\s+(\d+)", completed.stdout, re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts",
        completed.stdout,
        re.MULTILINE,
    )
    if not pages_match or not size_match:
        raise RuntimeError(f"cannot parse pdfinfo for {path}")
    return (
        int(pages_match.group(1)),
        float(size_match.group(1)),
        float(size_match.group(2)),
    )


def verify_release_package(checks: dict) -> None:
    rapid = load(RAPID_RESULT)
    add_check(
        checks,
        "preregistration_restored_and_pinned",
        PREREGISTRATION.is_file()
        and sha256(PREREGISTRATION) == PREREGISTRATION_SHA256
        and rapid["_meta"]["preregistration"].endswith(
            "development/nodes/node_006/preregistration.md"
        ),
        actual_sha256=sha256(PREREGISTRATION) if PREREGISTRATION.is_file() else None,
    )
    add_check(
        checks,
        "rapid_producer_code_identity",
        rapid["_meta"]["producer_sha256"]
        == "bee8fb78eb4fad2ac57aa244f716d2a6430406aab3e447b3b61df8c55fd169b8"
        and sha256(ROOT / "codes/gpu/eval_cube_distributional_rapid.py")
        == "464b52df572c136fe23c60b19a67de7f9564c16bbd81d0c97a82777d7460958f",
        origin_producer_sha256=rapid["_meta"]["producer_sha256"],
        path_sanitized_release_producer_sha256=sha256(
            ROOT / "codes/gpu/eval_cube_distributional_rapid.py"
        ),
    )

    figure_geometry: dict[str, dict[str, object]] = {}
    geometry_ok = True
    for stem in (
        "fig1_architecture",
        "fig2_generation",
        "fig3_propagation",
        "fig6_interface",
        "fig7_direct_traction",
        "fig8_closure_composition",
        "fig11_reachgated_composition",
        "fig14_slot_interface",
    ):
        png = ROOT / "manuscript/figures" / f"{stem}.png"
        pdf = ROOT / "manuscript/figures" / f"{stem}.pdf"
        with Image.open(png) as image:
            rgb = image.convert("RGB")
            width_px, height_px = rgb.size
            content_bounds = ImageChops.difference(
                rgb, Image.new("RGB", rgb.size, color=(255, 255, 255))
            ).getbbox()
        pages, width_pt, height_pt = pdf_info(pdf)
        content_contained = (
            content_bounds is not None
            and content_bounds[0] > 0
            and content_bounds[1] > 0
            and content_bounds[2] <= width_px
            and content_bounds[3] < height_px
        )
        passed = (
            pages == 1
            and 2000 <= width_px <= 4000
            and 1000 <= height_px <= 4400
            and 0.6 <= width_px / height_px <= 3.5
            and 450 <= width_pt <= 900
            and 250 <= height_pt <= 850
            and 0.6 <= width_pt / height_pt <= 3.5
            and content_contained
        )
        geometry_ok &= passed
        figure_geometry[stem] = {
            "pass": passed,
            "png_pixels": [width_px, height_px],
            "pdf_points": [width_pt, height_pt],
            "content_bbox": list(content_bounds) if content_bounds else None,
            "content_contained": content_contained,
        }
    add_check(
        checks,
        "figure_geometry_and_content_containment",
        geometry_ok,
        figures=figure_geometry,
    )

    manifest_text = (ROOT / "FIGURE_DATA_SHA256SUMS").read_text(encoding="utf-8")
    required_manifest_paths = [
        "requirements.txt",
        "SYSTEM_REQUIREMENTS.md",
        "codes/build_submission_source_data.py",
        "codes/derive_peer_review_audit.py",
        "codes/release_manifest.txt",
        "codes/reproduce_all.sh",
        "codes/probes/verify_submission_methodology.py",
        "codes/probes/verify_revision_semantics.py",
        "codes/probes/verify_level3_reader_consequences.py",
        "codes/probes/audit_level1_protocol.py",
        "manuscript/validate_submission_figures.py",
        "codes/figures/rebuild_submission_figures.py",
        "codes/figures/_submission.py",
        "codes/figures/fig1_architecture.py",
        "codes/figures/fig2_generation.py",
        "codes/figures/fig3_propagation.py",
        "codes/figures/fig6_interface.py",
        "codes/figures/fig7_direct_traction.py",
        "codes/figures/fig8_closure_composition.py",
        "codes/figures/fig11_reachgated_composition.py",
        "codes/figures/fig14_slot_interface.py",
        "manuscript/main.tex",
        "manuscript/supplementary.tex",
        "manuscript/main.pdf",
        "manuscript/supplementary_information.pdf",
    ]
    add_check(
        checks,
        "figure_manifest_covers_build_code",
        all(path in manifest_text for path in required_manifest_paths),
        required=required_manifest_paths,
    )

    exact_manifest = load(ROOT / "SUBMISSION_RELEASE_MANIFEST.json")
    manifest_entries = exact_manifest["entries"]
    manifest_paths = {entry["path"] for entry in manifest_entries}
    reproduction_paths = {
        entry["path"] for entry in manifest_entries if entry["role"] == "reproduction_case"
    }
    manuscript_source_paths = {
        entry["path"] for entry in manifest_entries if entry["role"] == "manuscript_source"
    }
    rendered_document_paths = {
        entry["path"] for entry in manifest_entries if entry["role"] == "rendered_document"
    }
    active_source_paths = {
        "manuscript/source_data/README.md",
        "manuscript/source_data/ACCESS_AND_LICENSES.md",
    }
    for directory in (
        "methods",
        "fig2",
        "fig3",
        "fig4",
        "current_fig5",
        "fig6",
        "fig7",
        "fig6_interface",
        "fig7_direct",
        "fig8_closure",
        "fig11_reachgated",
        "fig_slot",
        "fig1_v2",
        "diagnostic_e2",
        "review_audit",
        "figure_artist_manifests",
    ):
        root = ROOT / "manuscript/source_data" / directory
        active_source_paths.update(
            str(path.relative_to(ROOT)) for path in root.rglob("*") if path.is_file()
        )
    active_figure_paths = {
        f"manuscript/figures/{stem}.{extension}"
        for stem in (
            "fig1_architecture",
            "fig2_generation",
            "fig3_propagation",
            "fig6_interface",
            "fig7_direct_traction",
            "fig8_closure_composition",
            "fig11_reachgated_composition",
            "fig14_slot_interface",
        )
        for extension in ("pdf", "png")
    }
    enumerated_sources = {
        entry["path"] for entry in manifest_entries if entry["role"] == "source_data"
    }
    enumerated_figures = {
        entry["path"] for entry in manifest_entries if entry["role"] == "figure"
    }
    hashes_match = all(
        (ROOT / entry["path"]).is_file()
        and sha256(ROOT / entry["path"]) == entry["sha256"]
        and (ROOT / entry["path"]).stat().st_size == entry["bytes"]
        for entry in manifest_entries
    )
    add_check(
        checks,
        "exact_release_manifest",
        exact_manifest["schema"] == "gwt-submission-release-manifest-v1"
        and enumerated_sources == active_source_paths
        and enumerated_figures == active_figure_paths
        and manuscript_source_paths
        == {
            "manuscript/main.tex",
            "manuscript/sections/introduction.tex",
            "manuscript/sections/results.tex",
            "manuscript/sections/discussion.tex",
            "manuscript/sections/methods.tex",
            "manuscript/supplementary.tex",
            "manuscript/supplementary_information.tex",
            "manuscript/references.bib",
            "manuscript/sn-jnl.cls",
            "manuscript/sn-nature.bst",
            "manuscript/FIGURE_EVIDENCE_AUDIT.md",
            "manuscript/cover_letter.md",
            "manuscript/nature_communications_checklist.md",
        }
        and rendered_document_paths
        == {
            "manuscript/main.pdf",
            "manuscript/supplementary_information.pdf",
        }
        and hashes_match,
        source_count=len(enumerated_sources),
        figure_count=len(enumerated_figures),
        manuscript_source_count=len(manuscript_source_paths),
        rendered_document_count=len(rendered_document_paths),
        hashes_match=hashes_match,
    )
    required_reproduction_paths = {
        "codes/closure/wall_closure.py",
        "codes/closure/x2fam_weights.h",
        "codes/cube_les/cube.re2",
        "codes/cube_les/cube.udf",
        "codes/cube_les/cube.usr",
        "codes/cube_les/cube_prod.par",
        "codes/cube_les/extract_wall_pressure_shear.py",
        "codes/cube_les/gen_cube_mesh.py",
        "codes/cube_les/rasterize_cube.py",
        "codes/gpu/eval_causal_wall_history.py",
        "codes/gpu/eval_cube_3d_coupling.py",
        "codes/gpu/eval_cube_3d_coupling_adequate.py",
        "codes/gpu/eval_cube_distributional_rapid.py",
        "codes/gpu/eval_cube_native_yplus.py",
        "codes/gpu/eval_cube_periodic_topology.py",
        "codes/gpu/eval_cube_wiener_floor_bench.py",
        "codes/gpu/eval_e2_closure_composition.py",
        "codes/gpu/eval_e2_cube_reachgated.py",
        "codes/gpu/eval_e2_direct_traction.py",
        "codes/gpu/eval_e2_generality.py",
        "codes/gpu/eval_e2_slot_channel.py",
        "codes/gpu/eval_e2_traction_interface.py",
        "codes/gpu/eval_runtime_closure_conditioning.py",
        "codes/gpu/eval_wallstress_cond.py",
        "codes/gpu/finalize_cube_yplus_temporal.py",
        "codes/gpu/run_cube_les_and_coupling.py",
        "codes/gpu/run_wiener_bench_seed_chain.py",
        "codes/probes/build_contact_ledger.py",
        "codes/probes/complete_non_regression_node007.py",
        "codes/probes/freeze_node007.py",
        "codes/probes/verify_runtime_closure.py",
        "codes/results/causal_wall_history_results.json",
        "codes/results/cube3d_coupling_adequate_results.json",
        "codes/results/cube3d_coupling_results.json",
        "codes/results/cube3d_native_yplus_temporally_separated.json",
        "codes/results/cube_production_complete.json",
        "codes/results/e2_cube_reachgated_results.json",
        "codes/results/e2_grouped_hills_band_results.json",
        "codes/results/e2_traction_interface_results.json",
        "codes/results/runtime_closure_conditioning_results.json",
        "development/iteration_20260723T150839/nodes/node_006/preregistration.md",
        "development/nodes/node_004/PREREGISTRATION_E2_INTERFACE.md",
        "development/nodes/node_005/FROZEN_HASHES.json",
        "development/nodes/node_005/PREREGISTRATION_E2_DIRECT.md",
        "development/nodes/node_005/verify_node005.py",
        "development/nodes/node_006/DECISION_RULE_OUTCOME.json",
        "development/nodes/node_006/FROZEN_HASHES.json",
        "development/nodes/node_006/PREREGISTRATION_E2_CLOSURE.md",
        "development/nodes/node_006/apply_decision_rule.py",
        "development/nodes/node_006/verify_node006.py",
        "development/nodes/node_007/AMENDMENT_1_BLOCK_LENGTH.md",
        "development/nodes/node_007/CONTACT_LEDGER.json",
        "development/nodes/node_007/FROZEN_HASHES.json",
        "development/nodes/node_007/NON_REGRESSION_COMPLETION.json",
        "development/nodes/node_007/PREREGISTRATION_E2_GENERALITY.md",
        "development/nodes/node_007/apply_decision_rule_node007.py",
        "development/nodes/node_007/verify_node007.py",
        "development/nodes/node_008/CONTACT_LEDGER_PHYSICAL.json",
        "development/nodes/node_008/DECISION_RULE_OUTCOME.json",
        "development/nodes/node_008/DECISION_RULE_OUTCOME_REPAIR2.json",
        "development/nodes/node_008/DECISION_RULE_OUTCOME_REPAIR3.json",
        "development/nodes/node_008/FROZEN_HASHES.json",
        "development/nodes/node_011/DECISION_RULE_OUTCOME_NODE011.json",
        "development/nodes/node_011/EXPECTED_HASHES_FROZEN.json",
        "development/nodes/node_011/PREREGISTRATION_SLOT_INTERFACE.md",
        "development/nodes/node_011/PREREG_HASHES.txt",
        "development/nodes/node_011/PREREG_SNAPSHOT_v1.md",
        "development/nodes/node_011/PREREG_SNAPSHOT_v1a_postA1.md",
        "development/nodes/node_011/PREREG_SNAPSHOT_v2.md",
        "development/nodes/node_011/apply_decision_rule_node011.py",
        "development/peer_review_baseline.json",
    }
    add_check(
        checks,
        "exact_les_reproduction_case_released",
        reproduction_paths == required_reproduction_paths,
        released=sorted(reproduction_paths),
    )
    active_json_statuses: dict[str, list[str]] = {}
    active_json_text: dict[str, str] = {}
    for source_path in sorted(enumerated_sources):
        path = ROOT / source_path
        if path.suffix != ".json":
            continue
        active_json_text[source_path] = path.read_text(encoding="utf-8")
        active_json_statuses[source_path] = status_strings(load(path))
    add_check(
        checks,
        "active_source_statuses_nonprovisional",
        all(
            not re.search(r"provisional|preliminary", status, flags=re.IGNORECASE)
            for statuses in active_json_statuses.values()
            for status in statuses
        )
        and "manuscript/source_data/fig5/origins/node002_full_logo_harness.json"
        not in enumerated_sources,
        statuses=active_json_statuses,
    )
    add_check(
        checks,
        "active_source_json_quarantine_content",
        all(
            value not in text
            for text in active_json_text.values()
            for value in (
                "node002_full_logo_harness",
                "0.61667",
                "0.6166723436748704",
                "0.53176",
                "0.5317590917967943",
            )
        ),
        scanned=sorted(active_json_text),
    )
    layout_audit = load(ROOT / "FIGURE_LAYOUT_AUDIT.json")
    add_check(
        checks,
        "rendered_artist_collision_audit",
        layout_audit.get("schema") == "gwt-figure-layout-audit-v1"
        and layout_audit.get("all_pass") is True
        and len(layout_audit.get("figures", [])) == 8
        and all(
            figure.get("all_pass")
            and not figure.get("canvas_overflow")
            and not figure.get("text_text_collisions")
            and not figure.get("text_reference_line_collisions")
            and not figure.get("final_font_size_failures")
            and figure.get("final_width_mm") == 180.0
            and figure.get("effective_font_pt_range_at_180mm", [0, 99])[0] >= 6.4
            and figure.get("effective_font_pt_range_at_180mm", [0, 99])[1] <= 10.0
            for figure in layout_audit.get("figures", [])
        ),
        figures=layout_audit.get("figures", []),
    )
    add_check(
        checks,
        "stale_backups_excluded_from_release",
        all(
            not path.startswith("manuscript/figures/_")
            and not path.startswith("manuscript/source_data/_")
            and "main_preL1reseed_backup" not in path
            for path in manifest_paths
        )
        and len(exact_manifest["excluded_local_backup_patterns"]) == 4,
    )
    checksum_paths = {
        line.split("  ", 1)[1]
        for line in manifest_text.splitlines()
        if "  " in line
    }
    add_check(
        checks,
        "checksum_manifest_matches_release",
        manifest_paths.issubset(checksum_paths)
        and "SUBMISSION_RELEASE_MANIFEST.json" in checksum_paths,
    )

    release_manifest = ROOT / "codes/release_manifest.txt"
    release_paths = [
        line.strip()
        for line in release_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [path for path in release_paths if not (ROOT / path).is_file()]
    add_check(
        checks,
        "release_manifest_paths_exist",
        not missing,
        missing=missing,
    )
    text_suffixes = {
        ".bib",
        ".json",
        ".md",
        ".par",
        ".py",
        ".sh",
        ".tex",
        ".txt",
        ".udf",
        ".usr",
    }
    forbidden_release_content: dict[str, list[str]] = {}
    machine_path_pattern = re.compile(r"/(?:root|home|Users|workspace|mnt)/")
    secret_pattern = re.compile(
        r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|AKIA[0-9A-Z]{16}|"
        r"sk-[A-Za-z0-9]{20,}"
    )
    reviewed_nonsecret_paths = {
        "codes/gpu/eval_cube_wiener_floor_bench.py": [
            "/" + "root/autodl-tmp/cube_les/cube_ds2_float16.complete.npy"
        ],
        # Public release: development/peer_review_baseline.json ships with its
        # development-machine paths replaced by placeholders, so no allowance
        # for it is needed here.
    }
    for relative in release_paths:
        path = ROOT / relative
        if path.suffix not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for reviewed_path in reviewed_nonsecret_paths.get(relative, []):
            content = content.replace(reviewed_path, "")
        findings = []
        if machine_path_pattern.search(content):
            findings.append("absolute machine path")
        if secret_pattern.search(content):
            findings.append("credential-like token")
        if findings:
            forbidden_release_content[relative] = findings
    add_check(
        checks,
        "release_text_has_no_unreviewed_machine_paths_or_credentials",
        not forbidden_release_content,
        findings=forbidden_release_content,
        reviewed_nonsecret_paths=reviewed_nonsecret_paths,
    )
    exact_comparator_producer = ROOT / "codes/gpu/eval_cube_wiener_floor_bench.py"
    exact_comparator_text = exact_comparator_producer.read_text(encoding="utf-8")
    add_check(
        checks,
        "exact_comparator_producer_preserved_with_data_override",
        sha256(exact_comparator_producer)
        == "90b72cb3297be710155935d004131f968a93ffe60d8667af3954fdf5a0cc5d8d"
        and 'P.add_argument("--data"' in exact_comparator_text,
        producer_sha256=sha256(exact_comparator_producer),
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    add_check(
        checks,
        "readme_current_reproduction_contract",
        "bash codes/reproduce_all.sh" in readme
        and "94/94" not in readme
        and "86/86" not in readme
        and "verify_cube_distributional_rapid.py" not in readme,
    )

    main_pdf = ROOT / "manuscript/main.pdf"
    supplement_pdf = ROOT / "manuscript/supplementary_information.pdf"
    main_pages, main_w, main_h = pdf_info(main_pdf)
    supp_pages, supp_w, supp_h = pdf_info(supplement_pdf)
    newest_figure = max(
        path.stat().st_mtime for path in (ROOT / "manuscript/figures").glob("fig*_*.png")
    )
    add_check(
        checks,
        "compiled_pdf_contract",
        35 <= main_pages <= 60
        and 5 <= supp_pages <= 20
        and 500 <= main_w <= 700
        and 700 <= main_h <= 900
        and 500 <= supp_w <= 700
        and 700 <= supp_h <= 900
        and main_pdf.stat().st_mtime >= newest_figure
        and supplement_pdf.stat().st_mtime >= newest_figure,
        main=[main_pages, main_w, main_h],
        supplement=[supp_pages, supp_w, supp_h],
    )

    logs = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (
            ROOT / "manuscript/main.log",
            ROOT / "manuscript/supplementary_information.log",
        )
    )
    add_check(
        checks,
        "latex_logs_clean",
        not re.search(
            r"undefined references|Reference .* undefined|Citation .* undefined|^!|"
            r"constrained_layout not applied|axes sizes collapsed|"
            r"destination with the same identifier|multiply defined|"
            r"multiply-defined labels",
            logs,
            re.IGNORECASE | re.MULTILINE,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="codes/results/submission_verification.json",
    )
    args = parser.parse_args()
    checks: dict[str, dict] = {}
    verify_point(checks)
    verify_les_release(checks)
    verify_rapid(checks)
    verify_sources_and_text(checks)
    verify_release_package(checks)
    passed = sum(int(value["pass"]) for value in checks.values())
    report = {
        "schema": "gwt-submission-methodology-verification-v1",
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "all_pass": passed == len(checks),
        "artifacts": {
            "point_result_sha256": sha256(POINT_RESULT),
            "point_components_sha256": sha256(POINT_COMPONENTS),
            "rapid_result_sha256": sha256(RAPID_RESULT),
            "rapid_components_sha256": sha256(RAPID_COMPONENTS),
        },
    }
    requested_output = Path(args.output)
    output = (
        requested_output
        if requested_output.is_absolute()
        else ROOT / requested_output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"METHODOLOGY_VERIFICATION: "
        f"{'ALL_PASS' if report['all_pass'] else 'FAIL'} {passed}/{len(checks)}"
    )
    try:
        display_output = output.relative_to(ROOT)
    except ValueError:
        display_output = output
    print(f"report={display_output} sha256={sha256(output)}")
    if not report["all_pass"]:
        for name, value in checks.items():
            if not value["pass"]:
                print(f"[FAIL] {name}: {value}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
