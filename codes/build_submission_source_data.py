#!/usr/bin/env python3
"""Build compact, provenance-checked source data for the submission figures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCE = ROOT / "manuscript" / "source_data"
SOURCE = Path(os.environ.get("GWT_SOURCE_DATA", CANONICAL_SOURCE))


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


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checked_copy(src: Path, dst: Path, expected: str) -> None:
    require_hash(src, expected)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    require_hash(dst, expected)


def build_cube_sources() -> None:
    # The compact source-data tree is the public release authority. Development
    # result trees are deliberately not required for a clean-manifest rebuild.
    point_dir = CANONICAL_SOURCE / "fig3"
    rapid_dir = CANONICAL_SOURCE / "fig4"
    checked_copy(
        point_dir / "cube_periodic_topology_results.json",
        SOURCE / "fig3" / "cube_periodic_topology_results.json",
        "0fc302421d622879e00ef16a14636fb8849ee65c0680c85491659788807f914d",
    )
    checked_copy(
        point_dir / "cube_periodic_topology_components.npz",
        SOURCE / "fig3" / "cube_periodic_topology_components.npz",
        "28264e996586f961b1a3cd8c369f494f62d39d4d18d73f0625fc4983e4ab3d18",
    )
    checked_copy(
        CANONICAL_SOURCE / "fig2" / "cube3d_representative_fields.npz",
        SOURCE / "fig2" / "cube3d_representative_fields.npz",
        "a50f378be740b639da81b6f96b40c1ecd809dd9c8907e3de207e1c96a602a29f",
    )
    checked_copy(
        rapid_dir / "cube_distributional_rapid_results.json",
        SOURCE / "fig4" / "cube_distributional_rapid_results.json",
        "c70342f6fc4bf8e21f0883ba76de806f01eb8cfc34f9b0d52018443beba00588",
    )
    checked_copy(
        rapid_dir / "cube_distributional_rapid_components.npz",
        SOURCE / "fig4" / "cube_distributional_rapid_components.npz",
        "a7f1c74f909e1a805d4a3642db83285371ecd80e4a59cd11a873a14253010e2d",
    )


def build_les_sources() -> None:
    production = CANONICAL_SOURCE / "methods" / "cube_production_complete.json"
    native_yplus = (
        CANONICAL_SOURCE / "methods" / "cube3d_native_yplus_temporally_separated.json"
    )
    native = load_json(native_yplus)
    checked_copy(
        production,
        SOURCE / "methods" / "cube_production_complete.json",
        "44c28d22481e67bce5202af558ede1dee19d2dd0d4dd31de09ec09799e910c73",
    )
    checked_copy(
        native_yplus,
        SOURCE / "methods" / "cube3d_native_yplus_temporally_separated.json",
        "bddf92ed67c25250dced3569829603da4461abd9a8a001cf668bdad1c6bab54e",
    )
    case_files = {
        "mesh": ROOT / "codes" / "cube_les" / "cube.re2",
        "production_parameters": ROOT / "codes" / "cube_les" / "cube_prod.par",
        "udf": ROOT / "codes" / "cube_les" / "cube.udf",
        "usr": ROOT / "codes" / "cube_les" / "cube.usr",
        "mesh_generator": ROOT / "codes" / "cube_les" / "gen_cube_mesh.py",
        "rasterizer": ROOT / "codes" / "cube_les" / "rasterize_cube.py",
        "acquisition_driver": ROOT / "codes" / "gpu" / "run_cube_les_and_coupling.py",
        "native_yplus_producer": ROOT / "codes" / "gpu" / "eval_cube_native_yplus.py",
        "native_yplus_finalizer": ROOT / "codes" / "gpu" / "finalize_cube_yplus_temporal.py",
    }
    expected = {
        "mesh": "09a68e5c763ba500c5dfe9d7281ef2d34557b1c4632707fd36a03ee4ce0b4128",
        "production_parameters": (
            "99c9d69542b8ebcd72f075926d170b81bc32411b8379916d31a8b49a320a3e9d"
        ),
        "udf": "2452be22431242e4932bcd104c5d47cea62a5a13cf3dffcdd9d878a18be7ff51",
        "usr": "3c716f1771bcf4b5111e652c43c44274bf453711c8275e92ca6337fbcd22a079",
        "mesh_generator": (
            "9f8accbfd9acc2bdb51e85518858887839307f66281fba35dbf40128daf38514"
        ),
        "rasterizer": (
            "09f86291c2eb0bfe0b7d8a13d02000d25de8cc4209a93c9e511dedc20f1e8414"
        ),
        "acquisition_driver": (
            "884251a128e5307580edeab54282f2bcece068c66825290f7a7a84e173a55d80"
        ),
        "native_yplus_producer": (
            "8d4e3f812a0a5079306cfc142dd072e5af598a9d5637094b4e4d75b621b2f0ed"
        ),
        "native_yplus_finalizer": (
            "67548547f45a3dbaaa16c27c68464660fee863d7c0a19fde25182d672057f7e9"
        ),
    }
    for name, path in case_files.items():
        require_hash(path, expected[name])
    point = load_json(CANONICAL_SOURCE / "fig3/cube_periodic_topology_results.json")
    provenance = {
        "schema": "gwt-cube-les-release-provenance-v1",
        "claim_boundary": (
            "Exact evidence-producing LES case and compact audits; raw fields and "
            "checkpoints are not included."
        ),
        "case_files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": expected[name],
                "bytes": path.stat().st_size,
            }
            for name, path in case_files.items()
        },
        "mesh_semantics": {
            "elements": 14176,
            "polynomial_order": 7,
            "gll_points": 7258112,
            "domain_h": [2.0, 4.0, 2.0],
            "wall_faces": 2336,
            "periodic_faces": 2464,
            "unreciprocated_periodic_faces": 0,
        },
        "production": {
            "path": str(production.relative_to(ROOT)),
            "sha256": sha256(production),
            "stationarity_energy_drift_middle_to_final": point["_meta"][
                "stationarity_energy_drift_middle_to_final"
            ],
        },
        "native_yplus": {
            "path": str(native_yplus.relative_to(ROOT)),
            "sha256": sha256(native_yplus),
            "origin_sha256_before_path_sanitization": native["_meta"][
                "origin_artifact_sha256_before_path_sanitization"
            ],
            "field_times": [390.0003162817, 420.0001096262],
            "primary_record_end_time": 370.00048748,
            "separate_continuation": True,
            "classified_surfaces": [
                "floor_between_cubes",
                "cube_top",
                "cube_vertical_faces",
            ],
            "ceiling_classified": False,
            "historical_signal_specific_integral_time": native["_meta"][
                "historical_signal_specific_integral_time"
            ],
            "current_conservative_integral_time": native["_meta"][
                "current_conservative_integral_time"
            ],
            "retained_time_separation": native["_meta"]["retained_time_separation"],
            "historical_signal_specific_integral_time_separated": native["_meta"][
                "retained_fields_historical_signal_specific_integral_time_separated"
            ],
            "current_conservative_integral_time_separated": native["_meta"][
                "retained_fields_current_conservative_integral_time_separated"
            ],
            "temporal_independence_claim": native["_meta"][
                "temporal_independence_claim"
            ],
            "origin_machine_paths_redacted": native["_meta"][
                "origin_machine_paths_redacted"
            ],
        },
    }
    write_json(SOURCE / "methods" / "cube_les_provenance.json", provenance)


def build_grouped_sources() -> None:
    diffusion_path = ROOT / "codes" / "results" / "split_integrity_adequacy_l0_results.json"
    flow_path = ROOT / "codes" / "results" / "l2_families_results.json"
    compact_path = CANONICAL_SOURCE / "fig6" / "grouped_hill_supporting.json"
    compact_hash = "c798117f983a2aabf134a0c6ea7a253f35e17f7e490b44103a93081c0fc81754"
    if not diffusion_path.is_file() or not flow_path.is_file():
        # A compact public clone deliberately omits the larger forensic result
        # tree.  In that case the hash-pinned vendored extract is the release
        # authority and remains byte-identical to the full-project extraction.
        checked_copy(
            compact_path,
            SOURCE / "fig6" / "grouped_hill_supporting.json",
            compact_hash,
        )
        return
    require_hash(
        diffusion_path,
        "2ed0b8c3914f9faa28157a9a30e07de960f7ba2b6739f1af982254be03b8898f",
    )
    require_hash(
        flow_path,
        "439f311197c93375beac4b7d72bec94a82054210b178484175f7413147bd7053",
    )
    diffusion = load_json(diffusion_path)["models"]["adq"]
    flow = load_json(flow_path)["families"]["fm"]
    compact = {
        "schema": "gwt-grouped-hill-supporting-v1",
        "claim_boundary": (
            "Supporting physical-time-grouped hill evidence only; absolute skill is "
            "negative and the separately parameterized flow-matching replication fails."
        ),
        "split": {
            "train_physical_times": [0, 479],
            "validation_physical_times": [500, 569],
            "test_physical_times": [590, 959],
            "planes_grouped_by_physical_time": True,
            "effective_test_events": diffusion["boot_fluct"]["_n_eff"],
        },
        "sources": {
            "diffusion": {
                "path": str(diffusion_path.relative_to(ROOT)),
                "sha256": sha256(diffusion_path),
            },
            "flow_matching": {
                "path": str(flow_path.relative_to(ROOT)),
                "sha256": sha256(flow_path),
            },
        },
        "diffusion": {
            "arms": {
                name: values["R2_fluct"]
                for name, values in diffusion["arms"].items()
                if name in {"correct", "no_wall", "random", "wrong_swap"}
            },
            "bootstrap": diffusion["boot_fluct"],
            "checkpoint_sha256": diffusion["checkpoint_sha256"],
        },
        "flow_matching": {
            "arms": {
                name: values["R2_fluct"]
                for name, values in flow["arms"].items()
                if name in {"correct", "no_wall", "random", "wrong_swap"}
            },
            "bootstrap": flow["boot_fluct"],
            "checkpoint_sha256": flow["checkpoint_sha256"],
            "paired_regen_consistent": flow["pair"]["regen_consistent"],
        },
    }
    write_json(SOURCE / "fig6" / "grouped_hill_supporting.json", compact)
    require_hash(SOURCE / "fig6" / "grouped_hill_supporting.json", compact_hash)


def build_capacity_source() -> None:
    expected = "545232f088493f67c17922745ddb046b3a1a70e71676df6aa6a06a90f6b963df"
    src = CANONICAL_SOURCE / "fig7" / "capacity_crossed_adjudication.json"
    checked_copy(
        src,
        SOURCE / "fig7" / "capacity_crossed_adjudication.json",
        expected,
    )


def build_e2_diagnostic_source() -> None:
    """Expose source-invalid, smoke-scale and archived adequacy-failing E2 diagnostics."""

    checked_copy(
        CANONICAL_SOURCE / "diagnostic_e2/wallstress_cond_case1_bh1_results.json",
        SOURCE / "diagnostic_e2" / "wallstress_cond_case1_bh1_results.json",
        "f058a9990ebd2e02b059b6629c4aa2d3cc5c77594e6d4a27491b9e895bcb1073",
    )
    checked_copy(
        CANONICAL_SOURCE / "diagnostic_e2/case1_closure_inputs.json",
        SOURCE / "diagnostic_e2" / "case1_closure_inputs.json",
        "7474a8ed7277e77ca4cd900bf7c008553e893d7e1ed77f23d2cfe69802fde3fa",
    )
    checked_copy(
        CANONICAL_SOURCE / "diagnostic_e2/runtime_closure_conditioning_results.json",
        SOURCE / "diagnostic_e2" / "runtime_closure_conditioning_results.json",
        "d96b352e146f23f94bd16011337051c7993bea15b8ead7d638b96496163e6457",
    )
    checked_copy(
        CANONICAL_SOURCE / "diagnostic_e2/causal_wall_history_results.json",
        SOURCE / "diagnostic_e2" / "causal_wall_history_results.json",
        "0eb97c78b80d569b067dfec58f4ee4d9213335b57395eba1d2f7fadf819cf641",
    )


def build_composition_source() -> None:
    """Expose the exact ledgers and raster used by the withdrawn Case1 display."""

    records = {
        "l2_case1_closure_results.json": "653a8660a99cbba4a2e240f160f84d77a08878891dee23e508041d280a14474e",
        "wallstress_cond_case1_bh1_results.json": "f058a9990ebd2e02b059b6629c4aa2d3cc5c77594e6d4a27491b9e895bcb1073",
        "SOURCE_INVALID_evaluated_bridge_rendering.png": "aae86da0e26a47d546279fe29a01e3d8856d00ab23ac6058aef4f728ed6807f1",
        "INVALIDATED.json": "4c0ce376ed643806c137e5e1d00089b946f25bf0a63d045c4f6cac001b3ba800",
    }
    for name, expected in records.items():
        checked_copy(
            CANONICAL_SOURCE / "current_fig5" / name,
            SOURCE / "current_fig5" / name,
            expected,
        )


def build_peer_review_audit_source() -> None:
    """Expose the first-contact/comparator custody records and derived audit."""

    records = {
        "cube3d_coupling_results.json": (
            CANONICAL_SOURCE / "review_audit/cube3d_coupling_results.json",
            "61e7d09d297bd6fa6a5aa8d55476e1c2a6fe56ddda0e7efe93b7eab25c3ce2a2",
        ),
        "cube3d_coupling_adequate_results.json": (
            CANONICAL_SOURCE / "review_audit/cube3d_coupling_adequate_results.json",
            "6762e9a89855c38538463c217698dab0e33c587a00d2fac1f937cf5ce35e982b",
        ),
        "cube_wiener_floor_bench_results_seed2234.json": (
            CANONICAL_SOURCE / "review_audit/cube_wiener_floor_bench_results_seed2234.json",
            "77e6ff62f926130616cd8b09b044024c02a953f1f3d463758f85e5bca753f48c",
        ),
        "cube_wiener_floor_bench_results_seed3234.json": (
            CANONICAL_SOURCE / "review_audit/cube_wiener_floor_bench_results_seed3234.json",
            "1ee765378277f3264db777574bbe8ef8e91c75a1c2a20fb6d1ca1051af704918",
        ),
        "wiener_seed_chain_node005.json": (
            CANONICAL_SOURCE / "review_audit/wiener_seed_chain_node005.json",
            "52cac551f0d1f7f4661053a6bd81f4322c466dbb75457970189c8aa2093f2842",
        ),
        "cube_train_band_retrieval_results.json": (
            CANONICAL_SOURCE / "review_audit/cube_train_band_retrieval_results.json",
            "94fa34214b23c2ae2ed046e846ba84bbed642ce4c90618be04417c3e82d5674f",
        ),
        "cube_train_band_retrieval_components.npz": (
            CANONICAL_SOURCE / "review_audit/cube_train_band_retrieval_components.npz",
            "02b412daeaf5314c237291534c7c995876c66c81794ac07f19327eea93fec0ce",
        ),
        "case1_source_native_audit.json": (
            CANONICAL_SOURCE / "review_audit/case1_source_native_audit.json",
            "cec89f247e990462ce48b247c2659dde5aba9de0688cc831b1e98f564c883898",
        ),
        "closure_interface_custody.json": (
            CANONICAL_SOURCE / "review_audit/closure_interface_custody.json",
            "a9c415e59ef9fd8db71a9bad09996b4c02a083061f73e2fd018f97ffe5bc16bb",
        ),
        "legacy_metadata_supersession.json": (
            CANONICAL_SOURCE / "review_audit/legacy_metadata_supersession.json",
            "47d3b7053504c2524f72cf4319bafdb167f5c012f498bf995eedc3fb95b4f979",
        ),
        "publication_provenance_facts.json": (
            CANONICAL_SOURCE / "review_audit/publication_provenance_facts.json",
            "4f1713c2b581e8c530856e455313e519d8a6d702e68ec33488f94c8542527303",
        ),
        "publication_quantitative_contract.json": (
            CANONICAL_SOURCE / "review_audit/publication_quantitative_contract.json",
            "d3aee250a32bf1a17a507f051129b6533568efb037af8969543e771af83e3872",
        ),
        "publication_semantic_contract.json": (
            CANONICAL_SOURCE / "review_audit/publication_semantic_contract.json",
            # Re-pinned 2026-08-03: evidence_ladder_global amended in-file (E4
            # removed from the enumerated taxonomy; scope guarantee unchanged).
            # Re-pinned 2026-08-04 (peer-review round 2).  Five entries changed,
            # each carrying its own amendment_rationale in the contract file:
            #   - abstract_m0_headline -> abstract_quantitative_headline: the
            #     abstract was rewritten and no longer quotes the M0 aggregate, so
            #     the entry now binds the two claims it does make (cube wall-force
            #     0.09944 -> 0.06093; channel 0.11984 / 0.16497) and additionally
            #     REQUIRES the abstract to disclose that the hill transmits in only
            #     one of two generative families.
            #   - evidence_ladder_global: tokens now quote the plain-language
            #     taxonomy, the internal E1/E2/E3 labels having been removed from
            #     the manuscript on author instruction.
            #   - near_wall_prior_work_positioning: anchor tracks the compressed
            #     heading; all six substantive guarantees retained verbatim.
            #   - interface_gap_claim (NEW): keeps the relocated "no explicit
            #     wall-closure-to-generator interface" novelty claim anchored.
            #   - hill_evidence_identity (NEW): binds the hill headline to the
            #     released hierarchical-estimator JSON and requires its two
            #     caveats (N_eff ~ 3.3; non-separation from a scrambled traction),
            #     so the favourable number cannot be kept while a caveat is cut.
            # Re-pinned 2026-08-05: abstract_quantitative_headline retargeted after
            # the supervisor's abstract rewrite removed the phrase it was anchored
            # on; see amendment_rationale in the contract file.  Guarantees
            # unchanged: both quantitative claims and the adverse
            # one-of-two-families disclosure are still required.
            # Re-pinned 2026-08-10: the meeting-directed abstract rewrite trimmed
            # the propagation clause to "the stress propagates", so the same
            # entry's anchor is retargeted again (rationale in the contract
            # file); tokens and source bindings unchanged.
            "206a58dfbc00a795d020e284163292e1e58b61327f90424dd9d39fd050ae425f",
        ),
        "legacy_success_gate_disposition.json": (
            CANONICAL_SOURCE / "review_audit/legacy_success_gate_disposition.json",
            "bef54e5e60bb95d0f9f43f4cda2a5738b63a236f01b113c1c802689e896f222b",
        ),
    }
    destination = SOURCE / "review_audit"
    for name, (source, expected) in records.items():
        checked_copy(source, destination / name, expected)

    environment = os.environ.copy()
    environment["GWT_PEER_REVIEW_AUDIT_OUTPUT"] = str(
        destination / "derived_peer_review_statistics.json"
    )
    subprocess.run(
        [sys.executable, str(ROOT / "codes/derive_peer_review_audit.py")],
        check=True,
        env=environment,
    )


def build_reviewed_figure_sources() -> None:
    """Stage the reviewed-figure source directories, pinned by the checksum manifest."""
    manifest = {}
    for line in (ROOT / "FIGURE_DATA_SHA256SUMS").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition("  ")
        manifest[rel.strip()] = digest.strip()
    for directory in (
        "fig2",
        "fig6_interface",
        "fig7_direct",
        "fig8_closure",
        "fig11_reachgated",
        "fig_slot",
        "fig1_v2",
    ):
        canonical = CANONICAL_SOURCE / directory
        for path in sorted(canonical.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(ROOT))
            if rel not in manifest:
                raise RuntimeError(f"no pinned hash for staged source file {rel}")
            checked_copy(path, SOURCE / directory / path.relative_to(canonical), manifest[rel])


def main() -> None:
    build_cube_sources()
    build_les_sources()
    build_grouped_sources()
    build_capacity_source()
    build_e2_diagnostic_source()
    build_composition_source()
    build_reviewed_figure_sources()
    build_peer_review_audit_source()
    print("SOURCE_DATA_BUILD_PASS")


if __name__ == "__main__":
    main()
