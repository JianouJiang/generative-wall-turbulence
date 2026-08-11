#!/usr/bin/env python3
"""Rebuild seven active displays and one withdrawn Case1 diagnostic."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageChops


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "manuscript" / "source_data"
OUTPUT = PROJECT / "manuscript" / "figures"
LOCK = PROJECT / ".submission_release.lock"
RELEASE_MANIFEST = PROJECT / "SUBMISSION_RELEASE_MANIFEST.json"
CHECKSUM_MANIFEST = PROJECT / "FIGURE_DATA_SHA256SUMS"
LAYOUT_AUDIT = PROJECT / "FIGURE_LAYOUT_AUDIT.json"
SCRIPTS = [
    "fig1_architecture.py",
    "fig2_generation.py",
    "fig3_propagation.py",
    "fig6_interface.py",
    "fig7_direct_traction.py",
    "fig8_closure_composition.py",
    "fig11_reachgated_composition.py",
    "fig14_slot_interface.py",
]
EXPECTED = [
    "fig1_architecture",
    "fig2_generation",
    "fig3_propagation",
    "fig6_interface",
    "fig7_direct_traction",
    "fig8_closure_composition",
    "fig11_reachgated_composition",
    "fig14_slot_interface",
]
BUILD_FILES = [
    PROJECT / "README.md",
    PROJECT / "PROJECT_STATUS.md",
    PROJECT / "requirements.txt",
    PROJECT / "SYSTEM_REQUIREMENTS.md",
    PROJECT / "codes" / "build_submission_source_data.py",
    PROJECT / "codes" / "figures" / "build_fig2_time_pair.py",
    PROJECT / "codes" / "figures" / "stage_fig1_overview.py",
    PROJECT / "codes" / "derive_peer_review_audit.py",
    PROJECT / "codes" / "release_manifest.txt",
    PROJECT / "codes" / "reproduce_all.sh",
    PROJECT / "codes" / "probes" / "verify_submission_methodology.py",
    PROJECT / "codes" / "probes" / "verify_revision_semantics.py",
    PROJECT / "codes" / "probes" / "verify_level3_reader_consequences.py",
    PROJECT / "codes" / "probes" / "verify_clean_release.py",
    PROJECT / "codes" / "probes" / "audit_cube_train_band_retrieval.py",
    PROJECT / "codes" / "probes" / "audit_closure_interface_custody.py",
    PROJECT / "codes" / "probes" / "audit_level1_protocol.py",
    # Every probe that codes/reproduce_all.sh invokes must ship, or the release
    # cannot re-run its own regression suite from a clean tree.
    PROJECT / "codes" / "probes" / "recompute_fig6_conservative_blocks.py",
    PROJECT / "manuscript" / "validate_submission_figures.py",
    # Figure-8 producer script: the display for the paper's headline claim must
    # be rebuildable from the release set.
    PROJECT / "codes" / "figures" / "fig8_closure_composition.py",
    PROJECT / "codes" / "figures" / "fig9_generality.py",
    Path(__file__),
    Path(__file__).with_name("_submission.py"),
    PROJECT / "figure_drafts" / "fig45" / "fig5_composition_transfer_v3.py",
    *[Path(__file__).with_name(script) for script in SCRIPTS],
]
MANUSCRIPT_SOURCE_FILES = [
    PROJECT / "manuscript" / "main.tex",
    PROJECT / "manuscript" / "sections" / "introduction.tex",
    PROJECT / "manuscript" / "sections" / "results.tex",
    PROJECT / "manuscript" / "sections" / "discussion.tex",
    PROJECT / "manuscript" / "sections" / "methods.tex",
    PROJECT / "manuscript" / "supplementary.tex",
    PROJECT / "manuscript" / "supplementary_information.tex",
    PROJECT / "manuscript" / "references.bib",
    PROJECT / "manuscript" / "sn-jnl.cls",
    PROJECT / "manuscript" / "sn-nature.bst",
    PROJECT / "manuscript" / "FIGURE_EVIDENCE_AUDIT.md",
    PROJECT / "manuscript" / "cover_letter.md",
    PROJECT / "manuscript" / "nature_communications_checklist.md",
]
RENDERED_DOCUMENTS = [
    PROJECT / "manuscript" / "main.pdf",
    PROJECT / "manuscript" / "supplementary_information.pdf",
]
REPRODUCTION_FILES = [
    PROJECT / "development" / "peer_review_baseline.json",
    PROJECT
    / "development"
    / "iteration_20260723T150839"
    / "nodes"
    / "node_006"
    / "preregistration.md",
    PROJECT
    / "development"
    / "nodes"
    / "node_004"
    / "PREREGISTRATION_E2_INTERFACE.md",
    # E2 producers, preregistrations, freeze certificates, mechanically applied
    # decision rule and independent verifiers.  Without these the central
    # closure-composition result is not provenance-complete in the release set.
    PROJECT / "codes" / "gpu" / "eval_e2_direct_traction.py",
    PROJECT / "codes" / "gpu" / "eval_e2_closure_composition.py",
    PROJECT / "development" / "nodes" / "node_005" / "PREREGISTRATION_E2_DIRECT.md",
    PROJECT / "development" / "nodes" / "node_005" / "FROZEN_HASHES.json",
    PROJECT / "development" / "nodes" / "node_005" / "verify_node005.py",
    PROJECT / "development" / "nodes" / "node_006" / "PREREGISTRATION_E2_CLOSURE.md",
    PROJECT / "development" / "nodes" / "node_006" / "FROZEN_HASHES.json",
    PROJECT / "development" / "nodes" / "node_006" / "DECISION_RULE_OUTCOME.json",
    PROJECT / "development" / "nodes" / "node_006" / "apply_decision_rule.py",
    PROJECT / "development" / "nodes" / "node_006" / "verify_node006.py",
    # node 007: the generality experiment, its contact ledger, its freeze
    # certificate and the probes that discharge the non-regression mandate.
    PROJECT / "codes" / "gpu" / "eval_e2_generality.py",
    PROJECT / "codes" / "probes" / "build_contact_ledger.py",
    PROJECT / "codes" / "probes" / "complete_non_regression_node007.py",
    PROJECT / "codes" / "probes" / "freeze_node007.py",
    PROJECT / "development" / "nodes" / "node_007" / "PREREGISTRATION_E2_GENERALITY.md",
    PROJECT / "development" / "nodes" / "node_007" / "AMENDMENT_1_BLOCK_LENGTH.md",
    PROJECT / "development" / "nodes" / "node_007" / "FROZEN_HASHES.json",
    PROJECT / "development" / "nodes" / "node_007" / "CONTACT_LEDGER.json",
    PROJECT / "development" / "nodes" / "node_007" / "NON_REGRESSION_COMPLETION.json",
    PROJECT / "development" / "nodes" / "node_007" / "apply_decision_rule_node007.py",
    PROJECT / "development" / "nodes" / "node_007" / "verify_node007.py",
    PROJECT / "codes" / "cube_les" / "cube.re2",
    PROJECT / "codes" / "cube_les" / "cube_prod.par",
    PROJECT / "codes" / "cube_les" / "cube.udf",
    PROJECT / "codes" / "cube_les" / "cube.usr",
    PROJECT / "codes" / "cube_les" / "gen_cube_mesh.py",
    PROJECT / "codes" / "cube_les" / "rasterize_cube.py",
    PROJECT / "codes" / "cube_les" / "extract_wall_pressure_shear.py",
    PROJECT / "codes" / "gpu" / "run_cube_les_and_coupling.py",
    PROJECT / "codes" / "gpu" / "eval_cube_native_yplus.py",
    PROJECT / "codes" / "gpu" / "finalize_cube_yplus_temporal.py",
    PROJECT / "codes" / "gpu" / "eval_cube_3d_coupling.py",
    PROJECT / "codes" / "gpu" / "eval_cube_3d_coupling_adequate.py",
    PROJECT / "codes" / "gpu" / "eval_e2_traction_interface.py",
    PROJECT / "codes" / "gpu" / "eval_cube_periodic_topology.py",
    PROJECT / "codes" / "gpu" / "eval_cube_distributional_rapid.py",
    PROJECT / "codes" / "gpu" / "eval_cube_wiener_floor_bench.py",
    PROJECT / "codes" / "gpu" / "run_wiener_bench_seed_chain.py",
    PROJECT / "codes" / "gpu" / "eval_wallstress_cond.py",
    PROJECT / "codes" / "gpu" / "eval_runtime_closure_conditioning.py",
    PROJECT / "codes" / "gpu" / "eval_causal_wall_history.py",
    PROJECT / "codes" / "closure" / "wall_closure.py",
    PROJECT / "codes" / "closure" / "x2fam_weights.h",
    PROJECT / "codes" / "probes" / "verify_runtime_closure.py",
    PROJECT / "codes" / "results" / "cube3d_coupling_results.json",
    PROJECT / "codes" / "results" / "cube3d_coupling_adequate_results.json",
    PROJECT / "codes" / "results" / "e2_traction_interface_results.json",
    PROJECT / "codes" / "results" / "cube_production_complete.json",
    PROJECT / "codes" / "results" / "cube3d_native_yplus_temporally_separated.json",
    PROJECT / "codes" / "results" / "runtime_closure_conditioning_results.json",
    PROJECT / "codes" / "results" / "causal_wall_history_results.json",
    # node 008: reversed-time power-gated re-test (Fig. 7 of the paper).
    PROJECT / "codes" / "gpu" / "eval_e2_cube_reachgated.py",
    PROJECT / "development" / "nodes" / "node_008" / "FROZEN_HASHES.json",
    PROJECT / "development" / "nodes" / "node_008" / "CONTACT_LEDGER_PHYSICAL.json",
    PROJECT / "development" / "nodes" / "node_008" / "DECISION_RULE_OUTCOME.json",
    PROJECT / "development" / "nodes" / "node_008" / "DECISION_RULE_OUTCOME_REPAIR2.json",
    PROJECT / "development" / "nodes" / "node_008" / "DECISION_RULE_OUTCOME_REPAIR3.json",
    PROJECT / "codes" / "results" / "e2_cube_reachgated_results.json",
    PROJECT / "codes" / "results" / "e2_grouped_hills_band_results.json",
    # node 011: single-slot channel experiment (Fig. 8 of the paper).
    PROJECT / "codes" / "gpu" / "eval_e2_slot_channel.py",
    PROJECT / "development" / "nodes" / "node_011" / "PREREGISTRATION_SLOT_INTERFACE.md",
    PROJECT / "development" / "nodes" / "node_011" / "PREREG_SNAPSHOT_v1.md",
    PROJECT / "development" / "nodes" / "node_011" / "PREREG_SNAPSHOT_v1a_postA1.md",
    PROJECT / "development" / "nodes" / "node_011" / "PREREG_SNAPSHOT_v2.md",
    PROJECT / "development" / "nodes" / "node_011" / "PREREG_HASHES.txt",
    PROJECT / "development" / "nodes" / "node_011" / "EXPECTED_HASHES_FROZEN.json",
    PROJECT / "development" / "nodes" / "node_011" / "apply_decision_rule_node011.py",
    PROJECT / "development" / "nodes" / "node_011" / "DECISION_RULE_OUTCOME_NODE011.json",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@contextmanager
def release_lock():
    """Serialize standalone rebuilds; the full driver may hold this lock already."""

    if os.environ.get("GWT_SUBMISSION_LOCK_HELD") == "1":
        yield
        return
    LOCK.touch(exist_ok=True)
    with LOCK.open("r+") as handle:
        print(f"SUBMISSION_LOCK: waiting ({LOCK.name})")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print("SUBMISSION_LOCK: acquired")
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def active_source_files() -> list[Path]:
    """Return only claim-active sources; forensic trees remain on disk but outside release."""

    files = [
        SOURCE / "README.md",
        SOURCE / "ACCESS_AND_LICENSES.md",
    ]
    for directory in (
        "methods",
        "fig2",
        "fig3",
        "fig4",
        "current_fig5",
        "fig6",
        "fig6_interface",
        "fig7_direct",
        "fig7",
        "fig8_closure",
        "fig9_generality",
        "fig11_reachgated",
        "fig_slot",
        "fig1_v2",
        "diagnostic_e2",
        "review_audit",
        "figure_artist_manifests",
    ):
        root = SOURCE / directory
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(path for path in files if path.is_file())


def expected_figure_files(directory: Path = OUTPUT) -> list[Path]:
    return [
        directory / f"{stem}.{extension}"
        for stem in EXPECTED
        for extension in ("pdf", "png")
    ]


def validate_outputs(stage_output: Path) -> None:
    missing = [str(path) for path in expected_figure_files(stage_output) if not path.is_file()]
    if missing:
        raise RuntimeError("missing staged outputs: " + ", ".join(missing))

    # A dimension-only check missed clipped Fig. 3 text in an earlier release.
    # Require every PNG's non-white content to remain inside the raster canvas.
    failures: dict[str, object] = {}
    for path in expected_figure_files(stage_output):
        if path.suffix != ".png":
            continue
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            difference = ImageChops.difference(
                rgb, Image.new("RGB", rgb.size, color=(255, 255, 255))
            )
            bounds = difference.getbbox()
            if bounds is None:
                failures[path.name] = "blank figure"
                continue
            left, top, right, bottom = bounds
            width, height = rgb.size
            if left <= 0 or top <= 0 or right > width or bottom >= height:
                failures[path.name] = {
                    "content_bbox": list(bounds),
                    "canvas": [width, height],
                }
    if failures:
        raise RuntimeError(f"rendered content reaches canvas edge: {failures}")


def promote_tree(stage: Path, destination: Path) -> None:
    """Atomically replace each staged file while the release lock is held."""

    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)


def prune_managed_source_files(expected: set[Path]) -> None:
    """Remove obsolete active figure inputs while preserving docs and quarantine trees."""

    for directory in (
        "methods",
        "fig2",
        "fig3",
        "fig4",
        "fig5",
        "current_fig5",
        "fig6",
        "fig7",
        "fig8_closure",
        "fig9_generality",
        "fig11_reachgated",
        "fig_slot",
        "fig1_v2",
        "diagnostic_e2",
        "review_audit",
        "figure_artist_manifests",
    ):
        root = SOURCE / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file() and path.relative_to(SOURCE) not in expected:
                path.unlink()
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()


def write_release_manifests() -> None:
    source_files = active_source_files()
    figure_files = expected_figure_files()
    entries = []
    for role, files in (
        ("source_data", source_files),
        ("figure", figure_files),
        ("render_audit", [LAYOUT_AUDIT]),
        ("build_code", sorted(BUILD_FILES)),
        ("manuscript_source", sorted(MANUSCRIPT_SOURCE_FILES)),
        ("rendered_document", sorted(RENDERED_DOCUMENTS)),
        ("reproduction_case", sorted(REPRODUCTION_FILES)),
    ):
        for path in files:
            entries.append(
                {
                    "path": str(path.relative_to(PROJECT)),
                    "role": role,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }
            )
    payload = {
        "schema": "gwt-submission-release-manifest-v1",
        "selection": (
            "Exact active source-data, eleven PDF/PNG figure pairs, manuscript/Supplementary "
            "sources and rendered documents, environment specification, and build/audit code. "
            "Local backup trees are intentionally outside the release set."
        ),
        "entries": sorted(entries, key=lambda item: item["path"]),
        "excluded_local_backup_patterns": [
            "manuscript/figures/_quarantined_*",
            "manuscript/figures/_stale_*",
            "manuscript/source_data/_quarantined_*",
            "manuscript/main_preL1reseed_backup.tex",
        ],
    }
    RELEASE_MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checksum_files = [
        *source_files,
        *figure_files,
        LAYOUT_AUDIT,
        *BUILD_FILES,
        *MANUSCRIPT_SOURCE_FILES,
        *RENDERED_DOCUMENTS,
        *REPRODUCTION_FILES,
        RELEASE_MANIFEST,
    ]
    CHECKSUM_MANIFEST.write_text(
        "".join(
            f"{digest(path)}  {path.relative_to(PROJECT)}\n"
            for path in sorted(set(checksum_files))
        ),
        encoding="utf-8",
    )


def rebuild() -> None:
    stage_root = Path(
        tempfile.mkdtemp(prefix=".submission-build-", dir=str(PROJECT))
    )
    stage_source = stage_root / "source_data"
    stage_output = stage_root / "figures"
    stage_layout = stage_root / "layout"
    stage_source.mkdir(parents=True)
    stage_output.mkdir(parents=True)
    stage_layout.mkdir(parents=True)
    try:
        env = os.environ.copy()
        env["GWT_SOURCE_DATA"] = str(stage_source)
        env["GWT_FIGURE_OUTPUT"] = str(stage_output)
        env["GWT_LAYOUT_AUDIT_DIR"] = str(stage_layout)
        env["GWT_ARTIST_MANIFEST_DIR"] = str(
            stage_source / "figure_artist_manifests"
        )
        # Matplotlib embeds a PDF creation timestamp unless this epoch is fixed.
        env["SOURCE_DATE_EPOCH"] = "1784790000"
        subprocess.run(
            [sys.executable, str(PROJECT / "codes" / "build_submission_source_data.py")],
            check=True,
            env=env,
        )
        for script in SCRIPTS:
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name(script))],
                check=True,
                env=env,
            )
        validate_outputs(stage_output)
        artist_manifests = [
            json.loads(
                (
                    stage_source / "figure_artist_manifests" / f"{stem}.json"
                ).read_text(encoding="utf-8")
            )
            for stem in EXPECTED
        ]
        if len(artist_manifests) != len(EXPECTED) or not all(
            record.get("all_validated") and record.get("binding_count", 0) > 0
            for record in artist_manifests
        ):
            raise RuntimeError(
                f"producer-emitted artist provenance failed: {artist_manifests}"
            )
        layout_records = [
            json.loads((stage_layout / f"{stem}.json").read_text(encoding="utf-8"))
            for stem in EXPECTED
        ]
        if len(layout_records) != len(EXPECTED) or not all(
            record.get("all_pass") for record in layout_records
        ):
            raise RuntimeError(f"rendered layout audit failed: {layout_records}")
        staged_layout_audit = stage_root / LAYOUT_AUDIT.name
        staged_layout_audit.write_text(
            json.dumps(
                {
                    "schema": "gwt-figure-layout-audit-v1",
                    "figures": layout_records,
                    "all_pass": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expected_source = {
            path.relative_to(stage_source)
            for path in stage_source.rglob("*")
            if path.is_file()
        }
        promote_tree(stage_source, SOURCE)
        prune_managed_source_files(expected_source)
        promote_tree(stage_output, OUTPUT)
        os.replace(staged_layout_audit, LAYOUT_AUDIT)
        write_release_manifests()
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def main() -> None:
    with release_lock():
        if "--manifest-only" in sys.argv[1:]:
            write_release_manifests()
            print("SUBMISSION_MANIFEST: REFRESHED")
            return
        rebuild()
    print(f"SUBMISSION_FIGURES: ALL_PASS {2 * len(EXPECTED)}/{2 * len(EXPECTED)}")
    print(f"SUBMISSION_LAYOUT: ALL_PASS {len(EXPECTED)}/{len(EXPECTED)}")
    print(f"release_manifest={RELEASE_MANIFEST.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
