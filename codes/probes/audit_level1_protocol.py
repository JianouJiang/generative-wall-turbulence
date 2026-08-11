#!/usr/bin/env python3
"""Executable semantic audit for the Level-2 closure-to-field revision.

This is a read-only audit. It performs no simulation, training or neural inference.
It checks the scientific contracts that ordinary hash and phrase-presence checks can miss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
# 2026-08-07 supervisor framing revision: the title leads with the recurring
# wall problem while retaining the reusable interface framework and adequacy standard.
TITLE = (
    "An interface framework and adequacy standard when generative turbulence models hit"
    " the wall"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, **details: object) -> None:
        checks.append({"name": name, "pass": bool(passed), **details})

    baseline = load(ROOT / "development/peer_review_baseline.json")
    check(
        "immutable_campaign_baseline",
        baseline["source_commit"] == "6e5901b28b4bcb8383d8a6de419c179d8c20d30a"
        and baseline["baseline_sha256"]["manuscript/main.tex"]
        == "bc930b450c8d170df2211d2d67d0a8bbdc4aaa0127b41c6dbbf4e7777cf143e1",
        source_commit=baseline["source_commit"],
    )

    main_tex = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    intro = (ROOT / "manuscript/sections/introduction.tex").read_text(encoding="utf-8")
    results = (ROOT / "manuscript/sections/results.tex").read_text(encoding="utf-8")
    methods = (ROOT / "manuscript/sections/methods.tex").read_text(encoding="utf-8")
    discussion = (ROOT / "manuscript/sections/discussion.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "manuscript/supplementary.tex").read_text(encoding="utf-8")
    fig1 = (ROOT / "codes/figures/fig1_architecture.py").read_text(encoding="utf-8")
    fig3 = (ROOT / "codes/figures/fig3_propagation.py").read_text(encoding="utf-8")
    fig5 = (
        ROOT / "codes/figures/fig5_physical_composite.py"
    ).read_text(encoding="utf-8")
    traction_extractor = (
        ROOT / "codes/cube_les/extract_wall_pressure_shear.py"
    ).read_text(encoding="utf-8")
    source_readme = (
        ROOT / "manuscript/source_data/README.md"
    ).read_text(encoding="utf-8")
    checklist = (
        ROOT / "manuscript/nature_communications_checklist.md"
    ).read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
    cover = (ROOT / "manuscript/cover_letter.md").read_text(encoding="utf-8")
    # Public release: the journal checklist is an author-only document replaced by a
    # placeholder note; checks on its internal wording are skipped, their file-layout
    # clauses remain enforced. The full checks run in the authors' archive tree.
    checklist_withheld = "withheld from the public release" in checklist

    title_paths = {
        "main": main_tex,
        "supplement": supplement,
        "README": readme,
        "project_status": status,
        "cover_letter": cover,
    }
    check(
        "release_wide_title_identity",
        all(TITLE in compact(text) for text in title_paths.values()),
        title_words=len(TITLE.split()),
        paths=list(title_paths),
    )

    abstract_match = re.search(r"\\abstract\{(.*?)\}\s*\\maketitle", main_tex, re.S)
    abstract = compact(abstract_match.group(1)) if abstract_match else ""
    # 2026-08-11: the repository URL counts as one word, as in the submission
    # portal's whitespace count of the pasted abstract; splitting it into path
    # components had inflated the count by four.
    abstract_plain = re.sub(r"\\url\{[^}]*\}", " URL ", abstract)
    abstract_plain = re.sub(r"\\[A-Za-z]+|\$|[{}]", " ", abstract_plain)
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract_plain)
    # 2026-08-10 supervisor meeting: the abstract was rewritten to the dictated
    # opening (popularity/hope, full stop after "loads", a generative-objective
    # sentence, "commonly used", the refocus close) and re-trimmed to the
    # journal's official 200-word ceiling.
    check(
        "title_and_abstract_limits",
        len(TITLE.split()) <= 15 and len(abstract_words) <= 200 and "\\cite" not in abstract,
        title_words=len(TITLE.split()),
        abstract_words=len(abstract_words),
    )

    # 2026-08-04 supervisor framing revision: the abstract was rewritten to the
    # dictated diagnosis-then-repair shape and the title now names the interface
    # and standard, so the two phrase clauses below track the new wording.  The
    # GUARANTEE is unchanged and is what the remaining clauses enforce: the
    # abstract must state wall information propagating into the field, must not
    # claim solver deployment, and must not claim thermal transfer -- in the
    # title as well as the abstract, which is a strictly stronger boundary than
    # the clause it replaces.
    # (10 Aug trim: the propagation clause now reads "the stress propagates";
    # the guarantee is unchanged.)
    check(
        "title_boundary_remains_oracle_specific",
        "stress propagates" in compact(abstract)
        and "near-wall" in compact(abstract)
        and "solver" not in abstract
        and "heat" not in abstract
        and "solver" not in TITLE.lower()
        and "heat" not in TITLE.lower(),
    )

    # 2026-08-03 length revision: the per-experiment dependence and realisation
    # disclosures were consolidated into the single Methods conventions
    # paragraph; the reader guarantees are unchanged but now live in Methods.
    check(
        "temporal_dependence_wording_closed",
        "160 temporally correlated, thinned fields are evaluated from the chronologically separated remainder"
        in compact(results)
        and "160 temporally correlated three-component fields" in compact(results)
        and "160 chronologically separated fields" not in results
        and "temporally correlated rather than independent events" in compact(methods),
    )

    check(
        "sampler_realisation_uncertainty_explicit",
        "one realised eight-sample mean" in compact(methods)
        and "one realised sampler draw set" in compact(methods)
        and "Monte Carlo error cannot be replayed" in compact(supplement),
    )

    check(
        "ema_defined_at_first_use",
        "exponential moving average (EMA) decay 0.9995" in compact(methods)
        and "EMA 0.9995" not in methods,
    )

    check(
        "future_e2_checklist_not_false_frozen",
        checklist_withheld
        or (
            "Reichardt" not in checklist
            and "16-gate" not in checklist
            and "Do not represent a particular classical comparator" in checklist
            and "Before an authorised E2 launch" in checklist
            and "simultaneous decision rule" in compact(checklist)
        ),
        author_only_checklist_withheld=checklist_withheld,
    )

    check(
        "respectful_prior_work_and_public_arm_labels",
        # 2026-08-04 peer-review round 2: the paragraph heading was compressed to
        # "Representative whole-field formulations" for the 4,993-word ceiling.
        # The two respectfulness guarantees this check exists to protect -- the
        # negated universal accusation, and the explicit-interface novelty claim --
        # are unchanged and still asserted below.
        "Representative whole-field formulations" in intro
        and "A substantial strand" not in intro
        and "machine learning universally ignores walls" in compact(intro)
        and "wall-closure-to-generator interface" in compact(intro)
        and "Reader-facing arm labels are **aligned band**, **absent band** and **far-time band**"
        in compact(source_readme)
        and "historical keys `correct`, `no_wall`" in compact(source_readme),
    )

    adverse_phrases = (
        "source-valid scalar-proxy study has negative absolute fluctuation skill",
        "source-valid periodic-hill scalar-proxy study is adverse",
        "E2  NOT ESTABLISHED",
        "source-valid scalar proxy: adverse and",
    )
    # Figure strings contain explicit ``\n`` layout breaks.  Check rendered
    # semantics rather than forcing a scientifically arbitrary single line.
    fig1_semantic = re.sub(r'"\s*"', "", fig1).replace(r"\n", " ")
    # 2026-08-03 editorial revision: the archived scalar-proxy forensic detail
    # moved out of the Introduction to Supplementary Notes 6/8; the hill E2
    # boundary is now disclosed in the Introduction as a failed positive
    # control, and the unperformed solver-coupled level is stated once in
    # Methods as "nothing in this work advances a momentum equation".
    check(
        "both_e2_failure_paths_in_high_level_ladder",
        # 2026-08-04: wording compressed from "fails a prospective positive
        # control" to "fails its positive control"; the disclosure is unchanged.
        "fails its positive control" in compact(intro)
        and "two-dimensional irregular pipe" in compact(intro)
        and "excluded" in compact(intro)
        and "closure-side transfer" in compact(intro)
        and "nothing in this work advances a momentum equation" in compact(methods),
    )

    e2 = load(ROOT / "manuscript/source_data/diagnostic_e2/causal_wall_history_results.json")
    diff = e2["families"]["diffusion"]["arms"]["closure_history"]["R2_fluct"]
    flow = e2["families"]["flow_matching"]["arms"]["closure_history"]["R2_fluct"]
    check(
        "source_valid_e2_adverse_values_retained",
        diff < 0
        and flow < 0
        and e2["_gates"]["closure_beats_no_wall_both"] is False
        and e2["_gates"]["history_load_bearing_any"] is False,
        diffusion_closure_fluct=diff,
        flow_matching_closure_fluct=flow,
        gates=e2["_gates"],
    )

    custody_path = (
        ROOT
        / "manuscript/source_data/review_audit/closure_interface_custody.json"
    )
    custody = load(custody_path)
    custody_metrics = custody["metrics"]
    custody_contract = custody["frozen_header_contract"]
    custody_map = custody["actual_closure_map"]
    check(
        "frozen_closure_interface_custody",
        custody["status"]
        == "hard_launch_block_calibration_and_branch_contract_not_source_general"
        and custody_contract["contains_branch_network_weights"] is False
        and custody_contract["contains_set_normalisation"] is False
        and custody_metrics["pressure_surrogate_relative_rmse"] > 0.9
        and custody_metrics["closure_output_relative_rmse"] > 0.4
        and custody_metrics["closure_output_sign_changes"] == 1
        and custody_metrics["candidate_feature_cf_r2_all"]
        < custody_metrics["calibration_feature_cf_r2_all"]
        and "not transfer estimates" in compact(supplement)
        and "not source-general closure inference" in compact(supplement),
        custody_sha256=sha256(custody_path),
        pressure_surrogate_relative_rmse=custody_metrics[
            "pressure_surrogate_relative_rmse"
        ],
        closure_output_relative_rmse=custody_metrics[
            "closure_output_relative_rmse"
        ],
    )

    no_space = re.sub(r"\s+", "", methods)
    supplement_no_space = re.sub(r"\s+", "", supplement)
    check(
        "branch_beta_p_definition_complete",
        r"\beta_p=\frac{\delta^\ast_{\rmres}\,\partial_xp}{u_p^2}" in no_space
        and r"u_p=(\nu|\partial_xp|)^{1/3}" in no_space
        and r"y_m\leqy\leq\delta_{99}" in no_space
        and r"\beta_p=0" in methods
        and r"\beta_p=\frac{\delta^\ast_{\rmres}\,\partial_xp}{u_p^2}"
        in supplement_no_space
        and "is kinematic pressure" in compact(methods)
        and "is kinematic pressure" in compact(supplement)
        and "beta_p=dstar_resolved*dpdx/u_p^2"
        in custody_map["beta_p_definition"]
        and "same resolved" in source_readme,
        custody_definition=custody_map["beta_p_definition"],
    )

    check(
        "traction_sign_and_basis_contract",
        r"=-\mathbfP_t\!\left[\nu(" in no_space
        and r"\tau_1^{\,\mathrm{W\toF}}=-\nu" in no_space
        and "geometry-anchored unit tangent fixed independently" in methods
        and "both signed tangential components" in methods
        and "(tau_t1, tau_t2) = -mu * du_t/dn" in traction_extractor
        and "(-MU * samples[\"dudn\"][c][idx])" in traction_extractor
        and "(tau_t1, tau_t2) = mu * du_t/dn" not in traction_extractor,
    )

    native = load(
        ROOT
        / "manuscript/source_data/methods/"
        "cube3d_native_yplus_temporally_separated.json"
    )
    provenance = load(
        ROOT / "manuscript/source_data/methods/cube_les_provenance.json"
    )
    native_meta = native["_meta"]
    provenance_native = provenance["native_yplus"]
    check(
        "native_yplus_temporal_metadata_conservative",
        "retained_fields_integral_time_separated" not in native_meta
        and native_meta[
            "retained_fields_historical_signal_specific_integral_time_separated"
        ]
        is True
        and native_meta[
            "retained_fields_current_conservative_integral_time_separated"
        ]
        is False
        and native_meta["temporal_independence_claim"] is False
        and native_meta["retained_time_separation"]
        > native_meta["historical_signal_specific_integral_time"]
        and native_meta["retained_time_separation"]
        < native_meta["current_conservative_integral_time"]
        and provenance_native[
            "historical_signal_specific_integral_time_separated"
        ]
        is True
        and provenance_native[
            "current_conservative_integral_time_separated"
        ]
        is False
        and provenance_native["temporal_independence_claim"] is False,
        retained_time_separation=native_meta["retained_time_separation"],
        historical_integral_time=native_meta[
            "historical_signal_specific_integral_time"
        ],
        conservative_integral_time=native_meta[
            "current_conservative_integral_time"
        ],
    )

    manuscript_text = "\n".join(
        (main_tex, intro, results, methods, discussion, supplement)
    )
    forbidden_isolation_phrases = (
        "used to isolate whether",
        "M1-to-M2 change isolates",
        "isolates ordinary-convolution padding",
    )
    check(
        "release_wide_causal_and_display_wording",
        not any(phrase in manuscript_text for phrase in forbidden_isolation_phrases)
        and "tests whether a conditional generator changes outside the supplied support"
        in compact(results)
        and "differs from M1 only in using circular padding in the two periodic directions"
        in compact(results)
        and "isolates temporal alignment" not in discussion
        and "Intervals use conservative moving blocks" in compact(results)
        and "E3/E4" not in fig1
        and "WMLES" not in fig1
        and "closure-side transfer" in compact(intro)
        # 2026-08-03: E4 removed from the taxonomy; equivalent scope token.
        and "nothing in this work advances a momentum equation" in compact(methods),
        forbidden_isolation_phrases=list(forbidden_isolation_phrases),
    )

    check(
        "journal_policy_and_upload_boundary",
        (
            checklist_withheld
            or (
                "required, author-approved Methods disclosure" in checklist
                and "keeps exactly one" in checklist
                and "never stored" in checklist
                and "standard class, embedded references and no personal" in checklist
            )
        )
        and not (ROOT / "codes/prepare_single_file_submission.py").exists()
        and not (ROOT / "manuscript/main_submission.tex").exists()
        and not (ROOT / "manuscript/main_submission.pdf").exists(),
        author_only_checklist_withheld=checklist_withheld,
    )

    check(
        "arm_specific_condition_and_common_score_masks",
        r"c_a=[\mathbf o_a,\mathbf 1_{B_a},\mathbf 1_F]" in methods
        and r"B_{\rm absent})\gets(0,\varnothing)" in methods
        and "common score mask" in methods
        and r"U=F\setminus B" in methods
        and "fitted-conditional member" not in methods,
    )

    check(
        "two_cell_band_contract",
        "nominal two-cell band" in methods
        and "2(4/96)h=0.08333h" in methods
        and "nominal two-cell thickness" in supplement
        and "one-cell-thick band" not in readme + status + intro + methods,
    )

    invalidation = load(
        ROOT / "manuscript/source_data/current_fig5/INVALIDATED.json"
    )
    invalid_raster = (
        ROOT
        / "manuscript/source_data/current_fig5/"
        "SOURCE_INVALID_evaluated_bridge_rendering.png"
    )
    legacy_raster = (
        ROOT / "manuscript/source_data/current_fig5/evaluated_bridge_rendering.png"
    )
    check(
        "case1_machine_readable_invalidation",
        invalidation["status"] == "source_invalid_for_scientific_claims"
        and "closure-to-generator composition evidence"
        in invalidation["prohibited_use"]
        and "forensic custody" in invalidation["permitted_use"]
        and invalidation["rendering_status"]
        == "retained_with_unmistakable_source_invalid_filename"
        and "SOURCE_INVALID_evaluated_bridge_rendering.png"
        in invalidation["scope"]
        and invalid_raster.is_file()
        and sha256(invalid_raster)
        == "aae86da0e26a47d546279fe29a01e3d8856d00ab23ac6058aef4f728ed6807f1"
        and not legacy_raster.exists(),
        invalidation_sha256=sha256(
            ROOT / "manuscript/source_data/current_fig5/INVALIDATED.json"
        ),
    )

    supersession_path = (
        ROOT
        / "manuscript/source_data/review_audit/legacy_metadata_supersession.json"
    )
    supersession = load(supersession_path)
    authoritative = supersession["authoritative_dependence"]
    affected_hashes_match = all(
        sha256(ROOT / item["path"]) == item["sha256"]
        for item in supersession["affected_release_records"]
    )
    claim_surfaces = {
        "main": main_tex,
        "introduction": intro,
        "results": results,
        "methods": methods,
        "discussion": discussion,
        "supplement": supplement,
        "source_readme": source_readme,
        "project_readme": readme,
        "project_status": status,
    }
    legacy_phrases = (
        "97.592",
        "11.281",
        "one integral-correlation-time gap",
    )
    qualification_tokens = (
        "historical",
        "supersed",
        "custody",
        "not the current",
        "not current",
        "earlier signal-specific",
    )
    unqualified_legacy_mentions: list[str] = []
    for surface_name, text in claim_surfaces.items():
        lowered = text.lower()
        for phrase in legacy_phrases:
            start = 0
            while True:
                offset = lowered.find(phrase, start)
                if offset < 0:
                    break
                window = lowered[max(0, offset - 350) : offset + len(phrase) + 350]
                if not any(token in window for token in qualification_tokens):
                    unqualified_legacy_mentions.append(f"{surface_name}:{phrase}")
                start = offset + len(phrase)
    check(
        "legacy_dependence_metadata_superseded_machine_readably",
        supersession["status"]
        == "authoritative_review_values_supersede_preserved_producer_metadata"
        and sha256(ROOT / supersession["authoritative_source"]["path"])
        == supersession["authoritative_source"]["sha256"]
        and abs(authoritative["tau_integral_snapshots"] - 122.48173024910811)
        < 1e-12
        and abs(authoritative["gap_in_integral_times"] - 0.8001193304559283)
        < 1e-12
        and abs(
            authoritative["post_spinup_effective_events"] - 8.989095743183439
        )
        < 1e-12
        and affected_hashes_match
        and not unqualified_legacy_mentions
        and "legacy_metadata_supersession.json" in source_readme
        and "released supersession record" in compact(supplement),
        sidecar_sha256=sha256(supersession_path),
        affected_records=len(supersession["affected_release_records"]),
        unqualified_mentions=unqualified_legacy_mentions,
    )

    source_table1 = supplement.find(r"\label{tab:s-evidence}")
    clearpage = supplement.find(r"\clearpage", source_table1)
    source_table2 = supplement.find(r"\label{tab:s-interface}")
    check(
        "supplementary_table_source_order_forced",
        0 <= source_table1 < source_table2,
        source_offsets=[source_table1, source_table2],
    )

    pdf = ROOT / "manuscript/supplementary_information.pdf"
    pdf_order_ok = False
    pdf_offsets = [-1, -1]
    if pdf.is_file():
        run = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            text=True,
            capture_output=True,
            check=False,
        )
        if run.returncode == 0:
            pdf_offsets = [
                run.stdout.find("Supplementary Table 1:"),
                run.stdout.find("Supplementary Table 2:"),
            ]
            pdf_order_ok = 0 <= pdf_offsets[0] < pdf_offsets[1]
    check(
        "compiled_supplementary_table_order",
        pdf_order_ok,
        pdf_offsets=pdf_offsets,
    )

    check(
        "one_global_evidence_recommendation",
        # 2026-08-03 length revision: Discussion opener rephrased; same claim.
        "integrated closure-to-generative-field system" in compact(intro)
        and "evaluates one integrated system end to end" in compact(discussion),
    )

    report = {
        "schema": "gwt-level2-revision-contract-v1",
        "compute": "read-only retained-artifact and compiled-document audit",
        "checks": checks,
        "passed": sum(item["pass"] for item in checks),
        "total": len(checks),
        "all_pass": all(item["pass"] for item in checks),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
