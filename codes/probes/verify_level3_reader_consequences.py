#!/usr/bin/env python3
"""Verify what a reader can infer from every decision-bearing result and display.

This is deliberately not a census of typeset numbers.  It starts from the
scientific decisions exposed to a reader--support, absolute versus differential
skill, dependence, controls, distributional/physical outcomes, capacity and
evidence level--and checks those decisions against retained arrays and public
figure text.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "manuscript" / "source_data"
FIGURES = ROOT / "manuscript" / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, tolerance: float = 1.0e-7) -> bool:
    return bool(abs(float(actual) - float(expected)) <= tolerance)


def pdf_text(path: Path) -> str:
    process = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return re.sub(r"\s+", " ", process.stdout).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "codes" / "results" / "level3_reader_consequence_verification.json",
    )
    args = parser.parse_args()

    derived = load_json(SOURCE / "review_audit" / "derived_peer_review_statistics.json")
    m2 = load_json(SOURCE / "fig3" / "cube_periodic_topology_results.json")
    rapid = load_json(SOURCE / "fig4" / "cube_distributional_rapid_results.json")
    capacity = load_json(SOURCE / "fig7" / "capacity_crossed_adjudication.json")
    closure = load_json(SOURCE / "review_audit" / "closure_interface_custody.json")
    retrieval = load_json(SOURCE / "review_audit" / "cube_train_band_retrieval_results.json")
    layout = load_json(ROOT / "FIGURE_LAYOUT_AUDIT.json")

    checks: dict[str, dict[str, Any]] = {}

    representative = np.load(SOURCE / "fig2" / "cube3d_representative_fields.npz")
    fluid = int(np.count_nonzero(representative["fluid"]))
    band = int(np.count_nonzero(representative["band"]))
    unsupplied = fluid - band
    scored_percent = 100.0 * unsupplied / fluid
    band_percent = 100.0 * band / fluid
    checks["support_geometry"] = {
        "pass": fluid == 207360 and band == 14008 and unsupplied == 193352
        and close(scored_percent, 93.2445987654321, 1.0e-10),
        "fluid_cells": fluid,
        "supplied_band_cells": band,
        "unsupplied_scored_cells": unsupplied,
        "scored_percent": scored_percent,
        "band_percent": band_percent,
        "consequence": "All reported field scores exclude the 6.76% supplied band and cover the remaining 93.24% of fluid cells.",
    }

    m0 = derived["first_contact_m0"]
    m0_arithmetic = []
    for contrast, control in (("correct_minus_no_wall", "no_wall"), ("correct_minus_wrong_wall", "wrong_wall")):
        for region in ("full_support_excluded", "near_support_excluded_d_le_0p5h", "outer_d_gt_0p5h"):
            m0_arithmetic.append(
                close(
                    m0["differences"][contrast][region],
                    m0["arms"]["correct"][region] - m0["arms"][control][region],
                    1.0e-12,
                )
            )
    m0_aligned = m0["arms"]["correct"]
    m0_deltas = [value for contrast in m0["differences"].values() for value in contrast.values()]
    checks["m0_absolute_and_differential_consequence"] = {
        "pass": all(m0_arithmetic)
        and m0_aligned["full_support_excluded"] > 0
        and m0_aligned["near_support_excluded_d_le_0p5h"] > 0
        and m0_aligned["outer_d_gt_0p5h"] < 0
        and all(value > 0 for value in m0_deltas),
        "aligned_absolute_skill": m0_aligned,
        "paired_differences": m0["differences"],
        "consequence": "The aligned band improves every reported region over both controls, but its farther-region absolute skill remains negative.",
    }

    m2_eval = m2["evaluation"]
    m2_aligned = m2_eval["arms"]["correct"]
    m2_outer = m2_aligned["outer_d_gt_0p5h"]["R2_fluct_balanced"]
    m2_outer_delta = m2_eval["deltas"]["correct_minus_no_wall"]["outer_d_gt_0p5h"]
    checks["m2_interval_consequence"] = {
        "pass": m2_aligned["full_support_excluded"]["R2_fluct_balanced"] > 0
        and m2_aligned["near_support_excluded_d_le_0p5h"]["R2_fluct_balanced"] > 0
        and m2_outer < 0
        and m2_outer_delta["point"] > 0
        and m2_outer_delta["ci95"][0] > 0
        and not m2["topology_gates"]["positive_outer_absolute_skill"],
        "aligned_complete": m2_aligned["full_support_excluded"],
        "aligned_near": m2_aligned["near_support_excluded_d_le_0p5h"],
        "aligned_farther": m2_aligned["outer_d_gt_0p5h"],
        "aligned_minus_absent_farther": m2_outer_delta,
        "consequence": "Conservative block intervals preserve the differential farther-region gain while rejecting positive farther-region absolute skill.",
    }

    controls = derived["committed_comparators"]
    direct = controls["deterministic_direct_correct_band"]
    wiener = controls["linear_wiener"]
    shell = controls["quarantined_location_comparator"]
    retrieval_regions = retrieval["nearest_training_band"]["r2_fluctuation"]
    checks["method_attribution_controls"] = {
        "pass": all(row["complete"] > 0 and row["near"] > 0 and row["outer"] < 0 for row in direct)
        and wiener["near_wall_band"]["near"] > 0
        and wiener["near_wall_band"]["complete"] < 0
        and wiener["near_wall_band"]["outer"] < 0
        and all(value < 0 for value in retrieval_regions.values())
        and shell["active_claim"] is False
        and shell["complete_overlap_cells"] == shell["shell_cells"],
        "deterministic_seeds": direct,
        "linear_wiener_near_wall_band": wiener["near_wall_band"],
        "nearest_training_retrieval": retrieval_regions,
        "quarantined_shell_overlap_cells": shell["complete_overlap_cells"],
        "consequence": "Simple contacted controls reproduce only near-wall skill; retrieval fails everywhere; the overlapping interior comparator remains excluded.",
    }

    components = np.load(SOURCE / "fig4" / "cube_distributional_rapid_components.npz")
    test_idx = components["test_idx"].astype(int)
    donor_idx = components["donor_idx"].astype(int)
    dependence = derived["dependence"]
    test_steps = sorted(set(np.diff(test_idx).tolist()))
    donor_gaps = sorted(set(np.abs(donor_idx - test_idx).tolist()))
    checks["temporal_dependence_and_control_separation"] = {
        "pass": len(test_idx) == 160
        and test_steps == [2, 3]
        and donor_gaps == [172, 173]
        and close(dependence["gap_in_conservative_integral_times"], 98.0 / dependence["conservative_tau_snapshots"], 1.0e-12)
        and close(dependence["evaluation_span_in_conservative_integral_times"], 343.0 / dependence["conservative_tau_snapshots"], 1.0e-12),
        "evaluated_correlated_fields": len(test_idx),
        "test_index_steps": test_steps,
        "far_time_absolute_gaps": donor_gaps,
        "split_gap_integral_times": dependence["gap_in_conservative_integral_times"],
        "evaluation_span_integral_times_count_convention": dependence["evaluation_span_in_conservative_integral_times"],
        "consequence": "The 160 thinned fields are correlated observations over 2.800 integral times, not 160 independent events; far-time donors are 172--173 snapshots away.",
    }

    fair_expected = derived["fair_energy_score"]["block57_conservative"]["arms"]
    fair_actual: dict[str, float] = {}
    for arm in ("correct", "no_wall", "far_time_wall"):
        truth_distance = components[f"energy_truth_distance__{arm}"]
        pair_distance = components[f"energy_pair_distance__{arm}"]
        members = pair_distance.shape[1]
        diagonal = np.trace(pair_distance, axis1=1, axis2=2)
        off_diagonal = (pair_distance.sum(axis=(1, 2)) - diagonal) / (2 * members * (members - 1))
        fair_actual[arm] = float(np.mean(truth_distance.mean(axis=1) - off_diagonal))
    fair_close = [close(fair_actual[arm], fair_expected[arm]["mean"], 3.0e-8) for arm in fair_actual]
    fair_improvements = derived["fair_energy_score"]["block57_conservative"]["improvements"]
    checks["fair_energy_score_from_retained_arrays"] = {
        "pass": all(fair_close)
        and fair_actual["correct"] < fair_actual["no_wall"] < fair_actual["far_time_wall"]
        and all(item["ci95"][0] > 0 for item in fair_improvements.values()),
        "recomputed_means": fair_actual,
        "audited_means": {arm: record["mean"] for arm, record in fair_expected.items()},
        "paired_control_minus_aligned_improvements": fair_improvements,
        "consequence": "The unbiased finite-ensemble correction preserves the aligned-arm ordering, with both paired control-minus-aligned intervals strictly positive.",
    }

    physical = derived["physical_statistics"]["block57_conservative"]
    spectrum = physical["component_spectrum_log_rmse"]
    profile = physical["reynolds_stress_profile_nrmse"]
    checks["physical_corroboration_null_preserved"] = {
        "pass": spectrum["arms"]["correct"]["mean"] < spectrum["arms"]["no_wall"]["mean"]
        and profile["arms"]["correct"]["mean"] < profile["arms"]["no_wall"]["mean"]
        and spectrum["improvements"]["far_time_wall_minus_correct"]["ci95"][0] < 0
        and spectrum["improvements"]["far_time_wall_minus_correct"]["ci95"][1] > 0
        and profile["improvements"]["far_time_wall_minus_correct"]["ci95"][0] < 0
        and profile["improvements"]["far_time_wall_minus_correct"]["ci95"][1] > 0
        and derived["physical_statistics"]["withdrawn_family"]["used_for_adjudication"] is False
        and rapid["registered_gates"]["rapid_distributional_claim_pass"] is False,
        "spectrum": spectrum,
        "quadratic_profile": profile,
        "descriptive_hit_fractions": rapid["coverage_descriptives"],
        "consequence": "Both valid physical families beat absence but neither separates from far-time; interval hits remain descriptive and the registered distributional gate fails.",
    }

    slope = capacity["crossed_capacity_slopes"]["correct"]
    largest = capacity["points"]["120"]["correct"]
    checks["capacity_negative_result_visible"] = {
        "pass": largest["mean_across_seed_point_estimates"] < 0
        and largest["crossed_ci95"][1] < 0
        and slope["mean"] < 0
        and slope["crossed_ci95"][1] < 0
        and capacity["convergence"]["parity_pass"] is False
        and capacity["gates"]["scaling_claim_pass"] is False,
        "largest_width_aligned_skill": largest,
        "crossed_slope": slope,
        "convergence_parity": capacity["convergence"]["parity_pass"],
        "scaling_claim_pass": capacity["gates"]["scaling_claim_pass"],
        "consequence": "At fixed updates, absolute skill and its crossed slope are negative; absent convergence parity, no capacity-scaling claim is supported.",
    }

    checks["closure_interface_and_evidence_boundary"] = {
        "pass": closure["adjudication"]["source_general_branch_reconstruction_from_header_possible"] is False
        and closure["adjudication"]["fixed_embedding_transplant_permitted"] is False
        and closure["metrics"]["pressure_surrogate_relative_rmse"] > 0.9
        and closure["metrics"]["pressure_surrogate_pearson"] < 0.1
        and closure["status"] == "hard_launch_block_calibration_and_branch_contract_not_source_general",
        "pressure_surrogate_relative_rmse": closure["metrics"]["pressure_surrogate_relative_rmse"],
        "pressure_surrogate_pearson": closure["metrics"]["pressure_surrogate_pearson"],
        "closure_output_relative_rmse": closure["metrics"]["closure_output_relative_rmse"],
        "branch_weights_present": closure["frozen_header_contract"]["contains_branch_network_weights"],
        "set_normalisation_present": closure["frozen_header_contract"]["contains_set_normalisation"],
        "consequence": "The load-bearing closure-to-generator interface is not source-general; E2 remains unestablished and cannot be promoted to runtime or solver evidence.",
    }

    expected_figures = [
        "fig1_architecture", "fig2_generation", "fig3_propagation",
        "fig6_interface", "fig7_direct_traction", "fig8_closure_composition",
        "fig11_reachgated_composition", "fig14_slot_interface",
    ]
    public_text = {stem: pdf_text(FIGURES / f"{stem}.pdf") for stem in expected_figures}
    forbidden = {
        stem: [pattern for pattern in (
            r"\bcorrect-arm\b",
            r"\bcorrect-minus\b",
            r"\bno_wall\b",
            r"\bwrong wall\b",
            r"\bconditional-mean fluctuation\b",
        )
               if re.search(pattern, text, flags=re.IGNORECASE)]
        for stem, text in public_text.items()
    }
    required_public_tokens = {
        # 2026-08-07 supervisor revision: Figure 1 now names the stress-test
        # purpose and expresses equal footing as every control entering the
        # same slot; retain those reader-facing guarantees verbatim here.
        "fig1_architecture": [
            "one slot",
            "Three stress-test records",
            "controls through the SAME slot",
        ],
        "fig2_generation": ["8-sample mean", "absent-band", "equal-support", "far-time"],
        "fig3_propagation": ["realised 8-sample mean", "Support-excluded evaluation", "farther"],
        "fig6_interface": ["Absolute skill outside the supplied band", "Lift fidelity before sampling", "far-time band"],
        "fig7_direct_traction": ["complete-volume skill", "oracle first-cell traction", "sign-flipped traction"],
        "fig8_closure_composition": ["closure traction", "absolute skill", "a-priori traction", "paired contrasts"],
        # 2026-08-03 editorial revision: "gate" renamed to the standard term
        # "control" (oracle positive control); panel semantics unchanged.
        "fig11_reachgated_composition": ["hill: \u0394R 2 vs absent", "cube: control passes", "true traction"],
        "fig14_slot_interface": ["wall-normal reach", "field gain against measured traction fidelity"],
    }
    token_misses = {
        stem: [token for token in tokens if token not in public_text[stem]]
        for stem, tokens in required_public_tokens.items()
    }
    checks["public_figure_semantics"] = {
        "pass": all(not hits for hits in forbidden.values()) and all(not misses for misses in token_misses.values()),
        "forbidden_historical_label_hits": forbidden,
        "required_token_misses": token_misses,
        "consequence": "All eight release displays use public semantic labels; historical array keys do not escape into the reader-facing interpretation.",
    }

    layout_by_stem = {item["stem"]: item for item in layout["figures"]}
    checks["publication_size_layout"] = {
        "pass": layout["all_pass"] is True
        and set(expected_figures) == set(layout_by_stem)
        and all(item["all_pass"] for item in layout_by_stem.values())
        and all(item["minimum_effective_stroke_pt_at_180mm"] >= 0.98 for item in layout_by_stem.values())
        and all(not item["final_font_size_failures"] for item in layout_by_stem.values())
        and all(not item["final_stroke_failures"] for item in layout_by_stem.values())
        and all(not item["canvas_overflow"] for item in layout_by_stem.values()),
        "figures": {
            stem: {
                "all_pass": item["all_pass"],
                "font_range_pt_at_180mm": item["effective_font_pt_range_at_180mm"],
                "minimum_stroke_pt_at_180mm": item["minimum_effective_stroke_pt_at_180mm"],
            }
            for stem, item in layout_by_stem.items()
        },
        "consequence": "Every active or explicitly withdrawn release display passes the 180-mm font, stroke, overflow and collision audit.",
    }

    main_text = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    results_text = (ROOT / "manuscript" / "sections" / "results.tex").read_text(encoding="utf-8")
    supplement_text = (ROOT / "manuscript" / "supplementary.tex").read_text(encoding="utf-8")
    compact_results = re.sub(r"\s+", " ", results_text)
    # 2026-08-03 length revision: the realised-eight-sample-mean disclosure
    # was consolidated into the Methods statistical-conventions paragraph.
    compact_methods = re.sub(
        r"\s+",
        " ",
        (ROOT / "manuscript" / "sections" / "methods.tex").read_text(encoding="utf-8"),
    )
    checks["reader_wording_contract"] = {
        "pass": "when generative turbulence models hit the wall" in main_text
        and "one realised eight-sample mean" in compact_methods
        and "realised 8-sample mean" in public_text["fig3_propagation"]
        and "conditional-mean fluctuation" not in public_text["fig3_propagation"]
        and "conditional-mean fluctuation" not in (
            ROOT / "codes" / "figures" / "fig3_propagation.py"
        ).read_text(encoding="utf-8")
        and "closure $-$ absent band" in supplement_text
        and "closure-minus-absence" in supplement_text
        # 2026-08-03 editorial revision: the ledger's explicit "solver coupling
        # not run" row was removed with the E4 taxonomy entry; the same reader
        # guarantee is preserved by the composition row's scope column.
        and "not solver-coupled" in supplement_text,
        "consequence": "Title, main-text estimands and withdrawn Supplementary controls preserve the same evaluated contribution and evidence ladder.",
    }

    all_pass = all(record["pass"] for record in checks.values())
    payload = {
        "schema": "gwt-level3-reader-consequence-verification-v1",
        "all_pass": all_pass,
        "approach": "publication-outward scientific-consequence audit; not a numeric string census",
        "compute_boundary": "Deterministic inspection and arithmetic from committed arrays, JSON and built PDFs; no simulation, training or neural inference.",
        "checks": checks,
        "summary": {
            "passed": sum(record["pass"] for record in checks.values()),
            "total": len(checks),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))
    print(f"LEVEL3_READER_CONSEQUENCES: {'ALL_PASS' if all_pass else 'FAIL'}")
    print(f"output={args.output}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
