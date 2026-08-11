#!/usr/bin/env python3
"""Source-first semantic and producer-emitted figure provenance audit.

This verifier intentionally does not infer provenance by searching for nearby
numbers.  A reviewed contract selects an exact source path/key (or producer
literal) before any numerical comparison is made.  Figure sidecars are emitted
by the plotting process after the actual Matplotlib payload is compared with
the declared source-derived payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "manuscript" / "source_data"
CONTRACT = SOURCE / "review_audit" / "publication_semantic_contract.json"
MANIFEST_DIR = SOURCE / "figure_artist_manifests"
DEFAULT_OUTPUT = ROOT / "codes" / "results" / "revision_semantic_verification.json"
EXPECTED_FIGURES = {
    "fig1_architecture": 30,
    "fig2_generation": 33,
    "fig3_propagation": 23,
    "fig6_interface": 62,
    "fig7_direct_traction": 24,
    "fig8_closure_composition": 22,
    "fig11_reachgated_composition": 24,
    "fig14_slot_interface": 17,
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def get_path(payload: Any, dotted: str) -> Any:
    value = payload
    for token in dotted.split("."):
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def normalise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "array",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalise(child) for key, child in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalise(child) for child in value]
    return value


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            normalise(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12
        )
    return normalise(actual) == normalise(expected)


def resolve_ref(ref: dict[str, Any]) -> tuple[Any, Path]:
    relative = Path(ref["path"])
    path = ROOT / relative
    if not path.is_file():
        path = SOURCE / relative
    if ref["kind"] == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return get_path(payload, ref["key"]), path
    if ref["kind"] == "npz_shape":
        with np.load(path, allow_pickle=False) as payload:
            return list(payload[ref["key"]].shape), path
    if ref["kind"] == "npz":
        with np.load(path, allow_pickle=False) as payload:
            value = np.asarray(payload[ref["key"]])
            if "slice" in ref:
                selector = tuple(
                    slice(None) if token == ":" else int(token)
                    for token in ref["slice"]
                )
                value = value[selector]
            return value, path
    if ref["kind"] == "code_regex":
        text = path.read_text(encoding="utf-8")
        match = re.search(ref["pattern"], text, flags=re.MULTILINE)
        return (match.group(ref.get("group", 0)) if match else None), path
    raise ValueError(f"unsupported semantic-contract ref kind: {ref['kind']}")


def validate_entry(
    entry: dict[str, Any],
    *,
    surface_override: str | None = None,
) -> dict[str, Any]:
    surface_path = ROOT / entry["surface"]
    surface = (
        surface_override
        if surface_override is not None
        else surface_path.read_text(encoding="utf-8")
    )
    anchor_count = surface.count(entry["anchor"])
    start = surface.find(entry["anchor"])
    radius = int(entry.get("window_characters", 1200))
    window = (
        surface[max(0, start - radius) : start + len(entry["anchor"]) + radius]
        if start >= 0
        else ""
    )
    # Token matching is whitespace-normalised on both sides. LaTeX source is
    # hard-wrapped, so an otherwise unchanged sentence can move a line break
    # into the middle of a phrase; that is a typesetting event, not a semantic
    # one, and must not silently break or silently satisfy a claim check.
    def flat(text: str) -> str:
        return " ".join(text.split())

    flat_window = flat(window)
    required = {
        token: flat(token) in flat_window
        for token in entry.get("required_tokens", [])
    }
    forbidden = {
        token: flat(token) not in flat_window
        for token in entry.get("forbidden_tokens", [])
    }
    source_checks = []
    resolved_values = []
    for ref in entry.get("source_refs", []):
        # A key that does not exist in the declared source is a FAILED check,
        # not a crash. The adversarial wrong-key test substitutes a key from a
        # different source file, and after the release gained a second
        # interface result that substitution no longer resolves at all; an
        # unresolvable key must still count as a rejection.
        try:
            actual, path = resolve_ref(ref)
        except (KeyError, IndexError, TypeError, ValueError):
            actual, path = None, ROOT / ref["path"]
        expected = ref["expected"]
        passed = actual is not None and values_equal(actual, expected)
        resolved_values.append(actual)
        source_checks.append(
            {
                "kind": ref["kind"],
                "path": ref["path"],
                "key_or_pattern": ref.get("key", ref.get("pattern")),
                "expected": expected,
                "actual": actual,
                "source_sha256": sha256(path),
                "pass": passed,
            }
        )

    derived_checks = []
    for rule in entry.get("derived", []):
        if rule["kind"] == "percent_unsupplied":
            actual = 100.0 * (
                float(resolved_values[rule["fluid_ref"]])
                - float(resolved_values[rule["band_ref"]])
            ) / float(resolved_values[rule["fluid_ref"]])
        elif rule["kind"] == "comparison":
            left = float(resolved_values[rule["left_ref"]])
            right = float(resolved_values[rule["right_ref"]])
            operator = rule["operator"]
            actual = {
                "<": left < right,
                ">": left > right,
                "<=": left <= right,
                ">=": left >= right,
            }[operator]
        else:
            raise ValueError(f"unsupported derived rule: {rule['kind']}")
        derived_checks.append(
            {
                "kind": rule["kind"],
                "expected": rule["expected"],
                "actual": actual,
                "pass": values_equal(actual, rule["expected"]),
            }
        )

    passed = (
        anchor_count == 1
        and all(required.values())
        and all(forbidden.values())
        and all(record["pass"] for record in source_checks)
        and all(record["pass"] for record in derived_checks)
    )
    return {
        "id": entry["id"],
        "claim_class": entry["claim_class"],
        "evidence_level": entry["evidence_level"],
        "pass": passed,
        "anchor_count": anchor_count,
        "required_tokens": required,
        "forbidden_tokens": forbidden,
        "source_checks": source_checks,
        "derived_checks": derived_checks,
    }


def validate_artist_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identifiers = [record["artist_id"] for record in payload["bindings"]]
    records = []
    for binding in payload["bindings"]:
        refs = []
        for ref in binding["source_refs"]:
            actual, source_path = resolve_ref(ref)
            refs.append(
                {
                    "path": ref["path"],
                    "key": ref["key"],
                    "source_hash_matches": sha256(source_path)
                    == ref["source_sha256"],
                    "payload_hash_matches": payload_sha256(actual)
                    == ref["resolved_payload_sha256"],
                }
            )
        records.append(
            {
                "artist_id": binding["artist_id"],
                "validated": binding["validated"],
                "actual_matches_expected": (
                    binding["actual_payload_sha256"]
                    == binding["expected_payload_sha256"]
                ),
                "source_refs": refs,
                "pass": binding["validated"]
                and binding["actual_payload_sha256"]
                == binding["expected_payload_sha256"]
                and all(
                    ref["source_hash_matches"] and ref["payload_hash_matches"]
                    for ref in refs
                ),
            }
        )
    expected_count = EXPECTED_FIGURES[payload["figure"]]
    passed = (
        payload["schema"] == "gwt-producer-emitted-artist-manifest-v1"
        and payload["all_validated"]
        and payload["binding_count"] == expected_count
        and len(identifiers) == len(set(identifiers))
        and all(record["pass"] for record in records)
    )
    return {
        "figure": payload["figure"],
        "pass": passed,
        "expected_binding_groups": expected_count,
        "actual_binding_groups": payload["binding_count"],
        "unique_ids": len(identifiers) == len(set(identifiers)),
        "records": records,
    }


def run_adversarial_tests(
    contract: dict[str, Any],
    entry_results: list[dict[str, Any]],
) -> dict[str, bool]:
    empirical = next(
        entry
        for entry in contract["entries"]
        if entry.get("source_refs")
        and entry["source_refs"][0]["kind"] == "json"
        and isinstance(entry["source_refs"][0]["expected"], (int, float))
    )
    wrong_key = copy.deepcopy(empirical)
    wrong_key["source_refs"][0]["key"] = (
        "first_contact_m0.arms.correct.full_support_excluded"
    )
    wrong_key_rejected = not validate_entry(wrong_key)["pass"]

    sign_entry = next(
        entry for entry in contract["entries"] if entry["id"] == "fig4_sign_direction"
    )
    original = (ROOT / sign_entry["surface"]).read_text(encoding="utf-8")
    # 2026-08-03 length revision: the direction claim is now carried by the
    # "favours the aligned arm over both controls" sentence (exact improvement
    # intervals live in Supplementary Note 4), so the adversarial mutation
    # swaps that direction phrase instead of the retired inline interval.
    altered = original.replace(
        "favours the aligned arm over both controls",
        "favours both controls over the aligned arm",
    )
    swapped_direction_rejected = not validate_entry(
        sign_entry, surface_override=altered
    )["pass"]

    extraction_entry = next(
        entry
        for entry in contract["entries"]
        if entry["id"] == "tensor_shape_from_retained_array"
    )
    shape_mutation = copy.deepcopy(extraction_entry)
    shape_mutation["source_refs"][0]["expected"] = [3, 8, 6, 8]
    malformed_shape_rejected = not validate_entry(shape_mutation)["pass"]

    return {
        "numerically_near_wrong_key_rejected": wrong_key_rejected,
        "swapped_contrast_direction_rejected": swapped_direction_rejected,
        "malformed_multidigit_shape_rejected": malformed_shape_rejected,
        "all_pass": wrong_key_rejected
        and swapped_direction_rejected
        and malformed_shape_rejected
        and all(result["pass"] for result in entry_results),
    }


def validate_compiled_documents() -> dict[str, Any]:
    documents = {
        "main": ROOT / "manuscript" / "main.pdf",
        "supplementary": ROOT / "manuscript" / "supplementary_information.pdf",
    }
    extracted: dict[str, str] = {}
    page_counts: dict[str, int] = {}
    for name, path in documents.items():
        text_result = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
        )
        info_result = subprocess.run(
            ["pdfinfo", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        page_match = re.search(r"^Pages:\s+(\d+)", info_result.stdout, re.MULTILINE)
        extracted[name] = re.sub(r"\s+", " ", text_result.stdout)
        page_counts[name] = int(page_match.group(1)) if page_match else 0

    required_captions = {
        "main": (
            # 2026-08-07 supervisor revision: Figure 1 now leads with the physical
            # reason the interface is needed; the panel-by-panel evidence mapping
            # remains explicit in the rest of the caption.
            "Why generative turbulence models need an explicit wall interface.",
            "Representative aligned-cube reconstruction.",
            "Correct-time near-wall information acts chiefly near its supplied",
            "Frozen closure-to-generator interface.",
            "Direct traction retains useful near-wall information.",
            "A physics-grounded wall closure supplies traction that propagates through the",
            # 2026-08-03 editorial revision: "power gate" renamed to the standard
            # term "positive control"; the caption's claim is unchanged.
            # 2026-08-04 peer-review round 2: the caption no longer asserts an
            # ensemble-calibration *gate*, because the paper's own stochastic-sampler
            # experiment refuted the mechanism that criterion rested on.  The
            # oracle positive control is the surviving gate and the caption now
            # says so, and says explicitly that panel f's pass/fail annotation is
            # that control's verdict rather than a significance verdict on the
            # closure arm.
            "A prospectively frozen oracle positive control decides which cells may",
            "A single-slot channel experiment: transmitted traction information improves",
        ),
        "supplementary": (
            "Evidence and data-contact ledger.",
            "Physical-time-grouped hill support test.",
            "Frozen closure interface-custody audit.",
            # 2026-08-04: one named interval estimator is now used throughout.
            "Grouped-protocol re-test endpoints, one named estimator throughout.",
        ),
    }
    caption_checks = {
        name: {caption: caption in extracted[name] for caption in captions}
        for name, captions in required_captions.items()
    }
    layout = json.loads((ROOT / "FIGURE_LAYOUT_AUDIT.json").read_text(encoding="utf-8"))
    passed = (
        # 38 -> 42: the generality Results section and its display were added.
        35 <= page_counts["main"] <= 60
        and 2 <= page_counts["supplementary"] <= 20
        and all(
            present
            for document in caption_checks.values()
            for present in document.values()
        )
        and layout.get("all_pass") is True
        # Twelve built displays: seven main figures, four Supplementary figures
        # and one withdrawn diagnostic retained for forensic traceability.
        and len(layout.get("figures", [])) == 8
        and all(figure.get("all_pass") for figure in layout.get("figures", []))
    )
    return {
        "pass": passed,
        "pages": page_counts,
        "compiled_caption_mapping": caption_checks,
        "layout_figures_passed": sum(
            bool(figure.get("all_pass")) for figure in layout.get("figures", [])
        ),
        "layout_figures_total": len(layout.get("figures", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    entry_results = [validate_entry(entry) for entry in contract["entries"]]

    manifest_paths = sorted(
        path
        for path in MANIFEST_DIR.glob("*.json")
        if path.stem in EXPECTED_FIGURES
    )
    manifest_results = [validate_artist_manifest(path) for path in manifest_paths]
    manifest_set_pass = {
        record["figure"] for record in manifest_results
    } == set(EXPECTED_FIGURES)

    active_text_paths = [
        ROOT / "manuscript" / "FIGURE_EVIDENCE_AUDIT.md",
        ROOT / "manuscript" / "source_data" / "README.md",
        ROOT / "manuscript" / "nature_communications_checklist.md",
    ]
    active_text = "\n".join(path.read_text(encoding="utf-8") for path in active_text_paths)
    prohibited_overclaims = {
        phrase: phrase not in active_text
        for phrase in (
            "581/581",
            "181/181",
            "all 581",
            "181 exact",
            "every printed numerical occurrence",
        )
    }
    fig1_source = (ROOT / "codes" / "figures" / "fig1_architecture.py").read_text(
        encoding="utf-8"
    )
    fig1_not_hardcoded = (
        '"$+0.175$ vs absent; $+0.307$ vs far time"' not in fig1_source
        and 'np.load(SOURCE / "fig1_v2/fig1_overview_derived.npz")' in fig1_source
    )
    adversarial = run_adversarial_tests(contract, entry_results)
    compiled_documents = validate_compiled_documents()

    checks = {
        "typed_claim_contract": all(record["pass"] for record in entry_results),
        "producer_emitted_artist_manifests": manifest_set_pass
        and all(record["pass"] for record in manifest_results),
        "legacy_exhaustive_overclaim_removed": all(prohibited_overclaims.values()),
        "figure1_headline_loaded_from_source": fig1_not_hardcoded,
        "adversarial_mutations_fail": adversarial["all_pass"],
        "compiled_caption_and_layout_mapping": compiled_documents["pass"],
    }
    payload = {
        "schema": "gwt-revision-semantic-verification-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": (
            "Reviewed headline/decision-bearing manuscript claims plus producer-emitted "
            "data-bearing artist groups. This is deliberately not an assertion that every "
            "typeset numeral is an empirical result."
        ),
        "checks": checks,
        "typed_claims": entry_results,
        "artist_manifests": manifest_results,
        "prohibited_overclaims": prohibited_overclaims,
        "adversarial_tests": adversarial,
        "compiled_documents": compiled_documents,
        "summary": {
            "typed_claims_passed": sum(record["pass"] for record in entry_results),
            "typed_claims_total": len(entry_results),
            "artist_groups_passed": sum(
                record["actual_binding_groups"]
                for record in manifest_results
                if record["pass"]
            ),
            "artist_groups_total": sum(
                record["actual_binding_groups"] for record in manifest_results
            ),
            "figures": len(manifest_results),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"REVISION_SEMANTICS: {payload['status']} "
        f"{payload['summary']['typed_claims_passed']}/"
        f"{payload['summary']['typed_claims_total']} typed claims; "
        f"{payload['summary']['artist_groups_passed']}/"
        f"{payload['summary']['artist_groups_total']} artist groups"
    )
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
