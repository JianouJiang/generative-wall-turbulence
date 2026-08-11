#!/usr/bin/env bash
# Rebuild the eight active main figures and the withdrawn diagnostics.
set -euo pipefail

cd "$(dirname "$0")/.."
export SOURCE_DATE_EPOCH=1785430800
exec 9>.submission_release.lock
printf '%s\n' "== [lock] Waiting for the submission release lock =="
flock 9
export GWT_SUBMISSION_LOCK_HELD=1
printf '%s\n' "== [lock] Submission release lock acquired =="

printf '%s\n' "== [1/6] Verify sources; rebuild eight active displays plus one withdrawn diagnostic =="
python3 codes/figures/rebuild_submission_figures.py

printf '%s\n' "== [2/6] Validate active raster figures =="
python3 manuscript/validate_submission_figures.py

printf '%s\n' "== [3/6] Clean-build manuscript and Supplementary Information =="
(
  cd manuscript
  latexmk -C main.tex
  latexmk -C supplementary_information.tex
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_information.tex
)
printf '%s\n' "== [4/6] Refresh source/PDF hashes after compilation =="
python3 codes/figures/rebuild_submission_figures.py --manifest-only

printf '%s\n' "== [5/6] Replay endpoints and audit the release package =="
python3 codes/probes/audit_closure_interface_custody.py --verify-retained
python3 codes/probes/audit_level1_protocol.py
python3 codes/probes/verify_submission_methodology.py \
  --output "${GWT_VERIFICATION_OUTPUT:-codes/results/submission_verification.json}"
python3 codes/probes/verify_revision_semantics.py \
  --output "${GWT_REVISION_SEMANTIC_OUTPUT:-codes/results/revision_semantic_verification.json}"
python3 codes/probes/verify_level3_reader_consequences.py \
  --output "${GWT_LEVEL3_CONSEQUENCE_OUTPUT:-codes/results/level3_reader_consequence_verification.json}"

printf '%s\n' "== [6/6] Verify complete release manifest =="
sha256sum -c FIGURE_DATA_SHA256SUMS

printf '%s\n' "REPRODUCE_ALL: ALL_PASS"
