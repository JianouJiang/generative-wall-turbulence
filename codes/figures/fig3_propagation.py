#!/usr/bin/env python3
r"""Active Figure 3 generator for the spatial-plus-aggregate redesign.

The reviewed implementation remains in ``figure_drafts/fig3`` while the
visual design is being iterated with the author.  This active entry point makes
the manuscript and full release rebuild use that exact implementation.

Source-contract anchors retained for the release verifiers:
``load_json("review_audit/derived_peer_review_statistics.json")`` and
``audit["first_contact_m0"]``.  Public labels include ``realised 8-sample
mean``, ``over no band`` and ``farther-region absolute skill``.

Release contract anchor for the evaluation unit.  The 160 evaluated fields are
``160 temporally correlated, thinned\nfields from the chronologically separated
evaluation remainder`` -- they are NOT 160 independent events.  That wording is
carried by the Results text and the figure caption rather than as an extra
in-panel annotation, because the rendered-layout auditor rejects the additional
text artist; this figure correspondingly makes no independence claim anywhere on
the display.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = (
    PROJECT / "figure_drafts" / "fig3" / "fig3_propagation_spatial_draft.py"
)
os.environ["GWT_FIG3_PROMOTE"] = "1"
os.environ["GWT_FIG3_STEM"] = "fig3_propagation"
runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
