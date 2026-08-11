#!/usr/bin/env python3
"""Build Supplementary Figure 2 from Peidong's maintained scaling artwork.

The source retains Peidong's composition while resolving all quantitative
content from the current committed capacity-adjudication JSON.
"""

from pathlib import Path
import runpy


PROJECT = Path(__file__).resolve().parents[2]
runpy.run_path(
    str(PROJECT / "figure_drafts" / "fig8" / "fig8_scaling_v1.py"),
    run_name="__main__",
)
