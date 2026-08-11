#!/usr/bin/env python3
"""Release entry point for the a-priori closure-composition display."""

from pathlib import Path
import runpy


PROJECT = Path(__file__).resolve().parents[2]
runpy.run_path(
    str(PROJECT / "figure_drafts" / "fig45" / "fig5_composition_transfer_v3.py"),
    run_name="__main__",
)
