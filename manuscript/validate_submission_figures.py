#!/usr/bin/env python3
"""Read-only check of seven active figures and one withdrawn diagnostic."""
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
MIN_DOUBLE_COLUMN_WIDTH = 2080
EXPECTED = [
    "fig1_architecture.png",
    "fig2_generation.png",
    "fig3_propagation.png",
    "fig4_mechanism_robustness.png",
    "fig5_physical_composite.png",
    "fig6_composition.png",
    "fig6_3d.png",
    "fig7_scaling.png",
]


def main() -> int:
    missing = [name for name in EXPECTED if not (FIGURES / name).is_file()]
    undersized = []
    for index, name in enumerate(EXPECTED, start=1):
        path = FIGURES / name
        if path.is_file():
            with Image.open(path) as image:
                width, height = image.size
            if width < MIN_DOUBLE_COLUMN_WIDTH:
                undersized.append(name)
            status = f"{width}x{height}px, {path.stat().st_size / 1024:.0f} KiB"
        else:
            status = "MISSING"
        print(f"Figure {index}: {name:38s} {status}")
    print("Active main figures: 5; Supplementary figures: 2; withdrawn diagnostics: 1")
    print("Main tables: 2; algorithms: 1; active main display items: 8")
    if missing:
        print("Missing: " + ", ".join(missing))
        return 1
    if undersized:
        print(
            f"Below {MIN_DOUBLE_COLUMN_WIDTH}px double-column width: "
            + ", ".join(undersized)
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
