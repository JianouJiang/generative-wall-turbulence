#!/usr/bin/env python3
"""Render the withdrawn Case1 diagnostic for forensic custody.

The figure uses only frozen evaluated results: committed numerical ledgers
and crops from the committed bridge-composition rendering.  No simulation,
training, posterior regeneration or new statistical analysis is performed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "codes" / "figures"))
from _submission import bind_artist, configure, save  # noqa: E402

configure()

RESULTS = Path(os.environ.get(
    "GWT_SOURCE_DATA", ROOT / "manuscript" / "source_data"
)) / "current_fig5"


def load(name: str):
    with (RESULTS / name).open() as handle:
        return json.load(handle)


DIRECT = load("wallstress_cond_case1_bh1_results.json")
BRIDGE = load("l2_case1_closure_results.json")
SOURCE = plt.imread(RESULTS / "SOURCE_INVALID_evaluated_bridge_rendering.png")

BLUE = "#2C6FBB"
RED = "#C53B2C"
ORANGE = "#C66A3D"
GREEN = "#278A5B"
GREY = "#777777"
DARK = "#202020"
FCOL = {"diffusion": BLUE, "flow_matching": RED}
FLAB = {"diffusion": "Diffusion", "flow_matching": "Flow matching"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.51,
        "ytick.labelsize": 6.51,
        "legend.fontsize": 6.51,
        "axes.linewidth": 1.1,
        "savefig.dpi": 400,
    }
)

fig = plt.figure(figsize=(7.2, 5.0))
grid = fig.add_gridspec(
    2,
    3,
    left=0.065,
    right=0.955,
    bottom=0.09,
    top=0.94,
    height_ratios=[1.0, 1.05],
    hspace=0.18,
    wspace=0.54,
)


def panel(ax, letter, title):
    ax.text(-0.14, 1.09, letter, transform=ax.transAxes, fontsize=7.2,
            fontweight="bold", va="top")
    ax.set_title(title, loc="left", pad=6, fontweight="bold")


def raster_panel(ax, letter, title, crop):
    panel(ax, letter, title)
    ax.imshow(crop)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#505050")
        spine.set_linewidth(1.05)


# Evaluated fields from the committed equilibrium-bridge composition.
dns = SOURCE[610:1015, 755:1215]
generated = SOURCE[610:1015, 1370:1838]
stress_trace = SOURCE[118:505, 748:1223]

raster_panel(fig.add_subplot(grid[0, 0]), "a", "Source-record mean (mislabelled)", dns)
raster_panel(fig.add_subplot(grid[0, 1]), "b", "Derived ensemble mean (withdrawn)", generated)

# c | A larger local comparison of the separated shear-layer turn.
ax = fig.add_subplot(grid[0, 2])
panel(ax, "c", "Raster crop (no separation claim)")
dns_zoom = dns[155:395, 175:455]
gen_zoom = generated[155:395, 175:463]
canvas = np.ones((max(dns_zoom.shape[0], gen_zoom.shape[0]),
                  dns_zoom.shape[1] + gen_zoom.shape[1] + 12, 4))
canvas[:, :, :3] = 1.0
canvas[:, :, 3] = 1.0
canvas[:dns_zoom.shape[0], :dns_zoom.shape[1]] = dns_zoom
canvas[:gen_zoom.shape[0], dns_zoom.shape[1] + 12:] = gen_zoom
ax.imshow(canvas)
ax.axvline(dns_zoom.shape[1] + 5.5, color="white", lw=2)
ax.text(0.24, 0.06, "source", transform=ax.transAxes, ha="center", color=DARK,
        fontsize=6.6, bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"})
ax.text(0.76, 0.06, "derived", transform=ax.transAxes, ha="center", color=DARK,
        fontsize=6.6, bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"})
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("#505050")
    spine.set_linewidth(1.05)

fig.text(
    0.985,
    0.965,
    "WITHDRAWN: Case1 is irregular-pipe $(u,v,p)$; local wall-jet geometry/stress unsupported",
    ha="right",
    fontsize=6.8,
    color=RED,
    fontweight="bold",
)


# d | Evaluated signed stress feature in the a-priori two-dimensional composition.
ax = fig.add_subplot(grid[1, 0])
raster_panel(ax, "d", "Derived stress trace (source-invalid)", stress_trace)
ax.text(
    0.03,
    0.05,
    "computed corr$(\\tau_w)=0.90$\n"
    "sign agreement $=0.81$\n"
    "source-invalid; no physical claim",
    transform=ax.transAxes,
    fontsize=6.51,
    color=DARK,
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.2},
)
ax.text(
    0.04,
    0.90,
    "reference",
    transform=ax.transAxes,
    fontsize=6.51,
    color=BLUE,
    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.8},
)
ax.text(
    0.04,
    0.82,
    "closure",
    transform=ax.transAxes,
    fontsize=6.51,
    color=ORANGE,
    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.8},
)


# e | Bridge versus direct-stress features, shown as conditional intervals.
ax = fig.add_subplot(grid[1, 1])
panel(ax, "e", "Retained contrast (no physical claim)")
rows = [
    ("diffusion", "passthrough", "Diffusion\nreference"),
    ("diffusion", "closure", "Diffusion\nclosure"),
    ("flow_matching", "passthrough", "Flow matching\nreference"),
    ("flow_matching", "closure", "Flow matching\nclosure"),
]
y = np.arange(len(rows))[::-1]
for yi, (fam, arm, _) in zip(y, rows):
    key = f"d_{arm}_minus_no_wall"
    for ledger, marker, color, label in [
        (BRIDGE, "o", ORANGE, "equilibrium bridge"),
        (DIRECT, "s", FCOL[fam], "direct stress"),
    ]:
        stat = ledger["families"][fam]["boot_total"][key]
        mean = stat["mean"]
        lo, hi = stat["ci95"]
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt=marker,
            ms=4.7,
            color=color,
            mfc="white" if marker == "o" else color,
            mec=color,
            capsize=2.5,
            lw=1.05,
            zorder=3,
            label=label if yi == y[0] else None,
        )
ax.axvline(0, color=DARK, lw=1.05)
ax.set_yticks(y)
ax.set_yticklabels([r[2] for r in rows])
ax.set_xlabel(r"$\Delta R^2_{\rm total}$ over absent band")
ax.grid(axis="x", color="#DDDDDD", lw=1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right", handletextpad=0.4)
# f | CRPS comparison for both generator families.
ax = fig.add_subplot(grid[1, 2])
panel(ax, "f", "Pooled score (not calibration)")
for yi, fam in enumerate(("diffusion", "flow_matching")):
    uq = DIRECT["families"][fam]["uq"]["closure"]
    x0, x1 = uq["CRPS_ensemble"], uq["CRPS_point"]
    line = ax.plot([x0, x1], [yi, yi], color=FCOL[fam], lw=1.4, zorder=2)[0]
    bind_artist(
        fig,
        line,
        artist_id=f"withdrawn_fig6.f.{fam}_crps_segment",
        panel="f",
        source_refs=[
            {
                "kind": "json",
                "path": "current_fig5/wallstress_cond_case1_bh1_results.json",
                "key": f"families.{fam}.uq.closure.CRPS_ensemble",
            },
            {
                "kind": "json",
                "path": "current_fig5/wallstress_cond_case1_bh1_results.json",
                "key": f"families.{fam}.uq.closure.CRPS_point",
            },
        ],
        source_payload=[x0, x1],
        expected_payload={
            "type": "Line2D",
            "x": np.asarray([x0, x1], dtype=float),
            "y": np.asarray([yi, yi], dtype=float),
        },
        transform="connect retained ensemble and point CRPS for the same source-invalid arm",
        evidence="forensic source-invalid diagnostic; supports no physical claim",
    )
    ax.scatter(x0, yi, marker="o", s=38, color=FCOL[fam], edgecolor="white",
               zorder=3, label="conditional ensemble" if yi == 0 else None)
    ax.scatter(x1, yi, marker="D", s=31, facecolor="white", edgecolor=FCOL[fam],
               lw=1.2, zorder=3, label="ensemble mean" if yi == 0 else None)
    ax.text(
        x0,
        yi + 0.16,
        f"{100 * uq['crps_skill_vs_point']:.1f}% CRPS skill\n"
        f"descriptive hit fraction$_{{0.9}}={uq['coverage90']:.2f}$",
        ha="left",
        fontsize=6.51,
        color=FCOL[fam],
    )
ax.set_yticks([0, 1])
ax.set_yticklabels(["Diffusion", "Flow matching"])
ax.set_ylim(-0.45, 1.48)
ax.set_xlabel("CRPS (lower is better)")
ax.grid(axis="x", color="#DDDDDD", lw=1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right", handletextpad=0.3)

save(fig, "fig6_composition")
