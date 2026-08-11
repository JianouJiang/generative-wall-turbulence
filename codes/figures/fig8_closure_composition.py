#!/usr/bin/env python3
"""Figure: the source-valid closure-to-generator composition (paper Fig. 6).

Redesign 2026-08-02 v2: flow fields carry the figure.  Rows 1-2 show the
retained posterior-mean reconstructions on two planes (mid-span x-y and a
near-wall x-z plane), row 3 shows the wall-traction maps themselves (true,
closure-predicted, equilibrium) on the exposed floor, and the lower rows
carry the wall-load tracking and the registered statistics.  All field rows
show the same frame: record 760, the first strict held-out frame.  Every
value is bound to an exact key of the frozen producer output.  Statistical
panels are scored on the source-excluded region of `CONFIRM_STRICT` -- the
103 held-out frames no earlier experiment ever scored.
"""

import numpy as np

from _submission import (
    BLUE,
    GOLD,
    GREY,
    LIGHT_BLUE,
    LIGHT_RED,
    RED,
    SOURCE,
    bind_artist,
    configure,
    expected_bar_payload,
    expected_errorbar_payload,
    load_json,
    plt,
    save,
)

configure()
plt.rcParams.update({
    "font.size": 12.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 12.0,
})

SRC = "fig8_closure/e2_closure_composition_results.json"
NPZ = "fig8_closure/e2_closure_composition_components.npz"
doc = load_json(SRC)
npz = np.load(SOURCE / NPZ, allow_pickle=False)

UNIT = "CONFIRM_STRICT"
REGION = "full_srcex"
U = doc["units"][UNIT]
SEEDS = (8801, 8802, 8803)


def panel_label(ax, label, *, x=-0.13, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


ARMS = [
    ("K:absent", "absent", GREY),
    ("K:tau_eqwm", "eq. law", LIGHT_BLUE),
    ("K:tau_closure", "closure", GOLD),
    ("K:tau_native", "oracle", BLUE),
    ("K:tau_fartime", "far time", RED),
]
GAINS = [
    ("eqwm_minus_absent", "eq. law", LIGHT_BLUE, "s"),
    ("closure_minus_absent", "closure", GOLD, "o"),
    ("native_minus_absent", "oracle", BLUE, "D"),
    ("fartime_minus_absent", "far time", RED, "v"),
    ("closure_minus_fartime", "clo. $-$ far", GREY, "^"),
]
REGIONS = [
    ("full_srcex", "complete"),
    ("near_srcex", "near"),
    ("outer_srcex", "farther"),
]

fig = plt.figure(figsize=(8.75, 9.35))
# manual layout: three tight field rows left, one stats panel per row right,
# two large panels on the bottom row
TILE_X0, TILE_W, TILE_GAP = 0.055, 0.192, 0.011
ROW_Y = {1: 0.815, 2: 0.600, 3: 0.375}
TILE_H = 0.152
CBAR_X, CBAR_W = 0.664, 0.013
RX, RW = 0.785, 0.185


def tile_rect(row, j):
    return [TILE_X0 + j * (TILE_W + TILE_GAP), ROW_Y[row], TILE_W, TILE_H]

# --------------------------------------------- a,b: what the generator produces
# Retained posterior-mean fields of the first strict frame (record 760), seed
# 8801.  Streamwise fluctuation about the training mean on two planes.
MID_Z = 24
NEAR_Y = 4        # y = 0.19h, first rows above the two-cell supplied band
Y_TOP = 48        # y/h <= 2 of the 4h-tall domain
mean_u = np.asarray(npz["mean_field"])[0]
REP_TILES = [
    ("K:absent", "no wall information"),
    ("K:tau_closure", "closure traction"),
    ("K:tau_native", "oracle traction"),
]
rep_u = {key: np.asarray(npz[f"rep|{key}"])[0] for key, _ in REP_TILES}
side = {key: (rep_u[key][:, :Y_TOP, MID_Z] - mean_u[:, :Y_TOP, MID_Z]).T
        for key, _ in REP_TILES}
plan = {key: (rep_u[key][:, NEAR_Y, :] - mean_u[:, NEAR_Y, :]).T
        for key, _ in REP_TILES}
vlim = float(np.round(np.percentile(
    np.abs(np.stack(list(side.values()) + list(plan.values()))), 99.5), 1))
extent = (0.0, 2.0, 0.0, 2.0)
side_mask = np.zeros((Y_TOP, 48), dtype=bool)
side_mask[0:24, 12:36] = True
plan_mask = np.zeros((48, 48), dtype=bool)
plan_mask[12:36, 12:36] = True
field_cmap = plt.get_cmap("RdBu_r").copy()
field_cmap.set_bad("#d3d7da")


def field_tile(ax, values, mask, *, artist_id, panel, source_refs,
               source_payload, transform, evidence, cmap, limit):
    display = np.ma.array(values, mask=mask)
    image = ax.imshow(display, origin="lower", extent=extent, cmap=cmap,
                      vmin=-limit, vmax=limit, interpolation="nearest",
                      zorder=1)
    bind_artist(
        fig, image, artist_id=artist_id, panel=panel,
        source_refs=source_refs, source_payload=source_payload,
        expected_payload={"type": "AxesImage", "array": display,
                          "extent": list(extent), "clim": [-limit, limit]},
        transform=transform, evidence=evidence,
    )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("auto")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    return image


REP_EVIDENCE = ("retained posterior-mean reconstruction of the first strict "
                "frame (record 760), seed 8801; no truth field enters")
side_axes, plan_axes = [], []
for j, (key, title) in enumerate(REP_TILES):
    axS = fig.add_axes(tile_rect(1, j))
    side_axes.append(axS)
    field_tile(
        axS, side[key], side_mask,
        artist_id=f"fig8.a.side_{key.split(':')[1]}", panel="a",
        source_refs=[
            {"kind": "npz", "path": NPZ, "key": f"rep|{key}",
             "slice": ["0", ":", ":", str(MID_Z)]},
            {"kind": "npz", "path": NPZ, "key": "mean_field",
             "slice": ["0", ":", ":", str(MID_Z)]},
        ],
        source_payload=[rep_u[key][:, :, MID_Z], mean_u[:, :, MID_Z]],
        transform=("subtract the training mean from the retained posterior-"
                   "mean streamwise field, keep the mid-span slice below "
                   "y=2h, transpose and mask the solid cube"),
        evidence=REP_EVIDENCE, cmap=field_cmap, limit=vlim,
    )
    axS.add_patch(plt.Rectangle((0.5, 0.0), 1.0, 1.0, facecolor="none",
                                edgecolor="#263238", linewidth=1.2, zorder=3))
    axS.plot([0.0, 0.5, 0.5, 1.5, 1.5, 2.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
             color=GOLD, lw=2.6, solid_capstyle="round", zorder=4,
             clip_on=False)
    axS.set_title(title, fontsize=12, pad=7)
    axS.set_xticklabels([])
    if j == 0:
        axS.set_ylabel("$y/h$")
    else:
        axS.set_yticklabels([])

    axP = fig.add_axes(tile_rect(2, j))
    plan_axes.append(axP)
    field_tile(
        axP, plan[key], plan_mask,
        artist_id=f"fig8.c.plan_{key.split(':')[1]}", panel="c",
        source_refs=[
            {"kind": "npz", "path": NPZ, "key": f"rep|{key}",
             "slice": ["0", ":", str(NEAR_Y), ":"]},
            {"kind": "npz", "path": NPZ, "key": "mean_field",
             "slice": ["0", ":", str(NEAR_Y), ":"]},
        ],
        source_payload=[rep_u[key][:, NEAR_Y, :], mean_u[:, NEAR_Y, :]],
        transform=("subtract the training mean from the retained posterior-"
                   "mean streamwise field, keep the near-wall plane y=0.19h, "
                   "transpose and mask the solid cube"),
        evidence=REP_EVIDENCE, cmap=field_cmap, limit=vlim,
    )
    axP.add_patch(plt.Rectangle((0.5, 0.5), 1.0, 1.0, facecolor="none",
                                edgecolor="#263238", linewidth=1.2, zorder=3))
    axP.plot([0.5, 1.5, 1.5, 0.5, 0.5], [0.5, 0.5, 1.5, 1.5, 0.5],
             color=GOLD, lw=2.6, solid_capstyle="round", zorder=4)
    axP.set_xticklabels([])
    if j == 0:
        axP.set_ylabel("$z/h$")
    else:
        axP.set_yticklabels([])
cax = fig.add_axes([CBAR_X, ROW_Y[2], CBAR_W, ROW_Y[1] + TILE_H - ROW_Y[2]])
cbar = fig.colorbar(side_axes[0].images[0], cax=cax)
cbar.set_ticks([-vlim, 0.0, vlim])
cax.set_title("$u'$", fontsize=12, pad=7)
cbar.outline.set_linewidth(1.35)
cbar.ax.tick_params(width=1.35)
for coll in list(cbar.ax.collections):
    coll.set_linewidth(1.35)
panel_label(side_axes[0], "a", x=-0.28, y=1.20)
panel_label(plan_axes[0], "c", x=-0.28, y=1.15)


def row_tag(ax, label):
    """Small in-tile tag naming the plane a row of field tiles shows."""
    ax.text(0.035, 0.965, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, color="#20252a", zorder=20,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84,
                  "boxstyle": "round,pad=0.15"})


row_tag(side_axes[0], "mid-span")
row_tag(plan_axes[0], "near-wall\n$y=0.19h$")

# ----------------------- c: the wall-traction maps the composition runs on
# The producer stacks the 4,608 (cell, face) pairs face by face, the exposed
# floor first (1,728 cells in C order over x, z).  Scattering the first block
# back onto the 48x48 floor grid reproduces the streamwise traction map of
# the same frame the renders above show.
N_FLOOR = 1728
floor_mask2d = np.ones((48, 48), dtype=bool)
floor_mask2d[12:36, 12:36] = False        # cube base is solid
TAU_TILES = [
    ("apriori_tau_true", "true traction"),
    ("apriori_tau_closure", "closure prediction"),
    ("apriori_tau_eqwm", "equilibrium law"),
]
tau_floor = {}
for key, _ in TAU_TILES:
    arr = np.full((48, 48), np.nan, dtype=np.float64)
    arr[floor_mask2d] = np.asarray(npz[key])[0][:N_FLOOR]
    tau_floor[key] = 1e3 * arr.T
tlim = float(np.round(np.percentile(
    np.abs(np.concatenate([v[np.isfinite(v)] for v in tau_floor.values()])),
    99.5), 1))
tau_cmap = plt.get_cmap("PuOr_r").copy()
tau_cmap.set_bad("#d3d7da")
tau_axes = []
for j, (key, title) in enumerate(TAU_TILES):
    ax = fig.add_axes(tile_rect(3, j))
    tau_axes.append(ax)
    field_tile(
        ax, tau_floor[key], ~floor_mask2d.T,
        artist_id=f"fig8.e.{key.split('_')[-1]}", panel="e",
        source_refs=[{"kind": "npz", "path": NPZ, "key": key,
                      "slice": ["0", ":"]}],
        source_payload=[np.asarray(npz[key])[0]],
        transform=("scatter the floor block of the (cell, face) pair vector "
                   "back onto the 48x48 floor grid in producer order, take "
                   "the streamwise component, x1000, transpose and mask the "
                   "cube base"),
        evidence=("a-priori wall traction of the same strict frame (record "
                  "760); the closure read only matching-height state"),
        cmap=tau_cmap, limit=tlim,
    )
    ax.add_patch(plt.Rectangle((0.5, 0.5), 1.0, 1.0, facecolor="none",
                               edgecolor="#263238", linewidth=1.2, zorder=3))
    ax.set_title(title, fontsize=12, pad=7)
    if j == 0:
        ax.set_ylabel("$z/h$")
    else:
        ax.set_yticklabels([])
    if j == 1:
        ax.set_xlabel("$x/h$")
tax = fig.add_axes([CBAR_X, ROW_Y[3], CBAR_W, TILE_H])
tbar = fig.colorbar(tau_axes[0].images[0], cax=tax)
tbar.set_ticks([-tlim, 0.0, tlim])
tax.set_title("$\\tau_x\\!\\times\\!10^{3}$", fontsize=12, pad=7)
tbar.outline.set_linewidth(1.35)
tbar.ax.tick_params(width=1.35)
for coll in list(tbar.ax.collections):
    coll.set_linewidth(1.35)
panel_label(tau_axes[0], "e", x=-0.28, y=1.18)
row_tag(tau_axes[0], "floor, $y=0$")

# ----------------------------- d: the reconstructed wall load tracks the truth
axD = fig.add_axes([0.098, 0.058, 0.477, 0.225])
fx_all = np.asarray(npz["fx_all"], dtype=float)
idx = np.asarray(npz[f"{UNIT}|eval_idx"], dtype=int)
t_lo, t_hi = 755, 1101
truth_line = axD.plot(np.arange(t_lo, t_hi), 1e3 * fx_all[t_lo:t_hi],
                      color="#263238", lw=1.3, zorder=2, label="true load")
bind_artist(
    fig, truth_line[0], artist_id="fig8.g.fx_truth", panel="g",
    source_refs=[{"kind": "npz", "path": NPZ, "key": "fx_all"}],
    source_payload=[fx_all],
    expected_payload={"type": "Line2D",
                      "x": np.arange(t_lo, t_hi, dtype=float),
                      "y": 1e3 * fx_all[t_lo:t_hi]},
    transform="plot the true streamwise viscous wall force over the evaluation span, x1000",
    evidence="first-cell viscous surrogate of the frozen record; physical walls only",
)
for arm, colour, marker, label_key in [
    ("K:absent", LIGHT_RED, "^", "absent"),
    ("K:tau_closure", GOLD, "o", "closure"),
]:
    fx_arm = np.mean([np.asarray(npz[f"{UNIT}|fx|{arm}|{s}"]) for s in SEEDS],
                     axis=0)
    corr = doc["units"][UNIT]["wall_force"][arm]["correlation"]
    pts = axD.plot(idx, 1e3 * fx_arm, marker, ms=4.6, color=colour,
                   markeredgecolor="black", markeredgewidth=0.9, ls="none",
                   zorder=3, label=f"{label_key} ($r={corr:+.2f}$)")
    bind_artist(
        fig, pts[0], artist_id=f"fig8.g.fx_{label_key}", panel="g",
        source_refs=[{"kind": "npz", "path": NPZ, "key": f"{UNIT}|fx|{arm}|{s}"}
                     for s in SEEDS]
        + [{"kind": "npz", "path": NPZ, "key": f"{UNIT}|eval_idx"},
           {"kind": "json", "path": SRC,
            "key": f"units.{UNIT}.wall_force.{arm}.correlation"}],
        source_payload=[np.asarray(npz[f"{UNIT}|fx|{arm}|{s}"]) for s in SEEDS]
        + [idx, corr],
        expected_payload={"type": "Line2D", "x": np.asarray(idx, dtype=float),
                          "y": 1e3 * fx_arm},
        transform=("average the reconstructed wall force over the three seeds "
                   "and place it at each held-out record index, x1000"),
        evidence="wall load implied by the reconstructed near-wall field; offline",
    )
axD.set_xlabel("record index")
axD.set_ylabel("$F_x\\times 10^{3}$")
axD.set_title("viscous wall-load tracking", fontsize=12)
axD.set_xlim(t_lo - 6, t_hi + 6)
lo_y = 1e3 * min(fx_all[t_lo:t_hi].min(),
                 min(np.asarray(npz[f"{UNIT}|fx|K:absent|{s}"]).min()
                     for s in SEEDS))
hi_y = 1e3 * max(fx_all[t_lo:t_hi].max(),
                 max(np.asarray(npz[f"{UNIT}|fx|K:absent|{s}"]).max()
                     for s in SEEDS))
span_y = hi_y - lo_y
axD.set_ylim(lo_y - 0.08 * span_y, hi_y + 0.90 * span_y)
axD.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9,
           edgecolor="none", handletextpad=0.2, columnspacing=0.5,
           handlelength=1.0, borderpad=0.15, labelspacing=0.2)
axD.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
axD.set_xticks([800, 900, 1000, 1100])
panel_label(axD, "g", x=-0.145, y=1.20)

# --------------------------------------------------- e: what the closure predicts
axE = fig.add_axes([RX, ROW_Y[3], RW, TILE_H])
ap = doc["closure_apriori"]["confirm_strict"]
vals = [ap["R2_tau_eqwm"], ap["R2_tau_closure"]]
bars = axE.barh([0, 1], vals, height=0.56, color=[LIGHT_BLUE, GOLD],
                edgecolor="black", linewidth=1.2, zorder=2)
bind_artist(
    fig, bars, artist_id="fig8.f.closure_apriori", panel="f",
    source_refs=[{"kind": "json", "path": SRC,
                  "key": f"closure_apriori.confirm_strict.{k}"}
                 for k in ("R2_tau_eqwm", "R2_tau_closure")],
    source_payload=vals,
    expected_payload={"type": "BarContainer",
                      "centres": [v / 2.0 for v in vals],
                      "heights": [0.56, 0.56]},
    transform="plot a-priori traction R^2 of the two closures as horizontal bars",
    evidence="closure reads matching-height state only; never the wall-adjacent cell",
)
axE.axvline(0, color="black", lw=1.4, zorder=1)
axE.set_yticks([0, 1])
axE.set_yticklabels(["eq. law", "learned\nclosure"])
axE.invert_yaxis()
axE.set_xlim(0, max(vals) * 1.30)
axE.xaxis.set_major_locator(plt.MaxNLocator(3, prune="upper"))
axE.set_title("a-priori traction $R^2$", fontsize=12)
for y, v in zip([0, 1], vals):
    axE.text(v + 0.02 * max(vals), y, f"{v:.3f}", va="center", fontsize=12)
panel_label(axE, "f", x=-0.30, y=1.17)

# ------------------------------------------------------- f: absolute skill ladder
axF = fig.add_axes([RX, ROW_Y[1], RW, TILE_H])
keys = [k for k, _, _ in ARMS if k in U["arms"]]
labels = [lab for k, lab, _ in ARMS if k in U["arms"]]
colours = [c for k, _, c in ARMS if k in U["arms"]]
ypos = np.arange(len(keys))
values = [U["arms"][k][REGION]["R2_fluct_balanced"] for k in keys]
lo = [U["arms"][k][REGION]["ci95_conservative_block"][0] for k in keys]
hi = [U["arms"][k][REGION]["ci95_conservative_block"][1] for k in keys]
bars = axF.barh(ypos, values, height=0.66, color=colours, edgecolor="black",
                linewidth=1.2, zorder=2)
bind_artist(
    fig, bars, artist_id="fig8.b.absolute", panel="b",
    source_refs=[{"kind": "json", "path": SRC,
                  "key": f"units.{UNIT}.arms.{k}.{REGION}.R2_fluct_balanced"}
                 for k in keys],
    source_payload=values,
    expected_payload={"type": "BarContainer",
                      "centres": [v / 2.0 for v in values],
                      "heights": [0.66] * len(values)},
    transform="plot absolute source-excluded skill of each arm as horizontal bars",
    evidence="scored only outside supplied and closure-read cells",
)
ci_x, ci_y = [], []
for k, (l, h) in enumerate(zip(lo, hi)):
    ci_x += [l, h, np.nan]
    ci_y += [k, k, np.nan]
ci_line = axF.plot(ci_x, ci_y, color="black", lw=1.4, zorder=3)
bind_artist(
    fig, ci_line[0], artist_id="fig8.b.absolute_ci", panel="b",
    source_refs=[{"kind": "json", "path": SRC,
                  "key": f"units.{UNIT}.arms.{k}.{REGION}.ci95_conservative_block"}
                 for k in keys],
    source_payload=[[l, h] for l, h in zip(lo, hi)],
    expected_payload={"type": "Line2D", "x": np.asarray(ci_x, dtype=float),
                      "y": np.asarray(ci_y, dtype=float)},
    transform="draw the conservative-block interval of each arm as a horizontal segment",
    evidence="moving-block intervals conditional on this record",
)
axF.axvline(0, color="black", lw=1.4, zorder=1)
axF.set_yticks(ypos)
axF.set_yticklabels(labels)
axF.invert_yaxis()
axF.set_title("absolute skill, $R^2$", fontsize=12)
span = max(hi) - min(lo)
axF.set_xlim(min(lo) - 0.14 * span, max(hi) + 0.20 * span)
axF.set_xticks([-0.1, 0.0])
panel_label(axF, "b", x=-0.30, y=1.17)

# ------------------------------------------------------- g: paired gains by region
axG = fig.add_axes([0.678, 0.058, 0.307, 0.225])
xr = np.arange(len(REGIONS))
present = [g for g in GAINS if g[0] in U["deltas"]]
offs = np.linspace(-0.30, 0.30, max(len(present), 1))
allv = []
closure_off = 0.0
for i, (key, label, colour, marker) in enumerate(present):
    if key == "closure_minus_absent":
        closure_off = offs[i]
    pts = [U["deltas"][key][r]["delta"] for r, _ in REGIONS]
    clo = [U["deltas"][key][r]["ci95_conservative_block"][0] for r, _ in REGIONS]
    chi = [U["deltas"][key][r]["ci95_conservative_block"][1] for r, _ in REGIONS]
    allv += clo + chi
    sc = axG.errorbar(xr + offs[i], pts,
                      yerr=[np.array(pts) - np.array(clo),
                            np.array(chi) - np.array(pts)],
                      fmt=marker, ms=5.4, color=colour, ecolor=colour,
                      elinewidth=1.3, capsize=2.4, markeredgecolor="black",
                      markeredgewidth=1.1, markerfacecolor=colour,
                      label=label, zorder=3)
    bind_artist(
        fig, sc, artist_id=f"fig8.h.{key}", panel="h",
        source_refs=[{"kind": "json", "path": SRC,
                      "key": f"units.{UNIT}.deltas.{key}.{r}.delta"}
                     for r, _ in REGIONS]
        + [{"kind": "json", "path": SRC,
            "key": f"units.{UNIT}.deltas.{key}.{r}.ci95_conservative_block"}
           for r, _ in REGIONS],
        source_payload=pts + [[l, h] for l, h in zip(clo, chi)],
        expected_payload=expected_errorbar_payload(xr + offs[i], pts, clo, chi),
        transform="plot each registered paired contrast in three source-excluded regions",
        evidence="paired within-model intervention; the three regions overlap",
    )
per_seed = U["deltas"]["closure_minus_absent"][REGION]["per_seed"]
seed_x = np.full(len(per_seed), closure_off)
seed_dots = axG.plot(seed_x, per_seed, "o", ms=3.4, color="white",
                     markeredgecolor="black", markeredgewidth=1.0, ls="none",
                     zorder=4)
bind_artist(
    fig, seed_dots[0], artist_id="fig8.h.per_seed", panel="h",
    source_refs=[{"kind": "json", "path": SRC,
                  "key": f"units.{UNIT}.deltas.closure_minus_absent.{REGION}.per_seed"}],
    source_payload=[per_seed],
    expected_payload={"type": "Line2D", "x": seed_x.astype(float),
                      "y": np.asarray(per_seed, dtype=float)},
    transform=("overlay the three per-seed primary estimands at the "
               "complete-region closure position"),
    evidence="three seeds trained under one frozen mixture; clause 3 of the rule",
)
axG.axhline(0, color="black", lw=1.4, zorder=1)
axG.set_xticks(xr)
axG.set_xticklabels([lab for _, lab in REGIONS])
axG.set_ylabel("paired $\\Delta R^2$")
axG.set_title("paired contrasts", fontsize=12)
axG.legend(title="arm $-$ absent", title_fontsize=12, loc="upper right",
           ncol=2, frameon=False, handletextpad=0.15, labelspacing=0.18,
           columnspacing=0.55, handlelength=1.0, borderaxespad=0.1)
sp = max(allv) - min(allv)
axG.set_ylim(min(allv) - 0.10 * sp, max(allv) + 0.95 * sp)
panel_label(axG, "h", x=-0.20, y=1.20)

# --------------------------------------- h: distributional validity of the model
axH = fig.add_axes([RX, ROW_Y[2], RW, TILE_H])
dist = U.get("distributional", {})
dkeys = [k for k, _, _ in ARMS if k in dist]
crps = [dist[k]["crps_mean"] for k in dkeys]
ypos2 = np.arange(len(dkeys))
bars = axH.barh(ypos2, crps, height=0.66,
                color=[c for k, _, c in ARMS if k in dist],
                edgecolor="black", linewidth=1.2, zorder=2)
bind_artist(
    fig, bars, artist_id="fig8.d.crps", panel="d",
    source_refs=[{"kind": "json", "path": SRC,
                  "key": f"units.{UNIT}.distributional.{k}.crps_mean"}
                 for k in dkeys],
    source_payload=crps,
    expected_payload={"type": "BarContainer",
                      "centres": [v / 2.0 for v in crps],
                      "heights": [0.66] * len(crps)},
    transform="plot the continuous ranked probability score of each arm as horizontal bars",
    evidence="proper scoring rule over the 8 retained posterior members; lower is better",
)
axH.set_yticks(ypos2)
axH.set_yticklabels([lab for k, lab, _ in ARMS if k in dist])
axH.invert_yaxis()
axH.set_title("CRPS, lower better", fontsize=12)
axH.set_xlim(min(crps) * 0.955, max(crps) * 1.012)
axH.xaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
panel_label(axH, "d", x=-0.30, y=1.17)

save(fig, "fig8_closure_composition")
