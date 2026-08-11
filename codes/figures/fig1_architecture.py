#!/usr/bin/env python3
"""Figure 1: the near-wall band on the cube record, the one-slot wall-to-field
system, the three records, and the evidence ladder as three
picture-flows.  Every data-bearing artist is bound to the staged derived
arrays in manuscript/source_data/fig1_v2 (see stage_fig1_overview.py)."""

import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from _submission import (
    BLUE,
    GOLD,
    GREY,
    RED,
    SOURCE,
    bind_artist,
    configure,
    plt,
    save,
)

configure()

DARK = "#263238"
GREEN = "#2a8a4a"
STREAM = "#2c4a63"
SOLID_GREY = "#5b6770"

NPZ_REL = "manuscript/source_data/fig1_v2/fig1_overview_derived.npz"
D = np.load(SOURCE / "fig1_v2/fig1_overview_derived.npz")

vlim = float(D["vlim"])
tau_vlim = float(D["tau_vlim"])
v_gen = float(D["v_gen"])
vlim_tau_f = float(D["vlim_tau_f"])
vlim_ch = float(D["vlim_ch"])
vlim_hill = float(D["vlim_hill"])

mask_h = np.asarray(D["mask_h"], bool)
mask_v = np.asarray(D["mask_v"], bool)
solidH = np.asarray(D["solidH"], bool)


def nref(key):
    return {"path": NPZ_REL, "kind": "npz", "key": key}


# The blueprint was laid out on a 7.6-inch canvas; the approved figure crops
# the empty bottom strip.  All blueprint y-coordinates pass through TY()/HS()
# so every element keeps its exact physical size and position.
CUT = 0.055
SC = 1.0 / (1.0 - CUT)


def TY(y):
    return (y - CUT) * SC


def HS(h):
    return h * SC


fig = plt.figure(figsize=(8.75, 7.6 * (1.0 - CUT)))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def arrow(x0, y0, x1, y1, color=DARK, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=lw, color=color, zorder=4))


def panel(letter, x, y):
    ax.text(x, TY(y), letter, fontsize=12, fontweight="bold", va="top",
            zorder=5)


def txt(x, y, s, **kw):
    kw.setdefault("fontsize", 12)
    kw.setdefault("color", "black")
    ax.text(x, TY(y), s, **kw)


def bind_image(im, aid, pan, keys, payloads, transform, evidence):
    bind_artist(
        fig, im, artist_id=aid, panel=pan,
        source_refs=[nref(k) for k in keys],
        source_payload=payloads,
        expected_payload={
            "type": "AxesImage",
            "array": np.asanyarray(im.get_array()),
            "extent": [float(v) for v in im.get_extent()],
            "clim": [float(v) for v in im.get_clim()],
        },
        transform=transform, evidence=evidence)


def bind_line(ln, aid, pan, keys, payloads, transform, evidence):
    bind_artist(
        fig, ln, artist_id=aid, panel=pan,
        source_refs=[nref(k) for k in keys],
        source_payload=payloads,
        expected_payload={
            "type": "Line2D",
            "x": np.asarray(ln.get_xdata(orig=False), dtype=float),
            "y": np.asarray(ln.get_ydata(orig=False), dtype=float),
        },
        transform=transform, evidence=evidence)


cmap3 = plt.get_cmap("RdBu_r").copy()
cmap3.set_bad((0, 0, 0, 0))
cmap_tau = plt.get_cmap("PuOr_r").copy()
cmap_tau.set_bad((0, 0, 0, 0))
cmap_solid = plt.get_cmap("Greys").copy()
cmap_solid.set_bad((0, 0, 0, 0))

# ============================ a: why the wall matters ========================
panel("a", 0.012, 0.985)
txt(0.030, 0.985, "Thin wall band, large role", va="top", fontsize=12)
DX, DY = 0.55, 0.34
BT = 0.045
zc, dz = 0.30, 0.35
ox, oy = DX * zc, DY * zc
bx3, by3 = DX * dz, DY * dz
Y0 = 0.19
fv = 0.5
floor_pts = [(0.0, 0.0), (2.0, 0.0), (2.0 + DX, DY), (DX, DY)]

ax_a = fig.add_axes([0.014, TY(0.610), 0.310, HS(0.335)])
ax_a.set_xlim(-0.06, 2.66)
ax_a.set_ylim(-0.10, 2.46)
ax_a.set_aspect("equal")
ax_a.axis("off")

ax_a.add_patch(Polygon(floor_pts, closed=True, facecolor="#e2e5e7",
                       edgecolor="none", zorder=1.0))
ax_a.add_patch(Polygon(floor_pts, closed=True, facecolor=GOLD, alpha=0.5,
                       edgecolor="none", zorder=1.1))
ax_a.add_patch(Polygon([(0.5 + ox, oy), (1.5 + ox, oy),
                        (1.5 + ox + bx3, oy + by3),
                        (0.5 + ox + bx3, oy + by3)],
                       closed=True, facecolor="#c9ced1", edgecolor="none",
                       zorder=1.2))

img_plane_h = np.ma.array(np.asarray(D["plane_h"], float), mask=mask_h)
tr_h = mtransforms.Affine2D.from_values(1, 0, DX, DY, 0, Y0) + ax_a.transData
im = ax_a.imshow(img_plane_h, origin="lower", extent=(0.0, 2.0, 0.0, 1.0),
                 cmap=cmap3, vmin=-vlim, vmax=vlim, interpolation="nearest",
                 alpha=0.62, transform=tr_h, zorder=1.6)
bind_image(im, "a_near_wall_plane", "a", ["plane_h", "mask_h"],
           [D["plane_h"], D["mask_h"]],
           "masked array drawn obliquely at band height y=0.19h",
           "truth u-fluctuation plane at y-index 4 with the cube footprint "
           "masked, from the staged derived arrays")

xyb = np.asarray(D["stream_back_xy"], float)
ln, = ax_a.plot(xyb[:, 0], xyb[:, 1], color=STREAM, lw=1.25, alpha=0.42,
                zorder=2.5, solid_capstyle="round")
bind_line(ln, "a_streamlines_behind", "a", ["stream_back_xy"],
          [D["stream_back_xy"]],
          "NaN-separated projected polylines drawn as one line",
          "reconstructed 3-D streamlines seeded behind the cube "
          "(staging recipe: unit-speed tracer on the smoothed snapshot mean)")

img_plane_v = np.ma.array(np.asarray(D["plane_v"], float), mask=mask_v)
tr_v = mtransforms.Affine2D().translate(DX * fv, DY * fv) + ax_a.transData
im = ax_a.imshow(img_plane_v, origin="lower", extent=(0.0, 2.0, 0.0, 2.0),
                 cmap=cmap3, vmin=-vlim, vmax=vlim, interpolation="nearest",
                 alpha=0.60, transform=tr_v, zorder=2.0)
bind_image(im, "a_midspan_plane", "a", ["plane_v", "mask_v"],
           [D["plane_v"], D["mask_v"]],
           "masked array drawn at mid-depth of the oblique box",
           "truth u-fluctuation mid-span plane (x-index 24, lowest 48 "
           "heights) with the cube section masked")

ax_a.add_patch(Rectangle((0.5 + ox, oy), 1.0, 1.0, facecolor="#d3d7da",
                         edgecolor=DARK, linewidth=1.3, zorder=3))
ax_a.add_patch(Polygon([(0.5 + ox, 1.0 + oy), (1.5 + ox, 1.0 + oy),
                        (1.5 + ox + bx3, 1.0 + oy + by3),
                        (0.5 + ox + bx3, 1.0 + oy + by3)],
                       closed=True, facecolor="#c6cbce", edgecolor=DARK,
                       linewidth=1.3, zorder=3))
ax_a.add_patch(Polygon([(1.5 + ox, oy), (1.5 + ox, 1.0 + oy),
                        (1.5 + ox + bx3, 1.0 + oy + by3),
                        (1.5 + ox + bx3, oy + by3)],
                       closed=True, facecolor="#b3b9bd", edgecolor=DARK,
                       linewidth=1.3, zorder=3))
ax_a.add_patch(Rectangle((0.5 + ox - BT, oy), 1.0 + 2 * BT, 1.0 + BT,
                         facecolor=GOLD, alpha=0.45, edgecolor=GOLD,
                         linewidth=1.3, zorder=3.5))
ax_a.add_patch(Polygon([(0.5 + ox - BT, 1.0 + oy + BT),
                        (1.5 + ox + BT, 1.0 + oy + BT),
                        (1.5 + ox + BT + bx3, 1.0 + oy + BT + by3),
                        (0.5 + ox - BT + bx3, 1.0 + oy + BT + by3)],
                       closed=True, facecolor=GOLD, alpha=0.45,
                       edgecolor=GOLD, linewidth=1.3, zorder=3.6))
ax_a.add_patch(Polygon([(1.5 + ox + BT, oy), (1.5 + ox + BT, 1.0 + oy + BT),
                        (1.5 + ox + BT + bx3, 1.0 + oy + BT + by3),
                        (1.5 + ox + BT + bx3, oy + by3)],
                       closed=True, facecolor=GOLD, alpha=0.45,
                       edgecolor=GOLD, linewidth=1.3, zorder=3.6))

xyf = np.asarray(D["stream_front_xy"], float)
ln, = ax_a.plot(xyf[:, 0], xyf[:, 1], color=STREAM, lw=1.25, alpha=0.72,
                zorder=3.8, solid_capstyle="round")
bind_line(ln, "a_streamlines_front", "a", ["stream_front_xy"],
          [D["stream_front_xy"]],
          "NaN-separated projected polylines drawn as one line",
          "reconstructed 3-D streamlines seeded in front of or above the "
          "cube (same staging recipe as the behind group)")

for seg in np.asarray(D["stream_arrows"], float):
    ax_a.annotate("", xy=(seg[2], seg[3]), xytext=(seg[0], seg[1]),
                  arrowprops={"arrowstyle": "-|>", "color": STREAM,
                              "linewidth": 1.0, "mutation_scale": 7},
                  zorder=3.8)

for xs_, ys_ in (((0, 2), (0, 0)),
                 ((0, DX), (0, DY)), ((2, 2 + DX), (0, DY)),
                 ((DX, 2 + DX), (DY, DY)),
                 ((DX, DX), (DY, 2 + DY)), ((2 + DX, 2 + DX), (DY, 2 + DY)),
                 ((DX, 2 + DX), (2 + DY, 2 + DY)),
                 ((0, 0), (0, 2)), ((2, 2), (0, 2)),
                 ((0, 2), (2, 2)), ((0, DX), (2, 2 + DY)),
                 ((2, 2 + DX), (2, 2 + DY))):
    ax_a.plot(xs_, ys_, color=DARK, lw=1.25, alpha=0.75, zorder=4)

# ============================ b: the system, one slot ========================
panel("b", 0.320, 0.985)
txt(0.340, 0.985, "One-slot wall-to-field method", va="top", fontsize=12)

axB1 = fig.add_axes([0.343, TY(0.8207), 0.066, HS(0.0755)])
im = axB1.imshow(img_plane_h, origin="lower", cmap=cmap3, vmin=-vlim,
                 vmax=vlim, interpolation="nearest")
bind_image(im, "b_flow_state", "b", ["plane_h", "mask_h"],
           [D["plane_h"], D["mask_h"]],
           "masked array shown as the pipeline input thumbnail",
           "matching-height flow-state plane from the cube record")
axB1.set_xticks([]); axB1.set_yticks([])
for sp in axB1.spines.values():
    sp.set_visible(True); sp.set_edgecolor(BLUE); sp.set_linewidth(1.5)
txt(0.376, 0.800, "flow state", ha="center", va="top")

arrow(0.411, TY(0.8585), 0.457, TY(0.8585), lw=2.0)

ax.add_patch(FancyBboxPatch(
    (0.459, TY(0.808)), 0.074, HS(0.101),
    boxstyle="round,pad=0.004,rounding_size=0.008", linewidth=1.5,
    linestyle=(0, (4, 2.5)), edgecolor=DARK, facecolor="white", zorder=2))
axC = fig.add_axes([0.4645, TY(0.8145), 0.063, HS(0.088)])
axC.axis("off")
pn = np.asarray(D["profile_pn"], float)
yn = np.asarray(D["profile_yn"], float)
ln, = axC.plot(pn, yn, color=DARK, lw=1.6)
bind_line(ln, "b_closure_profile", "b", ["profile_pn", "profile_yn"],
          [D["profile_pn"], D["profile_yn"]],
          "normalised channel mean profile drawn as the closure glyph",
          "recorded channel mean-velocity profile (lower half, normalised "
          "by its maximum)")
ym = 0.20
um = float(np.interp(ym, yn, pn))
axC.plot([um, um], [0.05, ym], ls=":", color=GREY, lw=1.3)
axC.plot([um], [ym], marker="o", ms=5, color=BLUE, zorder=5)
axC.plot([0.0, 1.02], [0.0, 0.0], color=GOLD, lw=3.0, solid_capstyle="butt")
axC.annotate("", xy=(0.62, 0.09), xytext=(0.08, 0.09),
             arrowprops={"arrowstyle": "-|>", "color": GOLD,
                         "linewidth": 1.5, "mutation_scale": 9})
axC.set_xlim(-0.02, 1.06); axC.set_ylim(-0.04, 1.02)
txt(0.496, 0.800, "closure $\\mathcal{C}$", ha="center", va="top")

arrow(0.535, TY(0.8585), 0.581, TY(0.8585), lw=2.0)

img_tau = np.ma.array(np.asarray(D["tau_plane"], float), mask=mask_h)
axB2 = fig.add_axes([0.583, TY(0.8207), 0.066, HS(0.0755)])
im = axB2.imshow(img_tau, origin="lower", cmap=cmap_tau, vmin=-tau_vlim,
                 vmax=tau_vlim, interpolation="nearest")
bind_image(im, "b_traction", "b", ["tau_plane", "mask_h"],
           [D["tau_plane"], D["mask_h"]],
           "masked array shown as the slot thumbnail",
           "signed near-wall shear footprint on the floor (truth u at "
           "y-index 1, cube footprint masked)")
axB2.set_xticks([]); axB2.set_yticks([])
for sp in axB2.spines.values():
    sp.set_visible(True); sp.set_edgecolor(GOLD); sp.set_linewidth(1.5)
txt(0.616, 0.800, "traction $\\boldsymbol{\\tau}_w$", ha="center", va="top")

arrow(0.651, TY(0.8585), 0.697, TY(0.8585), color=GOLD, lw=2.8)
txt(0.674, 0.913, "one slot", ha="center", va="bottom")

ax.add_patch(FancyBboxPatch(
    (0.699, TY(0.808)), 0.074, HS(0.101),
    boxstyle="round,pad=0.004,rounding_size=0.008", linewidth=1.5,
    linestyle=(0, (4, 2.5)), edgecolor=DARK, facecolor="white", zorder=2))
axG = fig.add_axes([0.7035, TY(0.8115), 0.065, HS(0.094)])
axG.axis("off")
# Schematic latent-noise icon (fixed seed); an icon of the sampling process,
# not a data rendering, so it is deliberately left unbound.
noiseG = np.random.default_rng(7).standard_normal((22, 22))
axG.imshow(noiseG, extent=(0.02, 0.46, 0.54, 0.98), cmap="Greys",
           vmin=-2.8, vmax=2.8, interpolation="nearest", zorder=2)
axG.add_patch(Rectangle((0.02, 0.54), 0.44, 0.44, fill=False,
                        edgecolor=GREY, linewidth=1.25, zorder=3))
im = axG.imshow(np.asarray(D["gen_plane"], float), origin="lower",
                extent=(0.54, 0.98, 0.02, 0.46), cmap=cmap3, vmin=-v_gen,
                vmax=v_gen, interpolation="nearest", zorder=2)
bind_image(im, "b_generator_field", "b", ["gen_plane"], [D["gen_plane"]],
           "plain array shown as the generator-output icon",
           "generated-field plane above the cube (posterior mean, y-index "
           "40), scaled to its own 99th percentile")
axG.add_patch(Rectangle((0.54, 0.02), 0.44, 0.44, fill=False,
                        edgecolor=GREY, linewidth=1.25, zorder=3))
axG.annotate("", xy=(0.63, 0.35), xytext=(0.37, 0.63),
             arrowprops={"arrowstyle": "-|>", "color": DARK,
                         "linewidth": 1.3, "mutation_scale": 9}, zorder=4)
axG.set_xlim(0, 1); axG.set_ylim(0, 1); axG.set_aspect("auto")
txt(0.736, 0.800, "generator $\\mathcal{G}$", ha="center", va="top")

arrow(0.775, TY(0.8585), 0.836, TY(0.8585), lw=2.0)

axB3 = fig.add_axes([0.816, TY(0.778), 0.170, HS(0.140)])
axB3.set_xlim(-0.06, 2.66); axB3.set_ylim(-0.10, 2.50)
axB3.set_aspect("equal"); axB3.axis("off")
axB3.add_patch(Polygon(floor_pts, closed=True, facecolor="#e2e5e7",
                       edgecolor="none", zorder=1.0))
img_out3d = np.ma.array(np.asarray(D["out3d_v"], float), mask=mask_v)
tr_vm = mtransforms.Affine2D().translate(DX * fv, DY * fv) + axB3.transData
im = axB3.imshow(img_out3d, origin="lower", extent=(0.0, 2.0, 0.0, 2.0),
                 cmap=cmap3, vmin=-vlim, vmax=vlim, interpolation="nearest",
                 alpha=0.9, transform=tr_vm, zorder=2.0)
bind_image(im, "b_output_midspan", "b", ["out3d_v", "mask_v"],
           [D["out3d_v"], D["mask_v"]],
           "masked array drawn at mid-depth of the mini oblique box",
           "generated (posterior-mean) mid-span plane with the cube "
           "section masked")
axB3.add_patch(Rectangle((0.5 + ox, oy), 1.0, 1.0, facecolor="#d3d7da",
                         edgecolor=DARK, linewidth=1.25, zorder=3))
axB3.add_patch(Polygon([(0.5 + ox, 1.0 + oy), (1.5 + ox, 1.0 + oy),
                        (1.5 + ox + bx3, 1.0 + oy + by3),
                        (0.5 + ox + bx3, 1.0 + oy + by3)],
                       closed=True, facecolor="#c6cbce", edgecolor=DARK,
                       linewidth=1.25, zorder=3))
axB3.add_patch(Polygon([(1.5 + ox, oy), (1.5 + ox, 1.0 + oy),
                        (1.5 + ox + bx3, 1.0 + oy + by3),
                        (1.5 + ox + bx3, oy + by3)],
                       closed=True, facecolor="#b3b9bd", edgecolor=DARK,
                       linewidth=1.25, zorder=3))
for xs_, ys_ in (((0, 2), (0, 0)),
                 ((0, DX), (0, DY)), ((2, 2 + DX), (0, DY)),
                 ((DX, 2 + DX), (DY, DY)),
                 ((DX, DX), (DY, 2 + DY)), ((2 + DX, 2 + DX), (DY, 2 + DY)),
                 ((DX, 2 + DX), (2 + DY, 2 + DY)),
                 ((0, 0), (0, 2)), ((2, 2), (0, 2)),
                 ((0, 2), (2, 2)), ((0, DX), (2, 2 + DY)),
                 ((2, 2 + DX), (2, 2 + DY))):
    axB3.plot(xs_, ys_, color=DARK, lw=1.25, alpha=0.75, zorder=4)
txt(0.901, 0.924, "complete 3-D field", ha="center", va="bottom")

ax.add_patch(FancyBboxPatch(
    (0.343, TY(0.620)), 0.642, HS(0.135),
    boxstyle="round,pad=0.004,rounding_size=0.008", linewidth=1.5,
    linestyle=(0, (4, 2.5)), edgecolor=RED, facecolor="#fdf0f0", zorder=1.5))
txt(0.664, 0.750, "controls through the SAME slot", ha="center", va="top",
    fontweight="bold")

for i, (lab, key) in enumerate((("exact", "tau_plane"),
                                ("wrong instant", "tau_wi"),
                                ("shuffled", "tau_sh"),
                                ("equilibrium", "tau_eq"),
                                ("data-only", None))):
    cxv = 0.415 + i * 0.1244
    if key is not None:
        axv = fig.add_axes([cxv - 0.0217, TY(0.660), 0.0434, HS(0.050)])
        img_v = np.ma.array(np.asarray(D[key], float), mask=mask_h)
        im = axv.imshow(img_v, origin="lower", cmap=cmap_tau,
                        vmin=-tau_vlim, vmax=tau_vlim,
                        interpolation="nearest")
        bind_image(im, f"b_ctrl_{key}", "b", [key, "mask_h"],
                   [D[key], D["mask_h"]],
                   "masked array shown as a control-arm thumbnail",
                   f"traction control arm '{lab}' (staging recipe in the "
                   "fig1_v2 provenance ledger)")
        axv.set_xticks([]); axv.set_yticks([])
        for sp in axv.spines.values():
            sp.set_visible(True); sp.set_edgecolor(RED); sp.set_linewidth(1.3)
    else:
        ax.add_patch(Rectangle((cxv - 0.0217, TY(0.660)), 0.0434, HS(0.050),
                               fill=False, edgecolor=RED, linewidth=1.3,
                               linestyle=(0, (3, 2)), zorder=3))
        ax.plot([cxv - 0.0217, cxv + 0.0217], [TY(0.660), TY(0.710)],
                color=RED, lw=1.3, zorder=3)
    txt(cxv, 0.652, lab, ha="center", va="top")

arrow(0.674, TY(0.760), 0.674, TY(0.852), color=RED, lw=2.0)

# ========================== c: the three records ==========================
panel("c", 0.012, 0.592)
txt(0.030, 0.598, "Three stress-test records", va="top", fontsize=12)

ax.add_patch(FancyBboxPatch(
    (0.030, TY(0.419)), 0.182, HS(0.147),
    boxstyle="round,pad=0.004,rounding_size=0.008", linewidth=1.6,
    linestyle=(0, (4, 2.5)), edgecolor=GREEN, facecolor="white", zorder=1.5))
axCU = fig.add_axes([0.042, TY(0.433), 0.158, HS(0.121)])
im = axCU.imshow(img_plane_v[0:32, :], origin="lower",
                 extent=(0.0, 1.0, 0.0, 1.0), aspect="auto", cmap=cmap3,
                 vmin=-vlim, vmax=vlim, interpolation="nearest")
bind_image(im, "c_cube_section", "c", ["plane_v", "mask_v"],
           [D["plane_v"], D["mask_v"]],
           "masked array cropped to its lowest 32 rows so the tile keeps "
           "the true 3:2 aspect and the cube stays square",
           "truth mid-span section of the cube record")
axCU.add_patch(Rectangle((0.25, 0.0), 0.50, 0.75, facecolor=SOLID_GREY,
                         edgecolor="none", zorder=2))
axCU.plot([0.0, 0.25, 0.25, 0.75, 0.75, 1.0],
          [0.015, 0.015, 0.75, 0.75, 0.015, 0.015],
          color=GOLD, lw=3.0, zorder=3, solid_capstyle="butt")
axCU.set_xticks([]); axCU.set_yticks([])
for sp in axCU.spines.values():
    sp.set_visible(False)
axCU.set_xlim(0, 1); axCU.set_ylim(0, 1)
txt(0.121, 0.411, "pinned cube", ha="center", va="top", fontsize=12)

ax.add_patch(FancyBboxPatch(
    (0.245, TY(0.419)), 0.345, HS(0.147),
    boxstyle="round,pad=0.004,rounding_size=0.008", linewidth=1.6,
    linestyle=(0, (4, 2.5)), edgecolor=GREEN, facecolor="white", zorder=1.5))
axCH = fig.add_axes([0.257, TY(0.433), 0.321, HS(0.121)])
im = axCH.imshow(np.asarray(D["up_ch"], float), origin="lower",
                 extent=(0.0, 1.0, 0.0, 1.0), aspect="auto", cmap=cmap3,
                 vmin=-vlim_ch, vmax=vlim_ch, interpolation="nearest")
bind_image(im, "c_channel_slab", "c", ["up_ch"], [D["up_ch"]],
           "plain array shown as the channel-record tile",
           "channel u-fluctuation slab (u_plane minus mean profile)")
axCH.set_xticks([]); axCH.set_yticks([])
for sp in axCH.spines.values():
    sp.set_visible(False)
axCH.axhline(0.012, color=GOLD, lw=3.0)
axCH.axhline(0.988, color=GOLD, lw=3.0)
txt(0.4175, 0.411, "attached channel", ha="center", va="top", fontsize=12)

ax.add_patch(FancyBboxPatch(
    (0.620, TY(0.419)), 0.350, HS(0.147),
    boxstyle="round,pad=0.004,rounding_size=0.008", linewidth=1.6,
    linestyle=(0, (4, 2.5)), edgecolor=GREEN, facecolor="white", zorder=1.5))
img_hill = np.ma.array(np.asarray(D["u_stdH"], float), mask=solidH)
img_hill_solid = np.ma.array(solidH.astype(float), mask=~solidH)
bxy = np.asarray(D["hill_boundary_xy"], float)
axHI = fig.add_axes([0.632, TY(0.433), 0.326, HS(0.121)])
im = axHI.imshow(img_hill, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
                 aspect="auto", cmap=cmap3, vmin=-vlim_hill, vmax=vlim_hill,
                 interpolation="nearest")
bind_image(im, "c_hill_field", "c", ["u_stdH", "solidH"],
           [D["u_stdH"], D["solidH"]],
           "masked array shown as the hill tile",
           "standardised instantaneous hill field with the terrain masked")
im = axHI.imshow(img_hill_solid, origin="lower",
                 extent=(0.0, 1.0, 0.0, 1.0), aspect="auto", cmap=cmap_solid,
                 vmin=-1.2, vmax=1.6, interpolation="nearest", zorder=2)
bind_image(im, "c_hill_solid", "c", ["solidH"], [D["solidH"]],
           "terrain mask drawn as a flat grey overlay",
           "hill solid mask from the staged hill context frame")
ln, = axHI.plot(bxy[:, 0], bxy[:, 1], color=GOLD, lw=2.2, zorder=3)
bind_line(ln, "c_hill_boundary", "c", ["hill_boundary_xy"],
          [D["hill_boundary_xy"]],
          "0.5-level contour polyline of the terrain mask",
          "mapped hill boundary drawn in the band colour")
axHI.set_xticks([]); axHI.set_yticks([])
for sp in axHI.spines.values():
    sp.set_visible(False)
axHI.set_xlim(0, 1); axHI.set_ylim(0, 1)
txt(0.795, 0.411, "separating hill", ha="center", va="top", fontsize=12)

# ============================ d: evidence ladder =============================
panel("d", 0.012, 0.360)
txt(0.030, 0.365, "Evidence boundaries", va="top", fontsize=12)

ROW_Y = {"E1": 0.263, "E2": 0.170, "E3": 0.077}
TH_W, TH_H = 0.065, 0.073

img_tau_clos = np.ma.array(np.asarray(D["tau_clos_f"], float).T,
                           mask=np.isnan(np.asarray(D["tau_clos_f"]).T))
img_tau_true = np.ma.array(np.asarray(D["tau_true_f"], float).T,
                           mask=np.isnan(np.asarray(D["tau_true_f"]).T))
img_band = np.ma.array(np.asarray(D["band_plane"], float), mask=mask_h)


def d_thumb(x0, y0, img, aid, keys, payloads, evidence, edge, vmin_, vmax_,
            cmap_, hole=False):
    a = fig.add_axes([x0, TY(y0), TH_W, HS(TH_H)])
    im_ = a.imshow(img, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
                   aspect="auto", cmap=cmap_, vmin=vmin_, vmax=vmax_,
                   interpolation="nearest")
    bind_image(im_, aid, "d", keys, payloads,
               "array shown as an evidence-ladder thumbnail", evidence)
    if hole:
        a.add_patch(Rectangle((0.25, 0.25), 0.50, 0.50,
                              facecolor=SOLID_GREY, edgecolor="none",
                              zorder=2))
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(1.3)
    return a


def d_minibox(x0, y0, letter):
    ax.add_patch(FancyBboxPatch(
        (x0, TY(y0 + 0.0165)), 0.040, HS(0.040),
        boxstyle="round,pad=0.003,rounding_size=0.006", linewidth=1.3,
        edgecolor=DARK, facecolor="white", zorder=3))
    txt(x0 + 0.020, y0 + 0.0365, letter, ha="center", va="center", zorder=4)


y1 = ROW_Y["E1"]; c1 = y1 + TH_H / 2
txt(0.044, c1 + 0.026, "1", ha="center", va="center", fontweight="bold")
txt(0.044, c1 - 0.012, "positive\ncontrol", ha="center", va="center")
d_thumb(0.085, y1, img_band, "d_e1_band", ["band_plane", "mask_h"],
        [D["band_plane"], D["mask_h"]],
        "in-band truth plane (y-index 2) supplied to the generator",
        GOLD, -vlim, vlim, cmap3, hole=True)
arrow(0.158, TY(c1), 0.395, TY(c1), color=GOLD, lw=2.0)
d_minibox(0.400, y1, "$\\mathcal{G}$")
arrow(0.448, TY(c1), 0.685, TY(c1), lw=2.0)
d_thumb(0.690, y1, np.asarray(D["out30"], float), "d_e1_generated",
        ["out30"], [D["out30"]],
        "generated (posterior-mean) plane above the cube (y-index 30)",
        BLUE, -vlim, vlim, cmap3)
txt(0.785, c1, "$\\approx$", ha="center", va="center", fontweight="bold")
d_thumb(0.815, y1, np.asarray(D["tru30"], float), "d_e1_truth",
        ["tru30"], [D["tru30"]],
        "recorded truth plane above the cube (y-index 30)",
        DARK, -vlim, vlim, cmap3)
txt(0.933, c1, "wall signal\ndetectable", ha="center", va="center")

y2 = ROW_Y["E2"]; c2 = y2 + TH_H / 2
txt(0.044, c2 + 0.026, "2", ha="center", va="center", fontweight="bold")
txt(0.044, c2 - 0.012, "offline\nclosure", ha="center", va="center")
d_thumb(0.085, y2, img_plane_h, "d_e2_state", ["plane_h", "mask_h"],
        [D["plane_h"], D["mask_h"]],
        "matching-height flow state entering the closure",
        BLUE, -vlim, vlim, cmap3, hole=True)
arrow(0.158, TY(c2), 0.245, TY(c2), lw=2.0)
d_minibox(0.250, y2, "$\\mathcal{C}$")
arrow(0.298, TY(c2), 0.385, TY(c2), lw=2.0)
d_thumb(0.390, y2, img_tau_clos, "d_e2_tau_closure", ["tau_clos_f"],
        [D["tau_clos_f"]],
        "closure-predicted floor traction (closure-composition result file, frame 0)",
        GOLD, -vlim_tau_f, vlim_tau_f, cmap_tau, hole=True)
txt(0.485, c2, "$\\approx$", ha="center", va="center", fontweight="bold")
d_thumb(0.515, y2, img_tau_true, "d_e2_tau_true", ["tau_true_f"],
        [D["tau_true_f"]],
        "recorded floor traction (closure-composition result file, frame 0)",
        GOLD, -vlim_tau_f, vlim_tau_f, cmap_tau, hole=True)
arrow(0.588, TY(c2), 0.837, TY(c2), color=GOLD, lw=2.0)
txt(0.7365, 0.182, "into the same slot", ha="center", va="bottom")
d_minibox(0.842, y2, "$\\mathcal{G}$")
txt(0.944, c2, "signal\ntransmits", ha="center", va="center", fontsize=12)

y3 = ROW_Y["E3"]; c3 = y3 + TH_H / 2
txt(0.044, c3 + 0.026, "3", ha="center", va="center", fontweight="bold")
txt(0.044, c3 - 0.012, "closure\ntransfer", ha="center", va="center")
d_thumb(0.085, y3, img_plane_h, "d_e3_state", ["plane_h", "mask_h"],
        [D["plane_h"], D["mask_h"]],
        "matching-height flow state entering the closure",
        BLUE, -vlim, vlim, cmap3, hole=True)
arrow(0.158, TY(c3), 0.245, TY(c3), lw=2.0)
d_minibox(0.250, y3, "$\\mathcal{C}$")
arrow(0.298, TY(c3), 0.372, TY(c3), lw=2.0)

axg1 = fig.add_axes([0.378, TY(y3), 0.085, HS(TH_H)])
im = axg1.imshow(np.asarray(D["up_ch"], float), origin="lower",
                 extent=(0.0, 1.0, 0.0, 1.0), aspect="auto", cmap=cmap3,
                 vmin=-vlim_ch, vmax=vlim_ch, interpolation="nearest")
bind_image(im, "d_e3_channel", "d", ["up_ch"], [D["up_ch"]],
           "channel slab repeated as a transfer-row target",
           "channel u-fluctuation slab (u_plane minus mean profile)")
axg1.axhline(0.02, color=GOLD, lw=2.2)
axg1.axhline(0.98, color=GOLD, lw=2.2)
txt(0.5155, c3, "$+$", ha="center", va="center")
axg2 = fig.add_axes([0.568, TY(y3), 0.085, HS(TH_H)])
im = axg2.imshow(img_plane_v[0:32, :], origin="lower",
                 extent=(0.0, 1.0, 0.0, 1.0), aspect="auto", cmap=cmap3,
                 vmin=-vlim, vmax=vlim, interpolation="nearest")
bind_image(im, "d_e3_cube", "d", ["plane_v", "mask_v"],
           [D["plane_v"], D["mask_v"]],
           "cube section (lowest 32 rows) repeated as a transfer-row target",
           "truth mid-span section of the cube record")
axg2.add_patch(Rectangle((0.25, 0.0), 0.50, 0.75, facecolor=SOLID_GREY,
                         edgecolor="none", zorder=2))
axg2.plot([0.0, 0.25, 0.25, 0.75, 0.75, 1.0],
          [0.02, 0.02, 0.75, 0.75, 0.02, 0.02],
          color=GOLD, lw=2.2, zorder=3, solid_capstyle="butt")
txt(0.7055, c3, "$+$", ha="center", va="center")
axg3 = fig.add_axes([0.758, TY(y3), 0.085, HS(TH_H)])
im = axg3.imshow(img_hill, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
                 aspect="auto", cmap=cmap3, vmin=-vlim_hill, vmax=vlim_hill,
                 interpolation="nearest")
bind_image(im, "d_e3_hill", "d", ["u_stdH", "solidH"],
           [D["u_stdH"], D["solidH"]],
           "hill field repeated as a transfer-row target",
           "standardised instantaneous hill field with the terrain masked")
im = axg3.imshow(img_hill_solid, origin="lower",
                 extent=(0.0, 1.0, 0.0, 1.0), aspect="auto", cmap=cmap_solid,
                 vmin=-1.2, vmax=1.6, interpolation="nearest", zorder=2)
bind_image(im, "d_e3_hill_solid", "d", ["solidH"], [D["solidH"]],
           "terrain mask drawn as a flat grey overlay",
           "hill solid mask from the staged hill context frame")
ln, = axg3.plot(bxy[:, 0], bxy[:, 1], color=GOLD, lw=1.8, zorder=3)
bind_line(ln, "d_e3_hill_boundary", "d", ["hill_boundary_xy"],
          [D["hill_boundary_xy"]],
          "0.5-level contour polyline of the terrain mask",
          "mapped hill boundary drawn in the band colour")
txt(0.922, c3, "closure-side\nevidence only", ha="center", va="center",
    fontsize=12)
for axg in (axg1, axg2, axg3):
    axg.set_xticks([]); axg.set_yticks([])
    axg.set_xlim(0, 1); axg.set_ylim(0, 1)
    for sp in axg.spines.values():
        sp.set_visible(True); sp.set_edgecolor(GREY); sp.set_linewidth(1.3)

save(fig, "fig1_architecture")
