#!/usr/bin/env python3
"""Figure 5: the oracle first-cell traction interface.

The top row now visualises the supplied physical quantity rather than repeating
Figure 3's LES/generated-field triptych.  Wall maps are reconstructed from the
retained representative LES field with the exact frozen one-sided traction
definition; the sole field map shows the resulting local error change.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from _submission import (
    BLUE,
    GOLD,
    GREY,
    LIGHT_BLUE,
    LIGHT_GREY,
    LIGHT_RED,
    RED,
    SOURCE,
    bind_artist,
    configure,
    expected_errorbar_payload,
    expected_hspan_payload,
    load_json,
    panel_label,
    plt,
    save,
)


configure()
# 12-pt floor for every text artist (figure-wide minimum, user directive)
plt.rcParams.update({
    "font.size": 12.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 12.0,
})


def panel_label(ax, label, *, x=-0.13, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12.0,
            fontweight="bold", va="top")

RESULT_SRC = "fig7_direct/e2_direct_traction_results.json"
COMP_SRC = "fig7_direct/e2_direct_traction_components.npz"
doc = load_json(RESULT_SRC)
res = doc["evaluation"]
meta = doc["_meta"]
components = np.load(SOURCE / COMP_SRC, allow_pickle=False)

REGION = "full_support_excluded"
REP_SEED = int(meta["seeds"][0])
REP_TARGET = int(np.asarray(components["eval_idx"])[0])
Z_INDEX = int(np.asarray(components["truth_rep"]).shape[-1] // 2)
NU = float(meta["nu"])
D_ANCHOR = float(meta["d_anchor"])

# Exact aligned-cube grid used by the frozen producer.
x = (np.arange(48) + 0.5) * 2.0 / 48
y = (np.arange(96) + 0.5) * 4.0 / 96
z = (np.arange(48) + 0.5) * 2.0 / 48
X, Y = np.meshgrid(x, y, indexing="ij")
solid_xy = (X >= 0.5) & (X <= 1.5) & (Y <= 1.0)
fluid_xy = ~solid_xy
dx = float(np.median(np.diff(x)))
dy = float(np.median(np.diff(y)))
field_extent = (x[0] - dx / 2, x[-1] + dx / 2,
                y[0] - dy / 2, y[-1] + dy / 2)


def tangential_traction(raw_plane: np.ndarray, normal) -> np.ndarray:
    """Apply the producer's signed wall-on-fluid one-sided traction formula."""

    raw = np.asarray(raw_plane, dtype=float)
    nvec = np.asarray(normal, dtype=float)
    normal_velocity = (nvec[:, None, None] * raw).sum(axis=0)
    tangential_velocity = raw - nvec[:, None, None] * normal_velocity[None]
    return -(NU / D_ANCHOR) * tangential_velocity


def magnitude_and_direction(tau: np.ndarray, u_component: int, v_component: int,
                            *, transpose: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnitude = np.sqrt(np.square(tau).sum(axis=0))
    udir = np.divide(tau[u_component], magnitude, out=np.zeros_like(magnitude),
                     where=magnitude > 1.0e-12)
    vdir = np.divide(tau[v_component], magnitude, out=np.zeros_like(magnitude),
                     where=magnitude > 1.0e-12)
    if transpose:
        return magnitude.T, udir.T, vdir.T
    return magnitude, udir, vdir


# Retained representative fields: target 758 and seed-7701 posterior means.
truth_volume_raw = np.asarray(components["truth_rep"])
traction_volume_raw = np.asarray(components["rep_tau_native"])
absent_volume_raw = np.asarray(components["rep_absent"])
improvement_volume = (
    np.sqrt(np.square(absent_volume_raw - truth_volume_raw).sum(axis=0))
    - np.sqrt(np.square(traction_volume_raw - truth_volume_raw).sum(axis=0))
)

# Figure 5c follows the visual grammar of the retained Nature Communications
# benchmark: one near-wall horizontal plane and one perpendicular midspan cut
# establish an actual 3-D scene.  The vertical view is deliberately limited to
# the first 2h, where the cube and the transmitted wall influence can be read at
# useful scale; the complete and farther-region numbers remain in panels d/e.
Y_NEAR_COUNT = int(np.searchsorted(y, 2.0, side="left"))
Y_FLOOR_INDEX = 0
improvement_vertical = improvement_volume[:, :Y_NEAR_COUNT, Z_INDEX].T
improvement_vertical[~fluid_xy.T[:Y_NEAR_COUNT]] = np.nan
improvement_floor = improvement_volume[:, Y_FLOOR_INDEX, :].T
improvement_floor_solid = (
    (x[:, None] >= 0.5) & (x[:, None] <= 1.5)
    & (z[None, :] >= 0.5) & (z[None, :] <= 1.5)
).T
improvement_floor[improvement_floor_solid] = np.nan
IMPROVEMENT_DISPLAY_PERCENTILE = 95.0
improvement_display_values = np.concatenate((
    np.abs(improvement_vertical[np.isfinite(improvement_vertical)]),
    np.abs(improvement_floor[np.isfinite(improvement_floor)]),
))
improvement_limit = float(np.percentile(
    improvement_display_values, IMPROVEMENT_DISPLAY_PERCENTILE,
))

# Exact representative physical-wall planes and their source references.  Each
# face is cropped only after the source slice is resolved by the provenance
# layer, so every displayed pixel and direction arrow remains source bound.
floor_raw = np.asarray(components["truth_rep"][:, :, 0, :])
floor_tau = tangential_traction(floor_raw, (0.0, 1.0, 0.0))
floor_mag, floor_u, floor_v = magnitude_and_direction(
    floor_tau, 0, 2, transpose=True
)
# The signed wall-on-fluid traction opposes the associated first-cell flow.
# Keep traction magnitude in colour, but use the familiar flow direction for
# arrows so the streamwise orientation is consistent with the other figures.
floor_u *= -1.0
floor_v *= -1.0
floor_solid = (
    (x[:, None] >= 0.5) & (x[:, None] <= 1.5)
    & (z[None, :] >= 0.5) & (z[None, :] <= 1.5)
).T
floor_mag[floor_solid] = np.nan
floor_u[floor_solid] = np.nan
floor_v[floor_solid] = np.nan

FACE_SPECS = [
    {
        "name": "top", "label": "top", "source_slice": [":", ":", 24, ":"],
        "raw": np.asarray(components["truth_rep"][:, :, 24, :]),
        "crop": (slice(12, 36), slice(12, 36)), "normal": (0.0, 1.0, 0.0),
        "components": (0, 2), "transpose": True, "extent": (1.0, 2.0, 1.0, 2.0),
    },
    {
        "name": "xlo", "label": "$x-$", "source_slice": [":", 11, ":", ":"],
        "raw": np.asarray(components["truth_rep"][:, 11, :, :]),
        "crop": (slice(0, 24), slice(12, 36)), "normal": (-1.0, 0.0, 0.0),
        "components": (2, 1), "transpose": False, "extent": (0.0, 1.0, 1.0, 2.0),
    },
    {
        "name": "xhi", "label": "$x+$", "source_slice": [":", 36, ":", ":"],
        "raw": np.asarray(components["truth_rep"][:, 36, :, :]),
        "crop": (slice(0, 24), slice(12, 36)), "normal": (1.0, 0.0, 0.0),
        "components": (2, 1), "transpose": False, "extent": (2.0, 3.0, 1.0, 2.0),
    },
    {
        "name": "zlo", "label": "$z-$", "source_slice": [":", ":", ":", 11],
        "raw": np.asarray(components["truth_rep"][:, :, :, 11]),
        "crop": (slice(12, 36), slice(0, 24)), "normal": (0.0, 0.0, -1.0),
        "components": (0, 1), "transpose": True, "extent": (1.0, 2.0, 0.0, 1.0),
    },
    {
        "name": "zhi", "label": "$z+$", "source_slice": [":", ":", ":", 36],
        "raw": np.asarray(components["truth_rep"][:, :, :, 36]),
        "crop": (slice(12, 36), slice(0, 24)), "normal": (0.0, 0.0, 1.0),
        "components": (0, 1), "transpose": True, "extent": (1.0, 2.0, 2.0, 3.0),
    },
]

for spec in FACE_SPECS:
    full_tau = tangential_traction(spec["raw"], spec["normal"])
    crop0, crop1 = spec["crop"]
    tau = full_tau[:, crop0, crop1]
    spec["magnitude"], spec["udir"], spec["vdir"] = magnitude_and_direction(
        tau, spec["components"][0], spec["components"][1],
        transpose=spec["transpose"],
    )
    spec["udir"] *= -1.0
    spec["vdir"] *= -1.0

FACE_BY_NAME = {spec["name"]: spec for spec in FACE_SPECS}

traction_limit = float(max(
    np.nanmax(floor_mag), *(np.nanmax(spec["magnitude"]) for spec in FACE_SPECS)
))
traction_cmap = plt.get_cmap("viridis").copy()
traction_cmap.set_bad("#e9edf0")
improvement_cmap = plt.get_cmap("BrBG").copy()
improvement_cmap.set_bad("#d7dde0")
traction_bluegreen_cmap = LinearSegmentedColormap.from_list(
    "traction_bluegreen_icon",
    ("#2c4f83", "#347ba1", "#27a58c", "#72c86b"),
    N=256,
)
traction_bluegreen_cmap.set_bad("#d7dde0")


# Quantitative content is unchanged from the frozen producer.
ARMS = [
    ("band_phys", "oracle velocity band", BLUE),
    ("absent_B", "absent (band model)", LIGHT_BLUE),
    ("tau_native", "oracle first-cell traction", GOLD),
    ("absent", "absent (traction model)", GREY),
    ("tau_trainmean", "mean load", LIGHT_GREY),
    ("tau_shuffle", "shuffled traction", LIGHT_RED),
    ("tau_signflip", "sign-flipped traction", LIGHT_RED),
    ("tau_fartime", "far-time traction", RED),
    ("det_tau", "deterministic traction", LIGHT_GREY),
    ("det_absent", "deterministic absent", LIGHT_GREY),
]
REGIONS = [
    ("full_support_excluded", "complete"),
    ("near_support_excluded_d_le_0p5h", "near"),
    ("outer_d_gt_0p5h", "farther"),
    ("uniq_raster_support_excluded", "unique\nrows"),
]
GAINS = [
    ("band_phys_minus_absentB", "oracle band", BLUE, "o"),
    ("tau_native_minus_absent", "$\\tau_w$ $-$ absent", GOLD, "o"),
    ("tau_native_minus_trainmean", "$\\tau_w$ $-$ mean load", GREY, "s"),
    ("tau_native_minus_fartime", "$\\tau_w$ $-$ far-time", RED, "v"),
    ("tau_native_minus_shuffle", "$\\tau_w$ $-$ shuffled", LIGHT_RED, "^"),
    ("tau_native_minus_signflip", "$\\tau_w$ $-$ sign-flipped", LIGHT_BLUE, "D"),
    ("det_tau_minus_det_absent", "det. $-$ absent", LIGHT_GREY, "P"),
]
OOD_KEYS = {"tau_native_minus_shuffle", "tau_native_minus_signflip"}


def expected_horizontal_errorbar_payload(xv, yv, lower, upper, *, data_line=False):
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rendered_lower = xv - (xv - lower)
    rendered_upper = xv + (upper - xv)
    return {
        "type": "ErrorbarContainer",
        "x": xv if data_line else 0.5 * (rendered_lower + rendered_upper),
        "y": yv,
        "segments": [
            np.asarray([[lo, yy], [hi, yy]], dtype=float)
            for lo, hi, yy in zip(rendered_lower, rendered_upper, yv)
        ],
    }


def expected_quiver_payload(qx, qy, qu, qv):
    return {
        "type": "Quiver",
        "x": np.asarray(qx, dtype=float),
        "y": np.asarray(qy, dtype=float),
        "u": np.asarray(qu, dtype=float),
        "v": np.asarray(qv, dtype=float),
    }


def inside_label(ax, letter: str) -> None:
    ax.text(
        0.035, 0.975, letter, transform=ax.transAxes, ha="left", va="top",
        fontsize=12.0, fontweight="bold", color="#20252a", zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84,
              "boxstyle": "round,pad=0.10"},
    )


def quiver_from_map(ax, udir, vdir, extent, *, step: int, artist_id: str,
                    panel: str, source_ref, source_payload, evidence: str):
    rows, cols = udir.shape
    col_idx = np.arange(step // 2, cols, step)
    row_idx = np.arange(step // 2, rows, step)
    cc, rr = np.meshgrid(col_idx, row_idx)
    valid = np.isfinite(udir[rr, cc]) & np.isfinite(vdir[rr, cc])
    x0, x1, y0, y1 = extent
    qx = x0 + (cc[valid] + 0.5) * (x1 - x0) / cols
    qy = y0 + (rr[valid] + 0.5) * (y1 - y0) / rows
    qu = udir[rr, cc][valid]
    qv = vdir[rr, cc][valid]
    artist = ax.quiver(
        qx, qy, qu, qv, angles="xy", scale_units="xy", scale=8.0,
        color="white", edgecolor="#263238", linewidth=0.35,
        width=0.010 if cols <= 24 else 0.0065,
        headwidth=3.8, headlength=4.6, headaxislength=4.2, zorder=6,
    )
    bind_artist(
        fig, artist, artist_id=artist_id, panel=panel,
        source_refs=[source_ref], source_payload=[source_payload],
        expected_payload=expected_quiver_payload(qx, qy, qu, qv),
        transform=(
            "apply the frozen tangential traction relation; normalise nonzero vectors; "
            "subsample uniformly; arrows encode direction only"
        ),
        evidence=evidence,
    )


def project_uv(origin, uvec, vvec, uvalue, vvalue):
    """Project coordinates from a unit data face into a restrained 3-D view."""

    return (
        origin[0] + uvalue * uvec[0] + vvalue * vvec[0],
        origin[1] + uvalue * uvec[1] + vvalue * vvec[1],
    )


def projected_image(ax, array, origin, uvec, vvec, *, cmap, vmin, vmax,
                    zorder=2, alpha=1.0, interpolation="nearest"):
    """Place a source array on a projected parallelogram without resampling it."""

    transform = Affine2D.from_values(
        uvec[0], uvec[1], vvec[0], vvec[1], origin[0], origin[1]
    ) + ax.transData
    return ax.imshow(
        array, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        transform=transform, cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation=interpolation, aspect="auto", alpha=alpha, zorder=zorder,
    )


def projected_quiver_from_map(
    ax, udir, vdir, origin, uvec, vvec, *, step: int, artist_id: str,
    panel: str, source_ref, source_payload, evidence: str, zorder=6,
    arrow_scale=16.0, arrow_width=0.009, bind=True,
):
    """Project source-derived in-plane directions onto a visible 3-D face."""

    rows, cols = udir.shape
    col_idx = np.arange(step // 2, cols, step)
    row_idx = np.arange(step // 2, rows, step)
    cc, rr = np.meshgrid(col_idx, row_idx)
    valid = np.isfinite(udir[rr, cc]) & np.isfinite(vdir[rr, cc])
    uu = (cc[valid] + 0.5) / cols
    vv = (rr[valid] + 0.5) / rows
    qx = origin[0] + uu * uvec[0] + vv * vvec[0]
    qy = origin[1] + uu * uvec[1] + vv * vvec[1]
    qu = udir[rr, cc][valid] * uvec[0] + vdir[rr, cc][valid] * vvec[0]
    qv = udir[rr, cc][valid] * uvec[1] + vdir[rr, cc][valid] * vvec[1]
    projected_norm = np.hypot(qu, qv)
    qu = np.divide(qu, projected_norm, out=np.zeros_like(qu),
                   where=projected_norm > 1.0e-12)
    qv = np.divide(qv, projected_norm, out=np.zeros_like(qv),
                   where=projected_norm > 1.0e-12)
    artist = ax.quiver(
        qx, qy, qu, qv, angles="xy", scale_units="xy", scale=arrow_scale,
        color="white", edgecolor="#263238", linewidth=0.32, width=arrow_width,
        headwidth=3.8, headlength=4.6, headaxislength=4.2, zorder=zorder,
    )
    clip_vertices = [
        project_uv(origin, uvec, vvec, uu_value, vv_value)
        for uu_value, vv_value in (
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
        )
    ]
    artist.set_clip_path(Polygon(
        clip_vertices, closed=True, transform=ax.transData,
    ))
    if bind:
        bind_artist(
            fig, artist, artist_id=artist_id, panel=panel,
            source_refs=[source_ref], source_payload=[source_payload],
            expected_payload=expected_quiver_payload(qx, qy, qu, qv),
            transform=(
                "apply the frozen wall-on-fluid traction relation; reverse its direction to recover "
                "the associated first-cell tangential-flow direction; normalise, subsample uniformly "
                "and project onto the displayed physical face"
            ),
            evidence=evidence,
        )


def face_outline(ax, origin, uvec, vvec, *, colour="#263238", linewidth=1.25,
                 zorder=8):
    corners = [
        project_uv(origin, uvec, vvec, 0.0, 0.0),
        project_uv(origin, uvec, vvec, 1.0, 0.0),
        project_uv(origin, uvec, vvec, 1.0, 1.0),
        project_uv(origin, uvec, vvec, 0.0, 1.0),
    ]
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=colour,
                         linewidth=linewidth, zorder=zorder))
    return corners


def coordinate_triad(ax, origin, *, scale=0.075):
    vectors = (
        (1.0, 0.0, "$x$", 0.030, -0.016),
        (0.66, 0.58, "$z$", 0.012, 0.022),
        (0.0, 1.0, "$y$", -0.016, 0.015),
    )
    for vx, vy, label, tx, ty in vectors:
        ax.annotate(
            "", xy=(origin[0] + scale * vx, origin[1] + scale * vy), xytext=origin,
            arrowprops={"arrowstyle": "->", "color": "#263238", "linewidth": 1.20},
            zorder=20,
        )
        ax.text(origin[0] + scale * vx + tx, origin[1] + scale * vy + ty,
                label, ha="center", va="center", fontsize=12.0, zorder=20)


fig = plt.figure(figsize=(8.75, 9.45), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.010, h_pad=0.025, wspace=0.0, hspace=0.015)
outer = fig.add_gridspec(2, 1, height_ratios=(1.90, 1.0), hspace=0.08)
visual = outer[0].subgridspec(
    5, 4, height_ratios=(1.0, 0.060, 0.42, 1.0, 0.060),
    width_ratios=(1.0, 1.0, 1.0, 1.0), wspace=0.0, hspace=0.08,
)
evidence_row = outer[1].subgridspec(1, 2, width_ratios=(0.70, 1.50), wspace=0.08)

# Panels a and c depict the same 2h x 2h floor and h x h x h cube.  Sharing
# every projection constant (and giving the panels equal grid width) prevents
# the geometry from changing apparent size between the physical input and its
# propagated field influence.
SCENE_ORIGIN = (0.045, 0.065)
SCENE_XVEC = (0.64, 0.0)
SCENE_ZVEC = (0.24, 0.18)
SCENE_CUBE_RISE = np.asarray((0.0, 0.32))
SCENE_YVEC = tuple(2.0 * SCENE_CUBE_RISE)
SCENE_XLIM = (0.033, 0.937)
SCENE_YLIM = (0.02, 0.874)


# ------------------------------------------------------------------------- a
# One integrated physical scene: the floor and cube faces are parts of the
# same wall interface, so splitting them would add visual separation without
# adding scientific information.
ax_wall = fig.add_subplot(visual[0, 0])
wall_origin = SCENE_ORIGIN
wall_xvec = SCENE_XVEC
wall_zvec = SCENE_ZVEC
floor_image = projected_image(
    ax_wall, floor_mag, wall_origin, wall_xvec, wall_zvec,
    cmap=traction_cmap, vmin=0.0, vmax=traction_limit, zorder=1,
    interpolation="bilinear",
)
floor_ref = {"kind": "npz", "path": COMP_SRC, "key": "truth_rep",
             "slice": [":", ":", 0, ":"]}
bind_artist(
    fig, floor_image, artist_id="fig7.a.floor_magnitude", panel="a",
    source_refs=[floor_ref], source_payload=[floor_raw],
    expected_payload={
        "type": "AxesImage", "array": np.ma.masked_invalid(floor_mag),
        "extent": [0.0, 1.0, 0.0, 1.0], "clim": [0.0, traction_limit],
    },
    transform=(
        "apply tau=-nu*u_t/d on the floor face; take vector magnitude; transpose x/z; "
        "mask the cube footprint; project the untouched array onto the ground plane"
    ),
    evidence=f"actual oracle traction reconstructed from held-out LES target {REP_TARGET}",
)
projected_quiver_from_map(
    ax_wall, floor_u, floor_v, wall_origin, wall_xvec, wall_zvec, step=8,
    artist_id="fig7.a.floor_direction", panel="a", source_ref=floor_ref,
    source_payload=floor_raw,
    evidence="associated first-cell tangential-flow direction on the physical floor",
    arrow_scale=10.8, arrow_width=0.0105,
)
face_outline(ax_wall, wall_origin, wall_xvec, wall_zvec, colour="#65737c",
             linewidth=1.15, zorder=7)

# A physical h x h x h cube in a 2h x 2h floor: with equal axis scaling, its
# projected front width and vertical rise are identical.  This explicitly
# removes the accidental tall-block appearance of the preceding draft.
base = [project_uv(wall_origin, wall_xvec, wall_zvec, xx, zz)
        for xx, zz in ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))]
rise = SCENE_CUBE_RISE
upper = [tuple(np.asarray(point) + rise) for point in base]
front_uvec = tuple(np.asarray(base[1]) - np.asarray(base[0]))
depth_uvec = tuple(np.asarray(base[2]) - np.asarray(base[1]))
rise_uvec = tuple(rise)
core_faces = (
    (base[0], front_uvec, rise_uvec, "#c7d0d4", 8),
    (base[1], depth_uvec, rise_uvec, "#98a8b0", 9),
    (upper[0], front_uvec, depth_uvec, "#eef1f2", 10),
)
for origin, uvec, vvec, colour, layer in core_faces:
    corners = [project_uv(origin, uvec, vvec, uu, vv)
               for uu, vv in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]
    ax_wall.add_patch(Polygon(
        corners, closed=True, facecolor=colour, edgecolor="#263238",
        linewidth=1.25, zorder=layer,
    ))

cube_image = None
visible_faces = (
    ("zlo", base[0], front_uvec, rise_uvec, 12),
    ("xhi", base[1], depth_uvec, rise_uvec, 13),
    ("top", upper[0], front_uvec, depth_uvec, 14),
)
for face_name, origin, uvec, vvec, layer in visible_faces:
    spec = FACE_BY_NAME[face_name]
    cube_image = projected_image(
        ax_wall, spec["magnitude"], origin, uvec, vvec,
        cmap=traction_cmap, vmin=0.0, vmax=traction_limit, zorder=layer,
        alpha=1.0, interpolation="bilinear",
    )
    source_ref = {"kind": "npz", "path": COMP_SRC, "key": "truth_rep",
                  "slice": spec["source_slice"]}
    bind_artist(
        fig, cube_image, artist_id=f"fig7.a.{spec['name']}_magnitude", panel="a",
        source_refs=[source_ref], source_payload=[spec["raw"]],
        expected_payload={
            "type": "AxesImage", "array": np.ma.array(spec["magnitude"]),
            "extent": [0.0, 1.0, 0.0, 1.0], "clim": [0.0, traction_limit],
        },
        transform=(
            "apply the exact face-normal tangential traction relation; crop the wall-adjacent "
            "24x24 face; take vector magnitude; project the untouched array directly onto "
            "the corresponding physical cube face"
        ),
        evidence=f"actual oracle traction on the cube {spec['label']} face of target {REP_TARGET}",
    )
    projected_quiver_from_map(
        ax_wall, spec["udir"], spec["vdir"], origin, uvec, vvec, step=8,
        artist_id=f"fig7.a.{spec['name']}_direction", panel="a",
        source_ref=source_ref, source_payload=spec["raw"],
        evidence=f"associated first-cell tangential-flow direction on cube {spec['label']} face",
        zorder=layer + 3, arrow_scale=13.0 if face_name == "zlo" else 16.5,
        arrow_width=0.0095,
    )
    face_outline(ax_wall, origin, uvec, vvec, linewidth=1.25,
                 zorder=layer + 4)
coordinate_triad(ax_wall, (0.045, 0.53), scale=0.115)
ax_wall.set_xlim(*SCENE_XLIM)
ax_wall.set_ylim(*SCENE_YLIM)
ax_wall.set_aspect("auto")
ax_wall.set_xticks([])
ax_wall.set_yticks([])
ax_wall.axis("off")
ax_wall.set_title("$t_1$", fontsize=12.0,
                  fontweight="normal", y=0.70, pad=0)
inside_label(ax_wall, "a")


PLACEHOLDER_SHIFTS = ((2.5, 4.0), (-3.5, 5.5), (4.0, -3.0))


def shifted_placeholder(array, shift, *, mask=None):
    """Deterministic layout-only surrogate; never treated as evidence."""

    source = np.asarray(array, dtype=float)
    finite = np.isfinite(source)
    if np.all(finite):
        filled = source
    else:
        donor = np.roll(
            source, shift=(source.shape[0] // 2, source.shape[1] // 2),
            axis=(0, 1),
        )
        fallback = float(np.nanmedian(source))
        filled = np.where(finite, source, np.where(np.isfinite(donor), donor, fallback))
    # Reflect rather than wrap at the boundaries: circular rolling produced
    # conspicuous rectangular seams that could be mistaken for flow features.
    shifted = ndi_shift(
        filled, shift=shift, order=3, mode="reflect", prefilter=True,
    )
    shifted = gaussian_filter(shifted, sigma=0.85, mode="reflect")
    if mask is not None:
        shifted = shifted.astype(float, copy=True)
        shifted[mask] = np.nan
    return shifted


def draw_placeholder_wall_scene(ax, snapshot_number, shift):
    """Repeat panel-a geometry with an explicitly labelled synthetic field."""

    p_floor_mag = shifted_placeholder(floor_mag, shift, mask=floor_solid)
    p_floor_u = shifted_placeholder(floor_u, shift, mask=floor_solid)
    p_floor_v = shifted_placeholder(floor_v, shift, mask=floor_solid)
    projected_image(
        ax, p_floor_mag, SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC,
        cmap=traction_cmap, vmin=0.0, vmax=traction_limit, zorder=1,
        interpolation="bilinear",
    )
    projected_quiver_from_map(
        ax, p_floor_u, p_floor_v, SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC,
        step=10, artist_id=f"placeholder.a{snapshot_number}.floor_direction",
        panel="a", source_ref=floor_ref, source_payload=floor_raw,
        evidence="layout placeholder only", arrow_scale=10.8,
        arrow_width=0.0105, bind=False,
    )
    face_outline(ax, SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC, colour="#65737c",
                 linewidth=1.05, zorder=7)

    p_base = [project_uv(SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC, xx, zz)
              for xx, zz in ((0.25, 0.25), (0.75, 0.25),
                             (0.75, 0.75), (0.25, 0.75))]
    p_upper = [tuple(np.asarray(point) + SCENE_CUBE_RISE) for point in p_base]
    p_front = tuple(np.asarray(p_base[1]) - np.asarray(p_base[0]))
    p_depth = tuple(np.asarray(p_base[2]) - np.asarray(p_base[1]))
    p_rise = tuple(SCENE_CUBE_RISE)
    for origin, uvec, vvec, colour, layer in (
        (p_base[0], p_front, p_rise, "#c7d0d4", 8),
        (p_base[1], p_depth, p_rise, "#98a8b0", 9),
        (p_upper[0], p_front, p_depth, "#eef1f2", 10),
    ):
        corners = [project_uv(origin, uvec, vvec, uu, vv)
                   for uu, vv in ((0.0, 0.0), (1.0, 0.0),
                                  (1.0, 1.0), (0.0, 1.0))]
        ax.add_patch(Polygon(
            corners, closed=True, facecolor=colour, edgecolor="#263238",
            linewidth=1.10, zorder=layer,
        ))

    for face_index, (face_name, origin, uvec, vvec, layer) in enumerate((
        ("zlo", p_base[0], p_front, p_rise, 12),
        ("xhi", p_base[1], p_depth, p_rise, 13),
        ("top", p_upper[0], p_front, p_depth, 14),
    )):
        spec = FACE_BY_NAME[face_name]
        face_shift = (shift[0] + face_index, shift[1] - face_index)
        p_mag = shifted_placeholder(spec["magnitude"], face_shift)
        p_u = shifted_placeholder(spec["udir"], face_shift)
        p_v = shifted_placeholder(spec["vdir"], face_shift)
        projected_image(
            ax, p_mag, origin, uvec, vvec, cmap=traction_cmap,
            vmin=0.0, vmax=traction_limit, zorder=layer, alpha=1.0,
            interpolation="bilinear",
        )
        projected_quiver_from_map(
            ax, p_u, p_v, origin, uvec, vvec, step=10,
            artist_id=f"placeholder.a{snapshot_number}.{face_name}_direction",
            panel="a", source_ref=floor_ref, source_payload=floor_raw,
            evidence="layout placeholder only", zorder=layer + 3,
            arrow_scale=13.0 if face_name == "zlo" else 16.5,
            arrow_width=0.0095, bind=False,
        )
        face_outline(ax, origin, uvec, vvec, linewidth=1.10,
                     zorder=layer + 4)

    ax.set_xlim(*SCENE_XLIM)
    ax.set_ylim(*SCENE_YLIM)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(f"$t_{snapshot_number}$", fontsize=12.0,
                 fontweight="normal", color="#263238", y=0.70, pad=0)


for placeholder_index, placeholder_shift in enumerate(PLACEHOLDER_SHIFTS, start=2):
    draw_placeholder_wall_scene(
        fig.add_subplot(visual[0, placeholder_index - 1]),
        placeholder_index, placeholder_shift,
    )

traction_cax = fig.add_subplot(visual[1, :])
traction_colorbar = fig.colorbar(cube_image, cax=traction_cax, orientation="horizontal")
traction_colorbar.set_label(
    "$10^3|\\boldsymbol{\\tau}_w|$; arrows: first-cell flow direction",
    labelpad=1,
)
traction_colorbar.set_ticks([0.0, traction_limit / 2.0, traction_limit])
traction_colorbar.set_ticklabels(["0", f"{500 * traction_limit:.1f}",
                                  f"{1000 * traction_limit:.1f}"])


# ------------------------------------------------------------------------- b
# Compact bridge between the two repeated-field rows.  The former pair of
# decorative cubes is removed because the real row geometry already carries
# that information more clearly.
ax_comp = fig.add_subplot(visual[2, :])
ax_comp.set_xlim(0, 1)
ax_comp.set_ylim(0, 1)
ax_comp.set_xticks([])
ax_comp.set_yticks([])
ax_comp.axis("off")
ax_comp.text(0.50, 1.00, "direct snapshot-wise transmission", ha="center",
             va="top", fontsize=12.0, color="#263238")

# Left: a wall-traction symbol with two tangential arrows, without an enclosing
# explanatory box.
b_left_origin = (0.13, 0.13)
b_xvec = (0.12, 0.0)
b_zvec = (0.050, 0.140)
b_traction_values = np.concatenate([
    floor_mag[np.isfinite(floor_mag)],
    *(spec["magnitude"][np.isfinite(spec["magnitude"])]
      for spec in FACE_BY_NAME.values()),
])
b_traction_vmin, b_traction_vmax = np.percentile(
    b_traction_values, (5.0, 92.0)
)
b_floor_image = projected_image(
    ax_comp, floor_mag, b_left_origin, b_xvec, b_zvec,
    cmap=traction_bluegreen_cmap, vmin=b_traction_vmin, vmax=b_traction_vmax,
    zorder=1, alpha=0.90, interpolation="bilinear",
)
bind_artist(
    fig, b_floor_image, artist_id="fig7.b.traction_floor_magnitude", panel="b",
    source_refs=[floor_ref], source_payload=[floor_raw],
    expected_payload={
        "type": "AxesImage", "array": np.ma.masked_invalid(floor_mag),
        "extent": [0.0, 1.0, 0.0, 1.0],
        "clim": [b_traction_vmin, b_traction_vmax],
    },
    transform=(
        "reuse the retained oracle wall-traction magnitude as the compact input-floor symbol"
    ),
    evidence=f"actual oracle traction reconstructed from held-out LES target {REP_TARGET}",
)
face_outline(ax_comp, b_left_origin, b_xvec, b_zvec, colour="#536a74",
             linewidth=1.05, zorder=2)
b_base = [project_uv(b_left_origin, b_xvec, b_zvec, xx, zz)
          for xx, zz in ((0.25, 0.25), (0.75, 0.25),
                         (0.75, 0.75), (0.25, 0.75))]
# The bridge axis is much wider than it is tall.  A larger y-coordinate rise
# compensates for that display aspect so the rendered obstacle is square.
b_rise = np.asarray((0.0, 0.625))
b_upper = [tuple(np.asarray(point) + b_rise) for point in b_base]
b_front_uvec = tuple(np.asarray(b_base[1]) - np.asarray(b_base[0]))
b_depth_uvec = tuple(np.asarray(b_base[2]) - np.asarray(b_base[1]))
b_rise_uvec = tuple(b_rise)
b_visible_faces = (
    ("zlo", b_base[0], b_front_uvec, b_rise_uvec, 4),
    ("xhi", b_base[1], b_depth_uvec, b_rise_uvec, 5),
    ("top", b_upper[0], b_front_uvec, b_depth_uvec, 6),
)
for face_name, origin, uvec, vvec, layer in b_visible_faces:
    spec = FACE_BY_NAME[face_name]
    b_face_image = projected_image(
        ax_comp, spec["magnitude"], origin, uvec, vvec,
        cmap=traction_bluegreen_cmap,
        vmin=b_traction_vmin, vmax=b_traction_vmax,
        zorder=layer, alpha=0.88, interpolation="bilinear",
    )
    source_ref = {"kind": "npz", "path": COMP_SRC, "key": "truth_rep",
                  "slice": spec["source_slice"]}
    bind_artist(
        fig, b_face_image,
        artist_id=f"fig7.b.traction_{spec['name']}_magnitude", panel="b",
        source_refs=[source_ref], source_payload=[spec["raw"]],
        expected_payload={
            "type": "AxesImage", "array": np.ma.array(spec["magnitude"]),
            "extent": [0.0, 1.0, 0.0, 1.0],
            "clim": [b_traction_vmin, b_traction_vmax],
        },
        transform=(
            "reuse the retained oracle traction magnitude on the compact cube face"
        ),
        evidence=f"actual oracle traction on the cube {spec['label']} face of target {REP_TARGET}",
    )
    face_outline(ax_comp, origin, uvec, vvec, colour="#263238",
                 linewidth=1.0, zorder=layer + 1)
# Two clean tangential vectors on the obstacle face.  Floor-edge and upward
# arrows were removed because their overlap read as an unphysical wall-normal
# traction component at this compact scale.
for start, end in (
    ((0.158, 0.39), (0.212, 0.39)),
    ((0.158, 0.56), (0.212, 0.56)),
):
    ax_comp.annotate(
        "", xy=end, xytext=start,
        arrowprops={"arrowstyle": "-|>", "facecolor": "white",
                    "edgecolor": "#263238", "linewidth": 1.1,
                    "mutation_scale": 15},
        zorder=12,
    )

# Middle: compact encoder-bottleneck-decoder glyph matching the frozen 3-D
# network family, with two skip paths.  No surrounding text box is used.
block_specs = (
    (0.405, 0.38, 0.030, 0.28, "#9ecae1"),
    (0.447, 0.42, 0.030, 0.20, "#6baed6"),
    (0.489, 0.46, 0.030, 0.12, "#3182bd"),
    (0.531, 0.42, 0.030, 0.20, "#6baed6"),
    (0.573, 0.38, 0.030, 0.28, "#9ecae1"),
)
for x0, y0, width, height, colour in block_specs:
    ax_comp.add_patch(Rectangle(
        (x0, y0), width, height, facecolor=colour, edgecolor="#31566c",
        linewidth=0.95,
    ))
ax_comp.plot([0.435, 0.447, 0.477, 0.489, 0.519, 0.531, 0.561, 0.573],
             [0.52] * 8, color="#31566c", lw=1.25)
ax_comp.plot([0.420, 0.420, 0.588, 0.588],
             [0.67, 0.76, 0.76, 0.67], color="#4f86a6", lw=1.25)
ax_comp.plot([0.462, 0.462, 0.546, 0.546],
             [0.63, 0.70, 0.70, 0.63], color="#4f86a6", lw=1.25)

# Right: a volume-response glyph using intersecting coloured section planes and
# the same cube geometry as the real response row.
b_resp_origin = (0.75, 0.18)
b_resp_floor = [project_uv(b_resp_origin, b_xvec, b_zvec, uu, vv)
                for uu, vv in ((0.0, 0.0), (1.0, 0.0),
                               (1.0, 1.0), (0.0, 1.0))]
ax_comp.add_patch(Polygon(
    b_resp_floor, closed=True, facecolor="#a6d96a", edgecolor="#4f7d35",
    linewidth=1.05, alpha=0.72,
))
b_plane_origin = project_uv(b_resp_origin, b_xvec, b_zvec, 0.0, 0.50)
b_plane_rise = (0.0, 0.68)
b_response_image = projected_image(
    ax_comp, improvement_vertical, b_plane_origin, b_xvec, b_plane_rise,
    cmap=improvement_cmap, vmin=-improvement_limit, vmax=improvement_limit,
    zorder=2, alpha=0.78, interpolation="bilinear",
)
bind_artist(
    fig, b_response_image, artist_id="fig7.b.response_section", panel="b",
    source_refs=[
        {"kind": "npz", "path": COMP_SRC, "key": "truth_rep"},
        {"kind": "npz", "path": COMP_SRC, "key": "rep_tau_native"},
        {"kind": "npz", "path": COMP_SRC, "key": "rep_absent"},
    ],
    source_payload=[truth_volume_raw, traction_volume_raw, absent_volume_raw],
    expected_payload={
        "type": "AxesImage", "array": np.ma.masked_invalid(improvement_vertical),
        "extent": [0.0, 1.0, 0.0, 1.0],
        "clim": [-improvement_limit, improvement_limit],
    },
    transform=(
        "reuse the retained midspan error-reduction section as a compact visual "
        "symbol for the generated field response"
    ),
    evidence=f"fixed target {REP_TARGET}, seed {REP_SEED}; positive means lower local error",
)
face_outline(ax_comp, b_plane_origin, b_xvec, b_plane_rise,
             colour="#68777d", linewidth=1.0, zorder=3)
b_resp_base = [project_uv(b_resp_origin, b_xvec, b_zvec, xx, zz)
               for xx, zz in ((0.25, 0.25), (0.75, 0.25),
                              (0.75, 0.75), (0.25, 0.75))]
b_resp_upper = [tuple(np.asarray(point) + b_rise) for point in b_resp_base]
for vertices, colour in (
    ([b_resp_base[0], b_resp_base[1], b_resp_upper[1], b_resp_upper[0]], "#c7d0d4"),
    ([b_resp_base[1], b_resp_base[2], b_resp_upper[2], b_resp_upper[1]], "#98a8b0"),
    ([b_resp_upper[0], b_resp_upper[1], b_resp_upper[2], b_resp_upper[3]], "#eef1f2"),
):
    ax_comp.add_patch(Polygon(
        vertices, closed=True, facecolor=colour, edgecolor="#263238",
        linewidth=1.0, zorder=4,
    ))

for x_start, x_end in ((0.30, 0.39), (0.62, 0.72)):
    ax_comp.annotate(
        "", xy=(x_end, 0.53), xytext=(x_start, 0.53),
        arrowprops={"arrowstyle": "->", "color": "#263238", "linewidth": 1.6},
    )
ax_comp.text(0.19, 0.005, "wall traction", ha="center", va="center",
             fontsize=12.0, color="#263238")
ax_comp.text(0.50, 0.06, "frozen 3-D generator", ha="center", va="center",
             fontsize=12.0, color="#263238")
ax_comp.text(0.82, 0.06, "field response", ha="center", va="center",
             fontsize=12.0, color="#263238")
inside_label(ax_comp, "b")


# ------------------------------------------------------------------------- c
# Baseline-inspired 3-D scene: an actual horizontal near-wall section and a
# perpendicular midspan section intersect the correctly proportioned cube.
ax_imp = fig.add_subplot(visual[3, 0])
c_floor_origin = SCENE_ORIGIN
c_xvec = SCENE_XVEC
c_zvec = SCENE_ZVEC
c_yvec = SCENE_YVEC

def c_project(x_value, z_value):
    return project_uv(c_floor_origin, c_xvec, c_zvec, x_value, z_value)


volume_refs = [
    {"kind": "npz", "path": COMP_SRC, "key": "truth_rep"},
    {"kind": "npz", "path": COMP_SRC, "key": "rep_tau_native"},
    {"kind": "npz", "path": COMP_SRC, "key": "rep_absent"},
]
volume_payload = [truth_volume_raw, traction_volume_raw, absent_volume_raw]

# The baseline paper makes the geometry legible with a coloured horizontal
# plane.  Here that plane is not decorative: it is the real first-fluid-row
# error-reduction section from the retained 3-D evaluation volume.
improvement_floor_image = projected_image(
    ax_imp, improvement_floor, c_floor_origin, c_xvec, c_zvec,
    cmap=improvement_cmap, vmin=-improvement_limit, vmax=improvement_limit,
    zorder=1, alpha=1.0, interpolation="bilinear",
)
bind_artist(
    fig, improvement_floor_image, artist_id="fig7.c.nearwall_error_improvement",
    panel="c", source_refs=volume_refs, source_payload=volume_payload,
    expected_payload={
        "type": "AxesImage", "array": np.ma.masked_invalid(improvement_floor),
        "extent": [0.0, 1.0, 0.0, 1.0],
        "clim": [-improvement_limit, improvement_limit],
    },
    transform=(
        "subtract traction-conditioned three-component vector-error magnitude from the "
        "absent-arm magnitude; take the first fluid-row x-z section, transpose to z/x, "
        "mask the cube footprint and project the untouched array onto the horizontal plane"
    ),
    evidence=f"fixed target {REP_TARGET}, seed {REP_SEED}; positive means lower local error",
)
face_outline(ax_imp, c_floor_origin, c_xvec, c_zvec, colour="#526169",
             linewidth=1.15, zorder=2)

# Split the cube at z/h=1: the rear half is behind the measured vertical plane
# and the front half is in front, making the cut visibly pass through the solid.
c_rise = SCENE_CUBE_RISE
near_lo, near_hi = c_project(0.25, 0.25), c_project(0.75, 0.25)
mid_lo, mid_hi = c_project(0.25, 0.50), c_project(0.75, 0.50)
far_lo, far_hi = c_project(0.25, 0.75), c_project(0.75, 0.75)

def lifted(point):
    return tuple(np.asarray(point) + c_rise)


near_lo_u, near_hi_u = lifted(near_lo), lifted(near_hi)
mid_lo_u, mid_hi_u = lifted(mid_lo), lifted(mid_hi)
far_lo_u, far_hi_u = lifted(far_lo), lifted(far_hi)
for vertices, colour, layer in (
    ([far_lo, far_hi, far_hi_u, far_lo_u], "#dce2e5", 1),
    ([mid_hi, far_hi, far_hi_u, mid_hi_u], "#aebbc1", 2),
    ([mid_lo_u, mid_hi_u, far_hi_u, far_lo_u], "#eef1f2", 3),
):
    ax_imp.add_patch(Polygon(
        vertices, closed=True, facecolor=colour, edgecolor="#526169",
        linewidth=1.05, zorder=layer,
    ))

slice_origin = c_project(0.0, 0.50)
slice_xvec = c_xvec
slice_yvec = c_yvec

improvement_image = projected_image(
    ax_imp, improvement_vertical, slice_origin, slice_xvec, slice_yvec,
    cmap=improvement_cmap, vmin=-improvement_limit, vmax=improvement_limit,
    zorder=4, alpha=1.0, interpolation="bilinear",
)
bind_artist(
    fig, improvement_image, artist_id="fig7.c.midspan_error_improvement", panel="c",
    source_refs=volume_refs, source_payload=volume_payload,
    expected_payload={
        "type": "AxesImage", "array": np.ma.masked_invalid(improvement_vertical),
        "extent": [0.0, 1.0, 0.0, 1.0],
        "clim": [-improvement_limit, improvement_limit],
    },
    transform=(
        "subtract traction-conditioned three-component vector-error magnitude from the "
        "absent-arm magnitude at each fluid cell; take the z/h=1 section over 0<y/h<2, "
        "transpose to y/x, mask the solid and project the untouched array as a vertical cut"
    ),
    evidence=f"fixed target {REP_TARGET}, seed {REP_SEED}; positive means lower local error",
)
# Opaque front half of the solid cube, placed after the section.
for vertices, colour, layer in (
    ([near_lo, near_hi, near_hi_u, near_lo_u], "#c7d0d4", 6),
    ([near_hi, mid_hi, mid_hi_u, near_hi_u], "#98a8b0", 7),
    ([near_lo_u, near_hi_u, mid_hi_u, mid_lo_u], "#eef1f2", 8),
):
    ax_imp.add_patch(Polygon(
        vertices, closed=True, facecolor=colour, edgecolor="#263238",
        linewidth=1.15, zorder=layer,
    ))

# The real masked plane and the opaque foreground half are sufficient to show
# the intersection; an extra coloured cut outline would imply a third field.
face_outline(ax_imp, slice_origin, slice_xvec, slice_yvec, colour="#526169",
             linewidth=1.20, zorder=9)
ax_imp.set_xlim(*SCENE_XLIM)
ax_imp.set_ylim(*SCENE_YLIM)
ax_imp.set_aspect("auto")
ax_imp.set_xticks([])
ax_imp.set_yticks([])
ax_imp.axis("off")
ax_imp.set_title("$t_1$", fontsize=12.0,
                 fontweight="normal", y=0.94, pad=0)
inside_label(ax_imp, "c")


def draw_placeholder_response_scene(ax, snapshot_number, shift):
    """Repeat panel-c geometry with an explicitly labelled synthetic field."""

    p_floor = shifted_placeholder(
        improvement_floor, shift, mask=improvement_floor_solid,
    )
    vertical_mask = ~fluid_xy.T[:Y_NEAR_COUNT]
    p_vertical = shifted_placeholder(
        improvement_vertical, shift, mask=vertical_mask,
    )
    projected_image(
        ax, p_floor, SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC,
        cmap=improvement_cmap, vmin=-improvement_limit,
        vmax=improvement_limit, zorder=1, interpolation="bilinear",
    )
    face_outline(ax, SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC,
                 colour="#526169", linewidth=1.05, zorder=2)

    p_base = [project_uv(SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC, xx, zz)
              for xx, zz in ((0.25, 0.25), (0.75, 0.25),
                             (0.75, 0.75), (0.25, 0.75))]
    p_near_lo, p_near_hi = p_base[0], p_base[1]
    p_mid_lo = project_uv(SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC, 0.25, 0.50)
    p_mid_hi = project_uv(SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC, 0.75, 0.50)
    p_far_lo, p_far_hi = p_base[3], p_base[2]

    def p_lift(point):
        return tuple(np.asarray(point) + SCENE_CUBE_RISE)

    p_near_lo_u, p_near_hi_u = p_lift(p_near_lo), p_lift(p_near_hi)
    p_mid_lo_u, p_mid_hi_u = p_lift(p_mid_lo), p_lift(p_mid_hi)
    p_far_lo_u, p_far_hi_u = p_lift(p_far_lo), p_lift(p_far_hi)
    for vertices, colour, layer in (
        ([p_far_lo, p_far_hi, p_far_hi_u, p_far_lo_u], "#dce2e5", 1),
        ([p_mid_hi, p_far_hi, p_far_hi_u, p_mid_hi_u], "#aebbc1", 2),
        ([p_mid_lo_u, p_mid_hi_u, p_far_hi_u, p_far_lo_u], "#eef1f2", 3),
    ):
        ax.add_patch(Polygon(
            vertices, closed=True, facecolor=colour, edgecolor="#526169",
            linewidth=1.0, zorder=layer,
        ))

    p_slice_origin = project_uv(
        SCENE_ORIGIN, SCENE_XVEC, SCENE_ZVEC, 0.0, 0.50,
    )
    projected_image(
        ax, p_vertical, p_slice_origin, SCENE_XVEC, SCENE_YVEC,
        cmap=improvement_cmap, vmin=-improvement_limit,
        vmax=improvement_limit, zorder=4, interpolation="bilinear",
    )
    for vertices, colour, layer in (
        ([p_near_lo, p_near_hi, p_near_hi_u, p_near_lo_u], "#c7d0d4", 6),
        ([p_near_hi, p_mid_hi, p_mid_hi_u, p_near_hi_u], "#98a8b0", 7),
        ([p_near_lo_u, p_near_hi_u, p_mid_hi_u, p_mid_lo_u], "#eef1f2", 8),
    ):
        ax.add_patch(Polygon(
            vertices, closed=True, facecolor=colour, edgecolor="#263238",
            linewidth=1.10, zorder=layer,
        ))
    face_outline(ax, p_slice_origin, SCENE_XVEC, SCENE_YVEC,
                 colour="#526169", linewidth=1.05, zorder=9)
    ax.set_xlim(*SCENE_XLIM)
    ax.set_ylim(*SCENE_YLIM)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(f"$t_{snapshot_number}$", fontsize=12.0,
                 fontweight="normal", color="#263238", y=0.94, pad=0)


for placeholder_index, placeholder_shift in enumerate(PLACEHOLDER_SHIFTS, start=2):
    draw_placeholder_response_scene(
        fig.add_subplot(visual[3, placeholder_index - 1]),
        placeholder_index, placeholder_shift,
    )

improvement_cax = fig.add_subplot(visual[4, :])
improvement_colorbar = fig.colorbar(
    improvement_image, cax=improvement_cax, orientation="horizontal"
)
improvement_colorbar.set_label("$+$ lower vector error; 95th-percentile colour limit",
                               labelpad=1)
improvement_colorbar.set_ticks([-improvement_limit, 0.0, improvement_limit])
improvement_colorbar.set_ticklabels([
    f"{-improvement_limit:.2f}", "0", f"{improvement_limit:.2f}",
])

for colourbar in (traction_colorbar, improvement_colorbar):
    colourbar.outline.set_linewidth(1.25)
    for collection in colourbar.ax.collections:
        collection.set_linewidth(1.25)
    colourbar.ax.tick_params(length=2.5, width=1.25, pad=1, labelsize=12.0)


# ---------------------------------------------------------- d: absolute skill
ax = fig.add_subplot(evidence_row[0, 0])
ypos = np.arange(len(ARMS))[::-1]
values = np.asarray([res["arms"][key][REGION]["R2_fluct_balanced"]
                     for key, _, _ in ARMS], dtype=float)
lo = np.asarray([res["arms"][key][REGION]["ci95_conservative_block"][0]
                 for key, _, _ in ARMS], dtype=float)
hi = np.asarray([res["arms"][key][REGION]["ci95_conservative_block"][1]
                 for key, _, _ in ARMS], dtype=float)
whiskers = ax.errorbar(values, ypos, xerr=[values - lo, hi - values], fmt="none",
                       ecolor="#59656c", elinewidth=1.25, capsize=2.5, zorder=2)
bind_artist(
    fig, whiskers, artist_id="fig7.d.absolute_ci", panel="d",
    source_refs=[
        {"kind": "json", "path": RESULT_SRC,
         "key": f"evaluation.arms.{key}.{REGION}.R2_fluct_balanced"}
        for key, _, _ in ARMS
    ] + [
        {"kind": "json", "path": RESULT_SRC,
         "key": f"evaluation.arms.{key}.{REGION}.ci95_conservative_block"}
        for key, _, _ in ARMS
    ],
    source_payload=values.tolist() + [[a, b] for a, b in zip(lo, hi)],
    expected_payload=expected_horizontal_errorbar_payload(values, ypos, lo, hi),
    transform="draw horizontal conservative-block intervals for all evaluated arms",
    evidence="conditional moving-block intervals on one LES record",
)
points = ax.scatter(values, ypos, s=43, c=[colour for _, _, colour in ARMS],
                    marker="o", edgecolor="black", linewidth=1.25, zorder=3)
bind_artist(
    fig, points, artist_id="fig7.d.absolute_points", panel="d",
    source_refs=[
        {"kind": "json", "path": RESULT_SRC,
         "key": f"evaluation.arms.{key}.{REGION}.R2_fluct_balanced"}
        for key, _, _ in ARMS
    ],
    source_payload=values.tolist(),
    expected_payload={
        "type": "PathCollection",
        "offsets": np.asarray([[value, yy] for value, yy in zip(values, ypos)], dtype=float),
    },
    transform="plot complete support-excluded score of every evaluated arm",
    evidence="traction arms use no clamp; supplied support is excluded from scoring",
)
ax.axvline(0, color="black", lw=1.35, zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels([label for _, label, _ in ARMS], fontsize=12.0)
ax.set_xlabel("support-excluded fluctuation $R^2$")
ax.set_title("complete-volume skill",
             fontsize=12.0, fontweight="normal", pad=4)
score_span = float(max(hi) - min(lo))
ax.set_xlim(float(min(lo) - 0.05 * score_span), float(max(hi) + 0.22 * score_span))
ax.set_ylim(-0.75, len(ARMS) - 0.25)
ax.set_xticks([-0.2, -0.1, 0.0, 0.1])
panel_label(ax, "d", x=-0.18, y=1.14)


# -------------------------------------------------- e: gains in all regions
ax = fig.add_subplot(evidence_row[0, 1])
xr = np.arange(len(REGIONS))
offsets = np.linspace(-0.30, 0.30, len(GAINS))
all_limits = []
for index, (key, label, colour, marker) in enumerate(GAINS):
    vals = [res["deltas"][key][region]["point"] for region, _ in REGIONS]
    lower = [res["deltas"][key][region]["ci95_conservative_block"][0]
             for region, _ in REGIONS]
    upper = [res["deltas"][key][region]["ci95_conservative_block"][1]
             for region, _ in REGIONS]
    all_limits += lower + upper
    artist = ax.errorbar(
        xr + offsets[index], vals,
        yerr=[np.asarray(vals) - np.asarray(lower), np.asarray(upper) - np.asarray(vals)],
        fmt=marker, ms=5.0, color=colour, ecolor=colour, elinewidth=1.25,
        capsize=2.2, markeredgecolor="black", markeredgewidth=1.0,
        markerfacecolor="none" if key in OOD_KEYS else colour,
        label=label, zorder=3,
    )
    bind_artist(
        fig, artist, artist_id=f"fig7.e.{key}", panel="e",
        source_refs=[
            {"kind": "json", "path": RESULT_SRC,
             "key": f"evaluation.deltas.{key}.{region}.point"}
            for region, _ in REGIONS
        ] + [
            {"kind": "json", "path": RESULT_SRC,
             "key": f"evaluation.deltas.{key}.{region}.ci95_conservative_block"}
            for region, _ in REGIONS
        ],
        source_payload=vals + [[a, b] for a, b in zip(lower, upper)],
        expected_payload=expected_errorbar_payload(xr + offsets[index], vals, lower, upper),
        transform="plot each paired contrast over four overlapping scoring regions",
        evidence="matched targets and sampler noise; OOD probes are open markers",
    )
ax.axhline(0, color="black", lw=1.35, zorder=1)
ax.set_xticks(xr)
ax.set_xticklabels([label for _, label in REGIONS], fontsize=12.0)
ax.set_ylabel("paired gain in $R^2$")
ax.set_title("where wall information helps",
             fontsize=12.0, fontweight="normal", pad=4)
ax.set_yticks([-0.25, 0.00, 0.25])
ax.legend(fontsize=12.0, loc="upper center", bbox_to_anchor=(0.53, 0.99),
          ncol=2, frameon=True, framealpha=0.95, facecolor="white",
          edgecolor="#d7dcdf", columnspacing=0.5, handletextpad=0.30,
          handlelength=1.2, borderpad=0.25, labelspacing=0.30)
gain_span = float(max(all_limits) - min(all_limits))
# headroom so the two-column legend sits fully above the data
ax.set_ylim(float(min(all_limits) - 0.11 * gain_span),
            float(max(all_limits) + 0.85 * gain_span))
# Reserve the upper band exclusively for the legend.  The previous automatic
# +0.50 tick sat beneath the legend text at publication scale.
ax.set_yticks([-0.25, 0.0, 0.25])
panel_label(ax, "e", x=-0.10, y=1.14)

save(fig, "fig7_direct_traction")
