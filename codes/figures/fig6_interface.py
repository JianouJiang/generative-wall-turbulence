#!/usr/bin/env python3
"""Main Figure 4: what the frozen traction-to-band interface transmits.

The redesign keeps every frozen arm and interval while replacing the original
bar-chart-first layout with four distinct evidence views:

* a side-on interface schematic and real representative mid-plane errors;
* a forest plot of absolute support-excluded skill;
* a four-region matrix of paired gains over the absent-band branch; and
* target-wise a-priori lift-fidelity distributions.

No field is synthetic.  Representative maps and target-wise fidelity values
come from the retained interface components; quantitative conclusions use all
160 evaluation targets.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D

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

RESULT_SRC = "fig6_interface/e2_traction_interface_results.json"
CONS_SRC = "fig6_interface/e2_traction_interface_conservative_blocks.json"
COMP_SRC = "fig6_interface/e2_traction_interface_components.npz"

document = load_json(RESULT_SRC)
res = document["evaluation"]
cons = load_json(CONS_SRC)
components = np.load(SOURCE / COMP_SRC, allow_pickle=False)

REGION = "full_support_excluded"
ABSOLUTE_ARMS = [
    ("correct", "oracle band (all)", BLUE, "o"),
    ("correct_physwall", "oracle band (walls)", LIGHT_BLUE, "o"),
    ("no_wall", "absent band", GREY, "o"),
    ("wrong_wall", "far-time band", RED, "o"),
    ("tau_lift_oracle", "lift: target $u_\\tau$", GOLD, "s"),
    ("tau_lift_model_predicted", "lift: closure $u_\\tau$", "#a87900", "s"),
    ("tau_lift_trainmean", "lift: mean load", LIGHT_GREY, "s"),
    ("tau_lift_fartime", "lift: far-time", LIGHT_RED, "s"),
]
REGIONS = [
    ("full_support_excluded", "complete"),
    ("near_support_excluded_d_le_0p5h", "near wall"),
    ("outer_d_gt_0p5h", "farther"),
    ("raster_unique_support_excluded", "unique rows"),
]
GAINS = [
    ("correct_minus_no_wall", "oracle band, all", BLUE, "o"),
    ("correct_physwall_minus_no_wall", "oracle band, walls", LIGHT_BLUE, "o"),
    ("tau_lift_oracle_minus_no_wall", "lift: target $u_\\tau$", GOLD, "s"),
    ("tau_lift_model_predicted_minus_no_wall", "lift: closure $u_\\tau$", GREY, "s"),
    ("tau_lift_trainmean_minus_no_wall", "lift: mean load", LIGHT_GREY, "s"),
    ("tau_lift_fartime_minus_no_wall", "lift: far-time", RED, "s"),
]
FIDELITY = [
    ("lift_oracle", "target $u_\\tau$", GOLD),
    ("lift_model", "closure $u_\\tau$", GREY),
    ("lift_trainmean", "mean load", LIGHT_GREY),
]
FIELD_ARMS = [
    ("representative__correct", "oracle all", BLUE),
    ("representative__correct_physwall", "oracle walls", LIGHT_BLUE),
    ("representative__no_wall", "absent", GREY),
    ("representative__wrong_wall", "far-time", RED),
    ("representative__tau_lift_oracle", "target $u_\\tau$", GOLD),
    ("representative__tau_lift_model_predicted", "closure $u_\\tau$", "#a87900"),
    ("representative__tau_lift_trainmean", "mean load", "#68737b"),
    ("representative__tau_lift_fartime", "far-time", LIGHT_RED),
]


def expected_horizontal_errorbar_payload(xv, yv, lower, upper, *, data_line=True):
    xv = np.asarray(xv, dtype=float).reshape(-1)
    yv = np.asarray(yv, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
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


def rounded_box(ax, xy, width, height, *, face="white", edge="#7b858d",
                linewidth=1.35, radius=0.02, zorder=2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, colour="#2f3439", linewidth=1.45,
          style="-|>", mutation=11, zorder=4):
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=linewidth,
        color=colour,
        zorder=zorder,
        shrinkA=1.0,
        shrinkB=1.0,
    )
    ax.add_patch(patch)
    return patch


def npz_plane_ref(key: str, component: int, view: str, index: int):
    selection = (
        [component, ":", ":", index]
        if view == "xy"
        else [component, ":", index, ":"]
    )
    return {
        "kind": "npz",
        "path": COMP_SRC,
        "key": key,
        "slice": selection,
    }


# Exact retained aligned-cube geometry for three complementary real planes.
x = (np.arange(48) + 0.5) * 2.0 / 48
y = (np.arange(96) + 0.5) * 4.0 / 96
z = (np.arange(48) + 0.5) * 2.0 / 48
z_index = int(np.asarray(components["representative_truth"]).shape[-1] // 2)
y_count = int(np.searchsorted(y, 2.0, side="left"))
X, Y = np.meshgrid(x, y[:y_count], indexing="ij")
solid_xy = (X >= 0.5) & (X <= 1.5) & (Y <= 1.0)
y_near_index = int(np.argmin(np.abs(y - 0.25)))
y_outer_index = int(np.argmin(np.abs(y - 1.50)))
X_xz, Z_xz = np.meshgrid(x, z, indexing="ij")
solid_xz_near = (
    (X_xz >= 0.5) & (X_xz <= 1.5)
    & (Z_xz >= 0.5) & (Z_xz <= 1.5)
)

VIEWS = [
    ("xy", z_index, "mid-span $x$--$y$", "$0 \\leq y/h \\leq 2$"),
    ("xz_near", y_near_index, "near-wall $x$--$z$", "$y/h \\simeq 0.25$"),
    ("xz_outer", y_outer_index, "outer $x$--$z$", "$y/h \\simeq 1.50$"),
]


def volume_projection(x_value: float, y_value: float, z_value: float):
    """Project a unit volume into a compact angled view."""

    origin = np.asarray([0.055, 0.120])
    x_vector = np.asarray([0.540, -0.100])
    y_vector = np.asarray([0.000, 0.640])
    z_vector = np.asarray([0.280, 0.200])
    return origin + x_value * x_vector + y_value * y_vector + z_value * z_vector


def projected_plane_image(ax, array, view_key, *, cmap, vmin, vmax):
    """Map one exact source plane into its physical position in the volume."""

    if view_key == "xy":
        origin = volume_projection(0.0, 0.0, 0.5)
        u_vector = volume_projection(1.0, 0.0, 0.5) - origin
        v_vector = volume_projection(0.0, 1.0, 0.5) - origin
    else:
        height = 0.125 if view_key == "xz_near" else 0.750
        origin = volume_projection(0.0, height, 0.0)
        u_vector = volume_projection(1.0, height, 0.0) - origin
        v_vector = volume_projection(0.0, height, 1.0) - origin
    transform = Affine2D.from_values(
        u_vector[0], u_vector[1], v_vector[0], v_vector[1],
        origin[0], origin[1],
    ) + ax.transData
    image = ax.imshow(
        array,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        transform=transform,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="auto",
        alpha=1.0,
        zorder=5,
    )
    corners = np.asarray([
        origin,
        origin + u_vector,
        origin + u_vector + v_vector,
        origin + v_vector,
        origin,
    ])
    ax.plot(
        corners[:, 0], corners[:, 1], color="#263238", lw=1.35,
        zorder=7,
    )
    return image


def draw_projected_volume(ax):
    """Draw restrained volume faces and the solid obstacle behind the data plane."""

    def face(vertices, colour, alpha, layer):
        ax.add_patch(Polygon(
            [volume_projection(*vertex) for vertex in vertices],
            closed=True, facecolor=colour, edgecolor="none",
            alpha=alpha, zorder=layer,
        ))

    # Restrained translucent domain faces establish depth without obscuring
    # the source-derived plane.
    face(((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), "#d4e0e5", 0.96, 0)
    face(((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)), "#d7e1e6", 0.82, 0)
    face(((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)), "#e9eff2", 0.66, 0)

    # The physical h x h x h obstacle occupies the central footprint and the
    # lower half of the displayed 0<=y/h<=2 volume.
    lo, hi, top = 0.25, 0.75, 0.50
    obstacle_faces = (
        (((lo, top, lo), (hi, top, lo), (hi, top, hi), (lo, top, hi)), "#aeb9bf"),
        (((lo, 0, lo), (hi, 0, lo), (hi, top, lo), (lo, top, lo)), "#c4ccd1"),
        (((hi, 0, lo), (hi, 0, hi), (hi, top, hi), (hi, top, lo)), "#98a7af"),
    )
    for vertices, colour in obstacle_faces:
        face(vertices, colour, 0.98, 2)


truth_raw = np.asarray(components["representative_truth"])
truth = truth_raw.astype(float)
FIELD_ERRORS = {}
for key, _, _ in FIELD_ARMS:
    field = np.asarray(components[key]).astype(float)
    volume_error = np.sqrt(np.square(field - truth).sum(axis=0))
    error_xy = volume_error[:, :y_count, z_index].T
    error_xy[solid_xy.T] = np.nan
    error_near = volume_error[:, y_near_index, :].T
    error_near[solid_xz_near.T] = np.nan
    error_outer = volume_error[:, y_outer_index, :].T
    FIELD_ERRORS[(key, "xy")] = error_xy
    FIELD_ERRORS[(key, "xz_near")] = error_near
    FIELD_ERRORS[(key, "xz_outer")] = error_outer
field_values = np.concatenate(
    [array[np.isfinite(array)] for array in FIELD_ERRORS.values()]
)
field_limit = float(np.percentile(field_values, 99.0))
error_cmap = plt.get_cmap("magma").copy()
error_cmap.set_bad("#d7dce0")


fig = plt.figure(figsize=(8.75, 10.70))
grid = fig.add_gridspec(
    3,
    12,
    height_ratios=[2.25, 0.84, 0.98],
    left=0.205,
    right=0.98,
    bottom=0.072,
    top=0.975,
    hspace=0.42,
    wspace=0.42,
)

# ---------------------------------------------- a: 8 payloads x 3 real planes
ax_a = fig.add_subplot(grid[0, :])
ax_a.set_axis_off()
panel_label(ax_a, "a", x=-0.235, y=1.03)

# The matrix directly mirrors the eight arms quantified below.  Each row pairs
# two payloads, and each payload retains the same left-to-right triplet of
# mid-span, near-wall and outer planes.  This keeps all 24 source planes while
# limiting the visual density to six volumes per row.
matrix_left = -0.155
matrix_right = 1.015
column_gap = 0.010
column_width = (matrix_right - matrix_left - 5 * column_gap) / 6
row_bottoms = [0.760, 0.520, 0.280, 0.040]
row_height = 0.195
row_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

for y_centre, label, colour, lower, upper in (
    (0.738, "velocity-band\npayloads", BLUE, 0.514, 0.998),
    (0.258, "equilibrium-lift\npayloads", GOLD, 0.030, 0.507),
):
    ax_a.text(
        matrix_left - 0.062,
        y_centre,
        label,
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        fontsize=12.0,
        color="#202428",
    )
    ax_a.add_patch(FancyBboxPatch(
        (matrix_left - 0.015, lower),
        matrix_right - matrix_left + 0.020,
        upper - lower,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=ax_a.transAxes,
        facecolor="none",
        edgecolor=colour,
        linewidth=1.55,
        linestyle=(0, (5.0, 3.0)),
        zorder=10,
        clip_on=False,
    ))

last_image = None
for row_index, pair in enumerate(row_pairs):
    bottom = row_bottoms[row_index]
    for half, arm_index in enumerate(pair):
        key, title, colour = FIELD_ARMS[arm_index]
        first_column = 3 * half
        triplet_left = matrix_left + first_column * (column_width + column_gap)
        triplet_right = triplet_left + 3 * column_width + 2 * column_gap
        title_y = bottom + row_height + 0.006
        ax_a.plot(
            [triplet_left + 0.016, triplet_right - 0.016],
            [title_y, title_y],
            transform=ax_a.transAxes,
            color="#40474d",
            lw=1.30,
            zorder=11,
            clip_on=False,
        )
        ax_a.text(
            0.5 * (triplet_left + triplet_right),
            title_y,
            title,
            transform=ax_a.transAxes,
            ha="center",
            va="center",
            fontsize=12.0,
            color="#202428",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.4},
            zorder=12,
        )
        arm_raw = np.asarray(components[key])
        for view_index, (view_key, plane_index, _, _) in enumerate(VIEWS):
            column = first_column + view_index
            left = matrix_left + column * (column_width + column_gap)
            field_ax = ax_a.inset_axes([left, bottom, column_width, row_height])
            field_ax.set_xlim(0.0, 1.0)
            field_ax.set_ylim(0.0, 1.0)
            field_ax.set_aspect(1.62)
            field_ax.set_axis_off()
            draw_projected_volume(field_ax)
            error = FIELD_ERRORS[(key, view_key)]
            expected_image_array = (
                np.ma.masked_invalid(error)
                if np.isnan(error).any()
                else np.ma.MaskedArray(error, mask=np.False_)
            )
            image = projected_plane_image(
                field_ax,
                error,
                cmap=error_cmap,
                vmin=0.0,
                vmax=field_limit,
                view_key=view_key,
            )
            last_image = image
            source_refs = []
            source_payload = []
            for component in range(3):
                source_refs.append(
                    npz_plane_ref("representative_truth", component, view_key, plane_index)
                )
                if view_key == "xy":
                    source_payload.append(truth_raw[component, :, :, plane_index])
                else:
                    source_payload.append(truth_raw[component, :, plane_index, :])
            for component in range(3):
                source_refs.append(npz_plane_ref(key, component, view_key, plane_index))
                if view_key == "xy":
                    source_payload.append(arm_raw[component, :, :, plane_index])
                else:
                    source_payload.append(arm_raw[component, :, plane_index, :])
            bind_artist(
                fig,
                image,
                artist_id=f"fig6.a.error_map.{key}.{view_key}",
                panel="a",
                source_refs=source_refs,
                source_payload=source_payload,
                expected_payload={
                    "type": "AxesImage",
                    "array": expected_image_array,
                    "extent": [0.0, 1.0, 0.0, 1.0],
                    "clim": [0.0, field_limit],
                },
                transform=(
                    "three-component Euclidean error versus representative LES truth; "
                    f"{view_key} plane at retained index {plane_index}; solid cube masked"
                ),
                evidence=(
                    "one retained illustrative target; quantitative inference uses all targets"
                ),
            )

cax = ax_a.inset_axes([0.15, -0.010, 0.46, 0.022])
colourbar = fig.colorbar(last_image, cax=cax, orientation="horizontal")
colourbar.set_label("$|\\mathbf{u}_{gen}-\\mathbf{u}_{LES}|$", fontsize=12.0, labelpad=1.0)
colourbar.ax.tick_params(labelsize=12.0, length=2.5, width=1.2, pad=1.0)
colourbar.outline.set_linewidth(1.3)
colourbar.dividers.set_linewidth(1.3)

# --------------------------------------------------------- b: absolute forest
# Give the lift-fidelity panel enough room for its descriptive y-axis labels;
# this trims the absolute-skill panel by about ten per cent relative to the
# original two-thirds/one-third division.
middle_grid = grid[1, :].subgridspec(
    1, 2, width_ratios=[3.0, 2.0], wspace=0.18
)
ax_b = fig.add_subplot(middle_grid[0, 0])
panel_label(ax_b, "b", x=-0.30, y=1.08)
ax_b.axvspan(0.0, 0.14, color="#eaf3f8", zorder=0)
ax_b.axvline(0.0, color="#202428", lw=1.5, zorder=1)
ax_b.axhspan(3.5, 7.5, color="#f3f7fa", alpha=0.8, zorder=0)
ax_b.axhspan(-0.5, 3.5, color="#fff8eb", alpha=0.75, zorder=0)

for row, (key, label, colour, marker) in enumerate(reversed(ABSOLUTE_ARMS)):
    yrow = float(row)
    value = float(res["arms"][key][REGION]["R2_fluct_balanced"])
    lo, hi = cons["arms"][key][REGION]["ci95_conservative_block"]
    artist = ax_b.errorbar(
        [value],
        [yrow],
        xerr=[[value - lo], [hi - value]],
        fmt=marker,
        markersize=6.0,
        markerfacecolor=colour,
        markeredgecolor="#202428",
        markeredgewidth=1.2,
        color=colour,
        ecolor=colour,
        elinewidth=1.6,
        capsize=3.2,
        capthick=1.4,
        zorder=3,
    )
    bind_artist(
        fig,
        artist,
        artist_id=f"fig6.b.absolute.{key}",
        panel="b",
        source_refs=[
            {
                "kind": "json",
                "path": RESULT_SRC,
                "key": f"evaluation.arms.{key}.{REGION}.R2_fluct_balanced",
            },
            {
                "kind": "json",
                "path": CONS_SRC,
                "key": f"arms.{key}.{REGION}.ci95_conservative_block",
            },
        ],
        source_payload=[value, [lo, hi]],
        expected_payload=expected_horizontal_errorbar_payload(
            [value], [yrow], [lo], [hi]
        ),
        transform="complete support-excluded point estimate and conservative interval",
        evidence="frozen generator; only conditioning payload differs",
    )

ax_b.set_yticks(
    np.arange(len(ABSOLUTE_ARMS)),
    [entry[1] for entry in reversed(ABSOLUTE_ARMS)],
)
ax_b.set_xlim(-0.39, 0.13)
ax_b.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1])
ax_b.set_ylim(-0.55, 7.55)
ax_b.set_xlabel("support-excluded fluctuation $R^2$")
ax_b.set_title("Absolute skill outside the supplied band", fontweight="normal")
ax_b.tick_params(axis="y", length=0, pad=5)
# ------------------------------------------- d: target-wise lift fidelity
ax_d = fig.add_subplot(middle_grid[0, 1])
panel_label(ax_d, "d", x=-0.14, y=1.08)
ax_d.axvspan(0.0, 0.45, color="#eaf3f8", zorder=0)
ax_d.axvline(0.0, color="#202428", lw=1.5, zorder=1)
fid_doc = res["a_priori_interface"]["band_fluctuation_R2_of_lift_vs_true_band"]
for row, (key, label, colour) in enumerate(reversed(FIDELITY)):
    values = np.asarray(components[f"band_fidelity__{key}"], dtype=float)
    order = np.argsort(np.argsort(values))
    jitter = 0.22 * ((order % 17) / 16.0 - 0.5)
    offsets = np.column_stack([values, np.full(values.shape, row) + jitter])
    scatter = ax_d.scatter(
        offsets[:, 0],
        offsets[:, 1],
        s=10,
        facecolor=colour,
        edgecolor="none",
        linewidths=0.0,
        alpha=0.28,
        zorder=2,
    )
    bind_artist(
        fig,
        scatter,
        artist_id=f"fig6.d.targetwise.{key}",
        panel="d",
        source_refs=[{
            "kind": "npz", "path": COMP_SRC,
            "key": f"band_fidelity__{key}",
        }],
        source_payload=[values],
        expected_payload={"type": "PathCollection", "offsets": offsets},
        transform="deterministic rank jitter only; every retained target shown",
        evidence="a-priori band fidelity before generator sampling",
    )
    mean = float(fid_doc[key]["mean"])
    sd = float(fid_doc[key]["sd"])
    summary = ax_d.errorbar(
        [mean],
        [row],
        xerr=[[sd], [sd]],
        fmt="o",
        markersize=6.2,
        markerfacecolor=colour,
        markeredgecolor="#202428",
        markeredgewidth=1.2,
        color=colour,
        ecolor="#30353a",
        elinewidth=2.1,
        capsize=0,
        zorder=4,
    )
    bind_artist(
        fig,
        summary,
        artist_id=f"fig6.d.mean_sd.{key}",
        panel="d",
        source_refs=[
            {
                "kind": "json", "path": RESULT_SRC,
                "key": (
                    "evaluation.a_priori_interface."
                    f"band_fluctuation_R2_of_lift_vs_true_band.{key}.mean"
                ),
            },
            {
                "kind": "json", "path": RESULT_SRC,
                "key": (
                    "evaluation.a_priori_interface."
                    f"band_fluctuation_R2_of_lift_vs_true_band.{key}.sd"
                ),
            },
        ],
        source_payload=[mean, sd],
        expected_payload=expected_horizontal_errorbar_payload(
            [mean], [row], [mean - sd], [mean + sd]
        ),
        transform="mean with one standard deviation across evaluation targets",
        evidence="spread, not an inferential interval",
    )

ax_d.set_yticks(
    np.arange(len(FIDELITY)),
    [entry[1] for entry in reversed(FIDELITY)],
)
ax_d.set_xlim(-1.05, 0.42)
ax_d.set_xticks([-1.0, -0.5, 0.0])
ax_d.set_ylim(-0.55, 2.55)
ax_d.set_xlabel("lift-to-true-band fluctuation $R^2$")
ax_d.set_title("Lift fidelity before sampling", fontweight="normal")
ax_d.tick_params(axis="y", length=0, pad=4)
# --------------------------------------- c: four-region paired-gain matrix
gain_grid = grid[2, :].subgridspec(1, 4, wspace=0.12)
gain_axes = [fig.add_subplot(gain_grid[0, index]) for index in range(4)]
gain_rows = list(reversed(GAINS))

for region_index, ((region, title), ax_c) in enumerate(zip(REGIONS, gain_axes)):
    ax_c.axvspan(0.0, 0.52, color="#eaf3f8", zorder=0)
    ax_c.axvspan(-0.24, 0.0, color="#fff1ef", alpha=0.75, zorder=0)
    ax_c.axvline(0.0, color="#202428", lw=1.5, zorder=1)
    for row, (key, label, colour, marker) in enumerate(gain_rows):
        value = float(res["deltas"][key][region]["point"])
        lo, hi = cons["deltas"][key][region]["ci95_conservative_block"]
        artist = ax_c.errorbar(
            [value],
            [row],
            xerr=[[value - lo], [hi - value]],
            fmt=marker,
            markersize=5.2,
            markerfacecolor=colour,
            markeredgecolor="#202428",
            markeredgewidth=1.05,
            color=colour,
            ecolor=colour,
            elinewidth=1.4,
            capsize=2.6,
            capthick=1.3,
            zorder=3,
        )
        bind_artist(
            fig,
            artist,
            artist_id=f"fig6.c.{region}.{key}",
            panel="c",
            source_refs=[
                {
                    "kind": "json", "path": RESULT_SRC,
                    "key": f"evaluation.deltas.{key}.{region}.point",
                },
                {
                    "kind": "json", "path": CONS_SRC,
                    "key": f"deltas.{key}.{region}.ci95_conservative_block",
                },
            ],
            source_payload=[value, [lo, hi]],
            expected_payload=expected_horizontal_errorbar_payload(
                [value], [row], [lo], [hi]
            ),
            transform="paired gain over absent band with conservative interval",
            evidence="same targets, checkpoint, sampler noise and scoring mask",
        )
    ax_c.set_xlim(-0.24, 0.52)
    ax_c.set_xticks([-0.2, 0.0, 0.2, 0.4])
    ax_c.set_ylim(-0.55, len(GAINS) - 0.45)
    ax_c.set_title(title, fontweight="normal")
    ax_c.set_xlabel("paired gain")
    if region_index == 0:
        panel_label(ax_c, "c", x=-1.03, y=1.12)
        ax_c.set_yticks(
            np.arange(len(GAINS)),
            [entry[1] for entry in gain_rows],
        )
        ax_c.tick_params(axis="y", length=0, pad=5)
    else:
        ax_c.set_yticks(np.arange(len(GAINS)), [""] * len(GAINS))
        ax_c.tick_params(axis="y", length=0)

components.close()
save(fig, "fig6_interface")
