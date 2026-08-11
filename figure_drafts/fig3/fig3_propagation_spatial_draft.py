#!/usr/bin/env python3
"""Draft redesign of Figure 3 with spatial evidence above the M0 statistics.

The spatial row is deliberately derived from the already-frozen M2
representative cube volume.  It is illustrative and does not replace the M0
first-contact aggregate evidence in the lower row.  Output and provenance are
written inside ``figure_drafts/fig3`` so this design pass cannot disturb the
frozen submission package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if os.environ.get("GWT_FIG3_PROMOTE") != "1":
    os.environ.setdefault("GWT_FIGURE_OUTPUT", str(HERE))
    os.environ.setdefault("GWT_ARTIST_MANIFEST_DIR", str(HERE / "artist_manifests"))
    os.environ.setdefault("GWT_LAYOUT_AUDIT_DIR", str(HERE / "layout_audits"))
sys.path.insert(0, str(PROJECT / "codes" / "figures"))

from _submission import (  # noqa: E402
    BLUE,
    GREY,
    LIGHT_GREY,
    RED,
    bind_artist,
    configure,
    load_json,
    panel_label,
    plt,
    save,
    SOURCE,
)


configure()


def panel_label(ax, label, *, x=-0.13, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12.0,
            fontweight="bold", va="top")


# 12-pt floor for every text artist (figure-wide minimum = panel-a title size)
plt.rcParams.update({
    "font.size": 12.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 12.0,
})

# ---------------------------------------------------------------------- data
audit = load_json("review_audit/derived_peer_review_statistics.json")
m0 = audit["first_contact_m0"]
cube = np.load(SOURCE / "fig2/cube3d_representative_fields.npz")

z_index = cube["truth"].shape[-1] // 2
fluid_3d = np.asarray(cube["fluid"], dtype=bool)
band_3d = np.asarray(cube["band"], dtype=bool)
scored_3d = fluid_3d & (~band_3d)
x = np.asarray(cube["x"], dtype=float)
y = np.asarray(cube["y"], dtype=float)
dx = float(np.median(np.diff(x)))
dy = float(np.median(np.diff(y)))
extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)


def vector_slice(key: str) -> np.ndarray:
    """Return all three components on the retained mid-span plane."""

    return np.asarray(cube[key][:, :, :, z_index], dtype=float)


def component_slice(key: str) -> np.ma.MaskedArray:
    """Return the streamwise component in display orientation."""

    values = vector_slice(key)[0].T
    return np.ma.masked_where(~fluid_3d[:, :, z_index].T, values)


truth_vec = vector_slice("truth")
correct_vec = vector_slice("correct")
no_wall_vec = vector_slice("no_wall")
wrong_vec = vector_slice("wrong_wall")
truth_u = component_slice("truth")
correct_u = component_slice("correct")

# Positive local gain means that aligned wall information reduces vector
# squared error relative to the named control.  Supplied-band cells are masked:
# this visual evidence therefore obeys the same support exclusion as the scores.
gain_absent_raw = np.sum((no_wall_vec - truth_vec) ** 2, axis=0) - np.sum(
    (correct_vec - truth_vec) ** 2, axis=0
)
gain_far_raw = np.sum((wrong_vec - truth_vec) ** 2, axis=0) - np.sum(
    (correct_vec - truth_vec) ** 2, axis=0
)
scored_mid = scored_3d[:, :, z_index].T
gain_absent = np.ma.masked_where(~scored_mid, gain_absent_raw.T)
gain_far = np.ma.masked_where(~scored_mid, gain_far_raw.T)

velocity_values = np.concatenate(
    [np.asarray(truth_u.compressed()), np.asarray(correct_u.compressed())]
)
velocity_limit = float(np.percentile(np.abs(velocity_values), 99.5))
gain_values = np.concatenate(
    [np.abs(gain_absent.compressed()), np.abs(gain_far.compressed())]
)
gain_limit = float(np.percentile(gain_values, 98.5))

velocity_cmap = plt.get_cmap("RdBu_r").copy()
velocity_cmap.set_bad("#e9edf0")
gain_cmap = plt.get_cmap("BrBG").copy()
gain_cmap.set_bad("#e9edf0")
band_mid = band_3d[:, :, z_index].T

# --------------------------------------------------------------------- canvas
fig = plt.figure(figsize=(8.75, 8.60), constrained_layout=True)
outer = fig.add_gridspec(3, 1, height_ratios=(1.42, 0.70, 1.0), hspace=0.10)
spatial = outer[0].subgridspec(
    1, 6, width_ratios=(1, 1, 0.055, 1, 1, 0.055), wspace=0.09
)
logic = outer[1].subgridspec(1, 2, width_ratios=(0.96, 1.04), wspace=0.05)
quant = outer[2].subgridspec(1, 2, width_ratios=(1, 1), wspace=0.06)


def style_field_axis(ax, *, show_y: bool) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("$x/h$")
    if show_y:
        ax.set_ylabel("$y/h$")
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=12.0)
    ax.contour(
        x,
        y,
        band_mid,
        levels=[0.5],
        colors=["#e88900"],
        linewidths=1.45,
        origin="lower",
    )


# -------------------------------------------------------------- a: flow maps
flow_specs = [
    ("truth", truth_u, "LES target\n(M2 example)"),
    ("correct", correct_u, "correct-time\n8-sample mean"),
]
flow_image = None
for index, (key, values, title) in enumerate(flow_specs):
    ax = fig.add_subplot(spatial[index])
    flow_image = ax.imshow(
        values,
        origin="lower",
        extent=extent,
        cmap=velocity_cmap,
        vmin=-velocity_limit,
        vmax=velocity_limit,
        interpolation="nearest",
        aspect="equal",
    )
    raw = np.asarray(cube[key][0, :, :, z_index])
    bind_artist(
        fig,
        flow_image,
        artist_id=f"fig3draft.a.{key}_midspan_u",
        panel="a",
        source_refs=[
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": key,
                "slice": [0, ":", ":", z_index],
            },
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": "fluid",
                "slice": [":", ":", z_index],
            },
        ],
        source_payload=[raw, fluid_3d[:, :, z_index]],
        expected_payload={
            "type": "AxesImage",
            "array": values,
            "extent": list(extent),
            "clim": [-velocity_limit, velocity_limit],
        },
        transform=(
            "select producer-fixed M2 mid-span streamwise component; transpose x/y; "
            "mask solid cells; use a shared symmetric colour range"
        ),
        evidence="illustrative M2 field; quantitative first-contact evidence remains in panels c-d",
    )
    style_field_axis(ax, show_y=index == 0)
    ax.set_title(title, pad=9.0)
    if index == 0:
        panel_label(ax, "a", x=-0.22, y=1.13)

cax_velocity = fig.add_subplot(spatial[2])
cb_velocity = fig.colorbar(flow_image, cax=cax_velocity)
cb_velocity.set_ticks([-1.5, 0.0, 1.5])
cb_velocity.set_label("$u'/\sigma_u$", labelpad=2)
cb_velocity.ax.tick_params(labelsize=12.0)


# ---------------------------------------------------------- b: local gain maps
gain_specs = [
    (
        "no_wall",
        gain_absent,
        "correct vs\nno band",
        no_wall_vec,
    ),
    (
        "wrong_wall",
        gain_far,
        "correct vs\nwrong-time band",
        wrong_vec,
    ),
]
gain_image = None
for local_index, (control_key, values, title, control_vec) in enumerate(gain_specs):
    grid_index = 3 + local_index
    ax = fig.add_subplot(spatial[grid_index])
    gain_image = ax.imshow(
        values,
        origin="lower",
        extent=extent,
        cmap=gain_cmap,
        vmin=-gain_limit,
        vmax=gain_limit,
        interpolation="nearest",
        aspect="equal",
    )
    bind_artist(
        fig,
        gain_image,
        artist_id=f"fig3draft.b.aligned_gain_over_{control_key}",
        panel="b",
        source_refs=[
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": "truth",
                "slice": [":", ":", ":", z_index],
            },
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": "correct",
                "slice": [":", ":", ":", z_index],
            },
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": control_key,
                "slice": [":", ":", ":", z_index],
            },
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": "fluid",
                "slice": [":", ":", z_index],
            },
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": "band",
                "slice": [":", ":", z_index],
            },
        ],
        source_payload=[
            np.asarray(cube["truth"][:, :, :, z_index]),
            np.asarray(cube["correct"][:, :, :, z_index]),
            np.asarray(cube[control_key][:, :, :, z_index]),
            fluid_3d[:, :, z_index],
            band_3d[:, :, z_index],
        ],
        expected_payload={
            "type": "AxesImage",
            "array": values,
            "extent": list(extent),
            "clim": [-gain_limit, gain_limit],
        },
        transform=(
            "sum squared error over three velocity components for control minus aligned; "
            "select mid-span; transpose x/y; exclude solid and supplied-band cells; "
            "positive values mean aligned conditioning lowers local error"
        ),
        evidence="illustrative M2 local error consequence outside the supplied support",
    )
    style_field_axis(ax, show_y=local_index == 0)
    ax.set_title(title, pad=9.0)
    if local_index == 0:
        panel_label(ax, "b", x=-0.22, y=1.13)

cax_gain = fig.add_subplot(spatial[5])
cb_gain = fig.colorbar(gain_image, cax=cax_gain)
cb_gain.set_label(
    "control error$^2$ $-$ correct-time error$^2$\n(positive: correct-time better)",
    labelpad=2,
)
cb_gain.ax.tick_params(labelsize=12.0)


# ------------------------------------------------ c-d: experiment-specific logic
def schematic_box(
    ax,
    x0: float,
    y0: float,
    width: float,
    height: float,
    text: str,
    *,
    face: str = "#f4f6f7",
    edge: str = "#58656e",
    fontsize: float = 12.0,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.35,
    )
    ax.add_patch(patch)
    ax.text(
        x0 + width / 2,
        y0 + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="black",
        fontweight=weight,
        linespacing=1.10,
    )


def schematic_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=1.35,
            color="#39454c",
            shrinkA=2.0,
            shrinkB=2.0,
        )
    )


ax = fig.add_subplot(logic[0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(
    0.50,
    1.02,
    "Controlled wall-information intervention",
    ha="center",
    va="bottom",
    fontsize=12.0,
    fontweight="normal",
)
# The arm names are clearer than miniature input icons at final print size.
condition_specs = [
    (0.67, "correct-time\nband", "#e0edf8", BLUE),
    (0.40, "no band", "#edf0f2", GREY),
    (0.13, "wrong-time\nband", "#f8e3e6", RED),
]
for y0, label, face, edge in condition_specs:
    schematic_box(ax, 0.025, y0 - 0.025, 0.285, 0.25, label, face=face, edge=edge)

# Miniature of the implemented M2 backbone: a two-level 3-D U-Net with
# encoder widths 48/96, a 192-channel bottleneck, symmetric decoder and skips.
# The drawing follows the original U-Net convention: stacked feature maps,
# contracting/expanding legs, channel labels above, and horizontal skip paths.
ax.text(
    0.530, 0.115, "M2 periodic\n3-D U-Net", ha="center", va="center",
    fontsize=12.0, color="black",
)


def feature_stack(x0: float, y0: float, width: float, height: float, channels: int, face: str) -> dict[str, float]:
    offset = 0.007
    for layer in (2, 1, 0):
        ax.add_patch(
            Rectangle(
                (x0 + layer * offset, y0 + layer * offset), width, height,
                facecolor=face, edgecolor="#34434b", linewidth=1.3, zorder=3 + layer,
            )
        )
    centre_x = x0 + width / 2 + offset
    ax.text(
        centre_x, y0 + height + 0.018, str(channels), ha="center", va="bottom",
        fontsize=12.0, color="black", zorder=7,
    )
    return {
        "left": x0,
        "right": x0 + width + 2 * offset,
        "bottom": y0,
        "top": y0 + height + 2 * offset,
        "cx": centre_x,
        "cy": y0 + height / 2 + offset,
    }


encoder_0 = feature_stack(0.328, 0.500, 0.036, 0.125, 48, "#5a9bd4")
encoder_1 = feature_stack(0.405, 0.405, 0.043, 0.100, 96, "#78aed8")
bottleneck = feature_stack(0.490, 0.325, 0.050, 0.078, 192, "#d39a3e")
decoder_1 = feature_stack(0.580, 0.405, 0.043, 0.100, 96, "#62b89b")
decoder_0 = feature_stack(0.660, 0.500, 0.036, 0.125, 48, "#3b9e7d")
unet_stages = [encoder_0, encoder_1, bottleneck, decoder_1, decoder_0]
for first, second in zip(unet_stages[:-1], unet_stages[1:]):
    schematic_arrow(ax, (first["right"], first["cy"]), (second["left"], second["cy"]))
for source, target, y_skip in [
    (encoder_0, decoder_0, 0.648),
    (encoder_1, decoder_1, 0.536),
]:
    ax.add_patch(
        FancyArrowPatch(
            (source["right"], y_skip), (target["left"], y_skip),
            arrowstyle="-|>", mutation_scale=8.0, linewidth=1.3,
            color="#6d7b83", zorder=2,
        )
    )
for start_y, end_y in [(0.77, 0.590), (0.50, 0.560), (0.23, 0.530)]:
    schematic_arrow(ax, (0.312, start_y), (0.327, end_y))

# Only the eight-draw mean was retained.  The eight small cards therefore use
# distinct real sections of that retained volume as source-faithful texture,
# while the on-canvas label makes clear that the member cards are illustrative.
# The larger field below them is the actual retained mean used in the analysis.
ax.text(
    0.878,
    0.872,
    "8 random-noise\nseeds",
    ha="center",
    va="center",
    fontsize=12.0,
    color="black",
    linespacing=1.0,
)


def textured_volume_cube(
    x0: float,
    y0: float,
    size: float,
    field: np.ma.MaskedArray,
    *,
    edge: str = "#315767",
    line_width: float = 1.0,
    colour_map=None,
    value_min: float | None = None,
    value_max: float | None = None,
) -> tuple[object, tuple[float, float, float, float], tuple[float, float]]:
    """Draw an isometric volume whose front face carries a real field texture."""

    # Panel c is roughly three times wider than it is tall in display units;
    # compensate in axes coordinates so the rendered front face is square.
    face_height = 3.0 * size
    depth_x = 0.24 * size
    depth_y = 0.18 * face_height
    front_points = [
        (x0, y0),
        (x0 + size, y0),
        (x0 + size, y0 + face_height),
        (x0, y0 + face_height),
    ]
    top_points = [
        (x0, y0 + face_height),
        (x0 + depth_x, y0 + face_height + depth_y),
        (x0 + size + depth_x, y0 + face_height + depth_y),
        (x0 + size, y0 + face_height),
    ]
    right_points = [
        (x0 + size, y0),
        (x0 + size + depth_x, y0 + depth_y),
        (x0 + size + depth_x, y0 + face_height + depth_y),
        (x0 + size, y0 + face_height),
    ]
    if colour_map is None:
        colour_map = velocity_cmap
    if value_min is None:
        value_min = -velocity_limit
    if value_max is None:
        value_max = velocity_limit
    valid = np.asarray(field.compressed(), dtype=float)
    mean_value = float(np.mean(valid)) if valid.size else 0.0
    colour_position = np.clip((mean_value - value_min) / (value_max - value_min), 0.0, 1.0)
    side_colour = colour_map(colour_position)
    top_colour = tuple(0.70 * np.asarray(side_colour[:3]) + 0.30) + (1.0,)
    right_colour = tuple(0.82 * np.asarray(side_colour[:3]) + 0.18) + (1.0,)
    ax.add_patch(Polygon(top_points, closed=True, facecolor=top_colour, edgecolor=edge, linewidth=line_width, zorder=3))
    ax.add_patch(Polygon(right_points, closed=True, facecolor=right_colour, edgecolor=edge, linewidth=line_width, zorder=3))
    front_patch = Polygon(
        front_points,
        closed=True,
        facecolor="#e9edf0",
        edgecolor=edge,
        linewidth=line_width,
        zorder=4,
    )
    ax.add_patch(front_patch)
    image = ax.imshow(
        field,
        origin="lower",
        extent=(x0, x0 + size, y0, y0 + face_height),
        cmap=colour_map,
        vmin=value_min,
        vmax=value_max,
        interpolation="nearest",
        aspect="auto",
        zorder=5,
    )
    image.set_clip_path(front_patch)
    ax.add_patch(
        Polygon(
            front_points,
            closed=True,
            facecolor="none",
            edgecolor=edge,
            linewidth=line_width,
            zorder=6,
        )
    )
    return (
        image,
        (x0, x0 + size, y0, y0 + face_height),
        (x0 + size + depth_x, y0 + face_height / 2 + depth_y),
    )

illustrative_z_indices = np.linspace(3, cube["correct"].shape[-1] - 4, 8).round().astype(int)
for member, member_z in enumerate(illustrative_z_indices):
    member_col = member % 4
    member_row = member // 4
    member_x0 = 0.752 + 0.050 * member_col
    member_y0 = 0.635 - 0.150 * member_row
    member_field = np.ma.masked_where(
        ~fluid_3d[:, :, member_z].T,
        np.asarray(cube["correct"][0, :, :, member_z], dtype=float).T,
    )
    member_image, member_extent, _ = textured_volume_cube(
        member_x0,
        member_y0,
        0.034,
        member_field,
        line_width=0.9,
    )
    bind_artist(
        fig,
        member_image,
        artist_id=f"fig3draft.c.illustrative_draw_texture_{member + 1}",
        panel="c",
        source_refs=[
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": "correct",
                "slice": [0, ":", ":", int(member_z)],
            },
            {
                "kind": "npz",
                "path": "fig2/cube3d_representative_fields.npz",
                "key": "fluid",
                "slice": [":", ":", int(member_z)],
            },
        ],
        source_payload=[
            np.asarray(cube["correct"][0, :, :, member_z]),
            fluid_3d[:, :, member_z],
        ],
        expected_payload={
            "type": "AxesImage",
            "array": member_field,
            "extent": list(member_extent),
            "clim": [-velocity_limit, velocity_limit],
        },
        transform=(
            "select one streamwise z-section of the retained correct-time M2 mean; "
            "transpose x/y and mask solid cells; use only as texture for the "
            "explicitly illustrative stochastic-draw card"
        ),
        evidence=(
            "source-faithful illustrative texture from the retained mean; not a "
            "claim that an individual stochastic member was archived"
        ),
    )

ax.text(
    0.8525,
    0.052,
    "mean field",
    ha="center",
    va="center",
    fontsize=12.0,
    color="black",
)
mean_thumb, mean_extent, mean_right = textured_volume_cube(
    0.814,
    0.125,
    0.075,
    correct_u,
    line_width=1.3,
)
bind_artist(
    fig,
    mean_thumb,
    artist_id="fig3draft.c.correct_mean_midspan",
    panel="c",
    source_refs=[
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "correct", "slice": [0, ":", ":", z_index]},
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "fluid", "slice": [":", ":", z_index]},
    ],
    source_payload=[
        np.asarray(cube["correct"][0, :, :, z_index]),
        fluid_3d[:, :, z_index],
    ],
    expected_payload={
        "type": "AxesImage",
        "array": correct_u,
        "extent": list(mean_extent),
        "clim": [-velocity_limit, velocity_limit],
    },
    transform="select retained correct-time M2 eight-sample mean mid-span streamwise component; transpose x/y; mask solid cells",
    evidence="actual retained ensemble-mean output; individual member fields were not retained",
)
# Route the ensemble-reduction arrow out of the right side of the eight-volume
# group, down, then left into the right face of the retained mean volume.
ensemble_arrow_path = MplPath(
    [
        (0.944, 0.630),
        (0.958, 0.630),
        (0.958, mean_right[1]),
        (mean_right[0], mean_right[1]),
    ],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
)
ax.add_patch(
    FancyArrowPatch(
        path=ensemble_arrow_path,
        arrowstyle="-|>",
        mutation_scale=9.0,
        linewidth=1.35,
        color="#39454c",
        zorder=7,
    )
)
schematic_arrow(ax, (decoder_0["right"], decoder_0["cy"]), (0.744, 0.610))
panel_label(ax, "c", x=0.00, y=1.10)


ax = fig.add_subplot(logic[1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(
    0.50,
    1.02,
    "Support-excluded evaluation",
    ha="center",
    va="bottom",
    fontsize=12.0,
    fontweight="normal",
)

# Real generated and LES volumes replace the former prose input box.
ax.text(0.042, 0.785, "generated\nmean", ha="center", va="center", fontsize=12.0, color="black", linespacing=1.0)
generated_cube, generated_extent, _ = textured_volume_cube(
    0.028, 0.500, 0.052, correct_u, line_width=1.15,
)
bind_artist(
    fig,
    generated_cube,
    artist_id="fig3draft.d.generated_mean_cube",
    panel="d",
    source_refs=[
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "correct", "slice": [0, ":", ":", z_index]},
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "fluid", "slice": [":", ":", z_index]},
    ],
    source_payload=[np.asarray(cube["correct"][0, :, :, z_index]), fluid_3d[:, :, z_index]],
    expected_payload={
        "type": "AxesImage",
        "array": correct_u,
        "extent": list(generated_extent),
        "clim": [-velocity_limit, velocity_limit],
    },
    transform="show the retained correct-time M2 mean as an isometric generated-field cube",
    evidence="actual retained generated mean used by the support-excluded evaluation",
)

ax.text(0.222, 0.785, "LES\ntarget", ha="center", va="center", fontsize=12.0, color="black", linespacing=1.0)
les_cube, les_extent, _ = textured_volume_cube(
    0.185, 0.500, 0.052, truth_u, line_width=1.15,
)
bind_artist(
    fig,
    les_cube,
    artist_id="fig3draft.d.les_target_cube",
    panel="d",
    source_refs=[
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "truth", "slice": [0, ":", ":", z_index]},
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "fluid", "slice": [":", ":", z_index]},
    ],
    source_payload=[np.asarray(cube["truth"][0, :, :, z_index]), fluid_3d[:, :, z_index]],
    expected_payload={
        "type": "AxesImage",
        "array": truth_u,
        "extent": list(les_extent),
        "clim": [-velocity_limit, velocity_limit],
    },
    transform="show the matched LES target as an isometric field cube",
    evidence="actual retained LES target used by the support-excluded evaluation",
)
ax.add_patch(
    FancyArrowPatch(
        (0.102, 0.600), (0.176, 0.600), arrowstyle="<->",
        mutation_scale=9.0, linewidth=1.25, color="#39454c",
    )
)

# The middle cube is the real three-component error outside the supplied band;
# the orange cross makes the exclusion operation immediately visible.
error_mid_raw = np.sqrt(np.sum((correct_vec - truth_vec) ** 2, axis=0)).T
support_excluded_error = np.ma.masked_where(~scored_mid, error_mid_raw)
error_limit = float(np.percentile(support_excluded_error.compressed(), 99.0))
error_cmap = plt.get_cmap("magma").copy()
error_cmap.set_bad("#e9edf0")
ax.text(0.470, 0.785, "exclude support", ha="center", va="center", fontsize=12.0, color="black")
error_cube, error_extent, _ = textured_volume_cube(
    0.420,
    0.475,
    0.065,
    support_excluded_error,
    edge="#7d5a35",
    line_width=1.2,
    colour_map=error_cmap,
    value_min=0.0,
    value_max=error_limit,
)
bind_artist(
    fig,
    error_cube,
    artist_id="fig3draft.d.support_excluded_error_cube",
    panel="d",
    source_refs=[
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "correct", "slice": [":", ":", ":", z_index]},
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "truth", "slice": [":", ":", ":", z_index]},
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "fluid", "slice": [":", ":", z_index]},
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "band", "slice": [":", ":", z_index]},
    ],
    source_payload=[
        np.asarray(cube["correct"][:, :, :, z_index]),
        np.asarray(cube["truth"][:, :, :, z_index]),
        fluid_3d[:, :, z_index],
        band_3d[:, :, z_index],
    ],
    expected_payload={
        "type": "AxesImage",
        "array": support_excluded_error,
        "extent": list(error_extent),
        "clim": [0.0, error_limit],
    },
    transform=(
        "take the Euclidean three-component generated-minus-LES error at mid-span; "
        "exclude solid and supplied-band cells; render on an isometric cube"
    ),
    evidence="actual local reconstruction error outside the orange supplied support",
)
ax.text(0.453, 0.580, "×", ha="center", va="center", fontsize=12.0, color="#e88900", fontweight="bold", zorder=8)
schematic_arrow(ax, (0.258, 0.600), (0.392, 0.600))


def region_glyph(x0: float, label: str, kind: str, colour: str) -> None:
    """Compact spatial-mask glyph for complete, near and farther views."""

    y0, width, height = 0.100, 0.105, 0.185
    ax.add_patch(Rectangle((x0, y0), width, height, facecolor="#f4f6f7", edgecolor="black", linewidth=1.15))
    if kind == "all":
        ax.add_patch(Rectangle((x0 + 0.006, y0 + 0.006), width - 0.012, height - 0.012, facecolor=colour, alpha=0.70, edgecolor="none"))
    elif kind == "near":
        ax.add_patch(FancyBboxPatch((x0 + 0.016, y0 + 0.020), width - 0.032, height * 0.62, boxstyle="round,pad=0.002", facecolor=colour, alpha=0.78, edgecolor="none"))
    else:
        ax.add_patch(Rectangle((x0 + 0.006, y0 + 0.006), width - 0.012, height - 0.012, facecolor=colour, alpha=0.66, edgecolor="none"))
        ax.add_patch(FancyBboxPatch((x0 + 0.018, y0 + 0.018), width - 0.036, height * 0.60, boxstyle="round,pad=0.002", facecolor="#f4f6f7", edgecolor="none"))
    obstacle_width = 0.040
    ax.add_patch(Rectangle((x0 + (width - obstacle_width) / 2, y0), obstacle_width, height * 0.36, facecolor="#cbd2d6", edgecolor="black", linewidth=1.0, zorder=4))
    ax.add_patch(Rectangle((x0 + (width - obstacle_width) / 2 - 0.005, y0 + height * 0.36), obstacle_width + 0.010, 0.010, facecolor="#e88900", edgecolor="none", zorder=5))
    label_y = -0.035 if "\n" in label else -0.005
    ax.text(x0 + width / 2, label_y, label, ha="center", va="center", fontsize=12.0, color="black")


region_glyph(0.273, "all", "all", "#90c9aa")
region_glyph(0.403, "near\nwall", "near", "#5fb69a")
region_glyph(0.533, "farther", "far", "#9bc8dc")
schematic_arrow(ax, (0.455, 0.455), (0.455, 0.285))

# Make the two summaries self-explanatory at a glance.  The upper pictogram is
# the absolute score of one generated arm against its LES target.  The lower
# pictogram is the difference between two scores evaluated on the same target
# and with the same sampler seeds: correct-time minus its matched control.
ax.text(
    0.862, 0.895, "one field vs LES", ha="center", va="center",
    fontsize=12.0, color="black",
)
absolute_generated, absolute_generated_extent, _ = textured_volume_cube(
    0.768, 0.565, 0.033, correct_u, line_width=1.0,
)
bind_artist(
    fig,
    absolute_generated,
    artist_id="fig3draft.d.absolute_generated_icon",
    panel="d",
    source_refs=[
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "correct", "slice": [0, ":", ":", z_index]},
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "fluid", "slice": [":", ":", z_index]},
    ],
    source_payload=[np.asarray(cube["correct"][0, :, :, z_index]), fluid_3d[:, :, z_index]],
    expected_payload={
        "type": "AxesImage",
        "array": correct_u,
        "extent": list(absolute_generated_extent),
        "clim": [-velocity_limit, velocity_limit],
    },
    transform="repeat the retained generated mean as the absolute-score comparison icon",
    evidence="actual generated field entering the field-to-target R-squared score",
)

absolute_les, absolute_les_extent, _ = textured_volume_cube(
    0.831, 0.565, 0.033, truth_u, line_width=1.0,
)
bind_artist(
    fig,
    absolute_les,
    artist_id="fig3draft.d.absolute_les_icon",
    panel="d",
    source_refs=[
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "truth", "slice": [0, ":", ":", z_index]},
        {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "fluid", "slice": [":", ":", z_index]},
    ],
    source_payload=[np.asarray(cube["truth"][0, :, :, z_index]), fluid_3d[:, :, z_index]],
    expected_payload={
        "type": "AxesImage",
        "array": truth_u,
        "extent": list(absolute_les_extent),
        "clim": [-velocity_limit, velocity_limit],
    },
    transform="repeat the retained LES field as the absolute-score comparison icon",
    evidence="actual LES target entering the field-to-target R-squared score",
)
ax.add_patch(
    FancyArrowPatch(
        (0.805, 0.620), (0.827, 0.620), arrowstyle="<->",
        mutation_scale=7.5, linewidth=1.15, color="#39454c",
    )
)
schematic_arrow(ax, (0.873, 0.620), (0.890, 0.620))
ax.add_patch(Circle((0.932, 0.620), 0.037, facecolor="#dcebf3", edgecolor="#397b9c", linewidth=1.45))
ax.text(0.932, 0.620, "$R^2$", ha="center", va="center", fontsize=12.0, color="black", fontweight="bold")

ax.text(
    0.815, 0.485, "same target + seeds", ha="center", va="center",
    fontsize=12.0, color="black",
)
paired_nodes = [
    (0.700, r"$R^2_{C}$", "#dcebf3", "#397b9c"),
    (0.812, r"$R^2_{0}$", "#eceff1", "#6f7f87"),
    (0.930, r"$\Delta R^2$", "#dff0e8", "#26745e"),
]
for centre_x, symbol, face, edge in paired_nodes:
    radius = 0.032 if centre_x < 0.95 else 0.035
    ax.add_patch(Circle((centre_x, 0.330), radius, facecolor=face, edgecolor=edge, linewidth=1.4))
    ax.text(centre_x, 0.330, symbol, ha="center", va="center", fontsize=12.0, color="black", fontweight="bold")
ax.text(0.756, 0.330, "$-$", ha="center", va="center", fontsize=12.0, color="black")
ax.text(0.862, 0.330, "$=$", ha="center", va="center", fontsize=12.0, color="black")
ax.text(
    0.815, 0.108, "$C$: correct\n$0$: control", ha="center", va="center",
    fontsize=12.0, color="black",
    linespacing=1.0,
)

# The three region masks feed both the one-arm score and the matched score
# difference.  A shared trunk keeps this relation visible without another box.
ax.plot([0.628, 0.646, 0.646], [0.165, 0.165, 0.620], color="#39454c", linewidth=1.25)
schematic_arrow(ax, (0.646, 0.620), (0.763, 0.620))
schematic_arrow(ax, (0.646, 0.330), (0.660, 0.330))
panel_label(ax, "d", x=0.00, y=1.10)


# ------------------------------------------------------ e: M0 absolute skill
regions = [
    ("full_support_excluded", "all\ncells"),
    ("near_support_excluded_d_le_0p5h", "near wall\n$d/h\leq0.5$"),
    ("outer_d_gt_0p5h", "farther\n$d/h>0.5$"),
]
arms = [
    ("correct", "correct time", BLUE, "-"),
    ("no_wall", "no band", GREY, "--"),
    ("wrong_wall", "wrong time", RED, ":"),
]

ax = fig.add_subplot(quant[0])
positions = np.arange(len(regions))
offsets = (-0.18, 0.0, 0.18)
for offset, (key, label, color, line_style) in zip(offsets, arms):
    values = [m0["arms"][key][region] for region, _ in regions]
    artist = ax.plot(
        positions + offset,
        values,
        marker="o",
        markersize=7.2,
        color=color,
        markeredgecolor="black",
        markeredgewidth=1.0,
        linewidth=1.35,
        linestyle=line_style,
        label=label,
        zorder=3,
    )[0]
    bind_artist(
        fig,
        artist,
        artist_id=f"fig3draft.e.{key}_absolute_skill",
        panel="e",
        source_refs=[
            {
                "kind": "json",
                "path": "review_audit/derived_peer_review_statistics.json",
                "key": f"first_contact_m0.arms.{key}.{region}",
            }
            for region, _ in regions
        ],
        source_payload=values,
        expected_payload={
            "type": "Line2D",
            "x": positions + offset,
            "y": np.asarray(values),
        },
        transform="connect complete/near/farther M0 point estimates at fixed horizontal offsets",
        evidence="E1 first-contact terminal aggregates; no replayable M0 interval",
    )
ax.axhline(0, color="black", lw=1.35)
ax.set_xticks(positions, [label for _, label in regions])
ax.set_ylabel("realised 8-sample mean\nfluctuation $R^2$")
ax.set_title("First-contact M0 reconstruction skill")
ax.set_ylim(-0.58, 0.36)
ax.set_yticks([-0.4, -0.2, 0.0, 0.2])
handles, labels = ax.get_legend_handles_labels()
order = [0, 2, 1]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          fontsize=12.0, ncol=3, loc="lower center",
          bbox_to_anchor=(0.46, 0.005), handlelength=1.25,
          borderpad=0.25, columnspacing=0.9, handletextpad=0.5)
panel_label(ax, "e", x=-0.19, y=1.17)


# ---------------------------------------------------- f: M0 paired differences
ax = fig.add_subplot(quant[1])
controls = [
    ("correct_minus_no_wall", "over no band", GREY, -0.10, "--"),
    ("correct_minus_wrong_wall", "over wrong-time band", RED, 0.10, "-"),
]
for key, label, color, offset, line_style in controls:
    values = [m0["differences"][key][region] for region, _ in regions]
    artist = ax.plot(
        positions + offset,
        values,
        marker="o",
        markersize=7.2,
        color=color,
        markeredgecolor="black",
        markeredgewidth=1.0,
        linewidth=1.35,
        linestyle=line_style,
        label=label,
        zorder=3,
    )[0]
    bind_artist(
        fig,
        artist,
        artist_id=f"fig3draft.f.{key}",
        panel="f",
        source_refs=[
            {
                "kind": "json",
                "path": "review_audit/derived_peer_review_statistics.json",
                "key": f"first_contact_m0.differences.{key}.{region}",
            }
            for region, _ in regions
        ],
        source_payload=values,
        expected_payload={
            "type": "Line2D",
            "x": positions + offset,
            "y": np.asarray(values),
        },
        transform="connect paired M0 differences for complete/near/farther regions",
        evidence=(
            "E1 first-contact terminal aggregates; positive differences do not imply "
            "positive absolute skill"
        ),
    )
ax.axhline(0, color="black", lw=1.35)
ax.set_xticks(positions, [label for _, label in regions])
ax.set_ylabel("gain in fluctuation $R^2$")
ax.set_title("First-contact M0 band benefit")
ax.set_ylim(-0.03, 0.62)
ax.set_yticks([0.0, 0.2, 0.4, 0.6])
ax.legend(fontsize=12.0, loc="lower center", bbox_to_anchor=(0.42, 0.03),
          handlelength=1.65, borderpad=0.25, labelspacing=0.35)
panel_label(ax, "f", x=-0.19, y=1.17)

# Matplotlib creates a few 0.8-pt internal contour/colorbar collections even
# when the visible data contours are thicker.  Lift those technical strokes to
# the same publication floor as the plotted lines.
for collection in fig.findobj(match=LineCollection):
    widths = np.asarray(collection.get_linewidths(), dtype=float)
    if widths.size:
        collection.set_linewidths(np.maximum(widths, 1.30))

fig.set_constrained_layout_pads(h_pad=0.10, w_pad=0.05, hspace=0.05, wspace=0.05)

save(fig, os.environ.get("GWT_FIG3_STEM", "fig3_propagation_spatial_draft"))
