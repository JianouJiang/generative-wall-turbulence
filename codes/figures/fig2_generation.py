#!/usr/bin/env python3
"""Figure 2: representative genuine-3-D aligned-cube reconstruction.

The four audited velocity sections and their four aligned error maps are retained.
Two source-derived 3-D cutaways make the target-versus-donor temporal control
explicit, and two target-level profile diagnostics quantify how the
reconstructions differ.
"""

import numpy as np
from matplotlib.patches import Polygon

from _submission import (
    GREY,
    SOURCE,
    bind_artist,
    configure,
    load_json,
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
data = np.load(SOURCE / "fig2/cube3d_representative_fields.npz")
time_pair = np.load(SOURCE / "fig2/cube3d_time_pair.npz")
time_sequence = np.load(SOURCE / "fig2/cube3d_time_sequence.npz")
timeline = load_json("fig2/cube3d_timeline.json")
z_index = data["truth"].shape[-1] // 2
fluid_3d = np.asarray(data["fluid"], dtype=bool)
band_3d = np.asarray(data["band"], dtype=bool)
scored_3d = fluid_3d & (~band_3d)
fluid = fluid_3d[:, :, z_index].T
band = band_3d[:, :, z_index].T
x = np.asarray(data["x"], dtype=float)
y = np.asarray(data["y"], dtype=float)
z = np.asarray(data["z"], dtype=float)

# The conditioning support contains cube-, floor- and top-wall strips.  The
# spatial maps outline the complete supplied support (Figure 3 convention), so
# Figures 1-3 all show the same envelope; metrics exclude that same support.


def slice_field(key: str, plane: int = z_index) -> np.ndarray:
    field = data[key][0, :, :, plane].astype(float).T
    field[~fluid_3d[:, :, plane].T] = np.nan
    return field


truth = slice_field("truth")
fields = {
    "LES reference": truth,
    "aligned\noracle-band\n8-sample mean": slice_field("correct"),
    "absent-band\ndropout\n8-sample mean": slice_field("no_wall"),
    "equal-support\nfar-time\n8-sample mean": slice_field("wrong_wall"),
}
field_source_keys = {
    "LES reference": "truth",
    "aligned\noracle-band\n8-sample mean": "correct",
    "absent-band\ndropout\n8-sample mean": "no_wall",
    "equal-support\nfar-time\n8-sample mean": "wrong_wall",
}
time_pair_values = np.concatenate(
    [np.asarray(time_pair[key][0], dtype=float)[fluid_3d] for key in ("target", "donor")]
)
valid_values = np.concatenate(
    [value[np.isfinite(value)] for value in fields.values()] + [time_pair_values]
)
limit = float(np.max(np.abs(valid_values)))
error_fields = {name: np.abs(field - truth) for name, field in fields.items()}
error_limit = float(max(np.nanmax(error) for error in error_fields.values()))

dx = float(np.median(np.diff(x)))
dy = float(np.median(np.diff(y)))
extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)

velocity_cmap = plt.get_cmap("RdBu_r").copy()
velocity_cmap.set_bad("#e9edf0")
error_cmap = plt.get_cmap("magma").copy()
error_cmap.set_bad("#e9edf0")

ARM_STYLE = {
    "truth": ("LES reference", "#20252a", "-"),
    "correct": ("aligned wall", "#0072b2", "-"),
    "no_wall": ("absent wall", "#6f777d", "--"),
    "wrong_wall": ("far-time wall", "#d55e00", ":"),
}


def panel_label(ax, label: str, *, x_pos: float = -0.12, y_pos: float = 1.06) -> None:
    if hasattr(ax, "text2D"):
        ax.text2D(
            x_pos, y_pos, label, transform=ax.transAxes, fontsize=12.0,
            fontweight="bold", va="top",
        )
    else:
        ax.text(
            x_pos, y_pos, label, transform=ax.transAxes, fontsize=12.0,
            fontweight="bold", va="top",
        )


fig = plt.figure(figsize=(8.75, 8.90), constrained_layout=True)
outer = fig.add_gridspec(
    3,
    1,
    height_ratios=(1.90, 2.90, 2.90),
    hspace=0.10,
)
# Maps sit flush left in narrow columns; the freed right-hand third holds
# the profile panels, and each map row carries its colourbar underneath.
field_grid = outer[1].subgridspec(
    2, 5, width_ratios=(1, 1, 1, 1, 1.60), height_ratios=(1.0, 0.055),
    wspace=0.05, hspace=0.02,
)
error_grid = outer[2].subgridspec(
    2, 5, width_ratios=(1, 1, 1, 1, 1.60), height_ratios=(1.0, 0.055),
    wspace=0.05, hspace=0.02,
)


# ------------------------------------------------------------------------- a
# Two real 3-D LES volumes make the temporal mismatch visible.  Each cutaway
# contains three exact x-y sections, rather than several nearly coincident planes
# from one volume.  The projection is calibrated to the rendered panel aspect so
# the physical h x h x h obstacle reads as a cube.
ax_volume = fig.add_subplot(outer[0])
ax_volume.set_xlim(0, 1)
ax_volume.set_ylim(0, 1)
ax_volume.set_aspect("auto")
ax_volume.axis("off")

if not np.array_equal(np.asarray(time_pair["fluid"]), fluid_3d):
    raise RuntimeError("Figure 2 time-pair mask differs from the representative target mask")
if not np.array_equal(np.asarray(time_sequence["fluid"]), fluid_3d):
    raise RuntimeError("Figure 2 time-sequence mask differs from the representative target mask")

card_width, card_height = 0.090, 0.650
depth_x, depth_y = 0.090, 0.150
base_y = 0.035
# All three planes cut through the obstacle.  This is essential for showing the
# supplied near-wall band on the section where it actually lives.
section_planes = (13, z_index, 34)


def project(
    base_x: float, x_value: float, y_value: float, z_value: float
) -> tuple[float, float]:
    """Oblique projection of the 2h x 4h x 2h periodic volume."""

    return (
        base_x + card_width * x_value / 2.0 + depth_x * z_value / 2.0,
        base_y + card_height * y_value / 4.0 + depth_y * z_value / 2.0,
    )


def temporal_section(volume_key: str, plane: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(time_pair[volume_key][0, :, :, plane])
    section = raw.astype(float).T
    section[~fluid_3d[:, :, plane].T] = np.nan
    return section, raw


def draw_time_volume(volume_key: str, base_x: float, colour: str) -> None:
    card_corners: dict[int, tuple[float, float, float, float]] = {}
    # Back first, then the highlighted middle section, then the translucent front.
    for plane, alpha in (
        (section_planes[2], 0.74),
        (section_planes[1], 0.98),
        (section_planes[0], 0.74),
    ):
        x_origin, y_origin = project(base_x, 0.0, 0.0, float(z[plane]))
        card_extent = (
            x_origin, x_origin + card_width,
            y_origin, y_origin + card_height,
        )
        section, raw_section = temporal_section(volume_key, plane)
        section_artist = ax_volume.imshow(
            section, origin="lower", aspect="auto", cmap=velocity_cmap,
            vmin=-limit, vmax=limit, extent=card_extent, interpolation="nearest",
            alpha=alpha, zorder=4 if plane == z_index else 2,
        )
        expected_section = np.ma.masked_invalid(section)
        if np.all(np.isfinite(section)):
            expected_section = np.ma.array(section)
        bind_artist(
            fig,
            section_artist,
            artist_id=f"fig2.a.{volume_key}_section_z{plane}",
            panel="a",
            source_refs=[
                {"kind": "npz", "path": "fig2/cube3d_time_pair.npz",
                 "key": volume_key, "slice": [0, ":", ":", plane]},
                {"kind": "npz", "path": "fig2/cube3d_time_pair.npz",
                 "key": "fluid", "slice": [":", ":", plane]},
            ],
            source_payload=[raw_section, np.asarray(time_pair["fluid"][:, :, plane])],
            expected_payload={
                "type": "AxesImage", "array": expected_section,
                "extent": [float(value) for value in card_extent],
                "clim": [-limit, limit],
            },
            transform=(
                "streamwise component on an exact retained spanwise section; transpose x/y; "
                "mask solid cells; place in its source-derived temporal volume"
            ),
            evidence=f"real {volume_key} LES section from the frozen temporal pair",
        )
        edge_colour = "#263238" if plane == z_index else "#65737c"
        ax_volume.plot(
            [x_origin, x_origin + card_width, x_origin + card_width, x_origin, x_origin],
            [y_origin, y_origin, y_origin + card_height, y_origin + card_height, y_origin],
            color=edge_colour,
            linewidth=1.75 if plane == z_index else 1.25,
            zorder=5,
        )
        card_corners[plane] = card_extent

    front = card_corners[section_planes[0]]
    rear = card_corners[section_planes[2]]
    for front_point, rear_point in (
        ((front[0], front[2]), (rear[0], rear[2])),
        ((front[1], front[2]), (rear[1], rear[2])),
        ((front[0], front[3]), (rear[0], rear[3])),
        ((front[1], front[3]), (rear[1], rear[3])),
    ):
        ax_volume.plot(
            [front_point[0], rear_point[0]], [front_point[1], rear_point[1]],
            color="#77848c", linewidth=1.25, alpha=0.75, zorder=1,
        )

    cube_points = {
        name: project(base_x, xx, yy, zz)
        for name, (xx, yy, zz) in {
            "fbl": (0.5, 0.0, 0.5), "fbr": (1.5, 0.0, 0.5),
            "ftl": (0.5, 1.0, 0.5), "ftr": (1.5, 1.0, 0.5),
            "bbl": (0.5, 0.0, 1.5), "bbr": (1.5, 0.0, 1.5),
            "btl": (0.5, 1.0, 1.5), "btr": (1.5, 1.0, 1.5),
        }.items()
    }

    # Put the band where the data say it is: on every displayed x--y section.
    # Each filled U is the exact cube-adjacent thickness used by the producer,
    # evaluated at that plane.  It is not a decorative shell on the projection.
    band_thickness = 2.01 * (4.0 / len(y))
    x_low, x_high = 0.5 - band_thickness, 1.5 + band_thickness
    y_high = 1.0 + band_thickness
    for plane, opacity in zip(section_planes, (0.78, 0.94, 0.78)):
        plane_z = float(z[plane])
        # In this viewing direction the cube hides the left vertical leg.  The
        # top and right portions lie on the two exposed faces, so retain those
        # for every slice (including the highlighted middle section).
        band_layer = 18
        outline_layer = 19
        band_rectangles = (
            (((0.5, 1.0), (1.5, 1.0), (1.5, y_high), (0.5, y_high)), band_layer),
            (((1.5, 0.0), (x_high, 0.0), (x_high, y_high), (1.5, y_high)), band_layer),
            # floor strips flanking the obstacle sit behind the cube patches so
            # the obstacle occludes them exactly as it occludes the sections
            (((0.0, 0.0), (0.5, 0.0), (0.5, band_thickness), (0.0, band_thickness)), 14),
            (((x_high, 0.0), (2.0, 0.0), (2.0, band_thickness), (x_high, band_thickness)), 14),
            # computational-lid strip at the top of the domain
            (((0.0, 4.0 - band_thickness), (2.0, 4.0 - band_thickness), (2.0, 4.0), (0.0, 4.0)), band_layer),
        )
        for rectangle, layer in band_rectangles:
            projected = [project(base_x, xx, yy, plane_z) for xx, yy in rectangle]
            ax_volume.add_patch(
                Polygon(
                    projected, closed=True, facecolor="#f6a623",
                    edgecolor="none", alpha=opacity, zorder=layer,
                )
            )
        visible_path = [
            project(base_x, 0.5, y_high, plane_z),
            project(base_x, x_high, y_high, plane_z),
            project(base_x, x_high, 0.0, plane_z),
        ]
        ax_volume.plot(
            [point[0] for point in visible_path], [point[1] for point in visible_path],
            color="#e88900", linewidth=1.45, alpha=0.95, zorder=outline_layer,
        )

    # The cube remains opaque; the deliberately retained slice intersections
    # above are the visible top/right portions of the band on its exposed faces.
    for vertices, face_colour, layer in (
        (("fbr", "bbr", "btr", "ftr"), "#98a8b0", 15),
        (("ftl", "ftr", "btr", "btl"), "#eef1f2", 16),
        (("fbl", "fbr", "ftr", "ftl"), "#c7d0d4", 17),
    ):
        ax_volume.add_patch(
            Polygon(
                [cube_points[name] for name in vertices], closed=True,
                facecolor=face_colour, edgecolor="#263238", linewidth=1.5,
                alpha=0.98, zorder=layer,
            )
        )

    face_centre = np.mean(
        np.asarray([cube_points[name] for name in ("fbl", "fbr", "ftr", "ftl")]),
        axis=0,
    )
    ax_volume.text(
        face_centre[0], face_centre[1], "cube", fontsize=12.0,
        ha="center", va="center", color="#263238", zorder=21,
    )
    height_x = cube_points["fbl"][0] - 0.012
    ax_volume.annotate(
        "", xy=(height_x, cube_points["ftl"][1]),
        xytext=(height_x, cube_points["fbl"][1]),
        arrowprops={"arrowstyle": "|-|", "color": "#263238", "linewidth": 1.25},
        zorder=21,
    )
    ax_volume.text(
        height_x - 0.005,
        0.5 * (cube_points["fbl"][1] + cube_points["ftl"][1]),
        "$h$", fontsize=12.0, ha="right", va="center", zorder=21,
    )


def draw_mini_time_volume(sequence_index: int, base_x: float) -> None:
    """Draw one source-derived intermediate volume in the central timeline."""

    scale = 0.31
    mini_base_y = 0.365

    def mini_project(x_value: float, y_value: float, z_value: float) -> tuple[float, float]:
        return (
            base_x + scale * (card_width * x_value / 2.0 + depth_x * z_value / 2.0),
            mini_base_y
            + scale * (card_height * y_value / 4.0 + depth_y * z_value / 2.0),
        )

    card_corners: dict[int, tuple[float, float, float, float]] = {}
    for plane, alpha in (
        (section_planes[2], 0.72),
        (section_planes[1], 0.96),
        (section_planes[0], 0.72),
    ):
        x_origin, y_origin = mini_project(0.0, 0.0, float(z[plane]))
        card_extent = (
            x_origin,
            x_origin + scale * card_width,
            y_origin,
            y_origin + scale * card_height,
        )
        raw_section = np.asarray(
            time_sequence["intermediate_u"][sequence_index, :, :, plane]
        )
        section = raw_section.astype(float).T
        section[~fluid_3d[:, :, plane].T] = np.nan
        section_artist = ax_volume.imshow(
            section, origin="lower", aspect="auto", cmap=velocity_cmap,
            vmin=-limit, vmax=limit, extent=card_extent, interpolation="nearest",
            alpha=alpha, zorder=4 if plane == z_index else 2,
        )
        expected_section = np.ma.masked_invalid(section)
        if np.all(np.isfinite(section)):
            expected_section = np.ma.array(section)
        bind_artist(
            fig,
            section_artist,
            artist_id=f"fig2.a.intermediate_{sequence_index}_section_z{plane}",
            panel="a",
            source_refs=[
                {"kind": "npz", "path": "fig2/cube3d_time_sequence.npz",
                 "key": "intermediate_u", "slice": [sequence_index, ":", ":", plane]},
                {"kind": "npz", "path": "fig2/cube3d_time_sequence.npz",
                 "key": "fluid", "slice": [":", ":", plane]},
            ],
            source_payload=[raw_section, np.asarray(time_sequence["fluid"][:, :, plane])],
            expected_payload={
                "type": "AxesImage", "array": expected_section,
                "extent": [float(value) for value in card_extent],
                "clim": [-limit, limit],
            },
            transform=(
                "streamwise component on an exact retained intermediate section; "
                "transpose x/y; mask solid cells; scale into temporal sequence"
            ),
            evidence="real intermediate LES section between the displayed endpoints",
        )
        edge_colour = "#263238" if plane == z_index else "#65737c"
        ax_volume.plot(
            [x_origin, x_origin + scale * card_width, x_origin + scale * card_width,
             x_origin, x_origin],
            [y_origin, y_origin, y_origin + scale * card_height,
             y_origin + scale * card_height, y_origin],
            color=edge_colour, linewidth=1.25,
            zorder=5,
        )
        card_corners[plane] = card_extent

    front = card_corners[section_planes[0]]
    rear = card_corners[section_planes[2]]
    for front_point, rear_point in (
        ((front[0], front[2]), (rear[0], rear[2])),
        ((front[1], front[2]), (rear[1], rear[2])),
        ((front[0], front[3]), (rear[0], rear[3])),
        ((front[1], front[3]), (rear[1], rear[3])),
    ):
        ax_volume.plot(
            [front_point[0], rear_point[0]], [front_point[1], rear_point[1]],
            color="#77848c", linewidth=1.25, alpha=0.78, zorder=1,
        )

    cube_points = {
        name: mini_project(xx, yy, zz)
        for name, (xx, yy, zz) in {
            "fbl": (0.5, 0.0, 0.5), "fbr": (1.5, 0.0, 0.5),
            "ftl": (0.5, 1.0, 0.5), "ftr": (1.5, 1.0, 0.5),
            "bbl": (0.5, 0.0, 1.5), "bbr": (1.5, 0.0, 1.5),
            "btl": (0.5, 1.0, 1.5), "btr": (1.5, 1.0, 1.5),
        }.items()
    }
    for vertices, face_colour, layer in (
        (("fbr", "bbr", "btr", "ftr"), "#98a8b0", 15),
        (("ftl", "ftr", "btr", "btl"), "#eef1f2", 16),
        (("fbl", "fbr", "ftr", "ftl"), "#c7d0d4", 17),
    ):
        ax_volume.add_patch(
            Polygon(
                [cube_points[name] for name in vertices], closed=True,
                facecolor=face_colour, edgecolor="#263238", linewidth=1.25,
                alpha=0.98, zorder=layer,
            )
        )

    band_thickness = 2.01 * (4.0 / len(y))
    x_high = 1.5 + band_thickness
    y_high = 1.0 + band_thickness
    for plane, opacity in zip(section_planes, (0.78, 0.94, 0.78)):
        plane_z = float(z[plane])
        for rectangle, layer in (
            (((0.5, 1.0), (1.5, 1.0), (1.5, y_high), (0.5, y_high)), 18),
            (((1.5, 0.0), (x_high, 0.0), (x_high, y_high), (1.5, y_high)), 18),
            (((0.0, 0.0), (0.5, 0.0), (0.5, band_thickness), (0.0, band_thickness)), 14),
            (((x_high, 0.0), (2.0, 0.0), (2.0, band_thickness), (x_high, band_thickness)), 14),
            (((0.0, 4.0 - band_thickness), (2.0, 4.0 - band_thickness), (2.0, 4.0), (0.0, 4.0)), 18),
        ):
            ax_volume.add_patch(
                Polygon(
                    [mini_project(xx, yy, plane_z) for xx, yy in rectangle],
                    closed=True, facecolor="#f6a623", edgecolor="none",
                    alpha=opacity, zorder=layer,
                )
            )
        visible_path = [
            mini_project(0.5, y_high, plane_z),
            mini_project(x_high, y_high, plane_z),
            mini_project(x_high, 0.0, plane_z),
        ]
        ax_volume.plot(
            [point[0] for point in visible_path],
            [point[1] for point in visible_path],
            color="#e88900", linewidth=1.25, alpha=0.98, zorder=19,
        )


target_base_x, donor_base_x = 0.080, 0.720
draw_time_volume("target", target_base_x, "#0072b2")
draw_time_volume("donor", donor_base_x, "#d55e00")
for sequence_index, mini_base_x in enumerate((0.360, 0.472, 0.584)):
    draw_mini_time_volume(sequence_index, mini_base_x)

target_time = float(timeline["displayed_target"]["time_Ubar_over_h"])
donor_time = float(timeline["far_time_condition_for_displayed_target"]["time_Ubar_over_h"])
target_string = f"$t^*={target_time:.4f}$"
donor_string = f"$t^*={donor_time:.4f}$"
for x_position, heading, time_string, colour, artist_id, key, value, evidence in (
    (
        0.170, "reconstruction target", target_string, "#0072b2",
        "fig2.a.displayed_target_time", "displayed_target.time_Ubar_over_h",
        target_time, "producer-fixed first displayed evaluation target",
    ),
    (
        0.810, "mismatched wall donor", donor_string, "#d55e00",
        "fig2.a.far_time_donor_time",
        "far_time_condition_for_displayed_target.time_Ubar_over_h", donor_time,
        "far-time source of the deliberately mismatched wall condition",
    ),
):
    ax_volume.text(
        x_position, 0.975, heading, transform=ax_volume.transAxes,
        fontsize=12.0, color="#263238", ha="center", va="top",
    )
    time_artist = ax_volume.text(
        x_position, 0.885, time_string, transform=ax_volume.transAxes,
        fontsize=12.0, color="#263238", ha="center", va="top",
    )
    bind_artist(
        fig, time_artist, artist_id=artist_id, panel="a",
        source_refs=[{"kind": "json", "path": "fig2/cube3d_timeline.json", "key": key}],
        source_payload=[value],
        expected_payload={"type": "Text", "text": time_string},
        transform="format exact dimensionless time to four decimals",
        evidence=evidence,
    )

# The three intermediate states are samples from the full retained sequence;
# outer ellipses connect them visually to the two endpoint volumes.
for ellipsis_x in (0.315, 0.446, 0.558, 0.685):
    ax_volume.text(
        ellipsis_x, 0.490, "$\cdots$", transform=ax_volume.transAxes,
        fontsize=12.0, color="#65737c", ha="center", va="center",
    )
ax_volume.text(
    0.500, 0.640, "172 stored fields", transform=ax_volume.transAxes,
    fontsize=12.0, color=GREY, ha="center", va="center",
)

# A neutral, dot-free arrow carries the sequence's direction below the volumes.
time_left, time_right, time_y = 0.315, 0.685, 0.330
ax_volume.annotate(
    "", xy=(time_right, time_y), xytext=(time_left, time_y),
    arrowprops={"arrowstyle": "->", "color": "#4f5b62", "linewidth": 2.35},
    zorder=15,
)
definition_string = "$t^*\\equiv t\\overline{U}/h$"
definition_artist = ax_volume.text(
    0.500, 0.245, definition_string, transform=ax_volume.transAxes,
    fontsize=12.0, color="#263238", ha="center", va="center", fontweight="bold",
)
bind_artist(
    fig, definition_artist, artist_id="fig2.a.convective_time_definition", panel="a",
    source_refs=[{"kind": "json", "path": "fig2/cube3d_timeline.json",
                  "key": "definition.unit_meaning"}],
    source_payload=[timeline["definition"]["unit_meaning"]],
    expected_payload={"type": "Text", "text": definition_string},
    transform="express the source convective-time definition as dimensionless time",
    evidence="definition of the nondimensional time used for the two snapshots",
)
ax_volume.text(
    0.500, 0.005, "orange band = the only supplied information",
    transform=ax_volume.transAxes,
    fontsize=12.0, color="#263238", ha="center", va="bottom",
)

# Orientation of the oblique volume projection.
triad_origin = (0.025, 0.117)
for axis_name, axis_tip, label_position in (
    ("$x$", (0.069, 0.117), (0.075, 0.113)),
    ("$y$", (0.025, 0.217), (0.025, 0.232)),
    ("$z$", (0.056, 0.178), (0.062, 0.185)),
):
    ax_volume.annotate(
        "", xy=axis_tip, xytext=triad_origin, xycoords=ax_volume.transAxes,
        arrowprops={"arrowstyle": "->", "color": "#263238", "linewidth": 1.25},
        zorder=22,
    )
    ax_volume.text(
        label_position[0], label_position[1], axis_name,
        transform=ax_volume.transAxes, fontsize=12.0, color="#263238",
        ha="center", va="center", zorder=22,
    )

# Bulk-flow direction, kept outside the data planes and above the orientation
# triad so neither annotation obscures the velocity field.
ax_volume.annotate(
    "", xy=(0.098, 0.675), xytext=(0.030, 0.675),
    xycoords=ax_volume.transAxes,
    arrowprops={"arrowstyle": "->", "color": "#263238", "linewidth": 1.65},
    zorder=22,
)
ax_volume.text(
    0.064, 0.705, "$\overline{U}$", transform=ax_volume.transAxes,
    fontsize=12.0, color="#263238", ha="center", va="bottom", zorder=22,
)
panel_label(ax_volume, "a", x_pos=0.01, y_pos=0.98)


def style_field_axis(ax, *, bottom: bool, first: bool) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 2, 4])
    ax.tick_params(labelbottom=bottom, labelleft=first, pad=2)
    if bottom:
        ax.set_xlabel("$x/h$", labelpad=2)
    if first:
        ax.set_ylabel("$y/h$", labelpad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_color("#263238")


# ------------------------------------------------------------------------- b,c
field_axes = [fig.add_subplot(field_grid[0, column]) for column in range(4)]
error_axes = [fig.add_subplot(error_grid[0, column]) for column in range(4)]
velocity_cax = fig.add_subplot(field_grid[1, 0:4])
error_cax = fig.add_subplot(error_grid[1, 0:4])

for column, (title, field) in enumerate(fields.items()):
    source_key = field_source_keys[title]
    raw_field = np.asarray(data[source_key][0, :, :, z_index])
    raw_fluid = np.asarray(data["fluid"][:, :, z_index])

    ax = field_axes[column]
    image = ax.imshow(
        field, origin="lower", aspect="equal", cmap=velocity_cmap,
        vmin=-limit, vmax=limit, extent=extent, interpolation="nearest",
    )
    bind_artist(
        fig,
        image,
        artist_id=f"fig2.b.{source_key}_section",
        panel="b",
        source_refs=[
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz",
             "key": source_key, "slice": [0, ":", ":", z_index]},
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz",
             "key": "fluid", "slice": [":", ":", z_index]},
        ],
        source_payload=[raw_field, raw_fluid],
        expected_payload={
            "type": "AxesImage", "array": np.ma.masked_invalid(field),
            "extent": [float(value) for value in extent], "clim": [-limit, limit],
        },
        transform=(
            "streamwise component at producer-fixed middle-span section; transpose x/y; "
            "mask solid cells; shared symmetric full-range linear colour scale"
        ),
        evidence="contacted M2 illustrative field; LES reference is numerical, not truth",
    )
    ax.contour(
        x, y, band.astype(float), levels=[0.5],
        colors=["#e88900"], linewidths=1.45,
    )
    ax.set_title(title, fontsize=12.0, color="#263238", pad=10, linespacing=1.05)
    style_field_axis(ax, bottom=True, first=column == 0)

    error = error_fields[title]
    error[~fluid] = np.nan
    ax = error_axes[column]
    error_image = ax.imshow(
        error, origin="lower", aspect="equal", cmap=error_cmap,
        vmin=0, vmax=error_limit, extent=extent, interpolation="nearest",
    )
    truth_raw = np.asarray(data["truth"][0, :, :, z_index])
    bind_artist(
        fig,
        error_image,
        artist_id=f"fig2.c.{source_key}_absolute_error",
        panel="c",
        source_refs=[
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz",
             "key": source_key, "slice": [0, ":", ":", z_index]},
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz",
             "key": "truth", "slice": [0, ":", ":", z_index]},
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz",
             "key": "fluid", "slice": [":", ":", z_index]},
        ],
        source_payload=[raw_field, truth_raw, raw_fluid],
        expected_payload={
            "type": "AxesImage", "array": np.ma.masked_invalid(error),
            "extent": [float(value) for value in extent], "clim": [0.0, error_limit],
        },
        transform=(
            "absolute difference from the same LES section after transpose; mask solid "
            "cells; shared full-range linear error scale"
        ),
        evidence="illustrative error field; aggregate scores use all 160 targets",
    )
    ax.contour(
        x, y, band.astype(float), levels=[0.5],
        colors=["#e88900"], linewidths=1.45,
    )
    error_title = {
        "truth": "LES-reference\nerror",
        "correct": "aligned-wall\nerror",
        "no_wall": "absent-wall\nerror",
        "wrong_wall": "far-time-wall\nerror",
    }[source_key]
    ax.set_title(error_title, fontsize=12.0, color="#263238", pad=10)
    style_field_axis(ax, bottom=True, first=column == 0)

velocity_colorbar = fig.colorbar(image, cax=velocity_cax, orientation="horizontal")
velocity_colorbar.set_label("field   $u/\\overline{U}$", labelpad=2)
velocity_colorbar.set_ticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
error_colorbar = fig.colorbar(error_image, cax=error_cax, orientation="horizontal")
error_colorbar.set_label("error   $|u-u_{\\rm LES}|/\\overline{U}$", labelpad=2)
error_colorbar.set_ticks([0.0, 0.2, 0.4, 0.6])
for colorbar_artist in (velocity_colorbar, error_colorbar):
    colorbar_artist.outline.set_linewidth(1.25)
    for collection in colorbar_artist.ax.collections:
        collection.set_linewidth(1.25)
    colorbar_artist.ax.tick_params(length=3, width=1.25, pad=2, labelsize=12.0)
panel_label(field_axes[0], "b", x_pos=-0.36, y_pos=1.42)
panel_label(error_axes[0], "c", x_pos=-0.36, y_pos=1.52)


# ------------------------------------------------------------------------- d,e
# The two curves use the full 3-D target and score mask, not the displayed plane.
volumes = {key: np.asarray(data[key][0], dtype=float) for key in ARM_STYLE}


def mean_profile(volume: np.ndarray) -> np.ndarray:
    values = np.full(y.shape, np.nan, dtype=float)
    for y_index in range(len(y)):
        mask = scored_3d[:, y_index, :]
        if np.any(mask):
            values[y_index] = float(np.mean(volume[:, y_index, :][mask]))
    return values


def rmse_profile(volume: np.ndarray) -> np.ndarray:
    error = volume - volumes["truth"]
    values = np.full(y.shape, np.nan, dtype=float)
    for y_index in range(len(y)):
        mask = scored_3d[:, y_index, :]
        if np.any(mask):
            values[y_index] = float(np.sqrt(np.mean(error[:, y_index, :][mask] ** 2)))
    return values


ax_mean = fig.add_subplot(field_grid[0:2, 4])
for key, volume in volumes.items():
    label, colour, linestyle = ARM_STYLE[key]
    profile = mean_profile(volume)
    line, = ax_mean.plot(profile, y, color=colour, linestyle=linestyle, linewidth=1.8,
                         label=label)
    bind_artist(
        fig,
        line,
        artist_id=f"fig2.d.{key}_mean_profile",
        panel="d",
        source_refs=[
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz",
             "key": key, "slice": [0, ":", ":", ":"]},
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "fluid"},
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "band"},
        ],
        source_payload=[np.asarray(data[key][0]), fluid_3d, band_3d],
        expected_payload={"type": "Line2D", "x": profile, "y": y},
        transform="mean streamwise velocity over x-z cells outside the supplied envelope at each y",
        evidence="displayed producer-fixed 3-D target only; not the 160-time aggregate",
    )
ax_mean.set_xlabel("$\\langle u\\rangle_{xz}/\\overline{U}$")
ax_mean.set_xlim(0, 1.6)
ax_mean.set_xticks([0.0, 0.5, 1.0, 1.5])
ax_mean.set_ylabel("$y/h$")
ax_mean.set_ylim(0, 4)
ax_mean.set_yticks([0, 1, 2, 3, 4])
ax_mean.grid(True, color="#dfe3e6", linewidth=1.0)
ax_mean.legend(
    loc="upper left", bbox_to_anchor=(0.02, 0.99), ncol=1, fontsize=12.0,
    handlelength=1.3, handletextpad=0.4, borderpad=0.25, labelspacing=0.30,
    frameon=True, framealpha=0.94, facecolor="white", edgecolor="#d7dcdf",
)
ax_mean.set_title("Scored-volume mean", fontsize=12.0, pad=5)
panel_label(ax_mean, "d", x_pos=-0.15, y_pos=1.20)

ax_rmse = fig.add_subplot(error_grid[0:2, 4], sharey=ax_mean)
overall_rmse = {}
for key in ("correct", "no_wall", "wrong_wall"):
    volume = volumes[key]
    label, colour, linestyle = ARM_STYLE[key]
    profile = rmse_profile(volume)
    overall = float(np.sqrt(np.mean((volume[scored_3d] - volumes["truth"][scored_3d]) ** 2)))
    overall_rmse[key] = overall
    line, = ax_rmse.plot(
        profile, y, color=colour, linestyle=linestyle, linewidth=1.8,
        label=label.replace(" wall", ""),
    )
    bind_artist(
        fig,
        line,
        artist_id=f"fig2.e.{key}_rmse_profile",
        panel="e",
        source_refs=[
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz",
             "key": key, "slice": [0, ":", ":", ":"]},
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz",
             "key": "truth", "slice": [0, ":", ":", ":"]},
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "fluid"},
            {"kind": "npz", "path": "fig2/cube3d_representative_fields.npz", "key": "band"},
        ],
        source_payload=[
            np.asarray(data[key][0]), np.asarray(data["truth"][0]), fluid_3d, band_3d,
        ],
        expected_payload={"type": "Line2D", "x": profile, "y": y},
        transform="x-z root-mean-square streamwise error outside the supplied envelope at each y",
        evidence="displayed producer-fixed 3-D target only; legend gives its volume RMSE",
    )
ax_rmse.set_xlabel("$\\mathrm{RMSE}_{xz}(u)/\\overline{U}$")
ax_rmse.set_ylabel("$y/h$")
ax_rmse.set_ylim(0, 4)
ax_rmse.set_xlim(0, 0.28)
ax_rmse.set_xticks([0, 0.1, 0.2])
ax_rmse.grid(True, color="#dfe3e6", linewidth=1.0)
ax_rmse.legend(
    loc="upper right", bbox_to_anchor=(0.99, 0.86), borderaxespad=0.10,
    fontsize=12.0, handlelength=1.1, handletextpad=0.4,
    borderpad=0.25, labelspacing=0.30, frameon=True, framealpha=0.94,
    facecolor="white", edgecolor="#d7dcdf",
)
ax_rmse.set_title("Wall-normal RMSE", fontsize=12.0, pad=5)
panel_label(ax_rmse, "e", x_pos=-0.19, y_pos=1.16)

save(fig, "fig2_generation")
