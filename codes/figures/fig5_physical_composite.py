#!/usr/bin/env python3
"""Figure 5: registered cube physical-statistic families and null composite."""

import numpy as np

from _submission import (
    BLUE,
    GREY,
    RED,
    bind_artist,
    configure,
    expected_bar_payload,
    expected_errorbar_payload,
    load_json,
    panel_label,
    plt,
    save,
)


configure()
audit = load_json("review_audit/derived_peer_review_statistics.json")
secondary = audit["physical_statistics"]["block57_conservative"]
arm_order = [
    ("correct", "aligned\nband", BLUE),
    ("no_wall", "absent\nband", GREY),
    ("far_time_wall", "far-time\nband", RED),
]

fig, axes = plt.subplots(1, 3, figsize=(8.75, 3.48), constrained_layout=True)


def loss_panel(ax, metric, title, ylabel, label):
    stored = secondary[metric]
    values = np.array([stored["arms"][key]["mean"] for key, _, _ in arm_order])
    ci = np.array([stored["arms"][key]["ci95"] for key, _, _ in arm_order])
    positions = np.arange(3)
    bars = ax.bar(
        positions, values, color=[color for _, _, color in arm_order], alpha=0.9
    )
    whiskers = ax.errorbar(
        positions,
        values,
        yerr=[values - ci[:, 0], ci[:, 1] - values],
        fmt="none",
        ecolor="black",
        capsize=4,
        capthick=1.6,
        lw=1.6,
    )
    mean_refs = [
        {
            "kind": "json",
            "path": "review_audit/derived_peer_review_statistics.json",
            "key": (
                f"physical_statistics.block57_conservative.{metric}."
                f"arms.{key}.mean"
            ),
        }
        for key, _, _ in arm_order
    ]
    low_refs = [
        {
            "kind": "json",
            "path": "review_audit/derived_peer_review_statistics.json",
            "key": (
                f"physical_statistics.block57_conservative.{metric}."
                f"arms.{key}.ci95.0"
            ),
        }
        for key, _, _ in arm_order
    ]
    high_refs = [
        {
            "kind": "json",
            "path": "review_audit/derived_peer_review_statistics.json",
            "key": (
                f"physical_statistics.block57_conservative.{metric}."
                f"arms.{key}.ci95.1"
            ),
        }
        for key, _, _ in arm_order
    ]
    bind_artist(
        fig,
        bars,
        artist_id=f"fig5.{label}.{metric}_means",
        panel=label,
        source_refs=mean_refs,
        source_payload=values.tolist(),
        expected_payload=expected_bar_payload(positions, values),
        transform="bar heights are exact arm-wise means",
        evidence="E1 post-selection contacted physical diagnostic",
    )
    bind_artist(
        fig,
        whiskers,
        artist_id=f"fig5.{label}.{metric}_conditional_intervals",
        panel=label,
        source_refs=[*mean_refs, *low_refs, *high_refs],
        source_payload=[
            *values.tolist(),
            *ci[:, 0].tolist(),
            *ci[:, 1].tolist(),
        ],
        expected_payload=expected_errorbar_payload(
            positions, values, ci[:, 0], ci[:, 1], data_line=False
        ),
        transform="shared circular block-57 conditional interval about each arm mean",
        evidence="interval excludes model-selection and between-fit uncertainty",
    )
    ax.set_xticks(positions, [name for _, name, _ in arm_order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    panel_label(ax, label, y=1.23)
    if metric == "component_spectrum_log_rmse":
        ax.set_ylim(0.0, 0.39)
    else:
        ax.set_ylim(0.0, 0.64)
    far_ci = stored["improvements"]["far_time_wall_minus_correct"]["ci95"]
    ax.text(
        0.5,
        0.97,
        "beats absent; not far-time\n"
        "far-time conditional block interval\n"
        f"[{far_ci[0]:+.3f}, {far_ci[1]:+.3f}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="black",
        fontsize=8.1,
    )


loss_panel(
    axes[0],
    "component_spectrum_log_rmse",
    "Component spectra",
    "farther-region log-spectrum RMSE\n(lower is better)",
    "a",
)
loss_panel(
    axes[1],
    "reynolds_stress_profile_nrmse",
    "Instantaneous quadratic profiles",
    "plane-averaged quadratic-profile NRMSE\n(lower is better)",
    "b",
)

ax = axes[2]
coverage_block = secondary["coverage80"]["arms"]
values = np.array([coverage_block[key]["mean"] for key, _, _ in arm_order])
ci = np.array([coverage_block[key]["ci95"] for key, _, _ in arm_order])
positions = np.arange(3)
bars = ax.bar(
    positions, values, color=[color for _, _, color in arm_order], alpha=0.9
)
whiskers = ax.errorbar(
    positions,
    values,
    yerr=[values - ci[:, 0], ci[:, 1] - values],
    fmt="none",
    ecolor="black",
    capsize=4,
    capthick=1.6,
    lw=1.6,
)
coverage_mean_refs = [
    {
        "kind": "json",
        "path": "review_audit/derived_peer_review_statistics.json",
        "key": (
            "physical_statistics.block57_conservative.coverage80."
            f"arms.{key}.mean"
        ),
    }
    for key, _, _ in arm_order
]
coverage_low_refs = [
    {
        "kind": "json",
        "path": "review_audit/derived_peer_review_statistics.json",
        "key": (
            "physical_statistics.block57_conservative.coverage80."
            f"arms.{key}.ci95.0"
        ),
    }
    for key, _, _ in arm_order
]
coverage_high_refs = [
    {
        "kind": "json",
        "path": "review_audit/derived_peer_review_statistics.json",
        "key": (
            "physical_statistics.block57_conservative.coverage80."
            f"arms.{key}.ci95.1"
        ),
    }
    for key, _, _ in arm_order
]
bind_artist(
    fig,
    bars,
    artist_id="fig5.c.descriptive_hit_fraction_means",
    panel="c",
    source_refs=coverage_mean_refs,
    source_payload=values.tolist(),
    expected_payload=expected_bar_payload(positions, values),
    transform="bar heights are plug-in 0.1/0.9 interval hit fractions",
    evidence="descriptive only; no finite-eight-member calibration target",
)
bind_artist(
    fig,
    whiskers,
    artist_id="fig5.c.descriptive_hit_fraction_intervals",
    panel="c",
    source_refs=[*coverage_mean_refs, *coverage_low_refs, *coverage_high_refs],
    source_payload=[
        *values.tolist(),
        *ci[:, 0].tolist(),
        *ci[:, 1].tolist(),
    ],
    expected_payload=expected_errorbar_payload(
        positions, values, ci[:, 0], ci[:, 1], data_line=False
    ),
    transform="shared circular block-57 conditional interval about each hit fraction",
    evidence="descriptive block-selection variation; calibration family withdrawn",
)
ax.set_ylim(0, 0.9)
ax.set_xticks(positions, [name for _, name, _ in arm_order])
ax.set_ylabel("10–90% plug-in interval hit fraction")
ax.set_title("Descriptive interval hits")
panel_label(ax, "c", y=1.23)
for index, value in enumerate(values):
    ax.text(index, value + 0.025, f"{value:.1%}", ha="center", fontsize=8.1)
ax.text(
    0.5,
    0.82,
    "$M=8$; 51.4% hit fraction\nno finite-$M$ calibration target",
    transform=ax.transAxes,
    ha="center",
    va="top",
    color="black",
    fontsize=8.1,
)

fig.suptitle(
    "Physical corroboration is null: both valid families fail the two-control gate",
    fontsize=11.0,
    fontweight="bold",
)
save(fig, "fig5_physical_composite")
