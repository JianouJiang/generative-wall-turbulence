#!/usr/bin/env python3
"""Figure 4: registered farther-from-wall energy-score endpoint after model selection."""

import numpy as np

from _submission import (
    BLUE,
    GREY,
    RED,
    bind_artist,
    configure,
    expected_errorbar_payload,
    load_json,
    panel_label,
    plt,
    save,
)


configure()
audit = load_json("review_audit/derived_peer_review_statistics.json")
primary = audit["fair_energy_score"]["block57_conservative"]
arm_order = [
    ("correct", "aligned band", BLUE),
    ("no_wall", "absent band", GREY),
    ("far_time_wall", "far-time band", RED),
]

fig, axes = plt.subplots(1, 3, figsize=(8.75, 3.55), constrained_layout=True)

ax = axes[0]
positions = np.arange(3)
values = [primary["arms"][key]["mean"] for key, _, _ in arm_order]
errors = [
    [
        values[i] - primary["arms"][key]["ci95"][0]
        for i, (key, _, _) in enumerate(arm_order)
    ],
    [
        primary["arms"][key]["ci95"][1] - values[i]
        for i, (key, _, _) in enumerate(arm_order)
    ],
]
for position, value, lower, upper, (key, _, color) in zip(
    positions, values, errors[0], errors[1], arm_order
):
    errorbar = ax.errorbar(
        position,
        value,
        yerr=[[lower], [upper]],
        fmt="o",
        color=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=7.5,
        ecolor=color,
        capsize=4,
        capthick=1.6,
        lw=1.6,
    )
    interval = primary["arms"][key]["ci95"]
    bind_artist(
        fig,
        errorbar,
        artist_id=f"fig4.a.{key}_energy_score",
        panel="a",
        source_refs=[
            {
                "kind": "json",
                "path": "review_audit/derived_peer_review_statistics.json",
                "key": f"fair_energy_score.block57_conservative.arms.{key}.mean",
            },
            {
                "kind": "json",
                "path": "review_audit/derived_peer_review_statistics.json",
                "key": f"fair_energy_score.block57_conservative.arms.{key}.ci95.0",
            },
            {
                "kind": "json",
                "path": "review_audit/derived_peer_review_statistics.json",
                "key": f"fair_energy_score.block57_conservative.arms.{key}.ci95.1",
            },
        ],
        source_payload=[value, interval[0], interval[1]],
        expected_payload=expected_errorbar_payload(
            [position], [value], [interval[0]], [interval[1]]
        ),
        transform="identity mean and conditional block-57 interval on zero-based axis",
        evidence="E1 post-selection endpoint on contacted M2 targets",
    )
ax.set_xticks(positions, [label for _, label, _ in arm_order], rotation=13)
ax.set_ylabel("farther-region energy score\n($d/h>0.5$; lower is better)")
ax.set_ylim(0.0, 0.58)
ax.set_yticks([0.0, 0.2, 0.4, 0.58])
ax.set_title("Absolute score; zero-based axis")
panel_label(ax, "a", y=1.22)
for i, value in enumerate(values):
    ax.text(i, value - 0.055, f"{value:.3f}", ha="center", fontsize=8)

ax = axes[1]
controls = [
    ("no_wall_minus_correct", "absent\n− aligned", GREY),
    ("far_time_wall_minus_correct", "far-time\n− aligned", RED),
]
for i, (key, label, color) in enumerate(controls):
    item = primary["improvements"][key]
    value = item["mean"]
    errorbar = ax.errorbar(
        i,
        value,
        yerr=[[value - item["ci95"][0]], [item["ci95"][1] - value]],
        fmt="o",
        color=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=7.5,
        ecolor=color,
        capsize=4,
        capthick=1.6,
        lw=1.6,
    )
    bind_artist(
        fig,
        errorbar,
        artist_id=f"fig4.b.{key}",
        panel="b",
        source_refs=[
            {
                "kind": "json",
                "path": "review_audit/derived_peer_review_statistics.json",
                "key": f"fair_energy_score.block57_conservative.improvements.{key}.mean",
            },
            {
                "kind": "json",
                "path": "review_audit/derived_peer_review_statistics.json",
                "key": f"fair_energy_score.block57_conservative.improvements.{key}.ci95.0",
            },
            {
                "kind": "json",
                "path": "review_audit/derived_peer_review_statistics.json",
                "key": f"fair_energy_score.block57_conservative.improvements.{key}.ci95.1",
            },
        ],
        source_payload=[value, item["ci95"][0], item["ci95"][1]],
        expected_payload=expected_errorbar_payload(
            [i], [value], [item["ci95"][0]], [item["ci95"][1]]
        ),
        transform="control-minus-correct paired improvement with shared block-57 resampling",
        evidence="E1 post-selection contacted endpoint; lower energy score is better",
    )
ax.axhline(0, color="black", lw=1.6)
ax.set_ylim(-0.002, 0.038)
ax.set_yticks([0.0, 0.01, 0.02, 0.03])
ax.set_xticks(range(2), [item[1] for item in controls])
ax.get_xticklabels()[0].set_horizontalalignment("left")
ax.get_xticklabels()[1].set_horizontalalignment("right")
ax.set_ylabel("paired energy-score improvement")
ax.set_title("Both conditional intervals are positive")
panel_label(ax, "b", y=1.22)

ax = axes[2]
panel_label(ax, "c", y=1.22)
ax.axis("off")
ax.set_title("Evidence boundary")
facts = [
    ("ENDPOINT", "registered after checkpoint selection", BLUE),
    ("DATA CONTACT", "same interval used during development", RED),
    ("DEPENDENCE", "343 outputs: 2.800 count / 2.792 elapsed", GREY),
    ("INTERVALS", "conditional; not multiplicity-adjusted", GREY),
    ("PHYSICAL CHECKS", "two valid families: 0/2 pass both controls", RED),
]
for row, (tag, text, color) in enumerate(facts):
    y = 0.90 - row * 0.18
    ax.text(
        0.05,
        y,
        tag,
        color="black",
        fontweight="bold",
        fontsize=8.2,
        transform=ax.transAxes,
    )
    ax.text(0.05, y - 0.060, text, color="black", fontsize=8.2, transform=ax.transAxes)

fig.suptitle(
    "Fair finite-ensemble score preserves the selected model's ordering",
    fontsize=11.0,
    fontweight="bold",
)
save(fig, "fig4_mechanism_robustness")
