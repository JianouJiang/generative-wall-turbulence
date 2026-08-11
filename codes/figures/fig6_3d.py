#!/usr/bin/env python3
"""Figure S1: limited grouped-hill support and one-realization padding ablation."""

import numpy as np

from _submission import (
    BLUE,
    GOLD,
    GREY,
    RED,
    bind_artist,
    configure,
    expected_bar_payload,
    load_json,
    panel_label,
    plt,
    save,
)


configure()
grouped = load_json("fig6/grouped_hill_supporting.json")
cube = load_json("fig3/cube_periodic_topology_results.json")

fig, axes = plt.subplots(1, 3, figsize=(8.75, 3.60), constrained_layout=True)
arm_order = [
    ("correct", "aligned band", BLUE),
    ("no_wall", "absent band", GREY),
    ("random", "random band", GOLD),
    ("wrong_swap", "far-time band", RED),
]
arm_tick_labels = ["aligned", "absent", "random", "far-time"]

ax = axes[0]
diffusion = grouped["diffusion"]
values = [diffusion["arms"][key] for key, _, _ in arm_order]
bars = ax.bar(
    np.arange(4), values, color=[color for _, _, color in arm_order], alpha=0.9
)
bind_artist(
    fig,
    bars,
    artist_id="suppfig1.a.diffusion_arm_scores",
    panel="a",
    source_refs=[
        {
            "kind": "json",
            "path": "fig6/grouped_hill_supporting.json",
            "key": f"diffusion.arms.{key}",
        }
        for key, _, _ in arm_order
    ],
    source_payload=values,
    expected_payload=expected_bar_payload(np.arange(4), values),
    transform="identity grouped-hill fluctuation R2 by arm",
    evidence="supporting diffusion result; negative aligned-arm absolute skill",
)
ax.axhline(0, color="black", lw=1.6)
ax.set_xticks(np.arange(4), arm_tick_labels, fontsize=8.1)
ax.set_ylabel("grouped-hill fluctuation $R^2$")
ax.set_title("Diffusion supporting result")
panel_label(ax, "a", y=1.24)
ax.text(
    0.5,
    0.94,
    "aligned absolute skill < 0",
    transform=ax.transAxes,
    ha="center",
    color="black",
    fontweight="bold",
    fontsize=8.5,
)
diff_ci = diffusion["bootstrap"]["d_correct_minus_no_wall"]["ci95"]
ax.text(
    0.5,
    0.05,
    f"aligned-minus-absent interval\n[{diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}]",
    transform=ax.transAxes,
    ha="center",
    color="black",
    fontsize=8.2,
)

ax = axes[1]
flow = grouped["flow_matching"]
values = [flow["arms"][key] for key, _, _ in arm_order]
bars = ax.bar(
    np.arange(4), values, color=[color for _, _, color in arm_order], alpha=0.9
)
bind_artist(
    fig,
    bars,
    artist_id="suppfig1.b.flow_matching_arm_scores",
    panel="b",
    source_refs=[
        {
            "kind": "json",
            "path": "fig6/grouped_hill_supporting.json",
            "key": f"flow_matching.arms.{key}",
        }
        for key, _, _ in arm_order
    ],
    source_payload=values,
    expected_payload=expected_bar_payload(np.arange(4), values),
    transform="identity grouped-hill fluctuation R2 by arm",
    evidence="separately parameterised replication fails the aligned-minus-absent contrast",
)
ax.axhline(0, color="black", lw=1.6)
ax.set_xticks(np.arange(4), arm_tick_labels, fontsize=8.1)
ax.set_ylabel("grouped-hill fluctuation $R^2$")
ax.set_title("Flow-matching replication")
panel_label(ax, "b", y=1.24)
ci = flow["bootstrap"]["d_correct_minus_no_wall"]["ci95"]
ax.text(
    0.5,
    0.94,
    "replication fails",
    transform=ax.transAxes,
    ha="center",
    color="black",
    fontweight="bold",
    fontsize=8.5,
)
ax.text(
    0.5,
    0.05,
    f"aligned-minus-absent interval\n[{ci[0]:+.3f}, {ci[1]:+.3f}]",
    transform=ax.transAxes,
    ha="center",
    color="black",
    fontsize=8.2,
)

ax = axes[2]
change = cube["_meta"]["correct_R2_change_from_zero_padding"]
regions = [
    ("full_support_excluded", "complete"),
    ("near_support_excluded_d_le_0p5h", "near"),
    ("outer_d_gt_0p5h", "farther"),
]
values = [change[key] for key, _ in regions]
bars = ax.bar(np.arange(3), values, color=[BLUE, BLUE, BLUE], alpha=0.86)
bind_artist(
    fig,
    bars,
    artist_id="suppfig1.c.padding_change",
    panel="c",
    source_refs=[
        {
            "kind": "json",
            "path": "fig3/cube_periodic_topology_results.json",
            "key": f"_meta.correct_R2_change_from_zero_padding.{key}",
        }
        for key, _ in regions
    ],
    source_payload=values,
    expected_payload=expected_bar_payload(np.arange(3), values),
    transform="M2 minus M1 aligned-arm R2 by common support-excluded region",
    evidence="one-fit contacted-interval development sensitivity; no topology-law claim",
)
ax.axhline(0, color="black", lw=1.6)
ax.set_ylim(0, 0.06)
ax.set_yticks([0.00, 0.02, 0.04])
ax.set_xticks(np.arange(3), [label for _, label in regions])
ax.set_ylabel("change in aligned-arm $R^2$")
ax.set_title("Padding development sensitivity")
panel_label(ax, "c", y=1.24)
for index, value in enumerate(values):
    ax.text(index, value + 0.002, f"{value:+.3f}", ha="center", fontsize=8.1)
ax.set_xlabel(
    "same contacted interval; one fit\n(no topology-law claim)",
    color=GREY,
    fontsize=8.1,
)

fig.suptitle(
    "Grouped-hill support and model-development sensitivity\n"
    f"Grouped-hill test: {grouped['split']['effective_test_events']:.2f} effective events; supporting evidence only",
    color=GREY,
    fontsize=9.0,
    fontweight="bold",
)
save(fig, "fig6_3d")
