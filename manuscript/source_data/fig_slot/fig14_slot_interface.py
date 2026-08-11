#!/usr/bin/env python3
"""Figure: the single-slot, quality-agnostic, flux-consistent traction interface,
with the fidelity->gain dose-response as the attribution instrument (node_011).

Reads only byte-verified committed artifacts:
  codes/results/e2_slot_channel_final_results.json      (decisive FINAL unit)
  codes/results/block_inference_calibration.json        (frozen t*)
  development/nodes/node_011/V2_FROZEN_PARAMS.json      (frozen m5, C9 band)
Every plotted number is read from those files; nothing is typed in.

--phase rehearsal plots the development rehearsal instead (script testing only;
never released).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _submission import bind_artist, save as submission_save  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "codes" / "results"
NODE = ROOT / "development" / "nodes" / "node_011"
SOURCE = pathlib.Path(os.environ.get("GWT_SOURCE_DATA",
                                     ROOT / "manuscript" / "source_data"))
OUTPUT = pathlib.Path(os.environ.get("GWT_FIGURE_OUTPUT",
                                     ROOT / "manuscript" / "figures"))


def load_json_with_fallback(name):
    """Repository first; the staged release source package second."""
    for base in (RES, SOURCE / "fig_slot"):
        p = base / name
        if p.is_file():
            return json.loads(p.read_text())
    raise SystemExit(f"{name} not found in codes/results or source_data/fig_slot")

BLUE, RED, GREY, LGREY, GOLD, GREEN = ("#2166ac", "#b2182b", "#7a7a7a",
                                       "#c9c9c9", "#d99000", "#2a8a4a")
PURP = "#7b3294"

ap = argparse.ArgumentParser()
ap.add_argument("--phase", default="final", choices=("final", "rehearsal"))
A = ap.parse_args()


def linewidth_pt() -> float:
    log = ROOT / "manuscript" / "main.log"
    if log.is_file():
        m = re.findall(r"\\textwidth=([\d.]+)pt", log.read_text(errors="ignore"))
        if m:
            return float(m[-1])
    return 510.0


def panel_label(ax, s):
    ax.text(-0.24, 1.10, s, transform=ax.transAxes, fontsize=8.7,
            fontweight="bold", va="bottom", ha="left")


def main() -> None:
    R = load_json_with_fallback(f"e2_slot_channel_{A.phase}_results.json")
    CAL = load_json_with_fallback("block_inference_calibration.json")
    tstar = CAL["t_star_frozen"]
    v2_paths = (NODE / "V2_FROZEN_PARAMS.json",
                SOURCE / "fig_slot" / "V2_FROZEN_PARAMS.json")
    for vp in v2_paths:
        if vp.is_file():
            V2 = json.loads(vp.read_text())
            m5, c9band = V2["c5_margin_m5"], V2["c9_band"]
            break
    else:
        raise SystemExit("V2_FROZEN_PARAMS.json not found -- the frozen band/margin "
                         "must come from the preregistration, never a default")
    fid = R["fidelity_centred_r2"]
    con = R["contrasts"]
    dose = R["dose_response"]
    matched = R["_meta"]["matched_arm"]["name"]

    def c(a, b="absent", region="whole_scorable", fam="flow_matching"):
        return con.get(f"{fam}|{a}-{b}|{region}")

    W_IN = linewidth_pt() / 72.0
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.titlesize": 7.8, "axes.labelsize": 7.2,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 1.0,
        "lines.linewidth": 1.2, "savefig.dpi": 400,
    })
    fig = plt.figure(figsize=(W_IN, W_IN * 0.72), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.14, h_pad=0.12)
    gs = fig.add_gridspec(2, 3)

    # ---------------- a: the dose-response curve -----------------------------
    ax = fig.add_subplot(gs[0, 0:2])
    panel_label(ax, "a")
    curve = dose["curve"]
    xs = [e["fidelity"] for e in curve]
    ys = [e["gain"] for e in curve]
    (curve_line,) = ax.plot(xs, ys, "-", color=GREY, lw=1.2, zorder=1,
                            label="ladder interpolant (exact + graded noise)")
    if A.phase == "final":
        bind_artist(
            fig, curve_line,
            artist_id="fig14.a.dose_response_curve",
            panel="a",
            source_refs=[{"kind": "json",
                          "path": "fig_slot/e2_slot_channel_final_results.json",
                          "key": "dose_response.curve"}],
            source_payload=[curve],
            expected_payload={"type": "Line2D",
                              "x": np.asarray(xs, float),
                              "y": np.asarray(ys, float)},
            transform="plot each registered reference arm's (measured fidelity, "
                      "whole-scorable gain) pair in fidelity order",
            evidence="registered node_011 dose-response; ladder arms are "
                     "oracle-graded references, not deployable methods")
    ax.fill_between(xs, np.array(ys) - m5, np.array(ys) + m5, color=LGREY,
                    alpha=0.55, zorder=0, label=f"registered margin $\\pm m_5$")
    ax.plot(xs[:-1], ys[:-1], "o", color=BLUE, ms=4, zorder=3)
    ax.plot([1.0], [ys[-1]] if xs[-1] == 1.0 else [curve[-1]["gain"]], "s",
            color=BLUE, ms=5, zorder=3, label="exact traction (ceiling)")
    pts = [("closure", dose["closure_point"], RED, "D", "physics closure"),
           ("learned_dataonly", dose["learned_dataonly_point"], PURP, "v",
            "learned data-only"),
           ("equilibrium", dose["equilibrium_point"], GOLD, "^",
            "equilibrium law")]
    for _, p, col, mk, lab in pts:
        if p["gain"] is not None:
            ax.plot(p["fidelity"], p["gain"], mk, color=col, ms=5.5, zorder=4,
                    label=lab)
    mn = c(matched)
    if mn:
        ax.plot(fid[matched], mn["delta"], "P", color=GREEN, ms=6, zorder=4,
                label="fidelity-matched noise")
    for nm, mk in (("wrong_time", "x"), ("shuffle_z", "+")):
        e = c(nm)
        if e:
            ax.plot(fid[nm], e["delta"], mk, color="k", ms=5, zorder=4,
                    label=nm.replace("_", "-"))
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xlim(-0.58, 1.06)
    ax.set_ylim(-0.085, 0.20)
    ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([-0.05, 0.0, 0.05, 0.10, 0.15])
    ax.set_xlabel("measured a-priori traction fidelity, centred $R^2(\\tau)$")
    ax.set_ylabel("field gain over the absent arm\n(whole scorable volume)")
    ax.set_title("every wall model must land on the measured response curve")
    ax.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.4)

    # ---------------- b: region-resolved gains --------------------------------
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "b")
    regions = ["buffer_yp_lt30", "log_yp30_100", "outer_yp_gt100"]
    rlab = ["buffer", "log band", "outer"]
    ypos = np.arange(3)
    for off, (arm, col, lab) in enumerate(
            (("exact", BLUE, "exact"), ("closure", RED, "closure"))):
        ds = [c(arm, region=r)["delta"] for r in regions]
        ax.barh(ypos + (off - 0.5) * 0.36, ds, 0.34, color=col, label=lab)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(ypos, rlab)
    ax.set_xticks([0.0, 0.25, 0.5])
    ax.set_xlim(0, 0.56)
    ax.invert_yaxis()
    ax.set_xlabel("gain over the absent arm")
    ax.set_title("wall-normal reach")
    ax.legend(frameon=False, loc="lower right")

    # ---------------- c: primary block inference ------------------------------
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    names = [("exact", BLUE), ("closure", RED), (matched, GREEN),
             ("equilibrium", GOLD), ("learned_dataonly", PURP)]
    clabs = ["exact", "closure", "matched", "equil.", "data-only"]
    for i, (nm, col) in enumerate(names):
        e = c(nm)
        if e is None:
            continue
        bm = e["primary"]["block_means"]
        ax.plot(bm, [i] * len(bm), "o", color=col, ms=3.5, alpha=0.85)
        ax.plot([e["delta"]] * 2, [i - 0.24, i + 0.24], "-", color=col, lw=1.6)
        t_text = ax.text(0.185, i, f"t={e['primary']['t']:.1f}",
                         va="center", ha="left", fontsize=6.5, color=col)
        if A.phase == "final" and nm == "closure":
            bind_artist(
                fig, t_text,
                artist_id="fig14.c.closure_primary_t",
                panel="c",
                source_refs=[{"kind": "json",
                              "path": "fig_slot/e2_slot_channel_final_results.json",
                              "key": "contrasts.flow_matching|closure-absent|"
                                     "whole_scorable.primary.t"}],
                source_payload=[e["primary"]["t"]],
                expected_payload={"type": "Text",
                                  "text": f"t={e['primary']['t']:.1f}"},
                transform="round the registered primary t statistic to one decimal",
                evidence="registered C2 primary inference, 3 physical-time blocks, "
                         "calibrated t*")
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlim(-0.01, 0.245)
    ax.set_yticks(range(len(names)), clabs)
    ax.invert_yaxis()
    ax.set_xlabel("gain over absent (3 block means)")
    ax.set_title(f"calibrated $t^* = {tstar}$")

    # ---------------- d: engineering endpoint ---------------------------------
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    order = ["absent", "wrong_time", "shuffle_z", "equilibrium",
             "learned_dataonly", matched, "closure", "exact"]
    labs = ["none", "wrong-t", "shuffle-z", "equil.", "data-only", "matched",
            "closure", "exact"]
    key = f"flux_abserr_{c9band}"
    vals, cols = [], []
    for nm in order:
        v = [a[key] for k, a in R["arms"].items()
             if k.startswith(f"flow_matching|{nm}|")]
        vals.append(np.mean(v) if v else np.nan)
        cols.append(RED if nm == "closure" else BLUE if nm == "exact" else GREY)
    bars = ax.barh(range(len(order)), vals, 0.62, color=cols)
    if A.phase == "final":
        refs, pays = [], []
        for nm in order:
            for seed in R["_meta"]["fm_seeds"]:
                refs.append({"kind": "json",
                             "path": "fig_slot/e2_slot_channel_final_results.json",
                             "key": f"arms.flow_matching|{nm}|{seed}.{key}"})
                pays.append(R["arms"][f"flow_matching|{nm}|{seed}"][key])
        bind_artist(
            fig, bars,
            artist_id="fig14.d.flux_tracking_bars",
            panel="d",
            source_refs=refs,
            source_payload=pays,
            expected_payload={"type": "BarContainer",
                              # horizontal bars: the container's x-centre is
                              # value/2 (bars start at 0) and the patch height
                              # is the 0.62 bar thickness
                              "centres": [float(v) / 2.0 for v in vals],
                              "heights": [0.62 for _ in vals]},
            transform="average each arm's per-seed mean absolute momentum-flux "
                      "tracking error over the two flow-matching seeds; drawn "
                      "as horizontal bars from zero",
            evidence="registered C9 engineering endpoint, buffer band, outside "
                     "all conditioning")
    ax.set_yticks(range(len(order)), labs)
    ax.invert_yaxis()
    ax.set_xlabel("flux tracking error")
    ax.set_title(f"$\\langle u\'v\'\\rangle$, {c9band} band")

    # ---------------- e: robustness -------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "e")
    rows, labs2 = [], []
    k = c("closure")
    if k:
        rows += [k["per_seed_delta"][0], k["per_seed_delta"][1],
                 k.get("delta_wall0"), k.get("delta_wall1")]
        labs2 += ["fm seed 1", "fm seed 2", "wall 0", "wall 1"]
    d = c("closure", fam="diffusion")
    if d:
        rows += d["per_seed_delta"]
        labs2 += ["diff seed 1", "diff seed 2"]
    ax.barh(range(len(rows)), rows, 0.6,
            color=[RED] * 2 + [GREY] * 2 + [GOLD] * max(0, len(rows) - 4))
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(range(len(labs2)), labs2)
    ax.invert_yaxis()
    ax.set_xlabel("closure $-$ absent")
    ax.set_title("sign robustness")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if A.phase == "final":
        # manifest + rendered layout audit + save, all through the release helper
        submission_save(fig, "fig14_slot_interface")
    else:
        out = OUTPUT / "fig14_slot_interface"
        fig.savefig(f"{out}.pdf", facecolor="white")
        fig.savefig(f"{out}.png", facecolor="white")
        print(f"[fig14] written {out}.pdf (rehearsal test render)")


if __name__ == "__main__":
    main()
