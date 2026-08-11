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
from _submission import (bind_artist, expected_errorbar_payload,  # noqa: E402
                         save as submission_save)

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


def panel_label(ax, s, x=-0.24, y=1.16):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12.0,
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

    # house standard: 8.75-in canvas with the 12-pt text floor,
    # matching every other main figure in this repository
    W_IN = 8.75
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12.0,
        "axes.titlesize": 12.0, "axes.labelsize": 12.0,
        "xtick.labelsize": 12.0, "ytick.labelsize": 12.0,
        "legend.fontsize": 12.0, "axes.linewidth": 1.3,
        "lines.linewidth": 1.3, "savefig.dpi": 400,
    })
    fig = plt.figure(figsize=(W_IN, 10.40))
    # manual layout, no constrained engine
    R_A = [0.086, 0.815, 0.207, 0.148]
    RC_A = [0.301, 0.815, 0.010, 0.148]
    R_B = [0.402, 0.815, 0.213, 0.148]
    RC_B = [0.623, 0.815, 0.010, 0.148]
    R_C = [0.722, 0.815, 0.203, 0.148]
    RC_C = [0.933, 0.815, 0.010, 0.148]
    R_D = [0.085, 0.575, 0.195, 0.160]
    R_E = [0.370, 0.575, 0.235, 0.160]
    R_F = [0.735, 0.575, 0.240, 0.160]
    R_G = [0.098, 0.330, 0.517, 0.165]
    R_H = [0.735, 0.330, 0.240, 0.165]
    R_I = [0.100, 0.052, 0.400, 0.185]
    R_J = [0.585, 0.052, 0.400, 0.185]

    # ---------------- a: the record and how thin the slot is ------------------
    ctx = np.load(SOURCE / "fig_slot" / "channel_context_frame.npz",
                  allow_pickle=False)
    CTX = "fig_slot/channel_context_frame.npz"
    NPZC = "fig_slot/e2_slot_channel_final_components.npz"
    comp = np.load(SOURCE / NPZC, allow_pickle=False)
    prof = np.asarray(ctx["u_mean_profile"], np.float64)
    u_tau = float(ctx["u_tau"])
    dy = float(ctx["dy"])
    ax = fig.add_axes(R_A)
    panel_label(ax, "a", y=1.05)
    fluct = np.asarray(ctx["u_plane"], np.float64) - prof[:, None]
    f_disp = np.ma.masked_invalid(fluct)
    f_lim = float(np.round(np.percentile(np.abs(fluct), 99.5), 2))
    ext = (0.0, 100.0, 0.0, 1.0)
    im = ax.imshow(f_disp, origin="lower", extent=ext, cmap="RdBu_r",
                   vmin=-f_lim, vmax=f_lim, interpolation="nearest",
                   aspect="auto", zorder=1)
    if A.phase == "final":
        bind_artist(
            fig, im, artist_id="fig14.a.channel_plane", panel="a",
            source_refs=[{"kind": "npz", "path": CTX, "key": "u_plane"},
                         {"kind": "npz", "path": CTX, "key": "u_mean_profile"}],
            source_payload=[np.asarray(ctx["u_plane"]),
                            np.asarray(ctx["u_mean_profile"])],
            expected_payload={"type": "AxesImage", "array": f_disp,
                              "extent": list(ext), "clim": [-f_lim, f_lim]},
            transform=("subtract the eval-window mean profile from the "
                       "lower-wall streamwise plane of the first held-out "
                       "frame"),
            evidence="wall-resolved channel record; the frame is in the "
                     "never-scored FINAL window")
    ax.axhline(4 * dy, color=GOLD, lw=2.4, zorder=3)
    ax.annotate("slot ($y^+\!\leq\!3.2$)", xy=(52, 4 * dy),
                xytext=(30, 0.40), fontsize=12.0, color="k",
                ha="left", va="center",
                arrowprops={"arrowstyle": "->", "color": "k",
                            "linewidth": 1.3})
    ax.set_xlabel("streamwise cell")
    ax.set_ylabel("$y/\delta$")
    ax.set_title("held-out frame, $u'$")
    cb = fig.colorbar(im, cax=fig.add_axes(RC_A))
    cb.set_ticks([-f_lim, 0.0, f_lim])
    cb.outline.set_linewidth(1.35)
    cb.ax.tick_params(width=1.35)
    for coll in list(cb.ax.collections):
        coll.set_linewidth(1.35)

    # ---------------- b: the u'v' field the endpoint scores -------------------
    ax = fig.add_axes(R_B)
    panel_label(ax, "b", x=-0.06, y=1.05)
    v_prof = np.asarray(ctx["v_mean_profile"], np.float64)
    uv = 1e4 * ((np.asarray(ctx["u_plane"], np.float64) - prof[:, None])
                * (np.asarray(ctx["v_plane"], np.float64) - v_prof[:, None]))
    uv_disp = np.ma.masked_invalid(uv)
    uv_lim = float(np.round(np.percentile(np.abs(uv), 99.0), 0))
    im2 = ax.imshow(uv_disp, origin="lower", extent=ext, cmap="PiYG",
                    vmin=-uv_lim, vmax=uv_lim, interpolation="nearest",
                    aspect="auto", zorder=1)
    if A.phase == "final":
        bind_artist(
            fig, im2, artist_id="fig14.b.uv_plane", panel="b",
            source_refs=[{"kind": "npz", "path": CTX, "key": k}
                         for k in ("u_plane", "v_plane", "u_mean_profile",
                                   "v_mean_profile")],
            source_payload=[np.asarray(ctx["u_plane"]),
                            np.asarray(ctx["v_plane"]),
                            np.asarray(ctx["u_mean_profile"]),
                            np.asarray(ctx["v_mean_profile"])],
            expected_payload={"type": "AxesImage", "array": uv_disp,
                              "extent": list(ext),
                              "clim": [-uv_lim, uv_lim]},
            transform=("multiply the fluctuations of u and v about the "
                       "eval-window mean profiles, same frame and wall as "
                       "panel a, x1e4"),
            evidence="the instantaneous field of the registered turbulent-"
                     "shear endpoint")
    ax.axhline(4 * dy, color=GOLD, lw=2.4, zorder=3)
    ax.axhline(30.0 * float(ctx["nu"]) / float(ctx["u_tau"]), color=GOLD,
               lw=1.6, ls=(0, (3, 2.2)), zorder=3)
    ax.set_xlabel("streamwise cell")
    ax.set_xticks([0, 50])
    ax.set_yticks([0.0, 0.5])
    ax.set_yticklabels([])
    ax.set_title("same frame, $u\'v\'$", pad=8)
    cb2 = fig.colorbar(im2, cax=fig.add_axes(RC_B))
    cb2.set_ticks([-uv_lim, 0.0, uv_lim])
    cb2.outline.set_linewidth(1.35)
    cb2.ax.tick_params(width=1.35)
    for coll in list(cb2.ax.collections):
        coll.set_linewidth(1.35)

    # ---------------- c: where the streaks live -------------------------------
    ax = fig.add_axes(R_C)
    panel_label(ax, "c", x=-0.06, y=1.05)
    rms = np.asarray(ctx["u_rms_map"], np.float64)
    rms_disp = np.ma.masked_invalid(rms)
    rms_lim = float(np.round(np.percentile(rms, 99.5), 2))
    im3 = ax.imshow(rms_disp, origin="lower", extent=ext, cmap="plasma",
                    vmin=0.0, vmax=rms_lim, interpolation="nearest",
                    aspect="auto", zorder=1)
    if A.phase == "final":
        bind_artist(
            fig, im3, artist_id="fig14.c.rms_map", panel="c",
            source_refs=[{"kind": "npz", "path": CTX, "key": "u_rms_map"}],
            source_payload=[np.asarray(ctx["u_rms_map"])],
            expected_payload={"type": "AxesImage", "array": rms_disp,
                              "extent": list(ext), "clim": [0.0, rms_lim]},
            transform="plot the per-pixel standard deviation of u over the "
                      "46 held-out frames, lower wall",
            evidence="the unsteadiness of the never-scored window itself")
    ax.axhline(4 * dy, color=GOLD, lw=2.4, zorder=3)
    ax.set_xlabel("streamwise cell")
    ax.set_xticks([0, 50])
    ax.set_yticks([0.0, 0.5])
    ax.set_yticklabels([])
    ax.set_title("$u$ r.m.s., 46 frames", pad=8)
    cb3 = fig.colorbar(im3, cax=fig.add_axes(RC_C))
    cb3.set_ticks([0.0, rms_lim])
    cb3.outline.set_linewidth(1.35)
    cb3.ax.tick_params(width=1.35)
    for coll in list(cb3.ax.collections):
        coll.set_linewidth(1.35)

    # ---------------- b: the wall-resolved mean profile -----------------------
    ax = fig.add_axes(R_D)
    panel_label(ax, "d")
    y_plus = np.asarray(comp["y_plus"], np.float64)
    uplus = prof / u_tau
    (pl,) = ax.semilogx(y_plus, uplus, "-", color=BLUE, lw=1.6, zorder=3,
                        label="record mean")
    if A.phase == "final":
        bind_artist(
            fig, pl, artist_id="fig14.d.mean_profile", panel="d",
            source_refs=[{"kind": "npz", "path": CTX, "key": "u_mean_profile"},
                         {"kind": "npz", "path": CTX, "key": "u_tau"},
                         {"kind": "npz", "path": NPZC, "key": "y_plus"}],
            source_payload=[np.asarray(ctx["u_mean_profile"]),
                            np.asarray(ctx["u_tau"]),
                            np.asarray(comp["y_plus"])],
            expected_payload={"type": "Line2D", "x": y_plus, "y": uplus},
            transform="divide the eval-window mean profile by u_tau, plot "
                      "against the wall-unit coordinate",
            evidence="both walls and all 46 held-out frames enter the mean")
    yv = y_plus[y_plus <= 11.0]
    ax.semilogx(yv, yv, ls=(0, (3, 2.2)), color=GREY, lw=1.35, zorder=2,
                label="$U^+\!=\!y^+$")
    ax.axvspan(y_plus[0], 3.2, color=GOLD, alpha=0.25, zorder=1)
    ax.set_xlim(y_plus[0], 200)
    ax.set_xticks([1, 10, 100])
    ax.set_ylim(0, 21)
    ax.set_yticks([0, 10, 20])
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$U^+$")
    ax.set_title("mean profile; slot shaded")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4,
              handlelength=1.3, labelspacing=0.3)

    # ---------------- c: the traction signal entering the slot ----------------
    ax = fig.add_axes(R_E)
    panel_label(ax, "e")
    xcells = np.arange(np.asarray(ctx["tau_walls"]).shape[1], dtype=float)
    for wi, (col, lab) in enumerate((("#263238", "lower"),
                                     (GREY, "upper"))):
        tw = 1e3 * np.asarray(ctx["tau_walls"], np.float64)[wi]
        (tl,) = ax.plot(xcells, tw, "-", color=col, lw=1.5, zorder=3,
                        label=lab)
        if A.phase == "final":
            bind_artist(
                fig, tl, artist_id=f"fig14.e.tau_wall{wi}", panel="e",
                source_refs=[{"kind": "npz", "path": CTX, "key": "tau_walls",
                              "slice": [str(wi), ":"]}],
                source_payload=[np.asarray(ctx["tau_walls"])[wi]],
                expected_payload={"type": "Line2D", "x": xcells,
                                  "y": tw},
                transform="plot the first-cell viscous wall traction of the "
                          "held-out frame along the wall, x1000",
                evidence="the record's own traction: the strongest arm and "
                         "the training modality of the slot")
    ax.set_xlabel("streamwise cell")
    ax.set_ylabel(r"$\tau_w\times10^{3}$")
    ax.set_title("traction entering the slot")
    ax.set_xlim(0, 99)
    ax.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.9,
              edgecolor="none", handletextpad=0.2, handlelength=0.9,
              labelspacing=0.2, borderaxespad=0.1)

    # ---------------- d: the dose-response curve -----------------------------
    ax = fig.add_axes(R_G)
    panel_label(ax, "g", x=-0.11)
    curve = dose["curve"]
    xs = [e["fidelity"] for e in curve]
    ys = [e["gain"] for e in curve]
    (curve_line,) = ax.plot(xs, ys, "-", color=GREY, lw=1.35, zorder=1,
                            label="reference interpolant")
    if A.phase == "final":
        bind_artist(
            fig, curve_line,
            artist_id="fig14.g.dose_response_curve",
            panel="g",
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
    band_patch = ax.fill_between(xs, np.array(ys) - m5, np.array(ys) + m5,
                                 color=LGREY, alpha=0.55, zorder=0,
                                 label=f"margin $\\pm m_5$")
    ax.plot(xs[:-1], ys[:-1], "o", color=BLUE, ms=4, zorder=3)
    ax.plot([1.0], [ys[-1]] if xs[-1] == 1.0 else [curve[-1]["gain"]], "s",
            color=BLUE, ms=5, zorder=3, label="exact (ceiling)")
    pts = [("closure", dose["closure_point"], RED, "D", "physics closure"),
           ("learned_dataonly", dose["learned_dataonly_point"], PURP, "v",
            "data-only"),
           ("equilibrium", dose["equilibrium_point"], GOLD, "^",
            "equilibrium")]
    for _, p, col, mk, lab in pts:
        if p["gain"] is not None:
            ax.plot(p["fidelity"], p["gain"], mk, color=col, ms=5.5, zorder=4,
                    label=lab)
    mn = c(matched)
    if mn:
        ax.plot(fid[matched], mn["delta"], "P", color=GREEN, ms=6, zorder=4,
                label="matched noise")
    for nm, mk in (("wrong_time", "x"), ("shuffle_z", "+")):
        e = c(nm)
        if e:
            ax.plot(fid[nm], e["delta"], mk, color="k", ms=5, zorder=4,
                    label=nm.replace("_", "-"))
    ax.axhline(0, color="k", lw=1.35)
    ax.axvline(xs[0], color=GREY, lw=1.35, ls=(0, (3, 2.2)), zorder=0)
    ax.text(xs[0] + 0.035, -0.058, "measured reference domain",
            fontsize=12.0, color=GREY, ha="left", va="center")
    ax.set_xlim(-0.58, 1.06)
    ax.set_ylim(-0.085, 0.20)
    ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([-0.05, 0.0, 0.05, 0.10, 0.15])
    ax.set_xlabel("measured traction fidelity, centred $R^2(\\tau)$")
    ax.set_ylabel("gain over absent")
    ax.set_title("field gain against measured traction fidelity")
    # arms are identified by the shared colour code of panels c/d and the
    # caption; the in-panel legend carries only the reference elements
    ax.legend(handles=[curve_line, band_patch], loc="upper left",
              frameon=False, handletextpad=0.4, handlelength=1.3,
              labelspacing=0.3)

    # ---------------- e: wall-normal reach curve ------------------------------
    ax = fig.add_axes(R_F)
    panel_label(ax, "f")
    regions = ["buffer_yp_lt30", "log_yp30_100", "outer_yp_gt100"]
    rx = np.array([15.0, 55.0, 140.0])
    for arm, col, mk, lab in (("exact", BLUE, "s", "exact"),
                              ("closure", RED, "D", "closure")):
        m = [c(arm, region=r)["delta"] for r in regions]
        lo = [c(arm, region=r)["boot_ci95"][0] for r in regions]
        hi = [c(arm, region=r)["boot_ci95"][1] for r in regions]
        eb = ax.errorbar(rx, m, yerr=[np.array(m) - lo,
                                      np.array(hi) - np.array(m)],
                         marker=mk, color=col, ms=6, capsize=2.5, lw=1.5,
                         elinewidth=1.35, markeredgecolor="k",
                         markeredgewidth=1.0, label=lab, zorder=3)
        if A.phase == "final":
            bind_artist(
                fig, eb, artist_id=f"fig14.f.reach_{arm}", panel="f",
                source_refs=[{"kind": "json",
                              "path": "fig_slot/e2_slot_channel_final_results.json",
                              "key": f"contrasts.flow_matching|{arm}-absent|{r}.delta"}
                             for r in regions]
                + [{"kind": "json",
                    "path": "fig_slot/e2_slot_channel_final_results.json",
                    "key": f"contrasts.flow_matching|{arm}-absent|{r}.boot_ci95"}
                   for r in regions],
                source_payload=m + [[l, h] for l, h in zip(lo, hi)],
                expected_payload=expected_errorbar_payload(rx, m, lo, hi),
                transform="plot each registered band gain at a representative "
                          "wall-unit position with its bootstrap interval",
                evidence="the gain decays away from the wall in both arms")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=1.35)
    for edge in (30.0, 100.0):
        ax.axvline(edge, color=LGREY, lw=1.35, zorder=0)
    ax.set_xlim(8, 210)
    ax.set_xticks([10, 100])
    ax.set_xticklabels(["10", "100"])
    ax.set_ylim(-0.03, 0.62)
    ax.set_yticks([0.0, 0.3, 0.6])
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("gain over absent")
    ax.set_title("wall-normal reach")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.3,
              handlelength=1.2, labelspacing=0.25, borderaxespad=0.1)

    # ---------------- c: primary block inference ------------------------------
    ax = fig.add_axes(R_H)
    panel_label(ax, "h")
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
        t_text = ax.text(0.215, i, f"t={e['primary']['t']:.1f}",
                         va="center", ha="left", fontsize=12.0, color=col)
        if A.phase == "final" and nm == "closure":
            bind_artist(
                fig, t_text,
                artist_id="fig14.h.closure_primary_t",
                panel="h",
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
    ax.axvline(0, color="k", lw=1.35)
    ax.set_xlim(-0.01, 0.35)
    ax.set_xticks([0.0, 0.1, 0.2])
    ax.set_yticks(range(len(names)), clabs)
    ax.invert_yaxis()
    ax.set_xlabel("gain over absent")
    ax.set_title(f"calibrated $t^* = {tstar}$")

    # -------------- g: the buffer-band turbulent shear is tracked -------------
    ax = fig.add_axes(R_I)
    panel_label(ax, "i", x=-0.14, y=1.08)
    NPZC_PATH = "fig_slot/e2_slot_channel_final_components.npz"
    fm_seeds = [str(x) for x in R["_meta"]["fm_seeds"]]
    idxs = np.asarray(comp["eval_idx"], int)
    ft = np.asarray(comp[f"flow_matching|closure|{fm_seeds[0]}|flux_true_buffer"],
                    float)
    (tl,) = ax.plot(idxs, 1e4 * ft, "-o", color="#263238", lw=1.4, ms=3.6,
                    zorder=2, label="record")
    if A.phase == "final":
        bind_artist(
            fig, tl, artist_id="fig14.i.flux_truth", panel="i",
            source_refs=[
                {"kind": "npz", "path": NPZC_PATH,
                 "key": f"flow_matching|closure|{fm_seeds[0]}|flux_true_buffer"},
                {"kind": "npz", "path": NPZC_PATH, "key": "eval_idx"}],
            source_payload=[np.asarray(
                comp[f"flow_matching|closure|{fm_seeds[0]}|flux_true_buffer"]),
                np.asarray(comp["eval_idx"])],
            expected_payload={"type": "Line2D", "x": idxs.astype(float),
                              "y": 1e4 * ft},
            transform="plot the record's buffer-band spanwise-mean turbulent "
                      "shear at each held-out frame, x1e4",
            evidence="the registered C9 engineering endpoint's target series")
    for arm, col, mk, lab in (("absent", GREY, "^", "absent"),
                              ("closure", RED, "D", "closure")):
        pred = np.mean([np.asarray(
            comp[f"flow_matching|{arm}|{sd}|flux_pred_buffer"], float)
            for sd in fm_seeds], axis=0)
        (pl2,) = ax.plot(idxs, 1e4 * pred, mk, ms=4.6, color=col,
                         markeredgecolor="k", markeredgewidth=0.9, ls="none",
                         zorder=3, label=lab)
        if A.phase == "final":
            bind_artist(
                fig, pl2, artist_id=f"fig14.i.flux_{arm}", panel="i",
                source_refs=[{"kind": "npz", "path": NPZC_PATH,
                              "key": f"flow_matching|{arm}|{sd}|flux_pred_buffer"}
                             for sd in fm_seeds]
                + [{"kind": "npz", "path": NPZC_PATH, "key": "eval_idx"}],
                source_payload=[np.asarray(
                    comp[f"flow_matching|{arm}|{sd}|flux_pred_buffer"])
                    for sd in fm_seeds] + [np.asarray(comp["eval_idx"])],
                expected_payload={"type": "Line2D", "x": idxs.astype(float),
                                  "y": 1e4 * pred},
                transform="average the reconstructed buffer-band turbulent "
                          "shear over the two seeds per held-out frame, x1e4",
                evidence="strictly outside every conditioning and read row")
    ax.set_xlabel("record index")
    ax.set_ylabel("$\\langle u'v'\\rangle\\times10^{4}$")
    ax.set_title("buffer-band shear tracking")
    all_g = 1e4 * np.concatenate([ft] + [np.mean([np.asarray(
        comp[f"flow_matching|{a}|{sd}|flux_pred_buffer"], float)
        for sd in fm_seeds], axis=0) for a in ("absent", "closure")])
    sp_g = all_g.max() - all_g.min()
    ax.set_ylim(all_g.min() - 0.10 * sp_g, all_g.max() + 0.55 * sp_g)
    ax.set_yticks([-15, -10])
    ax.set_xlim(-12, 462)
    ax.set_xticks([0, 200, 400])
    ax.legend(frameon=False, loc="upper right", ncol=3, handletextpad=0.2,
              handlelength=1.0, columnspacing=0.5, borderaxespad=0.05)

    # ---------------- h: gain in every frame ----------------------------------
    ax = fig.add_axes(R_J)
    panel_label(ax, "j", x=-0.075, y=1.08)
    diff_seeds = [str(x) for x in R["_meta"]["diff_seeds"]]
    for fam, seeds, col, lab in (("flow_matching", fm_seeds, RED,
                                  "flow matching"),
                                 ("diffusion", diff_seeds, BLUE, "diffusion")):
        for k2, sd in enumerate(seeds):
            g_cl = np.asarray(comp[f"{fam}|closure|{sd}|whole_scorable"], float)
            g_ab = np.asarray(comp[f"{fam}|absent|{sd}|whole_scorable"], float)
            dser = g_cl - g_ab
            (ln,) = ax.plot(idxs, dser, marker="o", ms=3.0, lw=1.35, color=col,
                            alpha=1.0 if k2 == 0 else 0.55,
                            label=lab if k2 == 0 else None, zorder=3)
            if A.phase == "final":
                bind_artist(
                    fig, ln, artist_id=f"fig14.j.frames_{fam}_{sd}", panel="j",
                    source_refs=[
                        {"kind": "npz", "path": NPZC_PATH,
                         "key": f"{fam}|closure|{sd}|whole_scorable"},
                        {"kind": "npz", "path": NPZC_PATH,
                         "key": f"{fam}|absent|{sd}|whole_scorable"},
                        {"kind": "npz", "path": NPZC_PATH, "key": "eval_idx"}],
                    source_payload=[
                        np.asarray(comp[f"{fam}|closure|{sd}|whole_scorable"]),
                        np.asarray(comp[f"{fam}|absent|{sd}|whole_scorable"]),
                        np.asarray(comp["eval_idx"])],
                    expected_payload={"type": "Line2D",
                                      "x": idxs.astype(float), "y": dser},
                    transform="subtract the absent arm's per-frame "
                              "whole-scorable score from the closure arm's",
                    evidence="every frame, every seed and both families are "
                             "positive")
    ax.axhline(0, color="k", lw=1.35)
    ax.set_xlabel("record index")
    ax.set_ylabel("per-frame $\\Delta R^2$")
    ax.set_title("gain in every frame")
    ax.set_ylim(-0.02, 0.30)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlim(-12, 462)
    ax.set_xticks([0, 200, 400])
    ax.legend(frameon=False, loc="upper right", handletextpad=0.3,
              handlelength=1.2, labelspacing=0.25, borderaxespad=0.1)

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
