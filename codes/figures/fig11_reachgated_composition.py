#!/usr/bin/env python3
"""Figure: the positive-controlled two-regime evaluation (paper Fig. 7).

Redesign 2026-08-02 v3: field-dominant.  Rows 1-2 show the separating hill
as a geometry (test-window mean with its recirculation bubble, an
instantaneous frame, the fluctuation-intensity map) beside the closure's
wall-traction scatter; row 3 shows the reversed-time cube unit's floor
traction maps beside the failing hill control; the stats rows carry the
cube control, the near-wall ladder, the reach decay and the wall-load tracking.
Every value is bound to an exact key of the frozen producer outputs staged
with the source data.
"""

import numpy as np

from _submission import (
    BLUE,
    GOLD,
    GREY,
    LIGHT_BLUE,
    LIGHT_RED,
    RED,
    SOURCE,
    bind_artist,
    configure,
    expected_errorbar_payload,
    load_json,
    plt,
    save,
)

configure()
plt.rcParams.update({
    "font.size": 12.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 12.0,
})

GREEN = "#2e8b57"

D3 = load_json("fig11_reachgated/DECISION_RULE_OUTCOME_REPAIR3.json")
D2 = load_json("fig11_reachgated/DECISION_RULE_OUTCOME_REPAIR2.json")
R = load_json("fig11_reachgated/e2_cube_reachgated_results.json")
NPZ = "fig11_reachgated/e2_cube_reachgated_components.npz"
HILL = "fig11_reachgated/hill_context_frame.npz"
GH = "fig11_reachgated/e2_grouped_hills_components.npz"
D3_SRC = "fig11_reachgated/DECISION_RULE_OUTCOME_REPAIR3.json"
D2_SRC = "fig11_reachgated/DECISION_RULE_OUTCOME_REPAIR2.json"
R_SRC = "fig11_reachgated/e2_cube_reachgated_results.json"
npz = np.load(SOURCE / NPZ, allow_pickle=False)
hill = np.load(SOURCE / HILL, allow_pickle=False)
gh = np.load(SOURCE / GH, allow_pickle=False)


def panel_label(ax, label, *, x=-0.13, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


def lv(fam, reg, arm):
    return D3["levels"][f"{fam}|{reg}"][f"level|{arm}"]


def dl(fam, reg, key):
    return D3["deltas"][f"{fam}|{reg}"][key]


fig = plt.figure(figsize=(8.75, 10.95))
# manual layout, no constrained engine
AX_A = [0.055, 0.888, 0.400, 0.092]
AX_B = [0.487, 0.888, 0.400, 0.092]
AX_UCB = [0.905, 0.888, 0.011, 0.086]
AX_C = [0.055, 0.760, 0.400, 0.092]
AX_ICB = [0.470, 0.760, 0.011, 0.092]
AX_D = [0.660, 0.745, 0.190, 0.118]
T_X0, T_W, T_GAP, T_Y, T_H = 0.055, 0.142, 0.012, 0.562, 0.118
AX_TCB = [0.515, 0.562, 0.011, 0.118]
AX_F = [0.745, 0.559, 0.200, 0.112]
AX_G = [0.150, 0.352, 0.135, 0.130]
AX_H = [0.455, 0.352, 0.165, 0.130]
AX_I = [0.750, 0.352, 0.225, 0.130]
AX_J = [0.095, 0.055, 0.335, 0.215]
AX_K = [0.520, 0.055, 0.465, 0.215]

# --------------------------------- a,b: the separating regime as a geometry
h_val = float(hill["h"])
mu0, sd0 = float(hill["mu"][0]), float(hill["sd"][0])
solid = np.asarray(hill["mask"], np.float64) < 0.5
hx = np.asarray(hill["grid_x"], np.float64) / h_val
hy = np.asarray(hill["grid_y"], np.float64) / h_val
h_ext = (float(hx.min()), float(hx.max()), float(hy.min()), float(hy.max()))
u_inst = np.asarray(hill["u_std"], np.float64) * sd0 + mu0
u_mean = np.asarray(hill["u_mean_std"], np.float64) * sd0 + mu0
h_lim = float(np.round(np.abs(u_inst[~solid]).max(), 2))
hill_cmap = plt.get_cmap("RdBu_r").copy()
hill_cmap.set_bad("#d3d7da")

axA = fig.add_axes(AX_A)
mean_disp = np.ma.array(u_mean, mask=solid)
imA = axA.imshow(mean_disp, origin="lower", extent=h_ext, cmap=hill_cmap,
                 vmin=-h_lim, vmax=h_lim, interpolation="nearest",
                 aspect="auto", zorder=1)
bind_artist(
    fig, imA, artist_id="fig11.a.hill_mean", panel="a",
    source_refs=[{"kind": "npz", "path": HILL, "key": k}
                 for k in ("u_mean_std", "mask", "mu", "sd")],
    source_payload=[np.asarray(hill["u_mean_std"]), np.asarray(hill["mask"]),
                    np.asarray(hill["mu"]), np.asarray(hill["sd"])],
    expected_payload={"type": "AxesImage", "array": mean_disp,
                      "extent": list(h_ext), "clim": [-h_lim, h_lim]},
    transform=("de-standardise the test-window mean of the stored streamwise "
               "channel (plane 0, physical times 590-959) and mask the solid "
               "region below the curved wall"),
    evidence="public periodic-hill record; mean over the grouped test window",
)
XX, YY = np.meshgrid(hx, hy)
axA.contour(XX, YY, np.ma.array(u_mean, mask=solid), levels=[0.0],
            colors="black", linestyles="--", linewidths=1.6, zorder=3)
rev = np.argwhere((~solid) & (u_mean < 0))
rc_x = float(hx[np.clip(int(np.round(rev[:, 1].mean())), 0, len(hx) - 1)])
rc_y = float(hy[np.clip(int(np.round(rev[:, 0].mean())), 0, len(hy) - 1)])
axA.annotate("recirculation", xy=(rc_x, rc_y), xytext=(rc_x + 1.6, rc_y + 1.5),
             fontsize=12, color="black", ha="left", va="center",
             arrowprops={"arrowstyle": "->", "color": "black",
                         "linewidth": 1.3})
axA.set_title("mean flow, test window", fontsize=12, pad=6)
axA.set_xticklabels([])
axA.set_ylabel("$y/h$")
panel_label(axA, "a", x=-0.105, y=1.17)

axB = fig.add_axes(AX_B)
inst_disp = np.ma.array(u_inst, mask=solid)
imB = axB.imshow(inst_disp, origin="lower", extent=h_ext, cmap=hill_cmap,
                 vmin=-h_lim, vmax=h_lim, interpolation="nearest",
                 aspect="auto", zorder=1)
bind_artist(
    fig, imB, artist_id="fig11.b.hill_inst", panel="b",
    source_refs=[{"kind": "npz", "path": HILL, "key": k}
                 for k in ("u_std", "mask", "mu", "sd")],
    source_payload=[np.asarray(hill["u_std"]), np.asarray(hill["mask"]),
                    np.asarray(hill["mu"]), np.asarray(hill["sd"])],
    expected_payload={"type": "AxesImage", "array": inst_disp,
                      "extent": list(h_ext), "clim": [-h_lim, h_lim]},
    transform=("de-standardise the stored streamwise channel of the first "
               "test time (plane 0, physical time 590) and mask the solid "
               "region below the curved wall"),
    evidence="public periodic-hill record; the frame belongs to the test window",
)
axB.set_title("instantaneous, first test time", fontsize=12, pad=6)
axB.set_xticklabels([])
axB.set_yticklabels([])
panel_label(axB, "b", x=-0.045, y=1.17)

ucb = fig.colorbar(imB, cax=fig.add_axes(AX_UCB))
ucb.set_ticks([-h_lim, 0.0, h_lim])
fig.axes[-1].set_title("$u$", fontsize=12, pad=7)
ucb.outline.set_linewidth(1.35)
ucb.ax.tick_params(width=1.35)
for coll in list(ucb.ax.collections):
    coll.set_linewidth(1.35)

# ------------------------- c: where the unsteadiness lives (shear layer)
axCI = fig.add_axes(AX_C)
u_rms = np.asarray(hill["u_rms_std"], np.float64) * sd0
r_lim = float(np.round(np.percentile(u_rms[~solid], 99.5), 2))
rms_cmap = plt.get_cmap("plasma").copy()
rms_cmap.set_bad("#d3d7da")
rms_disp = np.ma.array(u_rms, mask=solid)
imC = axCI.imshow(rms_disp, origin="lower", extent=h_ext, cmap=rms_cmap,
                  vmin=0.0, vmax=r_lim, interpolation="nearest",
                  aspect="auto", zorder=1)
bind_artist(
    fig, imC, artist_id="fig11.c.hill_rms", panel="c",
    source_refs=[{"kind": "npz", "path": HILL, "key": k}
                 for k in ("u_rms_std", "mask", "sd")],
    source_payload=[np.asarray(hill["u_rms_std"]), np.asarray(hill["mask"]),
                    np.asarray(hill["sd"])],
    expected_payload={"type": "AxesImage", "array": rms_disp,
                      "extent": list(h_ext), "clim": [0.0, r_lim]},
    transform=("multiply the test-window standard deviation of the stored "
               "streamwise channel by its channel scale and mask the solid "
               "region below the curved wall"),
    evidence="public periodic-hill record; unsteadiness of the grouped test window",
)
axCI.contour(XX, YY, np.ma.array(u_mean, mask=solid), levels=[0.0],
             colors="white", linestyles="--", linewidths=1.6, zorder=3)
axCI.set_title("fluctuation intensity, test window", fontsize=12, pad=6)
axCI.set_xlabel("$x/h$")
axCI.set_ylabel("$y/h$")
panel_label(axCI, "c", x=-0.105, y=1.17)
icb = fig.colorbar(imC, cax=fig.add_axes(AX_ICB))
icb.set_ticks([0.0, r_lim])
fig.axes[-1].set_title("$u_{\\rm rms}$", fontsize=12, pad=7)
icb.outline.set_linewidth(1.35)
icb.ax.tick_params(width=1.35)
for coll in list(icb.ax.collections):
    coll.set_linewidth(1.35)

# ----------------- d: the closure predicts the hill wall traction it never saw
axC = fig.add_axes(AX_D)
tau_true = 1e3 * np.asarray(gh["tau_native_test"])[0, 0]
tau_clo = 1e3 * np.asarray(gh["tau_closure_test"])[0, 0]
tau_eq = 1e3 * np.asarray(gh["tau_eqwm_test"])[0, 0]
for vals, colour, lab, aid in (
        (tau_eq, LIGHT_BLUE, "eq. law", "eqwm"),
        (tau_clo, GOLD, "closure", "closure")):
    pts = axC.plot(tau_true, vals, "o", ms=4.0, color=colour,
                   markeredgecolor="black", markeredgewidth=0.8, ls="none",
                   label=lab, zorder=3)
    bind_artist(
        fig, pts[0], artist_id=f"fig11.d.scatter_{aid}", panel="d",
        source_refs=[
            {"kind": "npz", "path": GH, "key": "tau_native_test",
             "slice": ["0", "0", ":"]},
            {"kind": "npz", "path": GH, "key": f"tau_{aid}_test",
             "slice": ["0", "0", ":"]},
        ],
        source_payload=[np.asarray(gh["tau_native_test"])[0, 0],
                        np.asarray(gh[f"tau_{aid}_test"])[0, 0]],
        expected_payload={"type": "Line2D",
                          "x": np.asarray(tau_true, dtype=float),
                          "y": np.asarray(vals, dtype=float)},
        transform=("plot the predicted against the true streamwise wall "
                   "traction of the 234 curved-wall cells at the first test "
                   "time, x1000"),
        evidence=("the closure reads matching-height state only; the traction "
                  "is predicted on held-out times of the separating wall"),
    )
c_lim = float(np.round(1.05 * max(np.abs(tau_true).max(),
                                  np.abs(tau_clo).max(),
                                  np.abs(tau_eq).max()), 1))
axC.plot([-c_lim, c_lim], [-c_lim, c_lim], color="black", lw=1.4, ls="--",
         zorder=2)
axC.set_xlim(-c_lim, c_lim)
axC.set_ylim(-c_lim, c_lim)
axC.set_xlabel("true $\\tau_x\\times10^{3}$")
axC.set_ylabel("predicted $\\tau_x\\times10^{3}$")
axC.set_title("hill wall traction", fontsize=12)
axC.legend(frameon=False, loc="upper left", handletextpad=0.2,
           labelspacing=0.2, borderaxespad=0.1)
axC.set_xticks([-1, 0, 1])
axC.set_yticks([-1, 0, 1])
panel_label(axC, "d", x=-0.46, y=1.18)

# --------------------- e: the traction maps of the reversed-time cube unit
N_FLOOR = 1728
floor_mask2d = np.ones((48, 48), dtype=bool)
floor_mask2d[12:36, 12:36] = False
TAU_TILES = [
    ("tau_native_test", "true traction"),
    ("tau_closure_test", "closure prediction"),
    ("tau_eqwm_test", "equilibrium"),
]
tau_floor = {}
for key, _ in TAU_TILES:
    arr = np.full((48, 48), np.nan, dtype=np.float64)
    arr[floor_mask2d] = np.asarray(npz[key])[0, 0][:N_FLOOR]
    tau_floor[key] = 1e3 * arr.T
tlim = float(np.round(np.percentile(
    np.abs(np.concatenate([v[np.isfinite(v)] for v in tau_floor.values()])),
    99.5), 1))
tau_cmap = plt.get_cmap("PuOr_r").copy()
tau_cmap.set_bad("#d3d7da")
tau_axes = []
for j, (key, title) in enumerate(TAU_TILES):
    ax = fig.add_axes([T_X0 + j * (T_W + T_GAP), T_Y, T_W, T_H])
    tau_axes.append(ax)
    disp = np.ma.array(tau_floor[key], mask=~floor_mask2d.T)
    tile = ax.imshow(disp, origin="lower", extent=(0.0, 2.0, 0.0, 2.0),
                     cmap=tau_cmap, vmin=-tlim, vmax=tlim,
                     interpolation="nearest", aspect="auto", zorder=1)
    bind_artist(
        fig, tile, artist_id=f"fig11.e.{key.split('_')[1]}", panel="e",
        source_refs=[{"kind": "npz", "path": NPZ, "key": key,
                      "slice": ["0", "0", ":"]}],
        source_payload=[np.asarray(npz[key])[0, 0]],
        expected_payload={"type": "AxesImage", "array": disp,
                          "extent": [0.0, 2.0, 0.0, 2.0],
                          "clim": [-tlim, tlim]},
        transform=("scatter the floor block of the (cell, face) pair vector "
                   "of the first never-tested cube frame back onto the 48x48 "
                   "floor grid in producer order, streamwise component, "
                   "x1000, transpose and mask the cube base"),
        evidence=("reversed-time unit: the closure and generators never saw "
                  "frames 0-299; record index 0 shown"),
    )
    ax.add_patch(plt.Rectangle((0.5, 0.5), 1.0, 1.0, facecolor="none",
                               edgecolor="#263238", linewidth=1.2, zorder=3))
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_title(title, fontsize=12, pad=6)
    if j == 0:
        ax.set_ylabel("$z/h$")
    else:
        ax.set_yticklabels([])
    if j == 1:
        ax.set_xlabel("$x/h$")
tbar = fig.colorbar(tau_axes[0].images[0], cax=fig.add_axes(AX_TCB))
tbar.set_ticks([-tlim, 0.0, tlim])
fig.axes[-1].set_title("$\\tau_x\\!\\times\\!10^{3}$", fontsize=12, pad=7)
tbar.outline.set_linewidth(1.35)
tbar.ax.tick_params(width=1.35)
for coll in list(tbar.ax.collections):
    coll.set_linewidth(1.35)
panel_label(tau_axes[0], "e", x=-0.30, y=1.18)

# --------- f: the hill control, PAIRED against absent, in BOTH families
# The earlier version of this panel plotted only the flow-matching cell, as
# absolute arm levels with marginal intervals.  That hid the comparison that
# actually decides the positive control: the diffusion cell's marginal intervals
# overlap while its PAIRED contrast excludes zero.  Both cells are shown, and
# the quantity plotted is the paired contrast each arm makes against its own
# absent branch.
axD = fig.add_axes(AX_F)
AUD = load_json("fig11_reachgated/hill_positive_control_audit.json")
AUD_SRC = "fig11_reachgated/hill_positive_control_audit.json"
# Intervals come from the hierarchical seed-then-time bootstrap Methods declares,
# the same estimator the text quotes.  The audit file's own ci95 is the earlier
# seed-conditional estimator, which is narrower; plotting it here would have made
# the panel's error bars disagree with every bracket in the Results section.
HIER_SRC = "fig11_reachgated/hill_hierarchical_intervals.json"
HIER = load_json(HIER_SRC)
HIER_TAG = "e2_grouped_hills_band"
f_arms = [("band_oracle", GREEN, "oracle band"),
          ("band_closure", GOLD, "closure band"),
          ("band_fartime", RED, "far-time band")]
f_cells = [("H2", "diffusion"), ("H1", "flow matching")]
ypos, f_vals, f_lo, f_hi, f_cols, f_labs, f_refs = [], [], [], [], [], [], []
# Each family's verdict label sits on a clear row above that family's three bars.
# It used to be drawn across the bars themselves, where it covered the closure-band
# bar in both cells.  The bar rows are left exactly as they were: shifting them to
# non-dyadic values makes Matplotlib return a bar height of 0.6199999999999999 and
# breaks the artist-provenance digest.
f_head_y = []
row = 0
for cell, cname in f_cells:
    f_head_y.append(row - 0.8)
    reg = AUD["cells"][cell]["regions"]["full_srcex"]
    for arm, col, alab in f_arms:
        ypos.append(row)
        hci = (HIER["tags"][HIER_TAG]["cells"][cell]["regions"]
               ["full_srcex"][arm]["ci95_hierarchical"])
        f_vals.append(reg[arm]["delta_vs_absent"])
        f_lo.append(hci[0])
        f_hi.append(hci[1])
        f_cols.append(col)
        f_labs.append(alab)
        f_refs.append((cell, arm))
        row += 1
    row += 0.6
ypos = np.asarray(ypos, dtype=float)
bars = axD.barh(ypos, f_vals, height=0.62, color=f_cols, edgecolor="black",
                linewidth=1.2, zorder=2)
bind_artist(
    fig, bars, artist_id="fig11.f.hill_paired", panel="f",
    source_refs=[{"kind": "json", "path": AUD_SRC,
                  "key": f"cells.{c}.regions.full_srcex.{a}.delta_vs_absent"}
                 for c, a in f_refs],
    source_payload=f_vals,
    expected_payload={"type": "BarContainer",
                      "centres": [v / 2.0 for v in f_vals],
                      "heights": [0.62] * len(f_vals)},
    transform="plot each hill arm's paired contrast against its own absent branch",
    evidence="grouped leakage-free protocol; the control passes in the diffusion "
             "family and fails in the flow-matching family, which neither a "
             "calibrated stochastic sampler nor a finer raster repairs",
)
ci_x, ci_y = [], []
for k, (l, h) in enumerate(zip(f_lo, f_hi)):
    ci_x += [l, h, np.nan]
    ci_y += [ypos[k], ypos[k], np.nan]
ci = axD.plot(ci_x, ci_y, color="black", lw=1.4, zorder=3)
bind_artist(
    fig, ci[0], artist_id="fig11.f.hill_paired_ci", panel="f",
    source_refs=[{"kind": "json", "path": HIER_SRC,
                  "key": (f"tags.{HIER_TAG}.cells.{c}.regions.full_srcex.{a}"
                          ".ci95_hierarchical")}
                 for c, a in f_refs],
    source_payload=[[l, h] for l, h in zip(f_lo, f_hi)],
    expected_payload={"type": "Line2D", "x": np.asarray(ci_x, dtype=float),
                      "y": np.asarray(ci_y, dtype=float)},
    transform="draw the declared hierarchical 95% interval of each paired contrast",
    evidence="seeds resampled with replacement, then a circular moving-block "
             "resample in physical time; the estimator Methods declares and the "
             "text quotes",
)
axD.axvline(0, color="black", lw=1.4, zorder=1)
axD.set_yticks(ypos)
axD.set_yticklabels(f_labs)
axD.invert_yaxis()
for gi, (cell, cname) in enumerate(f_cells):
    ok = AUD["cells"][cell]["positive_control_passes"]
    # the verdict sits on its own clear row above the family's three bars, in
    # black: it must not be read off a colour, and it must not cover a bar
    axD.text(-0.06, f_head_y[gi],
             f"{cname}: control {'passes' if ok else 'fails'}",
             transform=axD.get_yaxis_transform(which="grid"),
             ha="left", va="center", fontsize=12, color="black",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
             zorder=4, clip_on=False)
axD.set_ylim(ypos[-1] + 0.5, f_head_y[0] - 0.5)   # inverted, with room for the labels
axD.set_title("hill: $\\Delta R^2$ vs absent", fontsize=12)
axD.xaxis.set_major_locator(plt.MaxNLocator(3, prune="upper"))
panel_label(axD, "f", x=-0.48, y=1.16)

# ------------------------------------- e: the power gate on the cube unit
axE = fig.add_axes(AX_G)
g_vals, g_lo, g_hi = [], [], []
for fam in ("F", "G"):
    d = dl(fam, "near_srcex", "delta|band_oracle-absent")
    g_vals.append(d["mean"])
    g_lo.append(d["ci95"][0])
    g_hi.append(d["ci95"][1])
ypos = np.arange(2)
bars = axE.barh(ypos, g_vals, height=0.55, color=[GREEN, LIGHT_BLUE],
                edgecolor="black", linewidth=1.2, zorder=2)
bind_artist(
    fig, bars, artist_id="fig11.g.gate", panel="g",
    source_refs=[{"kind": "json", "path": D3_SRC,
                  "key": f"deltas.{fam}|near_srcex.delta|band_oracle-absent.mean"}
                 for fam in ("F", "G")],
    source_payload=g_vals,
    expected_payload={"type": "BarContainer",
                      "centres": [v / 2.0 for v in g_vals],
                      "heights": [0.55, 0.55]},
    transform="plot the oracle-band positive control of both families as horizontal bars",
    evidence="no interface conclusion is drawn unless this control passes; both families pass",
)
ci_x, ci_y = [], []
for k, (l, h) in enumerate(zip(g_lo, g_hi)):
    ci_x += [l, h, np.nan]
    ci_y += [k, k, np.nan]
ci = axE.plot(ci_x, ci_y, color="black", lw=1.4, zorder=3)
bind_artist(
    fig, ci[0], artist_id="fig11.g.gate_ci", panel="g",
    source_refs=[{"kind": "json", "path": D3_SRC,
                  "key": f"deltas.{fam}|near_srcex.delta|band_oracle-absent.ci95"}
                 for fam in ("F", "G")],
    source_payload=[[l, h] for l, h in zip(g_lo, g_hi)],
    expected_payload={"type": "Line2D", "x": np.asarray(ci_x, dtype=float),
                      "y": np.asarray(ci_y, dtype=float)},
    transform="draw the conservative interval of each family's positive control",
    evidence="intervals exclude zero in both families",
)
axE.axvline(0, color="black", lw=1.4, zorder=1)
axE.set_yticks(ypos)
axE.set_yticklabels(["flow\nmatching", "diffusion"])
axE.invert_yaxis()
axE.set_title("cube: control passes", fontsize=12)
axE.xaxis.set_major_locator(plt.MaxNLocator(3, prune="upper"))
panel_label(axE, "g", x=-0.62, y=1.16)

# --------------------------------------------- f: near-wall arm ladder
axF = fig.add_axes(AX_H)
order = [("absent", GREY, "absent"),
         ("tau_fartime", RED, "wrong instant"),
         ("tau_shuffle", LIGHT_RED, "shuffled"),
         ("tau_eqwm", LIGHT_BLUE, "eq. law"),
         ("tau_native", BLUE, "oracle traction"),
         ("tau_closure", GOLD, "closure"),
         ("band_oracle", GREEN, "oracle band")]
l_vals = [lv("F", "near_srcex", a)["mean"] for a, _, _ in order]
l_lo = [lv("F", "near_srcex", a)["ci95"][0] for a, _, _ in order]
l_hi = [lv("F", "near_srcex", a)["ci95"][1] for a, _, _ in order]
ypos = np.arange(len(order))
bars = axF.barh(ypos, l_vals, height=0.66,
                color=[c for _, c, _ in order], edgecolor="black",
                linewidth=1.2, zorder=2)
bind_artist(
    fig, bars, artist_id="fig11.h.ladder", panel="h",
    source_refs=[{"kind": "json", "path": D3_SRC,
                  "key": f"levels.F|near_srcex.level|{a}.mean"}
                 for a, _, _ in order],
    source_payload=l_vals,
    expected_payload={"type": "BarContainer",
                      "centres": [v / 2.0 for v in l_vals],
                      "heights": [0.66] * len(l_vals)},
    transform="plot the near-wall absolute skill of every arm as horizontal bars",
    evidence="flow matching, two seeds; controls fall below absent",
)
ci_x, ci_y = [], []
for k, (l, h) in enumerate(zip(l_lo, l_hi)):
    ci_x += [l, h, np.nan]
    ci_y += [k, k, np.nan]
ci = axF.plot(ci_x, ci_y, color="black", lw=1.4, zorder=3)
bind_artist(
    fig, ci[0], artist_id="fig11.h.ladder_ci", panel="h",
    source_refs=[{"kind": "json", "path": D3_SRC,
                  "key": f"levels.F|near_srcex.level|{a}.ci95"}
                 for a, _, _ in order],
    source_payload=[[l, h] for l, h in zip(l_lo, l_hi)],
    expected_payload={"type": "Line2D", "x": np.asarray(ci_x, dtype=float),
                      "y": np.asarray(ci_y, dtype=float)},
    transform="draw the conservative interval of each arm",
    evidence="moving-block intervals conditional on the record",
)
axF.axvline(0, color="black", lw=1.4, zorder=1)
axF.set_yticks(ypos)
axF.set_yticklabels([lab for _, _, lab in order])
axF.invert_yaxis()
axF.set_title("near-wall skill, $R^2$", fontsize=12)
axF.xaxis.set_major_locator(plt.MaxNLocator(3, prune="upper"))
panel_label(axF, "h", x=-0.70, y=1.16)

# ------------------------------------------------------ g: reach decay
axG = fig.add_axes(AX_I)
regs = [("near015_srcex", "$\\leq0.15$"), ("near_srcex", "$\\leq0.5$"),
        ("full_srcex", "whole"), ("outer_srcex", "$>0.5$")]
xs = np.arange(len(regs))
for fam, col, mk, nm in (("F", GOLD, "o", "flow matching"),
                         ("G", LIGHT_BLUE, "s", "diffusion")):
    m = [dl(fam, r, "delta|tau_closure-absent")["mean"] for r, _ in regs]
    lo = [dl(fam, r, "delta|tau_closure-absent")["ci95"][0] for r, _ in regs]
    hi = [dl(fam, r, "delta|tau_closure-absent")["ci95"][1] for r, _ in regs]
    eb = axG.errorbar(xs, m, yerr=[np.array(m) - lo, np.array(hi) - np.array(m)],
                      marker=mk, color=col, ms=6, capsize=2.5, lw=1.4,
                      elinewidth=1.3, markeredgecolor="black",
                      markeredgewidth=1.1, label=nm, zorder=3)
    bind_artist(
        fig, eb, artist_id=f"fig11.i.reach_{fam}", panel="i",
        source_refs=[{"kind": "json", "path": D3_SRC,
                      "key": f"deltas.{fam}|{r}.delta|tau_closure-absent.mean"}
                     for r, _ in regs]
        + [{"kind": "json", "path": D3_SRC,
            "key": f"deltas.{fam}|{r}.delta|tau_closure-absent.ci95"}
           for r, _ in regs],
        source_payload=m + [[l, h] for l, h in zip(lo, hi)],
        expected_payload=expected_errorbar_payload(xs, m, lo, hi),
        transform="plot the frozen closure contrast in nested wall-distance shells",
        evidence="the gain decays monotonically away from the wall",
    )
m = [dl("F", r, "delta|band_oracle-absent")["mean"] for r, _ in regs]
ceil = axG.plot(xs, m, marker="^", ls="--", color=GREEN, ms=6, lw=1.4,
                markeredgecolor="black", markeredgewidth=1.1,
                label="oracle band", zorder=2)
bind_artist(
    fig, ceil[0], artist_id="fig11.i.reach_ceiling", panel="i",
    source_refs=[{"kind": "json", "path": D3_SRC,
                  "key": f"deltas.F|{r}.delta|band_oracle-absent.mean"}
                 for r, _ in regs],
    source_payload=m,
    expected_payload={"type": "Line2D", "x": np.asarray(xs, dtype=float),
                      "y": np.asarray(m, dtype=float)},
    transform="plot the oracle-band ceiling in the same shells",
    evidence="the contrast tracks its ceiling at every distance",
)
axG.axhline(0, color="black", lw=1.4, zorder=1)
axG.set_xticks(xs)
axG.set_xticklabels([n for _, n in regs])
axG.set_xlabel("$d/h$")
axG.set_ylabel("$\\Delta R^2$")
axG.set_ylim(-0.045, 0.40)
axG.set_yticks([0.0, 0.2, 0.4])
axG.set_title("reach of the correction", fontsize=12)
axG.legend(frameon=False, loc="upper right", handletextpad=0.3,
           labelspacing=0.2, handlelength=1.3, borderaxespad=0.1)
panel_label(axG, "i", x=-0.26, y=1.16)

# ------------------------------------------------- j: gain in every frame
axJ = fig.add_axes(AX_J)
sst = np.asarray(npz["sst_test|near_srcex"], dtype=float)
frames = np.arange(len(sst))
for fam, seeds, col, nm in (("F", (8801, 8802), GOLD, "flow matching"),
                            ("G", (9901, 9902), LIGHT_BLUE, "diffusion")):
    for k, sd_ in enumerate(seeds):
        sse_a = np.asarray(npz[f"{fam}|test|{sd_}|sse|absent|near_srcex"],
                           dtype=float)
        sse_c = np.asarray(npz[f"{fam}|test|{sd_}|sse|tau_closure|near_srcex"],
                           dtype=float)
        diff = (1 - sse_c / sst) - (1 - sse_a / sst)
        ln = axJ.plot(frames, diff, marker="o", ms=3.2, lw=1.35, color=col,
                      alpha=1.0 if k == 0 else 0.55,
                      label=nm if k == 0 else None, zorder=3)
        bind_artist(
            fig, ln[0], artist_id=f"fig11.j.frames_{fam}_{sd_}", panel="j",
            source_refs=[
                {"kind": "npz", "path": NPZ,
                 "key": f"{fam}|test|{sd_}|sse|tau_closure|near_srcex"},
                {"kind": "npz", "path": NPZ,
                 "key": f"{fam}|test|{sd_}|sse|absent|near_srcex"},
                {"kind": "npz", "path": NPZ, "key": "sst_test|near_srcex"},
            ],
            source_payload=[
                np.asarray(npz[f"{fam}|test|{sd_}|sse|tau_closure|near_srcex"]),
                np.asarray(npz[f"{fam}|test|{sd_}|sse|absent|near_srcex"]),
                np.asarray(npz["sst_test|near_srcex"]),
            ],
            expected_payload={"type": "Line2D",
                              "x": frames.astype(float), "y": diff},
            transform=("convert per-frame SSE/SST to R^2 and subtract the "
                       "absent arm from the closure arm"),
            evidence="paired within-seed difference on 25 disjoint frames",
        )
axJ.axhline(0, color=RED, lw=1.4, zorder=2)
axJ.set_xlabel("held-out frame")
axJ.set_ylabel("per-frame $\\Delta R^2$")
axJ.set_title("gain in every frame", fontsize=12)
axJ.set_ylim(-0.02, 0.30)
axJ.set_yticks([0.0, 0.1, 0.2, 0.3])
axJ.set_xlim(-1, 25.5)
axJ.set_xticks([0, 10, 20])
axJ.legend(frameon=False, loc="upper right", handletextpad=0.3,
           labelspacing=0.2, handlelength=1.3, borderaxespad=0.1)
panel_label(axJ, "j", x=-0.215, y=1.15)

# --------------------------------- k: wall-load tracking on the 25 frames
axI = fig.add_axes(AX_K)
fx_t = np.asarray(npz["fx_target_test"], dtype=float)
idx = np.asarray(npz["test_idx"], dtype=int)
truth = axI.plot(idx, 1e3 * fx_t, color="#263238", lw=1.4, marker="o", ms=4.2,
                 zorder=2, label="true load")
bind_artist(
    fig, truth[0], artist_id="fig11.k.fx_truth", panel="k",
    source_refs=[{"kind": "npz", "path": NPZ, "key": "fx_target_test"},
                 {"kind": "npz", "path": NPZ, "key": "test_idx"}],
    source_payload=[np.asarray(npz["fx_target_test"]),
                    np.asarray(npz["test_idx"])],
    expected_payload={"type": "Line2D", "x": idx.astype(float),
                      "y": 1e3 * fx_t},
    transform="plot the true streamwise viscous wall force of the 25 frames, x1000",
    evidence="first-cell viscous surrogate; physical walls only",
)
for arm, colour, mk, lab in (("absent", LIGHT_RED, "^", "absent"),
                             ("tau_closure", GOLD, "D", "closure")):
    fx_arm = np.mean([np.asarray(npz[f"F|test|{s}|fx|{arm}"], dtype=float)
                      for s in (8801, 8802)], axis=0)
    corr = R["wall_force"][f"F|{arm}"]["corr"]
    pts = axI.plot(idx, 1e3 * fx_arm, mk, ms=5.2, color=colour,
                   markeredgecolor="black", markeredgewidth=1.0, ls="none",
                   zorder=3, label=f"{lab} ($r={corr:+.2f}$)")
    bind_artist(
        fig, pts[0], artist_id=f"fig11.k.fx_{arm}", panel="k",
        source_refs=[{"kind": "npz", "path": NPZ, "key": f"F|test|{s}|fx|{arm}"}
                     for s in (8801, 8802)]
        + [{"kind": "npz", "path": NPZ, "key": "test_idx"},
           {"kind": "json", "path": R_SRC, "key": f"wall_force.F|{arm}.corr"}],
        source_payload=[np.asarray(npz[f"F|test|{s}|fx|{arm}"])
                        for s in (8801, 8802)]
        + [np.asarray(npz["test_idx"]), corr],
        expected_payload={"type": "Line2D", "x": idx.astype(float),
                          "y": 1e3 * fx_arm},
        transform=("average the reconstructed wall force over the two "
                   "flow-matching seeds at each never-tested frame, x1000"),
        evidence="offline wall-load consequence on the reversed-time unit",
    )
axI.set_xlabel("record index (never-tested)")
axI.set_ylabel("$F_x\\times 10^{3}$")
axI.set_title("viscous wall-load tracking", fontsize=12)
all_y = 1e3 * np.concatenate([fx_t] + [
    np.mean([np.asarray(npz[f"F|test|{s}|fx|{a}"], dtype=float)
             for s in (8801, 8802)], axis=0) for a in ("absent", "tau_closure")])
span_y = all_y.max() - all_y.min()
axI.set_ylim(all_y.min() - 0.10 * span_y, all_y.max() + 0.85 * span_y)
axI.set_xlim(-10, 298)
axI.set_xticks([0, 100, 200])
axI.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
axI.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9,
           edgecolor="none", handletextpad=0.18, columnspacing=0.5,
           handlelength=1.0, borderpad=0.15, labelspacing=0.2)
panel_label(axI, "k", x=-0.15, y=1.15)

save(fig, "fig11_reachgated_composition")
