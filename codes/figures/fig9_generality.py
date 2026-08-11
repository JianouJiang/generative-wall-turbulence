#!/usr/bin/env python3
"""Figure 8: does the closure-to-generator interface generalise across wall regime
and generative family?

Three cells of one unchanged protocol:

  C1  aligned wall-mounted cube  x  denoising diffusion   (family varied)
  C2  separating periodic hills  x  flow matching         (regime varied)
  C3  separating periodic hills  x  denoising diffusion   (both varied)

Every plotted number is bound to an exact key of the frozen node-007 producer
output through `bind_artist`, which re-reads the declared key from the released
source data and rejects the artist if it disagrees.  The decision panel (b) is
scored on `full_srcex`, the region excluding both every supplied wall cell and
every cell the closure read; the hill cells are decided on `U_STRICT`, the
strictly uncontacted gap window, and the cube cell on the matched node-006 unit.
Adverse cells, if any, are drawn with the same prominence as favourable ones.
"""

import numpy as np

from _submission import (
    BLUE,
    GOLD,
    GREEN,
    GREY,
    LIGHT_BLUE,
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
SRC = "fig9_generality/e2_generality_results.json"
doc = load_json(SRC)
A = doc["analysis"]

REGION = "full_srcex"
CELLS = [
    ("C1", "MATCHED_NODE006_UNIT", "cube\ndiffusion"),
    ("C2", "U_STRICT", "hills\nflow match."),
    ("C3", "U_STRICT", "hills\ndiffusion"),
]
ARMS = [("absent", "absent", GREY), ("tau_eqwm", "equilibrium", LIGHT_BLUE),
        ("tau_closure", "closure", GOLD), ("tau_native", "oracle", BLUE),
        ("tau_fartime", "far-time", RED)]


def bar_centres(pos, width):
    """Matplotlib stores a bar by its left edge and reports the centre as
    ``x + width/2``.  That round trip is not exact in binary floating point, so
    the declared expectation must follow the same path rather than the original
    bytes -- exactly as ``expected_errorbar_payload`` already does for whiskers."""
    pos = np.asarray(pos, dtype=float)
    return (pos - width / 2) + width / 2


def jkey(cell, unit, region, *rest):
    return ".".join(["analysis", cell, "by_subunit", unit, region, *rest])


def blk(cell, unit, region=REGION):
    return A[cell]["by_subunit"][unit][region]


fig, axes = plt.subplots(2, 3, figsize=(8.75, 7.0), constrained_layout=True)
(axa, axb, axc), (axd, axe, axf) = axes

# ------------------------------------------------ a: closure skill, BOTH definitions
ap = {"cube": ("cells.C1.closure_apriori", doc["cells"]["C1"]["closure_apriori"]),
      "hills": ("cells.hills_common.closure_apriori",
                doc["cells"]["hills_common"]["closure_apriori"])}
STATS = [("R2_centred_eqwm", LIGHT_BLUE, "equilibrium, centred $R^2$"),
         ("R2_centred_closure", GOLD, "closure, centred $R^2$"),
         ("skill_zero_ref_eqwm", "#cfd8dd", "equilibrium, zero-ref. skill"),
         ("skill_zero_ref_closure", "#e3c76a", "closure, zero-ref. skill")]
x = np.arange(len(ap))
for k, (stat, col, lab) in enumerate(STATS):
    vals = [ap[nm][1][stat] for nm in ap]
    pos = x + (k - 1.5) * 0.19
    bars = axa.bar(pos, vals, 0.19, color=col, label=lab, edgecolor="white", lw=1.3)
    bind_artist(
        fig, bars, artist_id=f"fig9.a.{stat}", panel="a",
        source_refs=[{"kind": "json", "path": SRC, "key": f"{ap[nm][0]}.{stat}"}
                     for nm in ap],
        source_payload=vals,
        expected_payload=expected_bar_payload(bar_centres(pos, 0.19), vals),
        transform="plot the closure's held-out wall-traction skill in both definitions",
        evidence="closure reads matching-height state only; never the wall-adjacent cell",
    )
axa.axhline(0, color=GREY, lw=1.5)
axa.set_xticks(x)
axa.set_xticklabels(list(ap))
axa.set_ylabel("wall-traction skill")
# headroom so the key never sits on top of a bar
axa.set_ylim(0, max(max(ap[nm][1][stat] for nm in ap) for stat, _, _ in STATS) * 1.55)
axa.legend(fontsize=8.0, frameon=False, loc="upper left")
axa.set_title("$L_A$ on held-out frames", fontsize=10.0)
panel_label(axa, "a", x=-0.20, y=1.04)

# ------------------------------------------------- b: THE registered contrast
xs = list(range(len(CELLS)))
pts, lo, hi = [], [], []
for cell, unit, _ in CELLS:
    d = blk(cell, unit)["deltas"]["closure_minus_absent"]
    pts.append(d["point"]); lo.append(d["ci_block"][0]); hi.append(d["ci_block"][1])
eb = axb.errorbar(xs, pts, yerr=[np.array(pts) - np.array(lo),
                                 np.array(hi) - np.array(pts)],
                  fmt="o", ms=5.5, capsize=3, lw=1.8, color=GREY, zorder=3)
bind_artist(
    fig, eb, artist_id="fig9.b.registered", panel="b",
    source_refs=[{"kind": "json", "path": SRC,
                  "key": jkey(c, u, REGION, "deltas", "closure_minus_absent", "point")}
                 for c, u, _ in CELLS],
    source_payload=pts,
    expected_payload=expected_errorbar_payload(xs, pts, lo, hi),
    transform="plot the registered closure-minus-absent contrast in each cell with its "
              "conservative moving-block interval",
    evidence="decision unit per cell; scored outside every supplied and closure-read cell",
)
for i, (c, u, _) in enumerate(CELLS):
    axb.plot([i], [pts[i]], "o", ms=6.5,
             color=GOLD if lo[i] > 0 else RED, zorder=4)
axb.axhline(0, color=GREY, lw=1.5, ls="--")
axb.set_xticks(xs)
axb.set_xticklabels([c[2] for c in CELLS], fontsize=8.4)
axb.set_ylabel("closure $-$ absent, $\\Delta R^2$")
# every point and interval is positive; keep the zero reference visible without
# generating a negative tick label that would collide with the cell names
axb.set_ylim(-0.04, max(hi) * 1.14)
axb.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
axb.set_title("registered contrast", fontsize=10.0)
panel_label(axb, "b", x=-0.20, y=1.04)

# ------------------------------------------------------- c: absolute skill levels
for j, (arm, lab, col) in enumerate(ARMS):
    vals = [blk(c, u)["arms"][arm]["R2_seedmean"] for c, u, _ in CELLS]
    pos = np.arange(len(CELLS)) + (j - 2) * 0.15
    bars = axc.bar(pos, vals, 0.15, color=col, label=lab, edgecolor="white", lw=1.3)
    bind_artist(
        fig, bars, artist_id=f"fig9.c.{arm}", panel="c",
        source_refs=[{"kind": "json", "path": SRC,
                      "key": jkey(c, u, REGION, "arms", arm, "R2_seedmean")}
                     for c, u, _ in CELLS],
        source_payload=vals,
        expected_payload=expected_bar_payload(bar_centres(pos, 0.15), vals),
        transform="plot absolute source-excluded skill of each arm in each cell",
        evidence="absolute skill is positive only in the hills flow-matching cell; in the cube and in hills x diffusion it stays negative and the gain is differential",
    )
axc.axhline(0, color=GREY, lw=1.5)
axc.set_xticks(range(len(CELLS)))
axc.set_xticklabels([c[2] for c in CELLS], fontsize=8.4)
axc.set_ylabel("absolute $R^2$")
axc.legend(fontsize=8.0, frameon=False, ncol=2)
axc.set_title("absolute skill: positive only\nin hills $\\times$ flow matching", fontsize=9.0)
panel_label(axc, "c", x=-0.20, y=1.04)

# --------------------------------------------------- d: information ladder
GAINS = [("eqwm_minus_absent", "equilib.", LIGHT_BLUE, "s"),
         ("closure_minus_absent", "closure", GOLD, "o"),
         ("native_minus_absent", "oracle", BLUE, "D"),
         ("fartime_minus_absent", "far-time", RED, "v")]
for j, (tag, lab, col, mk) in enumerate(GAINS):
    p, l, h = [], [], []
    for c, u, _ in CELLS:
        d = blk(c, u)["deltas"][tag]
        p.append(d["point"]); l.append(d["ci_block"][0]); h.append(d["ci_block"][1])
    pos = np.arange(len(CELLS)) + (j - 1.5) * 0.17
    eb = axd.errorbar(pos, p, yerr=[np.array(p) - np.array(l), np.array(h) - np.array(p)],
                      fmt=mk, ms=3.6, capsize=2, lw=1.4, color=col, label=lab)
    bind_artist(
        fig, eb, artist_id=f"fig9.d.{tag}", panel="d",
        source_refs=[{"kind": "json", "path": SRC,
                      "key": jkey(c, u, REGION, "deltas", tag, "point")}
                     for c, u, _ in CELLS],
        source_payload=p,
        expected_payload=expected_errorbar_payload(pos, p, l, h),
        transform="plot every arm's gain over absence in each cell",
        evidence="far-time is a wrong-instant control with a plausible traction marginal; "
                 "oracle is a reference arm read from the target, not a ceiling",
    )
axd.axhline(0, color=GREY, lw=1.5, ls="--")
axd.set_xticks(range(len(CELLS)))
axd.set_xticklabels([c[2] for c in CELLS], fontsize=8.4)
axd.set_ylabel("gain over absent, $\\Delta R^2$")
axd.legend(fontsize=8.0, frameon=False, ncol=2)
axd.set_title("controls", fontsize=10.0)
panel_label(axd, "d", x=-0.20, y=1.04)

# ------------------------------------------------------- e: region dependence
REGS = [("near_srcex", "near"), ("full_srcex", "complete"), ("outer_srcex", "farther")]
for i, (cell, unit, lab) in enumerate(CELLS):
    p, l, h = [], [], []
    for reg, _ in REGS:
        d = A[cell]["by_subunit"][unit][reg]["deltas"]["closure_minus_absent"]
        p.append(d["point"]); l.append(d["ci_block"][0]); h.append(d["ci_block"][1])
    pos = np.arange(len(REGS)) + (i - 1) * 0.17
    eb = axe.errorbar(pos, p, yerr=[np.array(p) - np.array(l), np.array(h) - np.array(p)],
                      fmt="os^"[i], ms=3.8, capsize=2, lw=1.4,
                      color=[BLUE, GREEN, GOLD][i], label=lab.replace("\n", " "))
    bind_artist(
        fig, eb, artist_id=f"fig9.e.{cell}", panel="e",
        source_refs=[{"kind": "json", "path": SRC,
                      "key": jkey(cell, unit, reg, "deltas",
                                  "closure_minus_absent", "point")}
                     for reg, _ in REGS],
        source_payload=p,
        expected_payload=expected_errorbar_payload(pos, p, l, h),
        transform="plot how the closure gain decays with wall distance in each cell",
        evidence="the farther region contains no cell adjacent to a source cell",
    )
axe.axhline(0, color=GREY, lw=1.5, ls="--")
axe.set_xticks(range(len(REGS)))
axe.set_xticklabels([r[1] for r in REGS])
axe.set_ylabel("closure $-$ absent, $\\Delta R^2$")
axe.legend(fontsize=8.0, frameon=False)
axe.set_title("reach with wall distance", fontsize=10.0)
panel_label(axe, "e", x=-0.20, y=1.04)

# ---------------------------------------- f: offline wall-load surrogate
for j, (arm, lab, col) in enumerate((("absent", "absent", GREY),
                                     ("tau_closure", "closure", GOLD))):
    vals, refs = [], []
    for cell, unit, _ in CELLS:
        k = f"fx_relRMSE_{arm}"
        vals.append(A[cell]["paired_endpoints"][k])
        refs.append({"kind": "json", "path": SRC,
                     "key": f"analysis.{cell}.paired_endpoints.{k}"})
    pos = np.arange(len(CELLS)) + (j - 0.5) * 0.32
    bars = axf.bar(pos, vals, 0.32, color=col, label=lab, edgecolor="white", lw=1.3)
    bind_artist(
        fig, bars, artist_id=f"fig9.f.{arm}", panel="f",
        source_refs=refs, source_payload=vals,
        expected_payload=expected_bar_payload(bar_centres(pos, 0.32), vals),
        transform="plot the relative RMSE of the reconstructed streamwise wall force "
                  "against the target's own, for absence and for the closure arm",
        evidence="offline first-cell viscous-force surrogate: excludes pressure and form "
                 "drag, inferred from a reconstructed field, not measured in a solver",
    )
axf.set_xticks(range(len(CELLS)))
axf.set_xticklabels([c[2] for c in CELLS], fontsize=8.4)
axf.set_ylabel("wall-force relative RMSE")
axf.legend(fontsize=8.0, frameon=False)
axf.set_title("offline wall-load surrogate", fontsize=10.0)
panel_label(axf, "f", x=-0.20, y=1.04)

for ax in (axa, axb, axc, axd, axe, axf):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.4)

fig.get_layout_engine().set(w_pad=0.06, h_pad=0.07, wspace=0.04, hspace=0.06)
save(fig, "fig9_generality")
