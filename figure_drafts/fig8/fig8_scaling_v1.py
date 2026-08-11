"""
Supplementary Figure 2 -- negative capacity-at-fixed-corpus diagnostic.

THE data file: fig7/capacity_crossed_adjudication.json (resolved to its
committed copy by figure_drafts/datapaths.py)
(The retained draft directory follows the earlier storyboard Figure 8 numbering.)
Three widths (48/96/120 channels; 4.78M/18.5M/28.8M parameters) x three
training seeds (1234/2234/3234) on ONE fixed hill corpus, fixed update
budget. The adjudicated verdict is scaling_claim_pass = false. This figure's
job is to make that absence of a capacity effect visually undeniable:
per-seed points + means + crossed 95% CIs, the interval COMMON to all three
correct-arm CIs shaded, the slope CIs against a zero line, and the gate
checklist verbatim from the file.

Template (narrated fully in fig8_scaling_v1_rationale.md): Oommen Fig 3
skeleton (a = schematic objective strip, then quantitative stat panels)
hybridised with Oommen Fig 2e's metric-vs-model-property scatter grammar;
deck/band/badge vocabulary from the house pattern fig1/fig_concept_v6.py
(CoNFiLD deck grammar). Landscape golden ratio phi:1, 7.2 x 4.4499 in
(183 mm Nature print width), unit system x 0-161.8 / y 0-100, divider at
y = 61.8 (panel a : stats row = 38.2 : 61.8).

HONESTY:
- Negative shown as negative: no trend lines through overlapping CIs, the
  correct arm's absolute skill stays visibly below zero, and the verdict is
  carried by a grey "verified negative" ladder badge (fig1 badge grammar)
  + the verbatim gate checklist.
- The diagnostic ran on the HILL corpus (evidence_level in the file); hill
  imagery is quarantined. Field imagery is therefore REAL held-out truth
  |u| from the verified aligned-cube package
  (fig2/cube3d_representative_fields.npz, truth
  array only -- never a model arm), labelled in-figure as case
  illustration, not model output. Identical colour limits for every width
  so no deck can look "better".
- Estimator wording literal to the file: per_seed_point_estimates,
  mean_across_seed_point_estimates, crossed_ci95 (10,000-replicate
  seed x circular-time-block bootstrap, block 116), R2 per natural-log
  parameter. Carried ONCE, in the b-d footnote.
- Source-flow / few-shot transfer axes DO NOT EXIST (Haolang pending):
  declared in the rationale only, no in-figure placeholder.

TEXT BUDGET (2026-07-29, Peidong: "substantially reduce the explanatory text
in the figures"): the figure carries labels, identifiers and honesty
disclosures only. Protocol prose (corpus counts, bootstrap recipe in the
eval box, dodging note), prose restatements of the result and parenthetical
glosses live in the print caption -- drafted in
fig8_scaling_v1_rationale.md Sec 8. Nothing that keeps the negative
undeniable was cut: zero-skill label, common-CI-band annotation, failed-gate
marks, verdict row, badge tagline, claim boundary, estimator wording and
the case-illustration labels on every field thumbnail all remain.
Space freed by the cuts went to the field thumbnails (deck chips now square
= grid-isotropic and ~30% larger in area; oracle chip ~3x area) and to the
b-d axes.

Audits: collision checker v3 + BOXES margin check (house pattern) AND a
print-floor audit (every rendered text incl. tick labels >= 5.5 pt).

Run:  py fig8_scaling_v1.py            -> fig8_scaling_v1.png (400 dpi) + .pdf
      FIG_WIREFRAME=1 py ...           -> fig8_scaling_v1_wireframe.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

matplotlib.rcParams["font.family"] = ["Arimo", "DejaVu Sans"]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datapaths import source   # committed-source resolver (issue #7 rule)
DATA_PATH = source("fig7/capacity_crossed_adjudication.json")
CUBE_PATH = source("fig2/cube3d_representative_fields.npz")
ASSETS = os.path.join(HERE, "assets_fig8_v1")
WIRE = os.environ.get("FIG_WIREFRAME", "0") == "1"

RED = "#c0392b"      # aligned arm (source key: correct)
GREY = "#8a8a8a"     # absent-band control (source key: no_wall)
DGREY = "#4d4d4d"    # far-time control (source key: wrong_wall)
DARK = "#1a1a1a"
ORANGE = "#e08214"   # wall band
BANDGREY = "#c9c9c9" # shared-CI band
WFRAME = "#b0b0b0"; WRED = "#cc2222"

PHI = 1.6180339887
W = 161.8
H = 100.0
GL = 61.8

# ------------------------------------------------------------------ data
with open(DATA_PATH, encoding="utf-8") as f:
    ADJ = json.load(f)

WIDTHS = ADJ["contract"]["widths"]                 # [48, 96, 120]
SEEDS = ADJ["contract"]["seeds"]                   # [1234, 2234, 3234]
NPAR = {w: ADJ["n_parameters"][str(w)] for w in WIDTHS}
PTS = ADJ["points"]
SLOPES = ADJ["crossed_capacity_slopes"]
GATES = ADJ["gates"]
NREP = ADJ["contract"]["bootstrap_replicates"]     # 10000
TBLOCK = ADJ["contract"]["circular_time_block"]    # 116
NTIME = ADJ["contract"]["physical_times"]          # 370
NPLANE = ADJ["contract"]["planes_grouped_per_time"]

ARMS = ["correct", "no_wall", "wrong_wall"]
ARM_COL = {"correct": RED, "no_wall": GREY, "wrong_wall": DGREY}
ARM_MK = {"correct": "o", "no_wall": "^", "wrong_wall": "s"}
ARM_DODGE = {"correct": 1.00, "no_wall": 0.85, "wrong_wall": 1.17}

def arm_mean(w, arm):
    return PTS[str(w)][arm]["mean_across_seed_point_estimates"]

def arm_ci(w, arm):
    return PTS[str(w)][arm]["crossed_ci95"]

def arm_seeds(w, arm):
    d = PTS[str(w)][arm]["per_seed_point_estimates"]
    return [d[str(s)] for s in SEEDS]

def con_mean(w, key):
    return PTS[str(w)][key]["mean"]

def con_ci(w, key):
    return PTS[str(w)][key]["crossed_ci95"]

# interval common to all three correct-arm crossed 95% CIs (the overlap
# device): non-empty ==> CI overlap across widths is visually undeniable
COMMON_LO = max(arm_ci(w, "correct")[0] for w in WIDTHS)
COMMON_HI = min(arm_ci(w, "correct")[1] for w in WIDTHS)
assert COMMON_LO < COMMON_HI, "correct-arm CIs no longer share an interval"

# --------------------------------------------------- real cube truth fields
# VERIFIED package, fig2 (aligned-cube case). TRUTH array only -- model-arm
# arrays are never rendered here. The capacity diagnostic itself ran on the
# quarantined hill corpus, so these are labelled case illustration.
_cube = np.load(CUBE_PATH)
_SP3 = np.sqrt((_cube["truth"].astype(np.float32) ** 2).sum(axis=0))
_FLUID = _cube["fluid"]
_BANDM = _cube["band"]

# nine wall-parallel near-wall slices (index axis 1 = wall-normal),
# interleaved across widths so every deck carries statistically equivalent
# truth imagery -- no deck can suggest a capacity effect
DECK_Y = {48: (1, 4, 7), 96: (2, 5, 8), 120: (3, 6, 9)}
def _wallpar(iy):
    return _SP3[:, iy, :].T, _FLUID[:, iy, :].T
_pool = np.concatenate([_SP3[:, iy, :][_FLUID[:, iy, :]]
                        for w in WIDTHS for iy in DECK_Y[w]])
CHIP_LO, CHIP_HI = np.percentile(_pool, 2), np.percentile(_pool, 99.5)
DECK_FIELDS = {w: [_wallpar(iy) for iy in DECK_Y[w]] for w in WIDTHS}

# oracle-band side view (wall-normal slice through the cube, cropped to the
# near-wall region) for the evaluation chip; orange = band mask (house)
_ZC = 16; _YCROP = 56
ORACLE_SP = _SP3[:, :_YCROP, _ZC].T
ORACLE_FL = _FLUID[:, :_YCROP, _ZC].T
ORACLE_BD = _BANDM[:, :_YCROP, _ZC].T
ORA_LO, ORA_HI = (np.percentile(ORACLE_SP[ORACLE_FL], 2),
                  np.percentile(ORACLE_SP[ORACLE_FL], 99.5))

SOLID_GREY = (0.851, 0.851, 0.851, 1.0)   # #d9d9d9, matching fig2/fig3/fig7


def _rgba(arr, mask, lo, hi):
    """Solid geometry is GREY, never white: white reads as missing data,
    grey reads as an obstacle. Same value across the whole figure set."""
    rgba = plt.cm.plasma(np.clip((arr - lo) / (hi - lo), 0, 1))
    rgba[~mask] = SOLID_GREY
    return rgba

# --------------------------------------------- synthetic band strip (fig1)
def synth(seed, n=256, slope=-1.55):
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(n)[:, None]; kz = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + kz ** 2); k[0, 0] = 1.0
    spec = k ** slope * np.exp(2j * np.pi * rng.random((n, n)))
    f = np.real(np.fft.ifft2(spec))
    return (f - f.mean()) / f.std()

BAND = synth(11)[:3, ::4]

# ------------------------------------------------------------------ canvas
fig = plt.figure(figsize=(7.2, 7.2 / PHI))
ov = fig.add_axes([0, 0, 1, 1]); ov.axis("off")
ov.set_xlim(0, W); ov.set_ylim(0, H)

CHECKS = []   # (text, parent_box, note, allow_imgs:set)
IMAGES = []   # (x0,y0,x1,y1,name) data units
RULES = []    # (x0,x1,y,name)
SEGS = []     # (x0,y0,x1,y1,name) -- empty in v1, kept for checker parity
ARROWS = []   # (x0,y0,x1,y1,name)
BOXES = []    # (x,y,w,h,name)

def reg_image(x0, y0, x1, y1, name):
    IMAGES.append((x0, y0, x1, y1, name))

def rule(x0, x1, y, name, color="#444444", lw=1.0):
    ov.plot([x0, x1], [y, y], color=color, lw=lw, zorder=2)
    RULES.append((x0, x1, y, name))

def rbox(x, y, w, h, ec, fc, lw=1.2, ls="-", z=2, name="box"):
    ov.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.25,rounding_size=1.2",
                 ec=ec, fc=fc, lw=lw, linestyle=ls, zorder=z))
    BOXES.append((x, y, w, h, name))
    return (x, y, w, h)

def arrow(x0, y0, x1, y1, color=DARK, lw=1.3, z=4, ms=9, name="arrow"):
    ov.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=ms, lw=lw, color=color,
                 shrinkA=0.5, shrinkB=0.5, zorder=z))
    ARROWS.append((x0, y0, x1, y1, name))

def txt(x, y, s, fs=6.0, color=DARK, ha="center", va="center", w="normal",
        style="normal", z=5, box=None, note="", allow=None):
    t = ov.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, weight=w,
                style=style, zorder=z)
    CHECKS.append((t, box, note or s[:26], allow or set()))
    return t

def field(x0, y0, w, h, arr, mask=None, lo=None, hi=None, alpha=1.0,
          name="field", z=3, register=True, band=None):
    """real truth chip: plasma |u| on white; solid-region mask rendered
    grey with a thin grey obstacle outline (house wall-outline grammar);
    optional orange band-mask contour."""
    if register:
        reg_image(x0, y0 - 0.55, x0 + w, y0 + h, name)
    if WIRE:
        ov.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=WFRAME, lw=0.8,
                               hatch="///", zorder=z))
        ov.text(x0 + w / 2, y0 + h / 2, name, fontsize=5.5, color=WRED,
                ha="center", va="center", zorder=z + 2)
        return
    if mask is None:
        mask = np.ones_like(arr, dtype=bool)
    if lo is None or hi is None:
        lo, hi = np.percentile(arr[mask], 2), np.percentile(arr[mask], 99.5)
    ov.imshow(_rgba(arr, mask, lo, hi), extent=[x0, x0 + w, y0, y0 + h],
              origin="lower", interpolation="bilinear", aspect="auto",
              alpha=alpha, zorder=z)
    X = np.linspace(x0, x0 + w, arr.shape[1])
    Y = np.linspace(y0, y0 + h, arr.shape[0])
    if not mask.all():
        ov.contour(X, Y, mask.astype(float), levels=[0.5], colors="#9a9a9a",
                   linewidths=0.6, zorder=z + 1)
    if band is not None:
        ov.contour(X, Y, band.astype(float), levels=[0.5], colors=ORANGE,
                   linewidths=0.7, zorder=z + 1)
    ov.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="#2f2f2f",
                           lw=0.7, zorder=z + 1))
    ov.plot([x0, x0 + w], [y0 - 0.5, y0 - 0.5], color="#9a9a9a", lw=1.6,
            alpha=alpha, zorder=z - 1, solid_capstyle="butt")

def deck(x0, y0, w, h, pairs, offset=1.3, name="deck"):
    """CoNFiLD-style stacked chips; stack axis = TRAINING SEEDS here (three
    per width), named by the header text -- honesty rule. All chips share
    CHIP_LO/CHIP_HI so no width can look better."""
    n = len(pairs)
    reg_image(x0, y0 - 0.55, x0 + (n - 1) * offset + w,
              y0 + (n - 1) * offset + h, name)
    for i, (arr, mask) in enumerate(pairs):
        ox = (n - 1 - i) * offset
        a = 1.0 if i == n - 1 else 0.92
        field(x0 + ox, y0 + ox, w, h, arr, mask=mask, lo=CHIP_LO, hi=CHIP_HI,
              alpha=a, name=f"{name}-{i}", z=3 + i, register=False)

def band_strip(x0, y0, w, h, z=3):
    reg_image(x0, y0, x0 + w, y0 + h, "band")
    if WIRE:
        ov.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=ORANGE,
                               lw=1.0, hatch="///", zorder=z))
        return
    ov.imshow(BAND, extent=[x0, x0 + w, y0, y0 + h], cmap="Oranges",
              interpolation="nearest", aspect="auto", zorder=z)
    for i in range(4):
        ov.plot([x0, x0 + w], [y0 + h * i / 3] * 2, color=ORANGE, lw=0.5,
                zorder=z + 1)
    ov.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=ORANGE, lw=1.3,
                           zorder=z + 1))

def badge(x, y, text, color, glyph, tw):
    """fig1 ladder-badge grammar: content-sized rounded box, coloured dot
    with glyph, bold label. tw = estimated text width in units."""
    bw = 4.6 + tw + 1.6
    b = rbox(x, y - 2.7, bw, 5.4, color, "white", lw=1.0,
             name=f"badge-{text}")
    ov.scatter([x + 2.75], [y + 0.05], s=78, c=color, zorder=6)
    txt(x + 2.75, y + 0.1, glyph, fs=5.5, color="white", w="bold", z=7)
    txt(x + 4.95, y + 0.1, text, fs=6.0, color=color, ha="left", w="bold",
        box=b)

def axes_stats(x0, y0, w, h, name, pad_l=6.0, pad_b=6.5, pad_t=2.0,
               pad_r=0.5):
    """inset axes registered WITH a tick-label pad so unit-layer text cannot
    sit on tick labels the checker would otherwise not see"""
    reg_image(x0 - pad_l, y0 - pad_b, x0 + w + pad_r, y0 + h + pad_t, name)
    ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    ax.tick_params(labelsize=5.5, length=2, width=0.5, pad=1.5)
    for s in ax.spines.values():
        s.set_edgecolor("#555555"); s.set_linewidth(0.6)
    if WIRE:
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, name, fontsize=5.5, color=WRED, ha="center",
                va="center", transform=ax.transAxes)
    return ax

# ================================================================== panel a
txt(2, 96.5, "a", fs=11, w="bold", ha="left")

# ---- fixed-corpus box -------------------------------------------------
CBOX = rbox(3, 64.8, 21, 27.5, "#888888", "white", lw=1.0, name="corpus")
txt(13.5, 89.6, "fixed training corpus", fs=6.2, w="bold", box=CBOX)
band_strip(6.0, 78.8, 15, 7.0)
txt(13.5, 76.4, "near-wall band", fs=5.5, color=GREY, box=CBOX)
txt(13.5, 72.0, "grouped-hill corpus", fs=5.8, box=CBOX)
txt(13.5, 68.3, "fixed update budget", fs=5.8, box=CBOX)
arrow(25.0, 79.0, 27.6, 79.0)

# ---- width decks (capacity axis anchor) -------------------------------
txt(56.7, 95.4, "three widths $\\times$ three training seeds", fs=6.2)
txt(56.7, 92.9, "aligned-cube DNS truth — context only, not capacity "
    "outputs", fs=5.5, color=GREY, style="italic")

deck(28.3, 72.6, 12.4, 12.4, DECK_FIELDS[48], offset=1.3, name="deck-w48")
deck(46.3, 71.8, 13.9, 13.9, DECK_FIELDS[96], offset=1.45, name="deck-w96")
deck(66.3, 70.9, 15.5, 15.5, DECK_FIELDS[120], offset=1.6, name="deck-w120")
for cx, wname, pname in ((35.8, "width 48", "4.78M parameters"),
                         (54.7, "width 96", "18.5M parameters"),
                         (75.7, "width 120", "28.8M parameters")):
    txt(cx, 68.6, wname, fs=6.0)
    txt(cx, 66.5, pname, fs=5.5, color=GREY)
arrow(86.5, 79.0, 89.1, 79.0)

# ---- evaluation chip --------------------------------------------------
EBOX = rbox(90, 64.8, 23, 27.5, "#888888", "white", lw=1.0, name="eval")
txt(101.5, 89.6, "held-out hill score", fs=6.2, w="bold", box=EBOX)
txt(101.5, 87.2, "near-wall-band", fs=5.8, box=EBOX)
txt(101.5, 85.1, "propagation $R^2$", fs=5.8, box=EBOX)
txt(101.5, 82.4, "cube DNS context", fs=5.5, color=GREY, style="italic",
    box=EBOX)
field(95.25, 66.0, 12.5, 14.6, ORACLE_SP, mask=ORACLE_FL, lo=ORA_LO,
      hi=ORA_HI, name="oracle-chip", band=ORACLE_BD)
arrow(113.7, 79.0, 116.0, 79.0)

# ---- adjudication gates (verbatim booleans from the file) -------------
GBOX = rbox(116.5, 64.8, 43, 27.5, "#666666", "white", lw=1.2, name="gates")
txt(138, 89.7, "adjudication gates", fs=6.2, w="bold", box=GBOX)
GATE_ROWS = [
    ("three_width_three_seed_matrix", "three-width $\\times$ three-seed matrix"),
    ("largest_correct_beats_no_wall", "largest aligned beats absent"),
    ("largest_correct_beats_wrong_wall", "largest aligned beats far-time"),
    ("nondecreasing_gain_wrong_wall", "gain vs far-time non-decreasing"),
    ("positive_size_slope", "positive size slope"),
    ("largest_correct_absolute_skill_positive",
     "largest aligned absolute skill > 0"),
    ("nondecreasing_gain_no_wall", "gain vs absent non-decreasing"),
    ("convergence_parity", "convergence parity"),
]
gy = 87.0
for key, label in GATE_ROWS:
    ok = GATES[key]
    glyph = "✓" if ok else "✕"
    gcol = GREY if ok else DGREY
    txt(119.4, gy, glyph, fs=5.8, color=gcol, w="bold", box=GBOX,
        note=f"glyph-{key}")
    txt(121.8, gy, label, fs=5.8, color=gcol if ok else DARK, ha="left",
        box=GBOX, note=f"gate-{key}")
    gy -= 2.42
rule(118.3, 157.7, 68.0, "gates-rule", color="#bbbbbb", lw=0.6)
vglyph = "✓" if GATES["scaling_claim_pass"] else "✕"
txt(119.4, 66.4, vglyph, fs=6.2, color=DGREY, w="bold", box=GBOX,
    note="verdict-glyph")
txt(121.8, 66.4, "scaling claim — NOT SUPPORTED", fs=6.2,
    color=DARK, w="bold", ha="left", box=GBOX, note="verdict-row")

# ---- claim boundary (literal quote from the file) ---------------------
txt(46.0, 63.3, "claim boundary: fixed-update capacity cannot establish "
    "scaling without convergence parity", fs=5.5,
    color=GREY, style="italic", note="claim-boundary")

# ================================================================== divider
rule(2, 159.8, GL, "golden divider", color="#bbbbbb", lw=0.8)

# ================================================================ stats row
txt(3.2, 56.8, "b", fs=11, w="bold", ha="left")
badge(8.0, 57.3, "negative diagnostic", GREY, "✕", 16.0)
txt(32.9, 57.4, "fixed-budget scaling not established", fs=5.8, color=GREY,
    ha="left", note="badge-tagline")
txt(71.5, 56.8, "c", fs=11, w="bold", ha="left")
txt(113.0, 56.8, "d", fs=11, w="bold", ha="left")

XLIM_B = (3.5e6, 7.0e7)

def draw_panel_b(ax, labeled=True, annotate=True):
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_B); ax.set_ylim(-0.35, 0.035)
    # shared correct-arm CI band FIRST (background)
    ax.axhspan(COMMON_LO, COMMON_HI, color=BANDGREY, alpha=0.5, lw=0,
               zorder=0)
    ax.axhline(0, color=DARK, lw=0.8, zorder=2)
    for arm in ARMS:
        col = ARM_COL[arm]; mk = ARM_MK[arm]; dg = ARM_DODGE[arm]
        for w in WIDTHS:
            x = NPAR[w] * dg
            # per-seed open markers (per_seed_point_estimates)
            for k, v in enumerate(arm_seeds(w, arm)):
                ax.plot(x * (0.965 + 0.035 * k), v, marker=mk, ms=2.3,
                        mfc="white", mec=col, mew=0.6, ls="none", zorder=3)
            # mean across seed point estimates + crossed 95% CI
            m = arm_mean(w, arm); lo, hi = arm_ci(w, arm)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=mk, ms=3.6,
                        mfc=col, mec=col, ecolor=col, elinewidth=0.9,
                        capsize=1.6, capthick=0.9, zorder=4)
    if annotate and not WIRE:
        ax.text(3.9e6, 0.014, "zero skill ($R^2$ = 0)", fontsize=5.5,
                color=DARK, ha="left", va="center")
        ax.annotate("interval common to\nall 3 aligned-arm CIs",
                    xy=(9.3e6, COMMON_HI + 0.004), xytext=(9.3e6, -0.105),
                    fontsize=5.5, color=DGREY, ha="center", va="top",
                    arrowprops=dict(arrowstyle="-|>", color=DGREY, lw=0.7,
                                    shrinkA=1, shrinkB=1))
        ax.text(3.3e7, arm_mean(120, "correct"), "aligned band",
                fontsize=5.5, color=RED, ha="left", va="center")
        ax.text(4.2e7, arm_mean(120, "wrong_wall") + 0.012, "far-time",
                fontsize=5.5, color=DGREY, ha="left", va="center")
        ax.text(3.2e7, arm_mean(120, "no_wall") - 0.035, "absent",
                fontsize=5.5, color=GREY, ha="left", va="center")
    if labeled:
        ax.set_xticks([NPAR[w] for w in WIDTHS])
        ax.set_xticklabels([f"{NPAR[w]/1e6:.3g}M\n(w{w})" for w in WIDTHS],
                           fontsize=5.5)
        ax.set_xticks([], minor=True)
        ax.set_yticks([-0.3, -0.2, -0.1, 0.0])
        ax.set_xlabel("model parameters (log scale)", fontsize=6.2,
                      labelpad=1.5)
        ax.set_ylabel("propagation skill $R^2$", fontsize=6.2, labelpad=1.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks([], minor=True)

CON_STYLE = {  # marker matches the control it subtracts (panel-b grammar)
    "correct_minus_no_wall": ("^", "-", 0.93),
    "correct_minus_wrong_wall": ("s", (0, (3, 2)), 1.08),
}

def draw_panel_c(ax, labeled=True, annotate=True):
    ax.set_xscale("log")
    ax.set_xlim(3.5e6, 4.2e7); ax.set_ylim(-0.02, 0.30)
    ax.axhline(0, color=DARK, lw=0.8, zorder=2)
    for key, (mk, ls, dg) in CON_STYLE.items():
        xs = [NPAR[w] * dg for w in WIDTHS]
        ms_ = [con_mean(w, key) for w in WIDTHS]
        los = [con_ci(w, key)[0] for w in WIDTHS]
        his = [con_ci(w, key)[1] for w in WIDTHS]
        ax.plot(xs, ms_, ls=ls, color=RED, lw=0.8, zorder=3)
        ax.errorbar(xs, ms_, yerr=[np.subtract(ms_, los),
                                   np.subtract(his, ms_)], fmt=mk, ms=3.4,
                    mfc=RED if mk == "^" else "white", mec=RED, ecolor=RED,
                    elinewidth=0.9, capsize=1.6, capthick=0.9, ls="none",
                    zorder=4)
    if annotate and not WIRE:
        ax.text(NPAR[96] * 0.93, 0.272, "aligned $-$ absent", fontsize=5.5,
                color=RED, ha="center", va="center")
        ax.text(NPAR[96] * 1.08, 0.055, "aligned $-$ far-time",
                fontsize=5.5, color=RED, ha="center", va="center")
    if labeled:
        ax.set_xticks([NPAR[w] for w in WIDTHS])
        ax.set_xticklabels([f"{NPAR[w]/1e6:.3g}M\n(w{w})" for w in WIDTHS],
                           fontsize=5.5)
        ax.set_xticks([], minor=True)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3])
        ax.set_xlabel("model parameters (log scale)", fontsize=6.2,
                      labelpad=1.5)
        ax.set_ylabel("$R^2$ contrast (aligned $-$ control)", fontsize=6.2,
                      labelpad=1.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks([], minor=True)

SLOPE_ROWS = [  # (file key, display label, y, marker)
    ("correct", "aligned arm", 3, "o"),
    ("gain_no_wall", "gain vs absent", 2, "^"),
    ("gain_wrong_wall", "gain vs far-time", 1, "s"),
]

def draw_panel_d(ax, labeled=True, annotate=True):
    ax.set_xlim(-0.052, 0.058); ax.set_ylim(0.3, 3.75)
    ax.axvspan(0, 0.058, color="#f2f2f2", lw=0, zorder=0)
    ax.axvline(0, color=DARK, lw=0.8, zorder=2)
    for key, label, y, mk in SLOPE_ROWS:
        m = SLOPES[key]["mean"]; lo, hi = SLOPES[key]["crossed_ci95"]
        col = RED if key == "correct" else DGREY
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=mk, ms=3.6,
                    mfc=col, mec=col, ecolor=col, elinewidth=0.9,
                    capsize=1.6, capthick=0.9, zorder=4)
        if annotate and not WIRE:
            ax.text(-0.050, y + 0.34,
                    f"{label}:  {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]",
                    fontsize=5.5, color=col, ha="left", va="center",
                    zorder=5, bbox=dict(boxstyle="square,pad=0.15",
                                        fc="white", ec="none", alpha=0.9))
    if annotate and not WIRE:
        ax.text(0.056, 0.62, "positive-slope region", fontsize=5.5,
                color=GREY, ha="right", va="center")
    if labeled:
        ax.set_yticks([])
        ax.set_xticks([-0.04, -0.02, 0.0, 0.02, 0.04])
        ax.set_xlabel("capacity slope ($R^2$ per natural-log parameter)",
                      fontsize=6.2, labelpad=1.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])

ax_b = axes_stats(9.5, 9.6, 60, 42, "panel-b")
ax_c = axes_stats(78, 9.6, 33, 42, "panel-c")
ax_d = axes_stats(116.5, 9.6, 42, 42, "panel-d", pad_l=2.0)
if not WIRE:
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)

txt(81, 1.7, "open: per-seed point estimates; filled: mean across seed "
    "point estimates; bars: crossed 95% CI "
    f"({NREP:,}-replicate seed $\\times$ circular-time-block bootstrap, "
    f"block {TBLOCK})", fs=5.5, color=GREY, note="estimator-footnote")

if WIRE:
    txt(80.9, 98.3, "WIREFRAME fig8 v1 — landscape phi:1; hatched = "
        "image asset", fs=5.8, color=WRED)

ov.set_xlim(0, W); ov.set_ylim(0, H); ov.set_aspect("auto")

# ------------------------------------------------------- collision checker v3
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
fw, fh = fig.canvas.get_width_height()
TOL = 2.0
viol = 0

def to_disp(x, y):
    return ov.transData.transform([[x, y]])[0]

def overlap(a, b, tol=1.5):
    return not (a[2] < b[0] + tol or b[2] < a[0] + tol
                or a[3] < b[1] + tol or b[3] < a[1] + tol)

def seg_cross(p1, p2, p3, p4, eps=1e-9):
    def o(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < eps else (1 if v > 0 else -1)
    o1, o2 = o(p1, p2, p3), o(p1, p2, p4)
    o3, o4 = o(p3, p4, p1), o(p3, p4, p2)
    return o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4)

img_disp = []
for (x0, y0, x1, y1, name) in IMAGES:
    (dx0, dy0) = to_disp(x0, y0); (dx1, dy1) = to_disp(x1, y1)
    img_disp.append((min(dx0, dx1), min(dy0, dy1),
                     max(dx0, dx1), max(dy0, dy1), name))
rule_disp = [(to_disp(x0, yy)[0], to_disp(x0, yy)[1], to_disp(x1, yy)[0],
              name) for (x0, x1, yy, name) in RULES]
seg_data = [(p[0], p[1], p[2], p[3], p[4]) for p in SEGS]

for t, box, note, allow in CHECKS:
    e = t.get_window_extent(renderer)
    te = (e.x0, e.y0, e.x1, e.y1)
    if box is not None:
        (bx0, by0) = to_disp(box[0], box[1])
        (bx1, by1) = to_disp(box[0] + box[2], box[1] + box[3])
        if (e.x0 < bx0 - TOL or e.x1 > bx1 + TOL
                or e.y0 < by0 - TOL or e.y1 > by1 + TOL):
            viol += 1; print(f"OVERFLOW text-vs-box: '{note}'")
    if e.x0 < -TOL or e.x1 > fw + TOL or e.y0 < -TOL or e.y1 > fh + TOL:
        viol += 1; print(f"OVERFLOW text-vs-canvas: '{note}'")
    for (ix0, iy0, ix1, iy1, iname) in img_disp:
        if iname in allow:
            continue
        if overlap(te, (ix0, iy0, ix1, iy1)):
            viol += 1
            print(f"COLLISION text-vs-image: '{note}' on '{iname}'")
    for (rx0, ry, rx1, rname) in rule_disp:
        if e.x0 < rx1 and e.x1 > rx0 and e.y0 - TOL < ry < e.y1 + TOL:
            viol += 1
            print(f"COLLISION text-vs-rule: '{note}' on '{rname}'")
    for (sx0, sy0, sx1, sy1, sname) in seg_data:
        (dx0, dy0) = to_disp(sx0, sy0); (dx1, dy1) = to_disp(sx1, sy1)
        sb = (min(dx0, dx1) - 1, min(dy0, dy1) - 1,
              max(dx0, dx1) + 1, max(dy0, dy1) + 1)
        if overlap(te, sb, tol=0.5):
            viol += 1
            print(f"COLLISION text-vs-seg: '{note}' on '{sname}'")
for i in range(len(img_disp)):
    for j in range(i + 1, len(img_disp)):
        a, b = img_disp[i], img_disp[j]
        if overlap(a[:4], b[:4], tol=0.5):
            viol += 1
            print(f"COLLISION image-vs-image: '{a[4]}' on '{b[4]}'")
for (ix0, iy0, ix1, iy1, iname) in img_disp:
    for (rx0, ry, rx1, rname) in rule_disp:
        if ix0 < rx1 and ix1 > rx0 and iy0 - 0.5 < ry < iy1 + 0.5:
            viol += 1
            print(f"COLLISION image-vs-rule: '{iname}' on '{rname}'")
MARGIN = 1.3
for (bx, by, bw, bh, bname) in BOXES:
    if (bx - 0.25 < MARGIN or by - 0.25 < MARGIN
            or bx + bw + 0.25 > W - MARGIN or by + bh + 0.25 > H - MARGIN):
        viol += 1
        print(f"MARGIN patch-vs-canvas: '{bname}'")
for (ax0, ay0, ax1, ay1, aname) in ARROWS:
    for (sx0, sy0, sx1, sy1, sname) in seg_data:
        if seg_cross((ax0, ay0), (ax1, ay1), (sx0, sy0), (sx1, sy1)):
            viol += 1
            print(f"CROSSING arrow-vs-seg: '{aname}' x '{sname}'")
    for (rx0, rx1_, ry, rname) in [(r[0], r[1], r[2], r[3]) for r in RULES]:
        if seg_cross((ax0, ay0), (ax1, ay1), (rx0, ry), (rx1_, ry)):
            viol += 1
            print(f"CROSSING arrow-vs-rule: '{aname}' x '{rname}'")

# --------------------------------------- print-floor audit (incl. ticks)
FLOOR = 5.5
sub_floor = []
for t in fig.findobj(mtext.Text):
    s = t.get_text()
    if not s or not t.get_visible():
        continue
    if t.get_fontsize() < FLOOR - 1e-6:
        sub_floor.append((s[:34], t.get_fontsize()))
for s, fsz in sub_floor:
    viol += 1
    print(f"PRINT-FLOOR sub-{FLOOR}pt text: '{s}' at {fsz}pt")
print(f"PRINT-FLOOR AUDIT (all text incl. tick labels >= {FLOOR}pt): "
      f"{len(sub_floor)} sub-floor")
print(f"COLLISION CHECK: {viol} violations")

suffix = "_wireframe" if WIRE else ""
fig.savefig(os.path.join(HERE, f"fig8_scaling_v1{suffix}.png"), dpi=400,
            facecolor="white")
if not WIRE:
    fig.savefig(os.path.join(HERE, "fig8_scaling_v1.pdf"), facecolor="white")
    manuscript_figures = os.path.join(REPO, "manuscript", "figures")
    os.makedirs(manuscript_figures, exist_ok=True)
    fig.savefig(os.path.join(manuscript_figures, "fig7_scaling.png"), dpi=400,
                facecolor="white")
    fig.savefig(os.path.join(manuscript_figures, "fig7_scaling.pdf"),
                facecolor="white")
print(f"wrote fig8_scaling_v1{suffix}.png" + ("" if WIRE else " + .pdf"))

# ------------------------------------------------------------------ assets
if not WIRE:
    def export_ax(name, drawer, figsize=(3.2, 2.4)):
        for kind in ("labeled", "clean"):
            d = os.path.join(ASSETS, kind); os.makedirs(d, exist_ok=True)
            f2 = plt.figure(figsize=figsize)
            a2 = f2.add_axes([0.16, 0.16, 0.80, 0.78])
            a2.tick_params(labelsize=7, length=2.5, width=0.6, pad=2)
            for s in a2.spines.values():
                s.set_edgecolor("#555555"); s.set_linewidth(0.8)
            drawer(a2, kind == "labeled")
            f2.savefig(os.path.join(d, name + ".png"), dpi=400,
                       transparent=True, bbox_inches="tight")
            plt.close(f2)

    export_ax("panel_b", lambda a, lab: draw_panel_b(a, labeled=lab,
                                                     annotate=lab))
    export_ax("panel_c", lambda a, lab: draw_panel_c(a, labeled=lab,
                                                     annotate=lab))
    export_ax("panel_d", lambda a, lab: draw_panel_d(a, labeled=lab,
                                                     annotate=lab))

    def export_img(name, drawer, title=None, figsize=(2.4, 2.0)):
        for kind in ("labeled", "clean"):
            d = os.path.join(ASSETS, kind); os.makedirs(d, exist_ok=True)
            f2 = plt.figure(figsize=figsize)
            a2 = f2.add_axes([0.02, 0.02, 0.96, 0.90])
            drawer(a2)
            if kind == "labeled" and title:
                a2.set_title(title, fontsize=8)
            f2.savefig(os.path.join(d, name + ".png"), dpi=400,
                       transparent=True, bbox_inches="tight")
            plt.close(f2)

    def draw_chip(a, arr, mask, lo, hi, band=None):
        a.imshow(_rgba(arr, mask, lo, hi), origin="lower",
                 interpolation="bilinear", aspect="auto")
        if not mask.all():
            a.contour(mask.astype(float), levels=[0.5], colors="#9a9a9a",
                      linewidths=0.8)
        if band is not None:
            a.contour(band.astype(float), levels=[0.5], colors=ORANGE,
                      linewidths=0.9)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#2f2f2f"); s.set_linewidth(0.8)

    for w in WIDTHS:
        for iy, (arr, mask) in zip(DECK_Y[w], DECK_FIELDS[w]):
            export_img(f"width{w}_deck_chip_y{iy}",
                       lambda a, arr=arr, mask=mask: draw_chip(
                           a, arr, mask, CHIP_LO, CHIP_HI),
                       f"width-{w} deck layer: held-out truth $|u|$\n"
                       f"(aligned-cube case illustration, slice {iy})")

    export_img("oracle_band_chip",
               lambda a: draw_chip(a, ORACLE_SP, ORACLE_FL, ORA_LO, ORA_HI,
                                   band=ORACLE_BD),
               "held-out oracle band region, truth $|u|$\n"
               "(aligned-cube case illustration)")

    def draw_band_asset(a):
        a.imshow(BAND, cmap="Oranges", interpolation="nearest",
                 aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(ORANGE); s.set_linewidth(1.3)
    export_img("band_strip", draw_band_asset,
               "near-wall-band conditioning (fixed corpus, schematic)",
               figsize=(3.2, 0.9))

    for orient in ("h", "v"):
        for kind in ("labeled", "clean"):
            d = os.path.join(ASSETS, kind); os.makedirs(d, exist_ok=True)
            f2 = plt.figure(figsize=(3.0, 0.7) if orient == "h"
                            else (0.8, 3.0))
            cax = f2.add_axes([0.05, 0.45, 0.9, 0.35] if orient == "h"
                              else [0.35, 0.05, 0.3, 0.9])
            sm = plt.cm.ScalarMappable(
                norm=Normalize(CHIP_LO, CHIP_HI), cmap="plasma")
            cb = f2.colorbar(sm, cax=cax, orientation="horizontal"
                             if orient == "h" else "vertical")
            cb.ax.tick_params(labelsize=7)
            if kind == "labeled":
                cb.set_label("$|u|$ (aligned-cube held-out truth)",
                             fontsize=8)
            else:
                cb.set_ticks([])
            f2.savefig(os.path.join(d, f"cbar_velocity_{orient}.png"),
                       dpi=400, transparent=True, bbox_inches="tight")
            plt.close(f2)
    print("assets ->", ASSETS)
