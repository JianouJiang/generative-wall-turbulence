# Preregistration — does the source-valid closure→generator interface generalise?
# (regime × generative family), and the completion of the mandated non-regression table

**Frozen before ANY held-out outcome of this experiment was observed.**
Node `development/nodes/node_007`, Level 3 attempt 5, 31 July 2026.

Hash-bound in `FROZEN_HASHES.json` together with the producer
`codes/gpu/eval_e2_generality.py`, the contact ledger builder
`codes/probes/build_contact_ledger.py` and the ledger it produced. The production
job is launched only after that freeze.

---

## 0. Full disclosure of prior outcome contact (read this first)

| Question | Answer |
|---|---|
| Was any held-out outcome of **this** experiment observed before this freeze? | **No.** |
| Were smoke runs executed before this freeze? | Yes — **two**. (i) local CPU, hill cells only, `--smoke --cells C2,C3`, tag `e2_generality_localsmoke`; (ii) `foshan` GPU, all three cells, `--smoke --cells C1,C2,C3`, tag `e2_generality_smoke`, log `foshan_20260731T232934.log`. |
| What did the smokes evaluate? | **Training frames only, in both regimes.** Under `--smoke` the producer hard-codes `train_idx_h = arange(64)`, `eval = arange(6)` for the hills and `strict_idx = train_idx[:6]` for the cube, so every evaluated frame lies inside that smoke's own training window. The printed `R²` values (≈ −3.2 / −25.8 / −37.7 after 60 optimiser steps at base width 16) are meaningless by construction and are used nowhere. |
| Is that machine-checkable? | Yes. `provenance.smoke == true` in both smoke result files, and every smoke evaluation index is a subset of the smoke training indices. `verify_node007.py` asserts it. |
| Did a smoke write any checkpoint the production run could inherit? | **No.** The producer's `torch.save` calls are guarded by `if not A.smoke`, and the smoke tags differ from the production tag, so no checkpoint path collides. |
| Was any hill closure or generator hyper-parameter selected against a held-out score? | **No.** Every architectural, optimiser, capacity, matching-height, mixture, sampler and bootstrap setting is inherited unchanged from the frozen node-006 producer. The only quantities derived from the hill record are its geometry, its viscosity (from `Re_h` and the record's own bulk velocity) and its integral time — all computed from the training window or from the record's stated physics, never from a score. |
| What node-006 outcomes are known? | All of them. Node 006 is published in this manuscript. That is precisely why **no node-006 quantity is a decision target here**: the decision rule below is evaluated on cells node 006 never ran. |

**Development discipline.** The hill closure `L_A^{H}` is fitted on snapshots
`[0, 1858)` and **model-selected on a validation window inside the training span**
(the last 10% of it). No held-out frame participates in its development. Unlike
node 006 — which the panel correctly caught monitoring a validation window while
always retaining the terminal checkpoint — the selection is genuine: the
lowest-validation-loss state is restored before freezing, and the selected step is
recorded in `closure_meta.selected_step`.

---

## 1. The defect this node exists to repair

Node 006 executed the composed chain

```
q^coarse --L_A (physics-grounded wall closure)--> tau_w --L_B (conditional generator)--> field
```

and obtained `Δ = R²(τ_closure) − R²(absent) = +0.0606 [+0.0569, +0.0636]`. All
three panel seats accepted that computation and rejected the level. Stripped of
the production defects (handled separately in this node, §10), the **scientific**
objection was unanimous and singular:

> one record, one wall regime, one generative family, `N_eff ≈ 3.2` — while
> `research/success_criteria.json` §§13–23, 67–103 registers a venue-shape
> contract requiring the central propagation claim in **at least two
> qualitatively different wall regimes** and **at least two generative
> families**.

Reviewer 2 additionally required a **machine-readable global contact ledger** and
a genuinely unscored unit, having found that 48 of node 006's 103 "untouched"
frames had in fact been scored by node 004.

This node answers exactly those two demands. It is **not** another closure
architecture search: not one design decision of the node-006 protocol is changed.

---

## 2. Design — three new cells, one protocol

| Cell | Wall regime | Generative family | Closure `L_A` |
|---|---|---|---|
| **C1** | aligned wall-mounted cube, `Re_h = 5000`, 3-D (node 006's record) | **G — denoising diffusion** (NEW) | node-006 checkpoint, **byte-identical, hash-gated, refitted not at all** |
| **C2** | **separating periodic hills, `Re_h = 2800`, 2-D** (NEW regime) | F — flow matching (node 006's family) | `L_A^{H}`, fitted on the hill training window only, then frozen |
| **C3** | separating periodic hills (NEW regime) | **G — denoising diffusion** (NEW) | the same frozen `L_A^{H}` |

C1 varies the family at fixed regime, closure, unit and arms — it is a *clean*
family contrast. C2 varies the regime at fixed family. C3 varies both.

**The two families are genuinely different generative classes**, and they are the
two this manuscript already uses throughout:

| | F — flow matching | G — denoising diffusion |
|---|---|---|
| forward process | linear interpolation `x_t = (1−t)x₀ + t z` | variance-preserving, cosine `ᾱ(t)` |
| training target | velocity `z − x₀` | noise `ε` |
| sampler | deterministic Euler along the probability flow | **stochastic** ancestral posterior sampling |

**Why periodic hills is a qualitatively different wall regime.** The cube is a
sharp-edged obstacle on a flat floor: separation is geometrically pinned at fixed
edges and the wall is piecewise flat with axis-aligned normals. The hill wall is
smooth and curved, and separation is *pressure-gradient-induced* with a mean
recirculation bubble whose detachment and reattachment points are free to move —
the regime in which equilibrium wall models are known to fail. It is the canonical
separated-flow wall benchmark, and it is the regime the paper's separation claims
are about. The record is 2-D; the tangent space at a 2-D wall is one-dimensional,
so `L_A^{H}` carries the bounded **magnitude** correction only and the direction
correction does not exist by dimension. This is stated, not hidden, and the
equilibrium special case `τ_eqwm` is unaffected.

---

## 3. Units, and the machine-verified contact ledger

`codes/probes/build_contact_ledger.py` scans **every** retained artefact in
`codes/results/` (282 scanned), harvests every stored evaluation index, and groups
them by physical record. Its output `CONTACT_LEDGER.json` is frozen with this
document. Two findings govern the units below.

**Cube.** 356 distinct frames spanning `[0, 1100]` have been scored by 18 retained
index sets across nodes 004–006. The cube test window is **fully contacted**. No
uncontacted cube unit exists locally, and the record cannot be extended: the nekRS
continuation blocks `cont_b001/b002` and their restart files were deleted from
`foshan` on 26 July 2026 (`/root/autodl-tmp/cube_les` now holds only the packed
arrays), and the case mesh/`.par`/`.udf` are gone with them.

> **Consequence, stated before the outcome.** C1 is evaluated on node 006's exact
> 103-frame unit **deliberately**, because a family contrast is only interpretable
> at a matched unit. It is labelled `MATCHED_NODE006_UNIT`. It is a
> **previously-contacted matched-unit family contrast, not an independent
> confirmation**, and no sentence in the manuscript will call it one. Every
> "untouched"/"uncontacted by any earlier experiment" claim attached to that unit
> is withdrawn in this node (§10.1).

**Hills.** No hill-record producer retained evaluation indices, so the ledger
resolves hill contact from the **declared split rule** instead — and every hill
producer in this project inherits exactly one, from
`codes/gpu/train_stats_l2.py:166–170`:

```
n_tr = int(0.76*T);  gap = max(80, 3*tau);  test = [n_tr + gap, T)
```

With `T = 2880` and the record's integral time `tau = 109.62` frames this gives
`n_tr = 2188`, `gap = 330`. Two consequences follow **mechanically**:

| Unit | Frames | n | Status |
|---|---|---|---|
| `U_STRICT` | `[2188, 2518)` | **330** | **STRICTLY UNCONTACTED.** Every prior producer trained on `[0, 2188)` and scored `[2518, 2880)`. All of them *discarded the decorrelation gap*. No producer in this project has ever trained on or scored these frames. |
| `U_PRIOR` | `[2518, 2880)` | 362 | Declared **previously-contacted replication unit**: scored by earlier producers, for other estimands and other models. Retained and reported, never called uncontacted. |

**This experiment shifts its own training window back by one full gap**, to
`[0, 1858)`, so that `U_STRICT` sits a complete decorrelation gap (330 frames ≈
3τ) behind its own training data as well as being untouched by history. Nothing
about `U_STRICT` was chosen by inspecting a score; it is the arithmetic complement
of a rule frozen in the repository since before this campaign began.

`N_eff(U_STRICT) = 330/109.62 = 3.01`. Reported for every interval. Moving-block
length `1.2551·τ = 138` frames, the conservative block of the frozen protocol.

---

## 4. Arms (identical in every cell)

| Arm | Conditioning | Role |
|---|---|---|
| `tau_closure` | traction **predicted** by the frozen closure from matching-height state only | **PRIMARY** |
| `absent` | conditioning zeroed | **PRIMARY CONTROL** (no-wall) |
| `tau_native` | traction read from the target's own wall-adjacent cell | **oracle reference arm** — not a closure result, and **not an empirical ceiling** (closure errors can encode matching-height outer-state information, so it can legitimately be exceeded) |
| `tau_eqwm` | the classical equilibrium wall model, i.e. the closure with its learned correction switched off | physics baseline at matched input information |
| `tau_fartime` | a real traction field from a far-time donor snapshot | **wrong-instant control**. It has a physically plausible traction *marginal* but deliberately breaks the learned condition–target *joint*; it is therefore a wrong-information control, **not** a distribution-shift-free control. Node 006's contrary wording is withdrawn. |

All five arms are training classes of the declared conditioning mixture
(`tau_native 0.40 / tau_closure 0.40 / absent 0.20`, inherited verbatim), so no
primary contrast is confounded by an out-of-distribution conditioning input.

**Scoring exclusions.** Every cell whose traction is *supplied* and every cell the
closure *reads* (matching offsets 2 and 4 along the face normal) is removed from
the region carrying the decision rule (`full_srcex`). `outer_srcex` (wall distance
> 0.5 h) is additionally reported: it contains no cell adjacent to any source cell
and is the strong non-adjacency test.

---

## 5. Metrics, and the six statistical repairs the panel required

Primary estimand, per cell:

```
Delta = R^2(tau_closure) - R^2(absent)     on `full_srcex`
```

`R²` is the fluctuation-balanced coefficient about the training-window mean field,
inherited byte-identically.

Repairs implemented in this producer, each answering a named panel finding:

1. **Traction skill is reported in both definitions, on every evaluated frame.**
   `R2_centred_*` is the conventional mean-centred coefficient of determination;
   `skill_zero_ref_*` is `1 − MSE/E[τ²]`, the zero-traction-reference statistic
   node 006 published as "`R²`". Node 006 also silently scored `strict_idx[::2]`
   (52 of 103 frames); here `n_frames_scored` is recorded and is the full unit.
2. **`tau_rms` is named as an RMS**, never as a standard deviation, everywhere it
   appears.
3. **The seed × block interval is a genuine hierarchical bootstrap**: seeds are
   resampled with replacement, a seed-mean replicate system is formed, and time
   blocks are resampled within it. Node 006 concatenated per-seed block
   distributions, which is not a crossed estimator.
4. **The energy score uses the off-diagonal `M(M−1)` estimator** of the Methods
   equation. Node 006's producer used the diagonal-inclusive `M²` form.
5. **CRPS, energy score and the wall-force endpoint carry paired,
   dependence-aware moving-block intervals** of the closure-minus-control
   contrast — not point orderings.
6. **Ensemble-size sensitivity is accumulated over the whole unit and every
   seed**, not the first batch of the first seed.

Engineering endpoint, per cell: the streamwise **viscous wall force** implied by
the reconstructed near-wall field, versus the target's own, as relative RMSE and
correlation with a paired block interval on the absolute-error contrast. It is an
**offline first-cell viscous-force surrogate**: it excludes pressure/form drag,
it is inferred from a reconstructed field, and it has not been tested in a solver.
Node 006's stronger wording ("the true instantaneous wall force", "the quantity a
wall-modelled computation consumes") is withdrawn.

---

## 6. The registered decision rule

Evaluated mechanically by `apply_decision_rule_node007.py` from the result file.
Cell C1 is decided on `MATCHED_NODE006_UNIT`; cells C2 and C3 are decided on
`U_STRICT`. Region `full_srcex` throughout.

| # | Clause | Pass condition |
|---|---|---|
| G1 | **Family generality** | C1: `Delta > 0` and its conservative-block 95% interval excludes 0 |
| G2 | **Regime generality** | C2: `Delta > 0` and its conservative-block 95% interval excludes 0, on `U_STRICT` |
| G3 | **Joint generality** | C3: `Delta > 0` and its conservative-block 95% interval excludes 0, on `U_STRICT` |
| G4 | **Wrong-information control** | in **every** cell, `Delta_closure > Delta_fartime` |
| G5 | **Seed robustness** | in **every** cell, `Delta` has the same sign on both training seeds |
| G6 | **Hierarchical uncertainty** | in **every** cell, the hierarchical seed × block 95% interval of `Delta` excludes 0 |
| G7 | **No reversal** | `min(Delta)` over the three cells is `> 0` |

**VERDICT = `GENERALITY_CONFIRMED` iff all seven clauses pass.**

Pre-declared partial dispositions, so that no outcome can be re-narrated after
the fact:

* **`REGIME_CONFIRMED_FAMILY_ADVERSE`** — G2, G3 pass but G1 fails: the interface
  transmits in a new regime but the effect is specific to the flow-matching
  family. Reported as such; the family-dependence becomes a stated limitation.
* **`FAMILY_CONFIRMED_REGIME_ADVERSE`** — G1 passes, G2/G3 fail: the effect is
  family-general but is a property of the aligned-cube regime. The manuscript
  then states that the composed chain does **not** transmit in a separated
  pressure-gradient regime, retains node 006 as a single-regime result, and the
  venue-shape contract is reported as **not met**.
* **`PARTIAL`** — any other mixture; every cell is reported with its sign and
  interval and no generality claim is made.
* **`ADVERSE`** — no new cell is positive: the manuscript reports that the node-006
  effect does not reproduce outside its own record and family. Node 006 is
  retained in full; the title, abstract and cover letter are narrowed to a
  single-regime single-family offline demonstration; and the frozen success
  criterion is reported as unmet rather than silently deferred.

**Stop rule.** One production run. A failing clause is reported as the result. The
run is **not** repeated with altered capacity, mixture, matching height, split,
units, arms or seeds. Cost ceiling **8 GPU-hours** on `foshan`; if exceeded, the
job is stopped and reported incomplete rather than re-scoped.

---

## 7. Non-regression — tolerances frozen HERE, before observation

### 7a. Exact-identity gate (nothing frozen may move)

| Endpoint | Requirement |
|---|---|
| SHA-256 of `e2_closure_composition_closure_LA.pt` | **unchanged**, and asserted inside the producer before it is used |
| SHA-256 of `e2_closure_composition_results.json` / `_components.npz` | unchanged |
| SHA-256 of `e2_direct_traction_*` (node 005) and `e2_traction_interface_*` (node 004) | unchanged |
| SHA-256 of every companion geometry / separation / thermal / 2-D-to-3-D / scaling source artefact | unchanged, recorded **pre** and **post** run |
| all new artefacts | written under the distinct prefix `e2_generality_*`; nothing frozen is overwritten |

### 7b. The oracle-band ceiling the mandate named and node 006 omitted

Node 005's `band_phys` and `absent_B` per-frame SSE arrays are retained in
`e2_direct_traction_components.npz` for all three seeds and all four regions. The
matched oracle-band ceiling `R²(band_phys) − R²(absent_B)` is therefore computable
from disk **without new computation**, on node 005's exact 240-frame unit, and is
completed in this node by `codes/probes/complete_non_regression_node007.py`.

These are *retained node-005 quantities recomputed*, not new outcomes. The
requirement frozen here is: the recomputation must reproduce node 005's published
band-ceiling values to within `1e-6` absolute, and the transmission ratio of the
repaired closure system to that ceiling must be reported with a paired block
interval in the manuscript.

### 7c. Repaired-system tolerances for the new cells

Family and regime are both changed, so identity is not expected. What is required
is that the repair does not *degrade* what the old system established.

| # | Endpoint (`full_srcex`, decision unit of the cell) | Reference | Tolerance frozen before observation |
|---|---|---|---|
| NRg1 | C1 `Delta` native − absent | node 006 K-family `+0.0667` | `>= +0.0400` |
| NRg2 | C1 `R²(absent)` | node 006 `−0.0892` | `>= −0.1400` |
| NRg3 | C1 ordering `native > absent` | holds | must still hold |
| NRg4 | C1 ordering `native > fartime` | holds | must still hold |
| NRg5 | every cell, `absent` is not the best arm | — | `R²(tau_closure) > R²(absent)` or the cell is reported adverse |
| NRg6 | node-006 K-family primary `Delta` on its own unit | `+0.06064` | **unchanged** — it is retained evidence and is not recomputed, retuned or replaced |
| NRg7 | full release regression suite `bash codes/reproduce_all.sh` | ALL_PASS | must still ALL_PASS after this node's edits |
| NRg8 | clean-tree release verifier `codes/probes/verify_clean_release.py` | currently **FAILS** (`fig8_closure` absent from the manifest) | must **PASS** — this node repairs it |

NRg1's tolerance is set at ~60% of the old effect: loose enough that a
differently-trained model of a different generative class is not failed for being
different, tight enough that a material loss of the established wall-to-field
effect is caught.

### 7d. Companion evidence — preservation by hash, never by re-claim

The geometry-transfer, separation, thermal-transfer, 2-D-to-3-D and
diversity-scaling strengths come from **separate frozen companion models**. This
node does not re-run them and **does not revalidate them**. It demonstrates
preservation only: every companion source artefact's SHA-256 is recorded before
the run and re-verified after it, and the manuscript states explicitly that the
cube/hill interface work is modular and leaves the companion evidence untouched
and unrevalidated. In particular, **cell C2/C3 evaluating a hill record does not
revalidate the companion geometry-transfer result**, which is a closure-side
result on different data with a different model.

---

## 8. Cost, inputs, outputs

* Inputs: `/root/autodl-tmp/cube_les/cube_ds2_float16.complete.npy` (retained on
  `foshan`); `codes/data/case3_grid_64x192_full.npz` (already resident on
  `foshan`, no upload); `codes/results/e2_closure_composition_closure_LA.pt`
  (frozen node-006 closure); `codes/results/e2_direct_traction_components.npz`
  (node-005 indices, read-only).
* Outputs: `codes/results/e2_generality_results.json`, `_components.npz`,
  `_results.sha256`, `e2_generality_closure_LA_hill.pt`, six generator
  checkpoints `e2_generality_C{1,2,3}_s*.pt`.
* Target: `foshan` only, through `cloud/gpu_run.sh --target foshan`.
* Ceiling: 8 GPU-hours. Expected ≈ 3–4.

---

## 9. What each outcome would and would not license

**A `GENERALITY_CONFIRMED` result would license**: the statement that the
composed, source-valid closure→generator interface transmits wall information
into the surrounding field in **two qualitatively different wall regimes** (a
sharp-edged aligned obstacle and a smooth curved separating wall) and under **two
distinct generative families**, offline, with matched no-wall and wrong-instant
controls.

**It would not license**: absolute field accuracy (node 006's absolute
source-excluded skill is negative and remains reported); solver-coupled WMLES;
closure accuracy in general; superiority over any prior architecture; any claim
that the hill cells revalidate the companion geometry, thermal, separation, 3-D
or scaling evidence; or a claim of independent confirmation on the cube, whose
unit is contacted and labelled so.

**If adverse**, the disposition of §6 applies verbatim. Node 006's positive result
is retained in full and unmodified, the adverse cells are reported with their
intervals, and the venue-shape contract is reported as unmet.

---

## 10. Concurrent production repairs (no new computation; frozen scope)

These are the remaining unanimous panel findings. They are executed in this node
from retained arrays and source files, and are listed here so their scope is fixed
in advance and cannot expand into outcome-dependent editing.

1. **Contribution-identity repair.** `manuscript/sections/methods.tex:13–21`,
   `manuscript/supplementary.tex:91–110`, `manuscript/FIGURE_EVIDENCE_AUDIT.md`,
   `manuscript/source_data/README.md`, root `README.md` and `PROJECT_STATUS.md`
   still assert that `L_A` is not evaluated and that no admissible E2 composition
   evidence exists, contradicting the title, abstract, Results and Fig. 1. Exactly
   one identity will hold everywhere: **`L_A ∘ L_B` is evaluated offline; E4 is
   not.** Every "untouched/uncontacted" claim on the cube unit is withdrawn.
2. **Release manifest.** `fig8_closure` and all claim-active E2 sources are added
   to `codes/figures/rebuild_submission_figures.py`, the verifier inventory and
   `SUBMISSION_RELEASE_MANIFEST.json`; `verify_clean_release.py` is added to
   `codes/reproduce_all.sh` so a clean-tree rebuild is part of the pass.
3. **Figure integration.** Panel numbers are corrected everywhere (the composition
   figure is Fig. 7, not Fig. 8) and the figure is placed beside its Results
   section instead of after the references.
4. **Statistical relabelling** in the manuscript to match §5.
5. **Supplementary metadata** title refresh and evidence/custody rows for nodes
   006–007.
6. **Freeze-timestamp erratum** for node 006's `FROZEN_HASHES.json` (recorded
   `19:15` against filesystem/launch times of `19:10`/`19:11`), recorded in an
   auditable sidecar rather than by rewriting frozen evidence.
