# Preregistration — source-valid closure-to-generator composition (E2), and the
# mandated old-versus-repaired non-regression table

**Frozen before ANY held-out generator outcome of this experiment was observed.**
Node `development/nodes/node_006`, Level 3 attempt 4, 31 July 2026.

This document is hash-bound in `FROZEN_HASHES.json` together with the producer
`codes/gpu/eval_e2_closure_composition.py`. The production job is launched only
after that freeze.

---

## 0. Full disclosure of prior outcome contact (read this first)

Node 005's preregistration was criticised — correctly — because a timing probe had
already printed a favourable held-out ordering before the document was frozen. The
corresponding disclosure for this node is:

| Question | Answer |
|---|---|
| Was any held-out **generator** outcome of this experiment observed before this freeze? | **No.** |
| Was any held-out **closure** outcome observed before this freeze? | **No.** |
| Was a smoke run executed before this freeze? | Yes — one, `e2_closure_smoke`, log `foshan_20260731T190533.log`. |
| What did the smoke evaluate? | Only **training** frames. The producer hard-codes `strict_idx = train_idx[:6]`, `prior_idx = train_idx[6:12]` whenever `--smoke` is set, so the strings `CONFIRM_STRICT` / `CONFIRM_ALL` in the smoke log denote snapshots 0–11, all of which are inside the training window. |
| Is that machine-checkable? | Yes. `provenance.smoke == true` in `e2_closure_smoke_results.json`, and the smoke path's evaluation indices are train indices. `verify_node006.py` asserts it. |
| Was the smoke's closure a-priori score on held-out data seen? | No. `closure_apriori.confirm_strict` in the smoke run was computed on `train_idx[0,2,4]`, not on held-out frames, for the same reason. |
| Model/hyperparameter selection against a held-out score? | None. Architecture, capacity, optimiser, steps, matching-height offsets, feature list and mixture weights were fixed from the frozen node-005 producer and from physics, not tuned. |

The smoke was run at `--base 16`, 60 generator steps and 2 members; its generator
numbers are meaningless by construction and are not used anywhere.

**Closure development discipline.** The closure `L_A` is fitted on snapshots
0–599 and model-selected against snapshots 600–659 — a validation window *inside
the training span*. No held-out frame participates in closure development.

---

## 1. The defect this node exists to repair

The manuscript's fixed story is one integrated system:

```
q^coarse --L_A (physics-grounded wall closure)--> tau_w --L_B (conditional generator)--> field
```

Three attempts have now probed it:

| Node | What was executed | Outcome | Why it did not establish the story |
|---|---|---|---|
| 004 | traction → equilibrium velocity-band lift → hard clamp → frozen generator | negative | confounded by the adapter (equilibrium model of an instantaneous field; every arm out of the generator's conditioning distribution) |
| 005 | **oracle** traction (read from the held-out target's own first cell) → generator, no adapter | positive, `Delta = +0.07415` | `L_A` was never evaluated. The input is the target's own wall-adjacent state, so this bounds what a closure could transmit; it is not a closure result |
| 006 (this node) | **closure-predicted** traction from matching-height state only → generator | to be determined | — |

All three panel seats identified the same blocking defect: **`L_A` is open.** This
node closes it.

---

## 2. `L_A` — the closure, defined completely

For every (wall cell, no-slip face) pair, with `n` the unit normal pointing from
the wall into the fluid:

**Inputs — matching-height outer-flow state only.** Velocity is read at wall
distances `2.5*Delta` and `4.5*Delta` (cell offsets 2 and 4 along `n`), i.e. the
standard WMLES matching-height seam. *The wall-adjacent cell is never read.* This
is the load-bearing difference from node 005: the closure has no access to the
quantity it is predicting, nor to any invertible function of it.

Availability was verified geometrically before the run: both matching cells are
in-bounds and fluid for **100%** of the 4,608 pairs on all six physical faces.

**Physics-grounded form.**

```
u_t(y_m) = u - (u.n) n           tangential velocity at the matching height
u_tau_eq : solve  |u_t(y_m1)| = u_tau * f_Reichardt(y_m1 u_tau / nu)     (bisection, 60 its)
e1 = u_t(y_m1)/|u_t(y_m1)| ,  e2 = n x e1                                (tangent frame)

tau_hat = -(u_tau_eq)^2 * exp(a) * [ cos(theta) e1 + sin(theta) e2 ]
```

with `kappa = 0.41`, `C = 7.8`, `nu = 2.0e-4`.

* Setting `a = theta = 0` recovers the **classical equilibrium wall model**
  exactly. That arm is evaluated as `tau_eqwm`.
* `(a, theta)` are a bounded learned non-equilibrium correction:
  `a = 1.5 tanh(.)`, `theta = (pi/3) tanh(.)`. The output layer is initialised to
  **zero**, so training starts *at* the equilibrium law and can only depart from
  it by earning it.
* Because `|theta| <= pi/3 < pi/2`, the sign convention `tau . u_t < 0` (the wall
  retards the fluid) holds **architecturally**, for every cell and every parameter
  value. It is not a diagnostic that could fail; it is a property of the model
  class. This is the manuscript's `-u_tau^2 t_hat` convention.

**Features** (8, all dimensionless or logarithmic, all available to a coarse-grid
solver, none derived from the wall): matching-point `y+`; `|u_t(y_m2)|/|u_t(y_m1)|`;
`u_n(y_m1)/|u_t(y_m1)|`; `u_n(y_m2)/|u_t(y_m1)|`; `cos` and `sin` of the skew
between the two heights; `log|u_t(y_m1)|`; and a local acceleration proxy
`log(|u_t(y_m1)| / boxmean_3x3x3 |u_t(y_m1)|)`. Plus a 6-way face embedding
(geometry, known a priori). Trunk: `14 -> 64 -> 64 -> 2`, SiLU.

**Loss / fit.** MSE against the native traction of the same snapshot, normalised
by the training-window traction s.d.; 12,000 AdamW steps on snapshots 0–599.

**Freeze.** The trained closure is written to
`codes/results/e2_closure_composition_closure_LA.pt`; its SHA-256 is recorded in
the result JSON. It is applied unchanged to every evaluated snapshot.

---

## 3. Evidence boundary — stated before the outcome, binding whatever it is

* This is an **a-priori (offline) wall-model protocol**. The closure consumes
  resolved-LES state at the matching height, which is state a coarse-grid solver
  carries. Nothing here advances a momentum equation, so it is **not** solver-coupled
  WMLES and must never be described as deployment.
* `tau_native` remains an **oracle** arm. It is the traction ceiling, not a closure.
* Every cell the closure **reads** (both matching layers) is excluded from the
  primary scoring region `full_srcex`, in addition to every cell whose traction is
  **supplied**. The decision rule runs on `full_srcex`.
* The record is one aligned-cube array at `Re_h = 5000`. One regime, one geometry,
  one generative family. No claim of generality follows from it.

---

## 4. Units — and the strictly uncontacted confirmation set

Split rule inherited byte-identically from the published producer:
`n = 1101`, train `0–659`, gap `98` snapshots (`tau_integral = 97.592`),
test `758–1100` (343 frames).

| Unit | Definition | n | Role |
|---|---|---|---|
| `CONFIRM_STRICT` | test frames **never scored by node 005**, computed mechanically as `setdiff1d(test_idx, node005.eval_idx)` | **103** | **the decision rule is evaluated here** |
| `CONFIRM_ALL` | node 005's exact 240 evaluation frames | 240 | required so the non-regression table compares *identical* units |

The 103 strict frames are the genuinely uncontacted unit the panel asked for. No
score of any kind has ever been computed on them by this project. The set
difference is recomputed from node 005's retained `eval_idx` inside the producer,
so it cannot be chosen by hand.

---

## 5. Arms

**Family K** — generator retrained with a **declared conditioning mixture**, so
that the decision-bearing arms are *in-distribution at evaluation time*. This is
the direct repair of the panel's causal-control objection.

Mixture (frozen): `tau_native 0.40 / tau_closure 0.40 / absent 0.20`.
Consequently `tau_closure`, `tau_native` and `absent` are all training classes; no
primary contrast carries a distribution-shift confound. Three seeds (8801–8803).

| Arm | Conditioning | Seeds | Status |
|---|---|---|---|
| `tau_closure` | **closure-predicted** traction, matching-height inputs only | 3 | **PRIMARY** — in-distribution |
| `absent` | conditioning zeroed | 3 | **PRIMARY CONTROL** — in-distribution |
| `tau_native` | oracle traction from the target's first cell | 3 | oracle ceiling — in-distribution |
| `tau_eqwm` | pure equilibrium closure, no learned correction | 3 | classical-physics baseline |
| `tau_fartime` | a **real** traction field from a far-time donor snapshot | 3 | **in-distribution wrong-information control** — the decision-bearing control |
| `tau_trainmean` | training-window time-mean traction | 1 | information-matched null |
| `tau_shuffle` | traction permuted across wall cells | 1 | **declared out-of-training probe** |
| `tau_signflip` | `-tau` | 1 | **declared out-of-training probe** |

`tau_fartime` is a genuine traction field carrying the wrong instant. It is
therefore inside the training conditioning distribution and isolates
*instantaneous correspondence* with no distribution-shift confound. It, not the
shuffle/sign probes, bears the decision-rule clause.

`tau_shuffle` and `tau_signflip` are **not** training classes. They are labelled
out-of-training sensitivity probes here, in the results and in the manuscript, and
support no causal statement. Node 005's language that the sign orientation is
"physically load-bearing" is withdrawn.

**Family T** — node 005's frozen checkpoints `e2_direct_traction_T_s7701..3.pt`,
reused **byte-for-byte, retrained not at all**. Arms `tau_closure`, `tau_native`,
`absent`, first seed. This asks a separate question: does the closure signal work
through the *already published* frozen interface, without any retraining?

---

## 6. Metrics

Primary estimand:

```
Delta_closure = R^2(tau_closure) - R^2(absent)      on `full_srcex`, unit CONFIRM_STRICT
```

`R^2` is the fluctuation-balanced coefficient about the training-window mean
field, identical in definition to the published producer. Reported on seven
regions: the four node-005 regions verbatim (for the non-regression table) plus
`full_srcex`, `near_srcex`, `outer_srcex` (closure inputs additionally excluded).

Uncertainty: circular moving-block bootstrap, 4,000 draws, at the release block
and at the conservative `1.2551*tau` block. **In addition**, and answering the
panel directly, every delta reports a **crossed seed x time-block** interval that
resamples training seeds together with time blocks. The conservative block governs
the decision rule. All intervals are conditional on this record and are labelled
as such; `n_effective` is reported for every unit.

New endpoints this node adds (node 005 had none of these):

* **Distributional**: per-target CRPS and per-cell 3-vector **energy score** over
  the 8 posterior members; **rank histogram** of the truth among members with its
  L1 reliability deviation. Computed on `full_srcex`.
* **Ensemble-size sensitivity**: posterior-mean SSE at 1, 2, 4 and 8 members.
* **Engineering consequence**: the streamwise viscous **wall force** implied by
  the reconstructed near-wall field, versus the target's, as relative bias,
  relative RMSE and correlation. This is the bounded industrial endpoint — a
  wall-load consequence — and it is offline, not solver-coupled.

---

## 7. The registered decision rule

Evaluated mechanically by `apply_decision_rule.py` from the result file, on
`full_srcex`, unit `CONFIRM_STRICT`.

| # | Clause | Pass condition |
|---|---|---|
| 1 | Primary sign | `Delta_closure > 0` |
| 2 | Significance | conservative-block 95% interval of `Delta_closure` excludes 0 |
| 3 | Seed robustness | same sign on all three training seeds |
| 4 | Crossed uncertainty | crossed seed x block 95% interval excludes 0 |
| 5 | Wrong-information control | `Delta_closure > Delta_fartime`, and the conservative-block interval of `closure_minus_fartime` excludes 0 |

**VERDICT = POSITIVE iff all five clauses pass.** Anything else is reported as the
outcome it is. Clauses 4 and 5 are new and are deliberately harder than node 005's.

Reported but **not** pass conditions (they are descriptive, and pre-declaring them
as gates would invite outcome tuning): the transmission ratio
`Delta_closure / Delta_native`; `Delta_eqwm`; the Family-T transfer deltas; the
distributional scores; the wall-force endpoint; absolute `R^2` levels.

**Stop rule.** One production run. If it fails a clause, the failure is reported
as the result; the run is not repeated with altered hyperparameters, altered
mixture, altered matching height or altered units. Cost ceiling **8 GPU-hours** on
`foshan`; if exceeded, the job is stopped and reported as incomplete rather than
re-scoped.

---

## 8. The mandated non-regression table — tolerances frozen HERE

Per the 31 July non-regression mandate. Shared endpoints are compared on
**identical units** (`CONFIRM_ALL` = node 005's exact 240 frames), identical region
definitions (node 005's four regions, verbatim), identical sampler noise seeds
(`9100 + j` in the same batch order), identical donor roll and identical
conditioning normalisation. The producer recomputes `tau_sd` on the unique-cell
summed traction precisely as node 005 did, so the reused Family-T arms are
reproducible rather than merely similar.

### 8a. Exact-identity gate (frozen Family T, no retraining)

| Endpoint | Requirement |
|---|---|
| `T:tau_native` `R^2`, all four node-005 regions | within **0.005** absolute of node 005's three-seed value |
| `T:absent` `R^2`, all four node-005 regions | within **0.005** absolute of node 005's three-seed value |
| Family-T checkpoint SHA-256 (×3) | **unchanged** from node 005 |

Only seed 7701 is re-evaluated for Family T within the cost ceiling; it is compared
against node 005's seed-7701 value, and the checkpoint hashes of all three are
verified unchanged.

### 8b. Repaired-system regression tolerances (Family K vs node-005 Family T)

Family K is a *different* model (new mixture, new seeds), so identity is not
expected. What is required is that the repair does not degrade the endpoints the
old system established. Tolerances, frozen before observation:

| # | Endpoint (`CONFIRM_ALL`, node-005 regions) | Old (node 005) | Tolerance — repaired system must satisfy |
|---|---|---|---|
| NR1 | `Delta` native−absent, `full_support_excluded` | +0.07415 | `>= +0.0500` |
| NR2 | `Delta` native−absent, `near_...` | +0.14024 | `>= +0.0950` |
| NR3 | `Delta` native−absent, `outer_d_gt_0p5h` | +0.00649 | `>= -0.0020` |
| NR4 | `Delta` native−absent, `uniq_raster...` | +0.09504 | `>= +0.0650` |
| NR5 | `R^2(tau_native)`, `full_support_excluded` | −0.01432 | `>= -0.0350` |
| NR6 | `R^2(absent)`, `full_support_excluded` | −0.08847 | `>= -0.1150` |
| NR7 | wrong-information ordering | native > fartime | must still hold, `full_support_excluded` |
| NR8 | no-wall ordering | native > absent | must still hold, all four regions |

Tolerances are set at roughly two-thirds of the old effect (NR1–NR4) and at
0.02–0.03 absolute on levels (NR5–NR6): loose enough that a differently-trained
model is not failed for being different, tight enough that a material loss of the
established wall-to-field effect is caught.

### 8c. Companion evidence — preservation by hash, not by re-claim

The geometry-transfer, separation, thermal-transfer, 2-D-to-3-D and
diversity-scaling strengths come from **separate frozen companion models**. This
node does not re-run them and **does not claim to have revalidated them**. It
demonstrates preservation only:

* every companion source artefact's SHA-256 is re-verified unchanged against
  `SUBMISSION_RELEASE_MANIFEST.json`;
* the complete release regression suite `bash codes/reproduce_all.sh` is re-run;
* the manuscript states explicitly that the cube interface repair is modular and
  leaves the companion evidence untouched and unrevalidated.

All new artefacts are written under the distinct prefix `e2_closure_composition_*`.
No frozen checkpoint, result or source-data file is overwritten.

---

## 9. What a POSITIVE result would and would not license

**Would.** That a physics-grounded wall closure, fed only matching-height outer
state, emits a wall traction that measurably improves a conditional generative
reconstruction of the surrounding turbulent field on strictly uncontacted held-out
data — i.e. the integrated `L_A ∘ L_B` chain transmits, offline, in this regime.

**Would not.** Closure accuracy in general; solver-coupled WMLES; absolute field
accuracy (the node-005 absolute skill was negative and any such result must be
reported); generality across regimes, geometries or generator families;
superiority over prior architectures.

**If ADVERSE.** The result is reported straight, the manuscript states that the
composed chain does not transmit in this regime, node 005's oracle result is
retained as the interface-feasibility bound, and the failing link is localised
using the arm ladder (`tau_eqwm` vs `tau_closure` vs `tau_native` separates
closure error from interface loss; the a-priori closure `R^2` separates estimation
from propagation).
