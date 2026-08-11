# Preregistration — node_011, Level 3 attempt 9 (v1)

## The single-slot, quality-agnostic, flux-consistent traction interface, with the
## fidelity→gain dose-response as the registered attribution instrument

**Frozen: 2026-08-02, before any node_011 producer outcome exists.**
This document has two freeze points. **v1 (this text)** freezes the design, arms,
endpoints, decision rules, final unit, cost ceilings and stop rules before the
rehearsal (development) run. **v2 (an appended amendment)** freezes the four
rehearsal-determined numerical parameters listed in §8 *after* the rehearsal run and
*before* the FINAL window is contacted. Immutable snapshots of both versions are kept
as separate files (`PREREG_SNAPSHOT_v1.md`, `PREREG_SNAPSHOT_v2.md`) whose SHA-256
values are recorded in `PREREG_HASHES.txt` — the node_010 defect (hashes recorded
without recoverable snapshots) cannot recur.

---

## 1. The three defects being repaired (all three node_010 referees, independently)

1. **Arm-exposure imbalance.** node_010 trained its generator on a seven-arm mixture
   (closure 0.24 vs learned control 0.12), so the attribution comparison was not
   budget-matched at the generator level.
2. **State side channel.** node_010's closure beat the exact traction in the log band
   (+0.178 vs +0.074) while having *worse* a-priori traction fidelity than the learned
   control (0.070 vs 0.084): the generator had learned closure-specific decodings, so
   the gain was matching-height state routing, not wall-information fidelity.
3. **No boundary-flux consistency.** Nothing tied the generated field's wall flux to
   the supplied traction; the registered engineering endpoint (C7) failed.

## 2. The repair, in one paragraph

The generator is trained on **exactly one traction-carrying condition**: the record's
own exact wall traction, corrupted by a **variance-preserving fidelity ladder**
`tau_r = mu + r (tau - mu) + sqrt(1-r^2) sd xi`, `r ~ U[0,1]`, with white or
spanwise-correlated `xi` (correlation length U[2,25] cells), and masked entirely with
probability 0.2 (the `absent` state). Verified numerically: this construction hits
centred fidelity `R^2 = 2r - 1` to three decimals while keeping the marginal scale
(variance ratio 0.997–1.002), so quality cannot be inferred from amplitude. **No
predictor's output is ever seen in training.** Every evaluated estimator
(physics closure, classical equilibrium law, learned data-only control, wrong-time,
spanwise-shuffle) enters at inference through the *same* slot with *identical (zero)*
generator-side training exposure: budgets are matched **by construction**. The
supplied traction is additionally enforced as a **hard signed-Neumann boundary
condition**: at every sampling step the support rows (rows 0–3, cell centres to
y+ = 3.2, strictly inside the viscous sublayer) are replaced, noise-consistently
(RePaint/DDNM-style), by the viscous-sublayer solution `u_i(y_j) = (du_i/dy)|_w y_j`
implied by the supplied traction, so the generated field's wall gradient **equals**
the supplied traction identically. The attribution instrument is the **measured
dose-response curve** gain(fidelity) traced by the exact+noise ladder: every estimator
must land ON the curve at its own measured fidelity; landing significantly ABOVE it is
the registered side-channel detector and fails the experiment.

## 3. Record, axis correction, and evidence label

`CoNFiLD Case2` (Du et al., *Nat. Commun.* **15**, 10416 (2024); Zenodo
10.5281/zenodo.14037782, CC-BY-4.0), the smooth-wall plane channel at
Re_tau = 183.8, wall-resolved (first cell y+ = 0.46), as established and
scaling-recovered by node_010 (`codes/data/build_case2_channel.py`, byte-identical,
sha `49bd3b7b…` for the derived array).

**Axis-identity correction (referee 2, confirmed by measurement).** The 100-point
in-plane tangential axis is **spanwise (z)**, not streamwise: the near-wall (y+ = 15)
u autocorrelation along it has the classic streak negative lobe (first zero at lag 6,
minimum −0.355 at lag 10), impossible for the streak-elongated streamwise direction.
The permutation control is therefore `shuffle_z`, and all statements about wall
integrals are labelled **spanwise line averages at one streamwise station**.

**Evidence label (frozen wording).** The closure reads contemporaneous target-record
velocities at matching heights y+ ≈ 30 and 60. This is an **offline, a-priori
composition on target-record matching-height states**; it is not solver-coupled
deployment, and no runtime-availability claim is attached.

## 4. The FINAL unit, and why it is genuinely untouched

Reversed-time split of the 1200-frame record:

| window | native frames | role |
|---|---|---|
| TRAIN | [746, 1199] (454) | generator + closure + normalisation + all statistics |
| REHEARSAL | [560, 740], stride 10 (19 frames) | development scoring (the never-scored GAP) |
| **FINAL** | **[0, 450], stride 10 (46 frames)** | scored **once**, after the v2 freeze |

`codes/probes/build_case2_contact_ledger.py` (retained, output
`codes/results/case2_contact_ledger.json`) swept **127 retained index-sets across
every artifact in `codes/results/`** plus all declared splits: the minimum frame any
retained producer ever *scored* on this timeline is **500**
(node_010 DEV [500,559]; node_010 TEST {746…1196}; `eval_l3_channel_absolute`
[744,1199]; rasterized-case2 producers ≥ 992). Frames [0,459] were used **only as
training data by now-discarded systems** (node_010 TRAIN [0,499];
`eval_l3_channel_absolute` TRAIN [0,719]; rasterized TRAIN [0,911]). No evaluation
outcome of any kind was ever computed there; no parameter of any of those systems
enters the node_011 system, which trains on [746,1199] only. Disclosed residual
contact: those discarded systems' training losses and node_010's normalisation
constants (six moments + a mean field) were observed; none is used here.

**Rehearsal-final separation.** The rehearsal window's closest frame (560) is 101
native frames (1.09 tau) from the final window's edge (459), and only ~9 of the 46
final frames lie within 2 tau of any rehearsal frame. The v2-frozen parameters are
structural (a correlation coefficient, a noise length, a band choice, a critical
value), not frame-specific; the residual adaptivity is disclosed rather than denied.

**Wrong-time donors.** Rehearsal: donor j = i + 370 (= 4.0 tau); FINAL: donor
j = i + 746 (= 8.1 tau). Every donor lies inside TRAIN, every lag meets the frozen
minimum exactly, donors are unique (no pile-up), and the producer asserts all three
properties — the node_010 clipping defect cannot recur.

## 5. Arms (all through the same slot, same trained generator, same seeds/noise)

| arm | slot content |
|---|---|
| `absent` | nothing (mask state) |
| `exact` | the record's own signed two-component wall traction (the ceiling **by construction**: it is the training modality) |
| `ladder_r0900/0750/0625/0550` | exact, corrupted at r = 0.90/0.75/0.625/0.55 (target centred fidelity 0.80/0.50/0.25/0.10), white noise |
| `matched_noise` (FINAL) | exact, corrupted at the **v2-frozen** (r_m, ell_c) matching the closure's rehearsal-measured fidelity and error correlation length: the information-matched control that preserves support, output dimension, marginal scale and error structure while carrying **no** state side channel |
| `closure` | physics-grounded closure: Reichardt equilibrium scale × bounded learned correction (|a|,|θ| ≤ 0.7 tanh), spanwise-convolutional trunk (circular padding), reading rows y+ ≈ 30, 60 only |
| `equilibrium` | classical Reichardt law alone |
| `learned_dataonly` | trunk-, input-, optimiser- and step-matched direct regression (no equilibrium scale, no wall law) |
| `wrong_time` | closure applied to the fixed-lag donor frame |
| `shuffle_z` | closure traction with spanwise stations permuted (fixed seed) |

Generators: flow matching (primary; seeds 8821/8822) and VP diffusion
`v_cosine_w128` (secondary; seeds 9921/9922; configuration inherited from node_010's
committed DEV selection, a pre-final design decision). Diffusion is evaluated on
{absent, exact, closure} as a sign-robustness check only. 8 members, 32 sampler
steps, backbone identical to node_010.

Scored mask (identical to node_010): rows ≥ 4, excluding closure-read rows 31–33 and
64–66 → 190 of 200 rows. The producer machine-verifies, **before any score exists**,
that every arm's wall-derived conditioning is identically zero on every scored cell,
and a static assert guarantees the Neumann enforcement writes only rows 0–3.

## 6. Endpoints and registered decision rules (conjunctive, falsifiable both ways)

Primary field endpoint: fluctuation R² on `whole_scorable` (support-excluded by
construction), walls averaged within each physical time, flow matching, mean of both
seeds. Primary inference: **three contiguous physical-time blocks** (final window:
15/16/15 strided frames = 150/160/150 native frames each, all above the conservative
1.2551 × max(92.5, tau_train) ≈ 116-native-frame Politis–White scale of the FIELD
series — the contrast series decorrelates faster still, tau_delta ≈ 53 native in
node_010), one-sided t on the three block means with critical value **t\*** whose
type-I error is **demonstrated by simulation at the exact design**
(`codes/probes/calibrate_block_inference.py`, sweep of AR(1) dependence up to the
theoretical strided lag-1 correlation 0.90; t\* frozen at v2 as the smallest value
with simulated one-sided size ≤ 0.05 across the sweep). Moving-block bootstrap ranges
are reported as secondary with an explicit no-coverage-claim label. Seeds are never
averaged before their per-seed deltas are individually reported and gated.

- **C0 — construction.** Support-integrity check passes; wall-gradient consistency
  `|LS-gradient(generated rows 0–2) − supplied traction|/s̄ < 10⁻³` for every
  traction arm. Failure ⇒ `CONSTRUCTION_FAILURE`.
- **C1 — power/ceiling gate.** `exact − absent > 0` on whole_scorable at t ≥ t\*,
  both walls agreeing in sign. Failure ⇒ `UNIT_BLIND`.
- **C2 — PRIMARY usefulness.** `closure − absent > 0` on whole_scorable at t ≥ t\*,
  **and** all three block means positive, **and** both walls positive, **and** both
  seeds positive.
- **C3 — absolute skill.** The closure arm's absolute fluctuation R² on
  whole_scorable > 0.
- **C4 — ceiling ordering (side-channel tripwire).** `closure − exact ≤ +0.01` on
  whole_scorable.
- **C5 — on-curve attribution.** The closure's (measured fidelity, gain) point lies
  **at or below** the ladder-interpolated dose-response curve + m₅ (margin frozen at
  v2 as max(0.02, 2 × rehearsal RMS deviation of the ladder points about their
  interpolant)). Landing above ⇒ `SIDE_CHANNEL_DETECTED` regardless of C2. The same
  on-curve diagnostic is *reported* (not gated) for `equilibrium` and
  `learned_dataonly`.
- **C6 — dose monotonicity.** Spearman rank correlation between measured fidelity and
  gain across {exact + four ladder arms} ≥ 0.7.
- **C7 — wrong-information controls.** `closure − wrong_time > 0` at t ≥ t\* and
  `closure − shuffle_z > 0` at t ≥ t\*; and `wrong_time − absent ≤ +0.02` (wrong
  information must not help).
- **C8 — beats the classical law.** `closure − equilibrium > 0` at t ≥ t\*.
  `closure` vs `learned_dataonly` is **reported head-to-head but is not a gate**: in
  this single-slot design any estimator's downstream gain is predicted by its
  a-priori fidelity (that is the point of the design), so the physics-vs-data
  question is settled a priori and by the companion transfer evidence, not by this
  in-record contrast. This is registered *now*, before any outcome.
- **C9 — engineering endpoint (co-primary).** Tracking error of the instantaneous
  spanwise-averaged tangential momentum flux `⟨u'v'⟩` in a scored band **outside the
  conditioning support**: per-frame absolute error, member-averaged, closure vs
  absent, one-sided t ≥ t\* on three block means; and `exact` must also improve
  (consistency). Band (buffer or log) frozen at v2 from rehearsal; both reported.
  The wall-load endpoint of node_010 is retired *by construction*: the generated wall
  flux now equals the supplied traction identically, so the honest industrial
  consequence lives strictly outside the support.
- **S1 — distributional (secondary, reported with block-t).** CRPS(closure) ≤
  CRPS(absent) direction; energy score and pooled rank histograms reported.
- **S2 — family robustness (secondary).** Diffusion `closure − absent` point estimate
  positive in both seeds; no significance requirement; labelled a sensitivity check.

### Registered outcome labels

| label | condition |
|---|---|
| `INTERFACE_TRANSMITS_FIDELITY` | C0–C9 all pass |
| `TRANSMITS_NO_ENGINEERING` | C0–C8 pass, C9 fails |
| `SIDE_CHANNEL_DETECTED` | C4 or C5 fails |
| `CLOSURE_INSUFFICIENT` | C1 passes, C2 fails (honest adverse: this closure's fidelity is too low to matter on this record) |
| `UNIT_BLIND` | C1 fails |
| `CONSTRUCTION_FAILURE` | C0 fails |

Every label is publishable and will be integrated into the manuscript at equal
prominence; no label is "narrated around". The decision applicator
(`apply_decision_rule_node011.py`, hash-frozen at v2) evaluates the clauses
mechanically from the result JSON.

## 7. Non-regression (frozen NOW, stricter than every predecessor)

Nothing old is retrained, rewritten or renamed. `freeze_node011_expected_hashes.py`
recorded, **before any node_011 outcome existed**, the SHA-256 of **1015 files**
spanning `codes/results/**`, `codes/data/*`, `manuscript/source_data/**` and
`manuscript/figures/*.pdf`. Verification requires every frozen file byte-identical to
its RECORDED hash (not merely present); new files must match declared
`e2_slot_channel_*` / node011 patterns; three documentation files
(source-data README, licences, figure-evidence audit) are whitelisted as mutable with
before/after hashes recorded. Because no shared endpoint is recomputed, no frozen
endpoint can move: the mandated non-regression table degenerates to exact byte
identity plus the complete release suite, which is re-run at the end. The companion
evidence (geometry, separation, thermal, 2-D-to-3-D, scaling) is byte-preserved and
is **not** revalidated by this experiment; the manuscript states the modular boundary.

## 8. Parameters that v2 will freeze (from rehearsal, before FINAL contact)

1. `matched_noise` parameters: r_m = (1 + R²_closure,rehearsal)/2 and ell_c
   (closure-error spanwise integral correlation length, measured at rehearsal).
2. C5 margin m₅ = max(0.02, 2 × rehearsal RMS ladder-about-interpolant deviation).
3. C9 band: buffer or log, whichever the rehearsal shows the exact arm improves more
   robustly (both always reported).
4. t\*: the calibrated critical value from the simulation study at the exact design.

v2 may also strengthen the closure *architecture/training* using rehearsal evidence
alone (rehearsal is development); any such change is recorded in the amendment with
its rationale, and the FINAL window remains uncontacted throughout.

## 9. Cost ceilings and stop rules

| run | target | ceiling |
|---|---|---|
| rehearsal (training + development scoring) | `foshan` | 7 GPU-h |
| FINAL (inference-only, frozen checkpoints) | `foshan` | 4 GPU-h |

One production run per phase. A run exceeding its ceiling is killed and reported
incomplete; no re-running with modified settings after outcomes are seen. A `--smoke`
crash-detection run precedes the rehearsal; its numbers are meaningless and are not
inspected beyond crash/no-crash. If the FINAL run crashes for an infrastructure
reason, it may be relaunched byte-identically; the FINAL window's outcomes are read
only from the completed run's harvested, hash-bound artifacts.

## 10. What this experiment cannot settle

Offline a-priori reconstruction, not solver deployment; one record, one Reynolds
number, marginal log layer (the 30 < y+ ≤ 100 band is named by wall units, not by an
asserted asymptotic log law); closure fitted and evaluated on the same record on
disjoint, contact-audited time windows — cross-record closure transfer remains
companion evidence. A positive result identifies fidelity-attributable propagation
beyond the supplied band; it does not establish that the generator's internal
mechanism is the physical one.

---

## AMENDMENT A1 — 2026-08-02, pre-outcome mechanical fix from the smoke run

The `--smoke` crash-detection run (whose scientific numbers are meaningless at 60
training steps and were not inspected) exposed one mechanical defect via the C0
construction fields: the VP-diffusion sampler's final replacement uses
`vp_alpha_bar(0) = 0.99984`, leaving ~1.1e-2 residual noise on the enforced support
rows, which would fail the frozen C0 wall-gradient threshold (1e-3) for the
diffusion family. Fix: an exact terminal projection `x[rows 0..N_C-1] = bc` after
the last step (the standard DDNM hard final step; flow matching already terminates
exactly, measured 1.7e-7). No estimand, split, arm, control, metric, threshold or
decision rule changed; no production outcome existed. The amended producer hash is
recorded alongside the original in `PREREG_HASHES.txt`.
