# Preregistration — direct native-traction conditioning (E2 repair)

**Frozen before any held-out outcome of this experiment was observed.**
Node `development/nodes/node_005`, Level 3 attempt 3, 31 July 2026.

## 1. What failed, and which link owns the failure

Node 004 executed the interface in the form the field reaches for first:

```
wall traction --(equilibrium/Reichardt lift)--> velocity band --(hard clamp)--> frozen generator
```

and it failed: even the target's own traction left the field worse than supplying
nothing (`-0.03248 [-0.04821, -0.02102]` against the absent band). The panel
returned that result as a **confounded** negative and named the two confounds.

| ID | Confound | Evidence from node 004 |
|----|----------|------------------------|
| C1 | The lift is an equilibrium *model* of an *instantaneous* field | lifted-band fluctuation `R^2 = -0.109 +/- 0.185` even with exact `u_tau`; worse than the band's own mean |
| C2 | Every lifted arm is out of the frozen generator's conditioning distribution | that generator only ever saw dense true velocity bands; the hard clamp injects adapter error along the whole trajectory |

Both confounds live in the **adapter**, not in the wall information. The panel
also established that the quantity node 004 called "oracle traction" was a
Reichardt inversion of the target's first-cell velocity, carrying a `+u_tau^2`
sign against the manuscript's `-u_tau^2` convention. That mislabelling is
withdrawn here and the calculation is renamed in every artefact.

## 2. The repair

**Delete the adapter.** Train the conditional generative model *prospectively,
from scratch,* to condition **directly on the signed wall-on-fluid traction
field**. No velocity reconstruction, no equilibrium law anywhere in the
interface, and no clamp: the traction enters only as a conditioning channel and
the sampler is free on every fluid cell. C1 is removed because no lift exists;
C2 is removed because the model is trained on exactly the conditioning it is
tested on.

This is the repair the panel named first ("preferably direct traction
conditioning ... so the evaluation is in distribution").

## 3. Native traction — definition, sign, and why no pressure is needed

For every fluid cell adjacent to a no-slip surface, with wall normal `n` pointing
from wall into fluid and first-cell distance `d = 0.5*Delta`:

```
u_t    = u - (u.n) n
tau_wf = -P_t[ nu (grad u + grad u^T) n ]  ~=  - nu u_t / d
```

* The tangential projector `P_t` annihilates the isotropic pressure term
  **identically**. The manuscript's Eq. (2) traction therefore requires no
  pressure channel, and node 004's blanket claim that the retained record cannot
  yield native traction is **withdrawn**. It is fully determined by the retained
  velocity record and the molecular viscosity `nu = 2.0e-4`.
* The sign is the physical one: the wall retards the fluid, so `tau_wf` is
  anti-parallel to the near-wall velocity, matching the Methods `-u_tau^2 t_hat`
  convention. The producer asserts this as a hard gate
  (`tau . u_t < 0` on every face).
* Cells touching two no-slip faces (96 edge cells of 6,816) sum the per-face
  tractions; the face count is a declared part of the map.

**Honest information statement, fixed in advance.** On a wall-resolved grid the
native viscous traction is an *invertible linear map* of the first-cell
tangential velocity. Supplying it is therefore exactly as informative as
supplying two velocity components on one wall-adjacent cell layer — strictly
less than the published oracle band (three components, two layers). This is
stated in the manuscript regardless of outcome. The traction is read from the
held-out target, so it is an **oracle** traction: it bounds what any closure
could transmit and is not a closure-accuracy measurement.

## 4. Source, units, split

| Item | Value |
|------|-------|
| Record | `/root/autodl-tmp/cube_les/cube_ds2_float16.complete.npy`, `(1101, 3, 48, 96, 48)` float16 |
| Regime | Coceal aligned cube array, `lambda_p = 0.25`, pitch `2h`, `Re_h = 5000` |
| Grid | uniform `Delta = 1/24`, domain `2 x 4 x 2`, cube `[0.5,1.5] x [0,1] x [0.5,1.5]` |
| Split rule | **byte-identical to the published producer**: first 60% train, one integral-time gap, chronological remainder test |
| Train | snapshots `0 .. 659` |
| Gap | `ceil(tau_integral)` snapshots, recomputed in-run from the kinetic-energy autocorrelation |
| Test | chronological remainder, **all** of it used (`nmax = 343`), against 160 in the published run |

The split is inherited, not chosen. No split, gap or region was tuned in this
node. Model development used the published architecture verbatim; no
hyperparameter was selected against a held-out score.

## 5. Arms — frozen before execution

Family **T** (traction-conditioned model, 25% conditioning dropout so that one
network serves both primary arms and the contrast carries no between-model
confound):

| Arm | Conditioning | Role |
|-----|--------------|------|
| `tau_native` | native signed wall-on-fluid traction | **PRIMARY** |
| `absent` | conditioning mask zeroed | **PRIMARY CONTROL** |
| `tau_trainmean` | training-window time-mean traction | information-matched null |
| `tau_fartime` | traction of a far-time donor snapshot | temporal scramble |
| `tau_signflip` | `-tau_native` | signed-convention control |
| `tau_shuffle` | `tau_native` permuted across wall cells | spatial-correspondence control |

Family **B** (physical-wall oracle velocity band, same budget, same seeds):
`band_phys`, `absent_B` — the **matched ceiling**, retrained under identical
capacity and compute so the transmission ratio is meaningful. The computational
lid is excluded from its support, carrying forward node 004's leak repair.

Family **D** (identical architecture trained as a deterministic regressor):
`det_tau`, `det_absent` — information-matched non-generative baseline.

All arms share targets, sampler noise, scoring masks, capacity and compute.

## 6. Metrics, uncertainty, and what counts as a pass

* Primary estimand `Delta_tau = R^2(tau_native) - R^2(absent)` on
  `full_support_excluded`; reported identically on `near`, `outer` and
  `uniq_raster` (node 004's raster-duplication repair).
* Every scored region **excludes the entire conditioning band**, so no supplied
  cell is ever scored.
* Uncertainty: circular moving-block bootstrap, 4,000 draws, seed 44, at both the
  release block and the **conservative** `1.2551 x tau` block, plus **3 training
  seeds** with the across-seed spread and the all-seeds-same-sign flag reported.
  Seed variance is a first-class quantity here because the panel showed the
  earlier intervals omitted it.
* Matched ceiling `Delta_band = R^2(band_phys) - R^2(absent_B)`; transmission
  ratio `Delta_tau / Delta_band` with its own interval and a
  denominator-excludes-zero flag.

**Decision rule, fixed in advance.**

* **POSITIVE** if `Delta_tau > 0` with the conservative-block 95% interval
  excluding zero on `full_support_excluded` **and** the same sign on all three
  seeds **and** `tau_native` beats `tau_trainmean`, `tau_fartime` and
  `tau_shuffle`. Interpretation: closure-supplyable surface information alone
  propagates into the field; the earlier negative was the adapter, not the
  information.
* **NULL/ADVERSE** otherwise. Interpretation: reported straight, with the
  transmission ratio and the deterministic baseline used to separate
  "information absent" from "generator cannot exploit it". No re-run, no arm
  added, no threshold moved after the fact.

Either outcome is written into the manuscript. The adverse node-004 result, the
2,304-cell leak correction and the raster audit are retained intact in both
branches.

## 7. Physical validation gates (hard, executed in-run before any arm)

1. `tau . u_t < 0` on **every** no-slip face — the signed convention is obeyed.
2. Streamwise wall-on-fluid viscous force is negative.
3. That force is compared against the **independent spectral-element quadrature**
   of the continuation blocks (`codes/results/wall_loads_audit_b001.json`,
   `b002.json`: `fx_viscous_mean = -0.018540`, `-0.017149`, gates all passing).
   Reported ratio; flagged if outside `[0.3, 3.0]`. This is an external check of
   the finite-difference extraction against a different code path on a different
   time window of the same stationary flow.
4. Traction is tangential by construction; pressure enters nowhere.

Gate 1 is fatal: if it fails the run is void and nothing is reported as traction.

## 8. Cost ceiling and stop rule

* Target: `foshan` only, via `cloud/gpu_run.sh --target foshan`. `orig` is not used.
* Frozen budget, fixed from a 400-update timing probe on the **training window**
  (measured 16 updates/s at base 32, batch 4):

  | Item | Value |
  |------|-------|
  | Trainings | 7 = family T x 3 seeds + family B x 3 seeds + family D x 1 seed |
  | Architecture | published `FlowUNet3D`, base width 32 |
  | Updates | 20,000, batch 4, AdamW lr 2e-4, EMA 0.9995 |
  | Held-out targets | 240 (published run used 160) |
  | Posterior | 8 members, 32 sampler steps |
  | Seeds per arm | 3 for `tau_native`, `absent`, `band_phys`, `absent_B`; 1 for the four scramble/null controls and the deterministic baseline |
  | Bootstrap | 4,000 draws, seed 44 |

  Ceiling **4 GPU-hours**. If exceeded the run is stopped and reported with
  whatever seeds completed, labelled as such.
* The control arms carry one seed rather than three. This is a declared cost
  bound, not a silent cap: it is reported with the results, and the primary
  estimand and the matched ceiling both carry the full three seeds.
* **One evaluation.** The held-out arms are scored once. No arm is re-run after
  its number is seen.

## 9. Declared prior contact — complete and exact

Two pre-freeze executions touched this pipeline. Both are declared in full,
including the held-out numbers that were visible, because partial disclosure of
prior contact is exactly what this preregistration exists to prevent.

**(1) Plumbing smoke** — `--smoke --tag e2_direct_smoke`, base 16, 60 updates,
8 targets, 2 members. It exposed two defects, both repaired before the probe:

* the hard signed-traction gate **failed** on the `top` face (dot product exactly
  `0.0`) because the lid-adjacent raster row is identically zero. The
  computational lid was therefore removed from the traction support, which also
  aligns it with the ceiling arm's physical-wall support. Gate 1 now passes
  strictly on all six physical faces;
* per-sample traction was being rebuilt on the CPU inside the training loop. It
  is now precomputed once over the 4,512 wall cells and scattered on the GPU.

Two provenance repairs were made at the same time: `integral_tau` was replaced by
the **byte-identical published implementation** so the split/gap/block are
inherited rather than re-derived, and the force-balance gate was re-pointed at
the **physical-wall** native quadrature (block total minus the lid).

**(2) Timing probe** — `--tag e2_direct_probe`, base 32, **400 updates**, 24
targets. Its purpose was to measure throughput (16 updates/s) so that §8's budget
fits the 4-hour ceiling. It also confirmed the repaired gates:
`tau . u_t < 0` on all six faces, and the finite-difference streamwise wall force
`-0.0076808` against the independent spectral-element quadrature `-0.007418`
(b001) and `-0.006391` (b002), i.e. ratios **1.035** and **1.202**.

**Held-out numbers seen during the probe.** Two arm scores were printed:
`tau_native R^2_full = -0.62948` and `absent R^2_full = -0.64402`, from a
**400-update** model — 2% of the production budget, far from convergence, on 24
of 240 targets, single seed, no bootstrap. They are recorded here rather than
omitted.

**What was and was not chosen from prior contact.** The arm set, the estimands,
the four scoring regions, the uncertainty protocol and the decision rule of §5
and §6 were written before either execution and are unchanged by them. The only
post-contact changes are the three listed above — a physical-validity repair, a
performance repair and two provenance repairs — plus the cost-bounded seed
allocation derived from the measured throughput. No threshold was moved, no arm
was added or removed, and the split was not touched.

Artefacts of both executions are retained under the `e2_direct_smoke` and
`e2_direct_probe` tags for audit.

## 10. Scope this experiment does not claim

* It is **not** a closure-accuracy measurement (`L_A` remains open; the traction
  is oracle).
* It is **not** solver-coupled WMLES.
* It does **not** supersede the closure-side companion evidence.
* A positive outcome licenses the statement that *closure-supplyable wall
  information propagates through a conditional generative model into the
  off-wall field*, offline, on this regime — and nothing beyond it.

## 11. Artefact binding

| Artefact | SHA256 |
|----------|--------|
| `codes/gpu/eval_e2_direct_traction.py` | recorded in `FROZEN_HASHES.json` at freeze time |
| record `cube_ds2_float16.complete.npy` | recomputed in-run into `_meta.data_memmap_sha256` |
