# Preregistration — smallest decisive test of the closure-to-generator interface

Written and hashed **before** any GPU job was launched. No result existed when this file was
frozen. Registered by the Worker acting as responding author, node 004, Level 3, attempt 2.

## 1. Which submission-blocking defect this addresses

All three Level-1/2/3 panel seats named the same blocker in writing:

- skeptic (node 003): *"the intended chain—coarse state, calibrated inputs, closure, signed
  two-component local-frame wall-on-fluid traction, lift and generator—is not evaluated"*; step 3
  of its required alternative approach is to *"preregister the smallest decisive offline E2
  evaluation ... Any heavy execution must use `cloud/gpu_run.sh --target foshan`"*.
- referee (node 003): *"The frozen closure export lacks the branch network ... no
  closure-to-traction-to-lift-to-generator result follows"*.
- champion (node 003): *"the cube experiment begins only at the final, target-derived
  velocity-band arrow"*.

Node 004's read-only recovery audit (`E2_ASSET_RECOVERY_AUDIT.json`, produced earlier in this same
node) established which E2 dependencies are recoverable and which are not:

| Dependency | Status after recovery audit |
|---|---|
| Frozen source-general closure branch weights + set normalisation | **NOT recoverable** |
| Source-native record with pressure for closure input features | **NOT recoverable** (raw nekRS `.f#####` fields pruned from `foshan:/root/autodl-tmp/cube_les`; retained record is velocity-only) |
| Wall geometry, wall-normal directions, tangent bases, viscosity | **Recoverable** — analytic Coceal geometry; `nu = 2e-4` pinned in `codes/data/cube_record/yplus_preflight.json` and `codes/cube_les/extract_wall_pressure_shear.py` |
| Signed two-component traction-to-generator lift | **Specifiable and freezable** — Reichardt equilibrium reconstruction already implemented in `codes/closure/wall_closure.py` (`reichardt_uplus`, `reconstruct_u`) |
| Frozen generator with retained per-time components | **Recoverable** — `codes/results/cube3d_coupling_adequate.pt`, sha256 `447b628e…` |

The chain therefore splits into two links:

- **L_A** coarse state → closure → traction. **Cannot** be executed: the closure's four input
  features require a pressure gradient that no retained artefact contains.
- **L_B** traction → lift → conditioning band → frozen generator → field. **Can** be executed
  exactly, and has never been evaluated anywhere in this project.

L_B is the load-bearing interface approximation of the paper's whole thesis. It is currently
asserted in prose and drawn in Fig. 1 but has never been measured. This preregistration covers
L_B only. It is the smallest calculation that can decide whether the paper's integrated
architecture is viable at all, because **L_B bounds L_A ∘ L_B from above**: if a perfect wall
closure cannot move the field through this interface, no closure can.

## 2. What is deliberately NOT claimed

- This is **not** solver-coupled WMLES (E4) and does not become it.
- This is **not** a runtime closure deployment: the traction in the primary new arm is derived
  from the held-out LES record and is therefore **target-derived / oracle**. It is an
  **upper bound on** what any closure could transmit, not a measured closure performance.
- No new LES, no new training, no fine-tuning. The generator is frozen and byte-identical to the
  published M1 posterior; only its conditioning input changes.
- E3 companion closure-side evidence is untouched.

## 3. Frozen protocol

**Record.** `codes/data/cube_record/cube_ds2_float16.npy`, 1101 post-spin-up snapshots,
(3, 48, 96, 48), t ∈ [40.20, 370.00], sha256 `8bac93f1537eab6667d692282b76c7bccd28f28965d35ea97668bcc2567bc45a`.

**Split.** Byte-identical to the published producer `eval_cube_3d_coupling.py`: energy integral
time τ = 97.59 snapshots; first 60 % (660) train; one-τ exclusion gap (98); chronological
remainder (343) test; evaluation thinned evenly to 160 targets; donor = `roll(test_idx, 80)`.
No re-selection.

**Generator.** `cube3d_coupling_adequate.pt` (base 48, 8.65 M parameters, 90 000 updates),
sha256 `447b628e8624b4be9a7af6d6a264d27744a7daff3fd540b53fcfc82584929c53`, loaded EMA-complete.
Sampler: 32 steps, 8 members, common noise across arms (`seed = 9100 + j`), batch 4 — identical to
the published run.

**Scoring.** Unchanged: channel-standardised fluctuation R² against the train-split mean field on
`full_support_excluded` (fluid ∧ ¬band), `near_support_excluded_d_le_0p5h`, `outer_d_gt_0p5h`.
The supplied band is always excluded from scoring. Circular moving-block bootstrap, block 49,
B = 4000, seed 44 — identical to the published run.

**Wall geometry (analytic, no fitting).** Eight no-slip surfaces of the Coceal cell: floor y = 0,
top y = 4, cube top y = 1, cube faces x = 0.5, x = 1.5, z = 0.5, z = 1.5. Normals point from solid
into fluid. Each band cell is assigned to the surface with the smallest perpendicular distance
among surfaces whose in-plane extent (dilated by two cells to cover edge-diagonal cells) contains
it; ties broken by fixed surface order. The anchor of a band cell is the first fluid cell of its
owning surface at perpendicular distance 0.5Δ, Δ = 1/24.

**Frozen traction estimator (equilibrium inversion).** At each surface anchor with tangential
velocity `u_a = |v − (v·n)n|` at `d_a = 0.5Δ`, solve for `u_τ`
```
u_a = u_τ · f_R(d_a u_τ / ν),   f_R(y⁺) = ln(1+κy⁺)/κ + C[1 − e^{−y⁺/11} − (y⁺/11) e^{−y⁺/3}]
```
by 80-step bisection on `u_τ ∈ [0, 10]`, with κ = 0.41, C = 7.8 (`wall_closure.reichardt_uplus`,
verbatim), ν = 2e-4. The signed two-component local-frame wall-on-fluid traction is
`τ = u_τ² t̂`, `t̂` the unit tangential direction of the anchor velocity, stored as both signed
components in the surface's tangent basis.

**Frozen lift (surface → volume).** For every band cell at wall distance d owned by that surface:
```
v_lift(d) = t̂ · u_τ · f_R(d u_τ / ν),   wall-normal component = 0.
```
Nothing else enters. Both κ, C and ν are fixed above and may not be tuned after any field result
is seen.

## 4. Arms

Reproduction guard (must match the published M1 values):

| Arm | Band content |
|---|---|
| `correct` | oracle full LES band (published 0.08949 / 0.24156 / −0.06623) |
| `no_wall` | absent (dropout branch) (published −0.08436 / −0.09042 / −0.07819) |
| `wrong_wall` | far-time donor band (published −0.32356 / −0.49552 / −0.14759) |

New interface arms:

| Arm | Band content | Role |
|---|---|---|
| `tau_lift_oracle` | frozen lift from the target's own wall traction | **primary**: ceiling of the closure→generator interface |
| `tau_lift_fartime` | frozen lift from the far-time donor's traction | adverse control matched to the interface |
| `tau_lift_trainmean` | frozen lift from the traction obtained by applying the frozen estimator to the **train-split mean field** (the mean wall load) | information-matched null: identical lift, identical mean wall load, **zero instantaneous wall information** |
| `tau_lift_model_predicted` | frozen lift from the traction inverted from the **`no_wall` posterior mean's own** anchor velocity | secondary: contains **no** target-derived wall information at any point |

## 5. Endpoints, fixed in advance

- **Primary**: `Δ_τ = R²(tau_lift_oracle) − R²(no_wall)` on `full_support_excluded`, with 95 %
  block-bootstrap CI.
- **Transmission ratio**: `ρ = Δ_τ / Δ_oracle`, `Δ_oracle = R²(correct) − R²(no_wall)`, same region.
- **Instantaneous-information test**: `R²(tau_lift_oracle) − R²(tau_lift_trainmean)` with CI.
- **Adverse control**: `R²(tau_lift_oracle) − R²(tau_lift_fartime)` with CI.
- **Deployability probe**: `R²(tau_lift_model_predicted) − R²(no_wall)` with CI.
- **A-priori interface fidelity** (no sampling): fluctuation R² of the lifted band against the
  true band over the 14 008 band cells, per component and pooled, plus the fraction of band
  variance the lift can represent by construction.
- Secondary regions `near_…` and `outer_…` reported for every arm without exception.

## 6. Interpretation rule, fixed in advance

1. If `Δ_τ > 0` with CI excluding 0 **and** `tau_lift_oracle` beats `tau_lift_trainmean` with CI
   excluding 0 → the declared wall-traction interface transmits genuine instantaneous wall
   information into the unsupplied volume; ρ is reported as the interface ceiling and the paper's
   integrated architecture is viable **up to closure accuracy**, which remains unmeasured (L_A).
2. If `Δ_τ ≤ 0`, or `tau_lift_oracle` is not separated from `tau_lift_trainmean` → the E1 gain is
   **not** transmissible through wall traction. This falsifies the paper's integrated claim at its
   load-bearing interface and must be reported as the headline negative; the submission hold
   stands and the manuscript is restructured around the measured negative, not narrowed in prose.
3. Any outcome between (partial transmission, region-dependent sign) is reported with all three
   regions and both controls, with no arm dropped.

## 7. Validity, cost and stop rule

- **Void condition**: if any reproduction-guard arm deviates from its published M1 value by more
  than 0.005 in `R2_fluct_balanced`, the run is void and no new number is reported.
- **Cost**: inference only. Published M1 evaluation was 949.5 s of GPU time for 3 arms × 160
  targets × 8 members × 32 steps; 7 arms ≈ 2.3× that, plus ~3 min of statistics. Budget ≤ 1.5 GPU
  hours on `foshan`. No second launch, no arm added, no hyperparameter changed after any field
  number is seen.
- **Stop rule**: one pass. The script writes one JSON with all seven arms and all three regions,
  a per-time components NPZ, and a sha256. If it fails, the failure is reported; it is not rerun
  with different settings.
- **Target**: `cloud/gpu_run.sh --target foshan` only. `orig` is reserved for another project and
  is not touched.

## 8. Producer

`codes/gpu/eval_e2_traction_interface.py` (sha256 recorded in `E2_INTERFACE_RESULTS.md` and in the
result JSON `_meta`).
