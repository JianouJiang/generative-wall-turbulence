# Figure and manuscript evidence audit

Date: 31 July 2026; evidence status refreshed 8 August 2026
Scope: Level-3 results, statistics and publication-size figure review, preserving the accepted
Level-1 source-native evidence boundary  
Baseline: commit `6e5901b28b4bcb8383d8a6de419c179d8c20d30a`  
Compute: deterministic CPU inspection, plotting and build only; no simulation, training or neural
inference.

## Technical decision

The paper retains one integrated scientific identity: a physics-grounded wall closure supplies
interface information and a conditional generator propagates its influence through the turbulent
field.

**Status as of 4 August 2026: that integrated contribution is established offline, on three
records, and is no longer the open item this audit was originally written against.** The
evaluated components are (i) dense contemporaneous near-wall velocity changing a conditional
generator outside the supplied support in the aligned-cube LES record; (ii) the closure-predicted
traction composing with the generator on that record under a registered rule
(`+0.061 [0.057,0.064]`) and again on its reversed-time re-test (`+0.103 [0.091,0.115]` near the
wall, positive absolute skill); (iii) the wall-resolved channel single-slot experiment
(`+0.120`, positive absolute skill); and (iv) the separating periodic hill in the diffusion
family (`+0.06223 [+0.01722,+0.10573]`), which is indicative rather than precise at
`N_eff ~ 3.3` and whose flow-matching cell fails its oracle positive control and adjudicates
nothing. Solver-coupled deployment remains absent and is claimed nowhere.
The Case1 closure-composition display is removed from the manuscript because its primary source is
a two-dimensional irregular pipe with stochastic forcing and `(u,v,p)`, while the local pipeline
introduced unsupported wall-jet geometry, obstacle scale, Reynolds number, viscosity and stress. The figure is
retained only as an unmistakably marked withdrawn diagnostic; its input raster is renamed
`SOURCE_INVALID_evaluated_bridge_rendering.png` and accompanied by a machine-readable
invalidation record so no favourable or unfavourable
result disappears.

## Active display audit

| Display | Scientific content | Evidence boundary |
|---|---|---|
| Main Fig. 1 | All-visual overview rendered from staged records (30 producer-emitted bindings to `source_data/fig1_v2`): a) thin wall support and three-dimensional cube field; b) the one-slot flow-state→closure→traction→generator method; c) pinned cube, attached channel and separating hill record tiles (outcomes narrated in the caption); d) positive-control, offline-composition and closure-transfer evidence rows | As a conceptual overview, the display contains no numerical result chips: exact gains, intervals and effective-sample qualifications remain in Results. Displayed traction is the record's first-cell shear footprint, and E2's closure-versus-recorded floor traction comes from the archived E2 result file. Derived visualisations are declared in the `fig1_v2` provenance ledger; no row represents solver deployment. |
| Main Fig. 8 | Single-slot channel experiment (node 011): graded-reference response, closure/data-only/equilibrium arms, adverse controls, block inference, buffer turbulent-shear endpoint | Narrowed per panel review: the ladder is a within-family sensitivity, not a universal law; closure read against the lowest measured rung (its fidelity is below the measured domain), residuals disclosed (+0.016 closure / +0.027 data-only vs frozen 0.02 margin); engineering endpoint labelled a buffer Reynolds-shear proxy, offline only; preregistration, frozen decision applicator and outcome archived in source_data/fig_slot |
| Main Fig. 2 | Representative cube field | Contacted M2 eight-sample mean estimate; LES numerical reference, not truth; visual only |
| Main Fig. 3 | M0 first-contact aggregates | Points only because per-time M0 arrays do not survive; complete/near positive, farther absolute negative |
| Main Fig. 4 | Post-selection farther-region energy score | Fair off-diagonal U-statistic, block 57, selected model and contacted targets |
| Main Fig. 5 | Physical-statistic corroboration | Both valid families fail; interval-hit fraction descriptive; no invalid 0.8 calibration line |
| Supplementary Fig. 1 | Grouped-hill and padding sensitivity | Negative/mixed replication and contacted one-realisation topology change |
| Supplementary Fig. 2 | Capacity adjudication | Negative absolute skill, negative slope and failed convergence parity |
| Withdrawn built display | Case1 fields and scalar-feature ledgers | Prominent source-invalid banner; absent from manuscript/SI and supports no physical claim |

## Page-scale and scientific checks

- A reviewed source-first semantic contract selects exact paths and keys before comparing
  headline and decision-bearing claims. Plot producers also emit manifests after comparing each
  actual Matplotlib payload with its source-derived expectation: 235 semantically grouped,
  data-bearing artist bindings span the eight released displays. Adversarial checks reject a
  numerically nearby wrong key, a reversed contrast and a malformed multidigit shape. The former
  value-nearest occurrence ledger and hand-counted artist census are retained only as invalidated
  forensic inventories; neither is a provenance guarantee.

- Title, abstract, Figure 1, Results, Discussion and conclusion preserve the integrated
  closure-to-field paper identity. Figure 1 presents the E1 oracle intervention and the offline
  E2 closure-to-generator composition as evaluated, and marks solver coupling as not performed.
- Figure 1 is conceptual and prints no numerical result chips; the Results retain exact values,
  intervals, controls, the adverse family verdict and deployment boundaries.
- The first relevant mention labels the condition as target-derived, dense, contemporaneous,
  three-component velocity and distinguishes it from a wall closure.
- Figure 1 distinguishes physics motivation, empirical closure calibration, oracle input, offline
  composition evidence and solver deployment. Its field thumbnails are re-rendered from the
  retained record through producer-emitted artist bindings (9 source-first groups): the mid-span
  LES section, the exact supplied-band mask, the matching-height plane, the frozen-formula
  first-cell traction magnitude and direction, and the oracle-traction posterior-mean sections;
  the on-figure texts disclose the LES-traction and posterior-mean provenance.
- Methods identify the frozen closure's member-specific branch embedding; a
  source-valid runtime construction for its family/set input is not claimed. Supplementary Note 7
  records that the header omits the branch network and set normalisation, and quantifies the
  material effect of the available extractor's two feature-definition substitutions.
- Methods, Supplementary Note 7 and the source-data map define the branch-only
  \(\beta_p\), its kinematic-pressure and sign conventions, zero-gradient handling and the same
  resolved displacement-thickness support used by the fourth feature.
- Figure 1 assigns the fitted closure path to offline composition, separates closure-side transfer
  from absent solver-coupled deployment, and spells out wall-modelled large-eddy simulation
  instead of relying on an undefined graphic acronym.
- Main Fig. 2 uses shared lossless scales, equal coordinate aspect and “LES reference”.
- Main Fig. 3 does not draw unreplayable M0 uncertainty.
- Main Fig. 3 labels its ordinate as a realised eight-sample mean rather than an exact
  conditional mean; the reader-consequence verifier rejects the stale phrase adversarially.
- Main Fig. 3 labels the 160 targets as “correlated, thinned fields”; the Fig. 2 caption supplies
  the full statement that they are temporally correlated thinned fields from a chronologically
  separated evaluation remainder, preventing mutual-independence ambiguity.
- Main Fig. 4 uses points and intervals; its absolute-score ordinate starts at zero, while the
  separate paired-difference panel retains the interval-scale resolution needed to show the
  small paired effects without visually magnifying the absolute scores.
- Main Fig. 5 uses block-57 intervals and names its profiles as instantaneous quadratic summaries,
  not Reynolds stresses.
- Aligned-band/absent-band/far-time-band arms use explicit information labels and
  blue/grey/red encoding; historical JSON keys remain unchanged for custody.
- Supplementary Figs. 1--2 and the withdrawn display now use the same public arm vocabulary as the
  main paper. Historical `correct`/`no_wall`/`wrong_wall` keys remain only in machine records;
  they no longer invite a physical-closure reading on those displays.
- M0 point estimates are explicitly conditional on the realised eight-draw sampler set at each
  target; member-level predictions do not survive for a separate Monte Carlo-error replay.
- Negative farther-region absolute skill, null physical corroboration, failed support/scaling
  tests and all Case1 adverse controls remain visible.
- Count-based chronology labels are distinguished from elapsed time: 98 omitted fields correspond
  to 0.800 integral times, the first test is 0.808 after the last training state, and the
  343-output remainder has count ratio 2.800 but elapsed duration 2.792.
- Moving-block intervals are labelled conditional and are not presented as simultaneous,
  multiplicity-controlled inference across regions, controls or diagnostic families.
- Historical producer-level dependence fields are preserved byte-for-byte but are governed by
  `source_data/review_audit/legacy_metadata_supersession.json`; no active text or display treats
  them as the current integral-time or independence result.
- Figure scripts emit vector PDFs and 300-dpi PNGs; publication-width font (minimum 6.4 pt at
  180 mm), collision and containment checks are executable. The stroke audit now includes axes
  spines as well as plotted artists; the explicitly withdrawn display was raised above the
  one-point production floor and its source-invalid banner was strengthened without changing any
  evidence.
- Figure 1 text must fit its rounded containers, not merely remain on canvas; Figure 5 annotation
  blocks are separated from bars and whiskers at the declared 180-mm display width.
- Supplementary Fig. 2 prints the strictly negative crossed slope interval to four decimals
  (`[-0.0345,-0.0003]`), avoiding the misleading visual boundary `-0.000`; the failed convergence
  and scaling gates remain adjacent to it.

## Reader-consequence release gate

`codes/probes/verify_level3_reader_consequences.py` is a publication-outward audit that is distinct
from the invalidated numeral census. It recomputes the supplied-versus-scored support and fair
off-diagonal energy score from retained arrays, verifies M0/M2 absolute and differential signs,
checks control attribution and temporal dependence, preserves the physical and capacity nulls,
tests the closure-interface blocker, extracts public text from all eight figure PDFs, and requires
the 180-mm layout record to pass. Its output states the scientific consequence of every check;
historical key names and value-nearest matching are not evidence inputs.

## Provenance anchors

- M0 result/producer `61e7d09d...` / `ebe012fb...`.
- M1 result/producer `6762e9a8...` / `4244262e...`.
- M2 result/components `0fc30242...` / `28264e99...`.
- Ensemble result/components `c70342f6...` / `a7f1c74f...`.
- Case1 raw coordinate/data `a79d1681...` / `098ba72f...`.
- Source-invalid Case1 raster/result `2e6549d2...` / `f058a999...`.
- Periodic-hill archived E2 result/producer `0eb97c78...` / `c6b251c1...`.
- Frozen closure header/interface audit `63e4941f...` /
  `source_data/review_audit/closure_interface_custody.json`.

The release-contained semantic audit is `codes/probes/audit_level1_protocol.py`; the complete
point-by-point review record is in the active development node. Authorship and
corresponding-author decisions remain fixed.
Affiliations, ORCIDs, CRediT roles, funding, acknowledgements, competing interests, permissions,
related-work disclosure and final data/code DOI/licence language require author confirmation.
