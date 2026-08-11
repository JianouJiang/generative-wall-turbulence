# An interface framework and adequacy standard when generative turbulence models hit the wall

Code, controls and source data for the manuscript of the above title, under review at
*Nature Communications*. This is the reader-facing release: it contains the manuscript and
Supplementary Information sources and PDFs, every figure's source data and build code, the
frozen closure interface, the LES case definition, and the audit/verification suite that
re-derives the reviewed release from its retained artifacts.

## Repository layout

- `manuscript/` — LaTeX sources, compiled PDFs, per-figure source data
  (`manuscript/source_data/`) and figure files.
- `codes/` — figure builders, verification probes, the frozen closure header
  (`codes/closure/`), the nekRS cube LES case (`codes/cube_les/`) and GPU evaluation
  scripts (`codes/gpu/`).
- `SUBMISSION_RELEASE_MANIFEST.json`, `FIGURE_DATA_SHA256SUMS` — hash manifests for the
  release set; verified by the reproduce chain.

Author-only submission documents (cover letter, journal checklist, development status) are
represented by short placeholder notes; the originals are retained in the authors' archive,
as are the raw LES volumes and model checkpoints, which are too large for this repository
and are available from the corresponding author on request (permanent archival deposit with
a DOI accompanies publication). A versioned archival release with an explicit open-source
licence will accompany publication; until then, all rights reserved.

## Claim boundary

1. **E1 — oracle propagation:** evaluated. An aligned-cube LES supplies a dense instantaneous
   three-component near-wall velocity band. Aligned, absent-band and far-time conditions are compared
   outside every supplied cell.
2. **E2 — offline composition:** evaluated offline, with a standing oracle positive control (no
   interface conclusion is drawn from a record unless conditioning on its own near-wall truth
   improves reconstruction there). On the cube record, a closure fitted on the training window and
   reading only matching-height state predicts a signed traction whose conditioning improves the
   source-excluded field under a decision rule frozen before execution; a reversed-time re-test on
   a fully disjoint unit and a prospectively registered single-slot channel experiment replicate
   the transmission with positive absolute skill on the channel. On the public separating
   periodic-hill record the positive control **passes in the denoising-diffusion family**
   (oracle near-wall band $+0.057$ [0.014, 0.101] over absent, with CRPS and the energy score
   agreeing) and **fails in the flow-matching family**, whose deterministic probability-flow
   sampler leaves the ensemble severely under-dispersed (rank-histogram extreme-bin mass 0.371,
   rising to 0.537 under conditioning, against a calibrated 0.222). Restoring that
   family's ensemble calibration with byte-identical weights was tested prospectively and did
   **not** repair it (the control failed more clearly), and neither did the finer raster.
   Ensemble calibration is therefore reported only as a diagnostic, not used as an adequacy
   criterion, and no mechanism is claimed. At the finer $96\times288$ raster the diffusion cell passes more
   strongly (oracle $+0.079$ [0.028, 0.131]; closure band $+0.059$ [$-0.011$, 0.122]).
   Re-derive with `codes/probes/audit_hill_positive_control.py` and
   `codes/probes/hill_hierarchical_intervals.py`. Separately, CoNFiLD Case1 is excluded: it is a two-dimensional irregular
   pipe with stochastic forcing and `(u,v,p)`, not the wall jet previously claimed.
3. **E3 — companion closure evidence:** closure side only; public citation requires author
   confirmation.

No solver-coupled computation is performed or claimed anywhere in the package; every evaluation
is offline and a priori.

The evaluated offline chain is coarse state → physics-grounded closure → local-frame wall-on-fluid
vector traction → declared interface → conditional generator, alongside oracle-band and
direct-traction controls. It does not establish runtime availability, momentum-boundary coupling
or performance in a time-evolving solver.

The frozen closure header is not a source-general inference package. It retains the local trunk
and one target-member-specific branch embedding, but not the learned branch network or set
normalisation. Its available local extractor also changes the calibrated pressure-gradient and
displacement-thickness definitions. A deterministic retained-profile audit gives closure-output
relative RMSE 0.408 and one sign change under those substitutions; this is an interface
sensitivity, not a transfer estimate.
The branch's fifth feature is the signed dimensionless coordinate
\(\beta_p=\delta^\ast_{\rm res}(\partial_xp)/(\nu|\partial_xp|)^{2/3}\), using kinematic pressure
and the same resolved \(y_m\)-to-\(\delta_{99}\) support as the fourth feature; the implementation
sets it to zero at zero pressure gradient.

## Evaluated results

In first-contact model M0, the aligned oracle band improves complete support-excluded fluctuation
\(R^2\) by 0.175 over the absent band and 0.307 over a far-time band. Aligned-arm skill is 0.103 over the
complete unsupplied volume, 0.243 near the wall and \(-0.040\) in the geometrically defined
\(d/h>0.5\) region.

The composition results are stated and bounded in the manuscript: closure-predicted traction
improves the cube's source-excluded field by \(+0.061\,[0.057,0.064]\) with the implied
wall-force error falling by almost two-fifths; the reversed-time cube re-test passes its oracle
positive control in both generative families and gains \(+0.103\) near the wall; and the
single-slot channel experiment gives \(+0.120\) for the closure against \(+0.165\) for the
record's own traction, with positive absolute skill and adverse wrong-information controls.

The limitations are load-bearing:

- 98 omitted fields give a 0.800 integral-time count ratio, while the first evaluated state is
  0.808 integral times after the last training state;
- the 343-output evaluation remainder has count ratio 2.800 and elapsed duration 2.792 integral
  times;
- M0 is conditional on one realised eight-draw sampler set per target; member-level outputs do not
  survive for a separate Monte Carlo-error replay;
- the two native-\(y^+\) continuation fields are not separated by the current conservative
  integral-time convention and support no independence claim;
- M1/M2 reuse the same interval during development;
- farther-region absolute skill is negative in every cube configuration;
- the no-band arm is a condition-dropout branch;
- deterministic controls prevent a matched generative-superiority claim;
- the interior comparator scores observed cells and is quarantined;
- physical-statistic corroboration and fixed-budget scaling fail; on the separating hill the
  oracle positive control passes in the diffusion family at both rasters but not in flow
  matching, whose failure neither a calibrated stochastic sampler nor a finer raster repaired,
  so that cell adjudicates nothing in either direction;
- resampling intervals are conditional and not multiplicity-adjusted across regions, controls or
  diagnostic families;
- the LES is a numerical reference without experimental or grid/filter validation.

## Case1 withdrawal

The release-contained semantic audit is `codes/probes/audit_level1_protocol.py`. It checks the
paper identity, both distinct E2 failure paths, arm-specific condition masks, traction sign,
two-cell band thickness, release-wide title consistency, whole-manuscript causal wording,
legacy-metadata supersession, invalidation markers and Supplementary table order. The legacy
Case1 raster is retained under the unmistakable filename
`SOURCE_INVALID_evaluated_bridge_rendering.png`. All favourable and unfavourable Case1 outputs remain in
`manuscript/source_data/diagnostic_e2/` and Supplementary Note 8, but support no physical
conclusion.

## Reproduce the reviewed package

```bash
bash codes/reproduce_all.sh
```

The command rebuilds the eight active main figures and one explicitly
withdrawn diagnostic display; compiles both documents; replays available lower-level cube
endpoints; checks a reviewed source-first contract for headline and decision-bearing claims;
verifies producer-emitted provenance for every declared data-bearing artist group; checks every
released display at publication width; and verifies the compact release manifest. It does not rerun
simulation, training or neural inference.

For the Level-1 semantic contract alone:

```bash
python3 codes/probes/audit_level1_protocol.py
python3 codes/probes/audit_closure_interface_custody.py --verify-retained
python3 codes/probes/verify_revision_semantics.py
```

Raw cube data, checkpoints, exact origin producers, final archive DOI/licence and author-only
metadata still require author-approved reviewer/public access. Legacy same-time-leaking
periodic-hill derivatives remain in named quarantine directories and support no claim.
