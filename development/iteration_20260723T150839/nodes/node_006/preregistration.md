# Preregistration — one-shot genuine-cube distributional confirmation

Frozen: 2026-07-23 Europe/London, before running the new arm inference.

## Scope and evidence level

This is the single rapid confirmatory analysis authorized by the 23 July charter. It reuses the
existing 1,101-field Coceal aligned-cube record and the terminal periodic rectified-flow checkpoint;
it generates no new LES/CFD data and performs no training, calibration, checkpoint selection,
wall-distance search, time-window search or seed search.

The retained checkpoint is a classifier-free conditional model shared by the three interventions,
not three separately selected models. The evidence remains **held-out LES oracle-band
propagation**: the correct arm receives the contemporaneous LES wall-adjacent velocity envelope.
It is not closure-conditioned deployment and not solver-coupled WMLES.

Frozen inputs:

- cube memmap SHA-256:
  `8bac93f1537eab6667d692282b76c7bccd28f28965d35ea97668bcc2567bc45a`;
- cube physical-time array SHA-256:
  `99e8e0a45cf6c361bcefc10b251ceac15bd60992d42255cf76c945eae9655482`;
- periodic checkpoint SHA-256:
  `6f507fd1fb97a7e52dd60a631507dbc37d2005fdd12abd856f165eed8f6135c2`;
- terminal periodic result SHA-256:
  `0fc302421d622879e00ef16a14636fb8849ee65c0680c85491659788807f914d`;
- frozen test-index/component file SHA-256:
  `28264e996586f961b1a3cd8c369f494f62d39d4d18d73f0625fc4983e4ab3d18`.

## Frozen evaluation contract

The evaluation uses exactly the 160 chronological test indices in the terminal component file, its
half-test-span donor map for the far-time-wall arm, eight ensemble members, 32 rectified-flow
steps, and common initial Gaussian noise for all three interventions at each physical time.

The arms are:

1. `correct`: the same-time wall envelope;
2. `no_wall`: no wall-envelope condition;
3. `far_time_wall`: the frozen half-test-span donor envelope (called `wrong_wall` in the older
   result for historical compatibility).

All scores exclude the supplied wall envelope and solids. The primary region is fixed at
`d/h > 0.5`; no other wall-distance cut is tested. Channel scales and the mean field are computed
from the original chronological training block only.

The 160 evaluated fields span only 3.515 estimated kinetic-energy integral times. Circular
moving-block inference uses the already frozen 49-evaluation-sample block and 4,000 resamples with
seed 20260723. These intervals describe this short record; resampling does not increase its
approximately 3.5 independent test events.

## Primary proper score

The sole primary endpoint is the multivariate **energy score** of the complete standardized
outer-volume velocity ensemble. For outer vector dimension \(D\), truth \(y\), members \(x_m\) and
\(M=8\),

\[
 {\rm ES}={1\over M}\sum_m {\|x_m-y\|_2\over\sqrt D}
 -{1\over 2M^2}\sum_{m,m'}{\|x_m-x_{m'}\|_2\over\sqrt D}.
\]

Lower is better. The rapid distributional gate passes only if the paired improvements
`ES(no_wall) - ES(correct)` and `ES(far_time_wall) - ES(correct)` both have strictly positive
two-sided 95% moving-block percentile intervals. Absolute outer posterior-mean \(R^2\) is reported
from the terminal result but is not a gate for this distributional claim.

## Frozen physical-statistics family

All three secondary families are reported; none may be dropped after inference.

1. **Component spectra:** RMSE between truth and ensemble-mean
   \(\log_{10}(E_{uu},E_{vv},E_{ww})\), using streamwise Fourier modes 1–12 on complete outer
   \(x\)-\(z\) planes and training-mean/channel-standardized fluctuations.
2. **Reynolds-stress profiles:** NRMSE of the ensemble-mean six-component
   \((uu,vv,ww,uv,uw,vw)\) instantaneous plane profiles against LES on the same complete outer
   planes, using training-mean/channel-standardized fluctuations.
3. **One-point coverage:** absolute error of the empirical central 80% member interval from nominal
   0.8, over every component and outer cell; raw coverage and mean interval width are also retained.

Lower loss is better for all three families. A family is a corroborating win only when the correct
arm has a positive 95% block interval for improvement over **both** controls. The physical
corroboration gate requires at least two of the three preregistered families. All six secondary
contrasts are explicitly multiplicity-labelled; their unadjusted intervals are not interpreted as
six independent primary tests.

The terminal rapid result is called positive only when both the primary proper-score gate and the
physical corroboration gate pass. A null is terminal for this rapid lane and does not erase the
already measured complete-volume/near-wall genuine-3-D capability result.

## Reproducibility outputs

The producer must write:

- one JSON with all frozen definitions, arm scores, contrasts, gates and input hashes;
- one NPZ with memberwise truth/pair distances for independent energy-score recomputation,
  spectra, Reynolds-stress profiles and coverage counts;
- a compact figure;
- SHA-256 sidecars and a literal `=== done ===` marker.

A separately implemented CPU verifier must recompute the scores, block intervals and gates from the
NPZ without loading the model or trusting the producer JSON.
