# Amendment 1 to the node-007 preregistration — moving-block length

**Raised and applied before any node-007 held-out outcome existed.**
31 July 2026, 22:45 UTC.

## What happened, in order

| Time (UTC) | Event |
|---|---|
| 22:32:49 | `FROZEN_HASHES.json` written (preregistration + producer + ledger frozen). |
| 22:33:05 | Production job launched on `foshan` (`foshan_20260731T233305.log`). |
| ~22:40 | While the job was still in the cube closure/generator **training** phase, the local retained-array probe `complete_non_regression_node007.py` produced *zero-width* bootstrap intervals on node 006's arrays. |
| 22:44 | Cause identified as an arithmetic degeneracy in the block length (below). Job log inspected: it had printed only `[cube split]` and `[cube] frozen-closure traction materialised`. **No held-out score of any kind had been computed or printed.** |
| 22:45 | Job killed (`KILLED`, GPU returned to 0% / 0 MiB). |
| 22:45–22:55 | Fix applied, re-verified, this amendment written. |
| — | Producer re-frozen and the run relaunched from scratch. |

No node-007 evaluation outcome was observed before this amendment. The evidence
is the job log itself, which is retained and whose last line before the kill is
the traction-materialisation notice, not a score.

## The defect

The circular moving-block bootstrap draws `nb = ceil(n/block)` blocks of length
`block` and truncates the concatenation to `n` indices. When `block >= n` this
gives `nb = 1`: a single block of length `>= n`, truncated to `n`, is a **cyclic
permutation containing every index exactly once**. Every replicate is then the
complete unit, the resampling distribution is a point mass, and the reported
"95% interval" has zero width.

The frozen node-007 producer computed the conservative block as
`ceil(1.2551 * tau)` in **record-frame** units. For the cube unit that is
`ceil(1.2551 x 97.59) = 123`, while the unit has `n = 103` frames — degenerate.

The frozen node-006 producer does **not** have this defect, because it divides by
the unit's *stride*: its evaluation unit is an interleaved subset of the test
window with a median spacing of three record frames, so a decorrelation time of
97.59 record frames is `97.59/3 = 32.5` unit indices. Node 006 accordingly
publishes `block = 33`, `block_conservative = 41`
(`e2_closure_composition_results.json → units.CONFIRM_STRICT`). The node-007
producer had simply omitted that stride division.

## The fix

A single helper, `conservative_block(idx, tau)`, used identically by the producer
and by the retained-array probe:

```python
stride = max(1.0, median(diff(idx)))                 # unit-index spacing
b      = ceil(1.2551 * tau / stride)                 # stride correction
b      = min(b, max(1, (n - 1) // 2))                # >= 3 blocks per replicate
```

Both terms are **pure arithmetic on the unit's index set and the record's integral
time**. Neither depends on any score, any arm, any seed or any outcome. There is
no degree of freedom here that could be turned toward a result.

## Verification

| Unit | n | stride | block | blocks/replicate |
|---|---|---|---|---|
| cube `MATCHED_NODE006_UNIT` | 103 | 3 | **41** | 3 |
| hills `U_STRICT` | 330 | 1 | 138 | 3 |
| hills `U_PRIOR` | 362 | 1 | 138 | 3 |
| hills whole unit | 692 | 1 | 138 | 6 |

The cube value **reproduces node 006's published `block_conservative = 41`
exactly**, which is the strongest available check that the corrected rule is the
frozen protocol's own rule rather than a new one.

Two independent confirmations that the correction is right and not
result-favourable:

* On node 006's retained arrays the corrected estimator gives a genuine
  hierarchical seed × block interval of `[0.05595, 0.06399]` for its primary
  estimand. Reviewer 2 independently recomputed that interval as approximately
  `[0.05595, 0.06410]`. The two agree to four decimal places.
* The corrected interval is **wider**, not narrower, than the degenerate one it
  replaces (which had zero width), so the amendment can only make a claim harder
  to sustain.

## Scope

This amendment changes the interval estimator only. It does **not** change the
design, the cells, the units, the arms, the closure, the generators, the seeds,
the estimand, the decision rule of §6, the non-regression tolerances of §7, the
cost ceiling or the stop rule. `FROZEN_HASHES.json` from 22:32:49 is retained
unmodified as `FROZEN_HASHES_v1.json`; the re-freeze is recorded as
`FROZEN_HASHES.json` with `supersedes` pointing at it.

Node 006's registered decision rule and its published numbers are untouched. The
corrected interval for its primary estimand is reported in
`NON_REGRESSION_COMPLETION.json` as a labelled post-registration correction, and
node 006's own registered value stands as published.
