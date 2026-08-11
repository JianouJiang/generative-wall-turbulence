# Figure-data handoff

This project contains the compact, reproducible source package for all seven
publication figures in the GWT paper. The package excludes multi-gigabyte raw LES
fields and private model checkpoints.

From this project directory, rebuild and verify the figures with:

```bash
bash codes/reproduce_all.sh
```

The command regenerates seven PNG/PDF figure pairs in `manuscript/figures/`, rebuilds
the main manuscript and Supplementary Information, replays the reported endpoints
from lower-level sufficient statistics, and verifies the exact release hashes.

Figure-ready inputs are indexed in `manuscript/source_data/README.md`.
`FIGURE_DATA_SHA256SUMS` pins the figure, source-data, code, and audit files used by
the release. `SUBMISSION_RELEASE_MANIFEST.json` records the exact public submission
package.

The supplemental simulations completed after the manuscript freeze are documented
in `PAPER_STATUS_20260726.md`. They are robustness/validity checks for this same
paper, not a replacement result set or a new mandatory campaign.
