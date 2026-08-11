# Tested reproduction environment

The complete CPU audit and document rebuild were tested with:

- Python 3.10.12;
- `latexmk` 4.76;
- pdfTeX 3.141592653-2.6-1.40.22 (TeX Live 2022);
- Poppler `pdfinfo`/`pdffonts` 22.02.0;
- util-linux `flock` 2.37.2; and
- GNU coreutils `sha256sum` 8.32.

Python packages are pinned in `requirements.txt`. The one-command review rebuild is:

```text
bash codes/reproduce_all.sh
```

The command performs deterministic CPU derivation, plotting, source/provenance validation and
LaTeX compilation. It does not retrain a model, rerun LES or reproduce the non-bitwise GPU
training history. The raw record and checkpoints require the separately confirmed access route
described in `manuscript/source_data/ACCESS_AND_LICENSES.md`.

The exact historical linear/Wiener comparator producer is included for protocol inspection and
retains the remote data default recorded at execution. Supply an authorised local copy with its
`--data` option; the historical path is not a distribution route.
