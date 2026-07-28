# PACLIC codebase delivery report

This repository is a complete replacement revision of the uploaded `paclic26_codebase-main` project. It implements all 17 requested methodological and reproducibility changes. The detailed mapping is in `docs/IMPLEMENTATION_REPORT.md`.

## What is included

- corrected corpus preprocessing and target-specific reading-time filtering;
- exact item/sentence construction with punctuation-preserving transformer context;
- rolling-window GPT-2 surprisal;
- token-wise and whole-word BERT/RoBERTa pseudo-surprisal;
- external-corpus interpolated Kneser-Ney n-gram baseline;
- explicit lexical, GPT-2, BERT, RoBERTa, n-gram, spillover, masking-sensitivity and combined feature sets;
- sentence-held-out and participant-held-out out-of-fold evaluation;
- fold-local imputation, scaling and ridge regression on common samples;
- paired cluster-bootstrap confidence intervals, sign-flip tests and BH-adjusted q-values;
- robustness filters, unique-item correlations and optional mixed-effects confirmation;
- generated manuscript tables and publication-readable figures;
- unit tests, synthetic end-to-end smoke test, pinned environments and one-command run scripts;
- complete run, result-interpretation, manuscript and Git migration documentation.

## Validation completed in the delivery environment

```text
python -m compileall -q src scripts     PASS
python -m pytest -q                     PASS (9 tests)
./scripts/run_debug.sh                  PASS
```

The debug pipeline used deterministic synthetic data and successfully exercised preprocessing, feature merging, sentence and participant splits, OOF metrics, bootstrap intervals, correlations, figures and tables.

## What was not run

The real GECO analysis and transformer inference were not run in the delivery environment because the raw GECO workbook, external n-gram training corpus and downloaded model checkpoints were not part of the uploaded repository. Consequently:

- this archive contains no claimed corrected PACLIC result values;
- the manuscript's old numerical results should not be retained automatically;
- the real pipeline must be run and the manuscript tables/claims updated from the generated output files.

## Start here

1. Read `docs/RUNBOOK.md` and configure `configs/geco.yaml` against the actual workbook columns.
2. Run `python -m pytest` and `./scripts/run_debug.sh`.
3. Run transformer scoring on Ada and the regression/statistical stages on CPU.
4. Validate outputs using `docs/RESULTS_GUIDE.md`.
5. Revise manuscript claims using `docs/MANUSCRIPT_REVISION_MAP.md`.
6. Replace the old repository safely using `docs/PUSH_TO_GITHUB.md`.
