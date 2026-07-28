# Changelog

## Reproducibility overhaul

- replaced in-sample headline R² with pooled out-of-fold R²;
- added sentence-held-out and leave-one-participant-out evaluation;
- added explicit GPT-2, BERT, RoBERTa, combined, n-gram, spillover and masking-sensitivity feature sets;
- fixed target-specific filtering and unique-item sentence construction;
- preserved punctuation for model context;
- added rolling causal and target-centred masked scoring;
- added token-wise and whole-word pseudo-surprisal;
- replaced add-alpha debug baseline with external-corpus interpolated Kneser-Ney;
- added fold-local imputation, scaling and nested ridge-alpha selection;
- added paired cluster-bootstrap uncertainty for all comparisons;
- added robustness filters, item-level correlations and optional mixed-effects models;
- added auditable manifests, tests, run scripts, figures, tables and documentation;
- removed stale generated result files that were not reproducible from the uploaded source.
