# Implementation report: all 17 requested changes

This report maps each requested correction to concrete files and outputs. The changes were implemented against the uploaded `paclic26_codebase-main` repository.

## 1. Genuine held-out R²

**Problem:** the old `analyse.py` reported in-sample OLS R² while cross-validation returned only RMSE and MAE.

**Implementation:**

- `src/evaluation.py` generates exactly one out-of-fold prediction for every target-valid row.
- Pooled held-out R² is computed from all OOF predictions.
- `delta_r2_vs_lexical` is the OOF R² difference on the identical sample.
- OOF predictions are saved under `results/main/oof_predictions/`.

**Audit:** `model_metrics.csv`, `fold_metrics.csv`, OOF files and `run_manifest.json`.

## 2. Genuine participant-held-out evaluation

**Problem:** the old code always grouped by sentence.

**Implementation:**

- `configs/analysis.yaml` defines sentence `GroupKFold(5)` and participant `LeaveOneGroupOut`.
- `src/evaluation.py` uses the configured grouping column and split method.
- Participant runs therefore contain one outer fold per reader.

**Audit:** participant rows in `fold_metrics.csv`; fold labels begin with `heldout_`.

## 3. Explicit GPT-2, BERT, RoBERTa and combined models

**Problem:** the old code selected the first AR and first MLM column by column order.

**Implementation:**

- `configs/analysis.yaml` maps named aliases to exact columns.
- Every model feature set is explicitly enumerated.
- Missing core features cause a clear error unless `--allow-missing-core` is used.

**Audit:** resolved predictors for every model are recorded in `run_manifest.json`.

## 4. Baseline agreement and previous-word lexical controls

**Problem:** paper and code listed different lexical controls.

**Implementation:** the baseline is now exactly:

- word length;
- Zipf frequency;
- sentence position;
- sentence length;
- sentence-initial and sentence-final indicators;
- previous-word length;
- previous-word Zipf frequency.

`src/prepare_corpus.py` derives previous lexical controls at the unique item level before merging them back to participant rows.

## 5. Target-specific reading-time filtering

**Problem:** one invalid target deleted the row from every target analysis.

**Implementation:** `apply_target_specific_filters()` marks only that target missing and preserves all other measurements.

**Audit:** `data_audit.json` reports raw, valid and invalid counts separately for FFD, GD, go-past and TRT.

## 6. Unique sentence reconstruction and pandas failure correction

**Problem:** participant duplication could reconstruct each word once per reader, and the old `fillna(np.arange(...))` operation was invalid on recent pandas.

**Implementation:**

- `src/common.py::build_item_table()` deduplicates by sentence and word before position assignment or sentence reconstruction.
- It rejects conflicting item tokens and duplicate positions.
- Stable fallback sorting no longer uses invalid array-valued `fillna`.

**Tests:** `tests/test_prepare_corpus.py`.

## 7. Preserve punctuation and exact stimulus context

**Problem:** punctuation was stripped before transformer scoring.

**Implementation:**

- `stimulus_token` preserves the raw displayed token.
- `lexical_form` is a separate cleaned value used only for lexical predictors.
- punctuation-only items remain in the sentence context but `is_analysis_token=0` excludes them from the default regression sample.

## 8. Precise MLM scoring and whole-word sensitivity

**Problem:** the exact multi-subword masking procedure was undocumented and only token-wise masking existed.

**Implementation:** `src/surprisal_mlm.py` supports:

- `tokenwise`: mask one subword while other pieces remain visible;
- `whole_word`: mask all target pieces simultaneously;
- `both`: produce both feature columns in one run.

It also emits per-item subword counts and alignment diagnostics.

## 9. Long-context handling and alignment audits

**Problem:** MLM input was silently truncated at 512 tokens and GPT-2 had no explicit context policy.

**Implementation:**

- GPT-2 uses rolling left-context windows and scores every token exactly once.
- MLM scoring uses a target-centred window that always contains the complete target word.
- Model limits are resolved from tokenizer/model configuration or `--max-length`.
- Every scorer writes sentence and summary alignment audits, including token lengths, windowed items, missing alignments and subword statistics.

## 10. Fold-local imputation and common evaluation samples

**Problem:** model-specific `dropna()` caused feature sets to be evaluated on different rows, and the paper claimed imputation that the code did not perform.

**Implementation:**

- A target/split/robustness sample is created once and shared by every model.
- `SimpleImputer(strategy="median", add_indicator=True)` is fitted inside each training fold.
- `StandardScaler` and `Ridge` are also fitted inside each training fold.
- Missing feature values therefore do not silently alter the test sample.

## 11. Complete uncertainty analysis

**Problem:** selected confidence intervals and p-values were not reproducible.

**Implementation:**

- `bootstrap_oof_differences()` performs a paired cluster bootstrap over sentence units or participants using saved OOF residuals.
- It reports R² intervals and delta-R², delta-RMSE and delta-MAE intervals for every model and target.
- A paired cluster sign-flip p-value is reported for every non-lexical comparison, with Benjamini–Hochberg adjustments; bootstrap intervals remain the primary uncertainty summary.

**Output:** `bootstrap_intervals.csv`.

## 12. Robustness, correlations, figures and tables

**Implementation:**

- full data;
- sentence-initial exclusion;
- sentence-final exclusion;
- both-edge exclusion;
- content-word-only analysis using NLTK Penn tags (NN/VB/JJ/RB families);
- Pearson and Spearman correlations on unique item rows rather than duplicated participant rows;
- publication-readable R² and delta-R² figures with bootstrap error bars;
- generated CSV and LaTeX tables.

**Files:** `src/analyse.py`, `src/make_plots.py`, `src/make_tables.py`.

## 13. Non-leaky Kneser-Ney n-gram baseline

**Problem:** the old add-alpha model did not truly back off and silently trained on GECO when external text was missing.

**Implementation:**

- `src/ngram_baseline.py` uses NLTK interpolated Kneser-Ney smoothing.
- `--train-text` is required for a paper run.
- corpus training is possible only with the explicit `--allow-corpus-training` debug flag and is marked as leaky in its audit.

## 14. Corrected research framing

The code now supports the narrower defensible questions:

1. Does transformer predictability improve held-out reading-time prediction beyond lexical controls?
2. Does incremental value differ across eye-tracking measures?
3. How do left-to-right surprisal and bidirectional pseudo-surprisal compare?
4. Do masked scores add beyond GPT-2 on the same observations?

The code does not assume that masked scores uniquely measure integration. See `docs/MANUSCRIPT_REVISION_MAP.md`.

## 15. High-value strengthening experiments

Implemented experiments include:

- previous-word surprisal for all model families;
- non-leaky n-gram comparisons;
- token-wise versus whole-word MLM scoring;
- combined and ablation feature sets;
- paired uncertainty for all model differences;
- optional confirmatory mixed-effects models with participant and item random intercepts.

**Mixed effects:** `src/mixed_effects.py`.

## 16. Auditable pipeline outputs

The revised pipeline writes:

- `data_audit.json`;
- scorer alignment audits and sentence audits;
- `merge_audit.json`;
- `model_metrics.csv`;
- `fold_metrics.csv`;
- `alpha_selections.csv`;
- per-run OOF prediction files;
- `bootstrap_intervals.csv`;
- `surprisal_correlations.csv`;
- optional `mixed_effects.csv`;
- generated figures and tables;
- a complete `run_manifest.json`.

## 17. Repository reliability and supplementary-material readiness

Implemented repository work:

- pinned Python package versions and Python 3.11 marker;
- separate CPU, base and development requirement files;
- one-command full and debug pipelines;
- high-risk unit tests;
- model revision and environment-version recording;
- functional compatibility entry points replacing broken scripts;
- removal of stale, untraceable result figures;
- revised README, runbook, result guide, manuscript map and Git push guide;
- ACL/PACLIC paper shell without altered page geometry.

## Validation performed on this delivered repository

- `python -m compileall -q src scripts` completed successfully.
- `python -m pytest -q` passed all nine tests.
- `./scripts/run_debug.sh` completed end to end and generated OOF metrics, bootstrap intervals, figures and tables for both sentence and participant splits.

The actual GECO and transformer run was not executed because the raw corpus and model downloads are not included in the uploaded repository. Therefore no corrected paper numbers are claimed in this report.
