# Results and validation guide

The corrected pipeline was not run on the real GECO workbook in the delivered environment. Therefore this repository does not promise that the manuscript's old numbers will be reproduced. Several corrected choices—target-specific filtering, preserved punctuation, common samples, proper held-out R², explicit models and long-context handling—can change the values.

## Files you should get

### Preprocessing

`data/processed/data_audit.json` must report:

- participant-word rows;
- number of readers;
- sentence/trial units;
- unique items;
- separate valid counts for FFD, GD, go-past and TRT;
- zero duplicate keys.

The four valid target counts no longer need to be identical. Identical counts are acceptable only if the raw data and thresholds genuinely produce them.

### Model scoring

Each feature file has a sidecar alignment audit. Check:

- resolved model revision is recorded;
- scored items equal expected items, apart from justified non-analysis alignments;
- missing item count is zero or investigated;
- mean and maximum subword counts are plausible;
- long sentences are marked as rolling/windowed rather than silently truncated.

### Merge

`merge_audit.json` must show:

- participant row count unchanged;
- no duplicate participant-item keys;
- expected current and previous surprisal columns;
- missingness for each feature.

### Main metrics

`model_metrics.csv` contains pooled out-of-fold R², RMSE and MAE. Every model within a target/split/robustness block must have the same `rows` and `groups` values.

The lexical row has delta-R² exactly zero. For another model:

```text
delta_r2_vs_lexical = model OOF R² - lexical OOF R²
```

Negative held-out R² is a valid result and must be reported honestly. It means the OOF predictions are worse than the held-out-sample mean benchmark.

### Participant evaluation

For participant-held-out runs, `fold_metrics.csv` should contain one fold per reader for every model. With 14 GECO monolingual readers, that means 14 outer folds per target/model/robustness block if the prepared subset indeed contains 14 readers.

### Confidence intervals

`bootstrap_intervals.csv` reports all comparisons, not selected ones. Interpret a positive incremental result as most reliable when:

- the point delta-R² is positive;
- the 95% cluster-bootstrap interval is above zero;
- the direction is similar under both sentence and participant splits;
- the effect is not driven only by sentence edges;
- it remains directionally similar under relevant sensitivity analyses.

Do not treat the cluster sign-flip p-value or its BH-adjusted q-value as a substitute for effect size and interval width.

## Qualitative patterns to check, not assume

The old manuscript described negligible transformer gains for first fixation and larger gains for cumulative measures, especially total reading time. The corrected run should test that pattern rather than force it.

Possible outcomes and manuscript implications:

- **Same pattern survives:** retain the early-versus-cumulative interpretation, with exact corrected values.
- **Only GPT-2 survives:** frame the paper around left-to-right predictability rather than complementary masked scores.
- **Masked whole-word and token-wise results differ:** make tokenisation/masking sensitivity a central methodological result.
- **N-gram absorbs transformer gains:** narrow the claim to language-model predictability generally, not transformer-specific information.
- **Participant split weakens results:** emphasise text generalisation and state limited cross-reader stability.
- **Previous-word surprisal matters:** discuss spillover and revise any claim that current-word surprisal alone explains the measure ordering.
- **Intervals include zero:** describe the result as uncertain rather than stable.

## Sanity checks

1. GPT-2, BERT and RoBERTa current-word features should be positively but not perfectly correlated.
2. BERT and RoBERTa may correlate more strongly with one another, but this must be read from `surprisal_correlations.csv`.
3. Whole-word pseudo-surprisal can differ substantially for multi-subword items.
4. First-position previous-word features should be missing and imputed inside folds, never copied from the prior sentence.
5. Punctuation must influence model context while being excluded from the default target sample.
6. Every OOF row must have one prediction per evaluated model.
7. The combined model's advantage must be compared with GPT-2 directly, not only with the lexical baseline.

## What not to expect

- Exact reproduction of old Table 2 values before the real corrected run.
- Guaranteed positive delta-R² for every measure.
- Guaranteed significance for the small effects.
- Fast mixed-effects fitting on a laptop.
- GPU acceleration for scikit-learn ridge or the bootstrap stage.
