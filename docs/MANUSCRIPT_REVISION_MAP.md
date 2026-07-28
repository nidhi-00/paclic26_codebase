# Manuscript revision map supported by the corrected code

## Recommended title

**Transformer Predictability Beyond Lexical Controls in GECO Reading Times**

A more explicit alternative is:

**Autoregressive Surprisal and Masked-LM Pseudo-Surprisal Across GECO Eye-Tracking Measures**

## Research questions

1. Do transformer-derived predictability measures improve held-out reading-time prediction beyond lexical controls?
2. Does their incremental value differ across first fixation, gaze duration, go-past time and total reading time?
3. How does GPT-2 left-to-right surprisal compare with BERT and RoBERTa bidirectional pseudo-surprisal under identical samples and evaluation splits?
4. Do masked-model scores contribute predictive information beyond GPT-2 and a non-neural n-gram baseline?
5. Are the conclusions robust to spillover predictors, sentence edges, content-word restriction and MLM masking definition?

## Claim discipline

The code supports predictive association claims. It does not by itself establish that:

- a language model is a cognitive process model;
- masked pseudo-surprisal represents online expectation;
- higher total-reading-time fit proves a unique integration mechanism;
- the combined model's superiority proves distinct cognitive systems.

Use “bidirectional contextual compatibility” for BERT/RoBERTa when discussing cognitive interpretation.

## Methods language

### Data and filtering

> Reading-time validity was determined separately for each target. Values outside the prespecified range were set to missing for that measure without deleting otherwise usable observations from the other analyses. Exact stimulus tokens, including punctuation, were retained for language-model context; punctuation-only rows were excluded from the default regression target sample.

### Item construction

> Language-model features were computed once per unique sentence-position item and merged back to participant-level observations. This prevented repeated participant rows from duplicating words during sentence reconstruction or model scoring.

### Autoregressive surprisal

> GPT-2 word surprisal was computed in bits by summing token-level negative log probabilities over all subword pieces associated with the word. Sequences exceeding the model context limit were scored with rolling left-context windows, and every content token was assigned exactly one score.

### Masked pseudo-surprisal

> For BERT and RoBERTa, token-wise pseudo-surprisal masked each subword piece separately while retaining the other pieces. A whole-word sensitivity analysis masked all pieces of the target simultaneously. Target-centred windows retained the complete target and as much bilateral context as allowed by the model.

### Baseline and spillover

> The lexical baseline included current-word length and Zipf frequency, sentence position and length, sentence-edge indicators, and previous-word length and frequency. Additional models introduced current- and previous-word language-model surprisal features.

### Evaluation

> Ridge models were evaluated on common target-valid samples. Median imputation, missingness indicators, standardisation and ridge fitting were learned inside each training fold. The main text split used sentence-grouped five-fold cross-validation; reader generalisation used leave-one-participant-out evaluation. Reported R², RMSE and MAE were calculated from pooled out-of-fold predictions.

### Uncertainty

> Incremental metrics were calculated relative to the lexical model on the same observations. Confidence intervals were obtained with paired cluster bootstrap resampling over the held-out grouping unit—sentence units for text generalisation and participants for reader generalisation.

## Required table updates

The manuscript should report at least:

- lexical, GPT-2, BERT, RoBERTa and combined-transformer rows;
- both split strategies;
- all four measures;
- OOF R² and delta-R²;
- 95% cluster-bootstrap intervals;
- row and group counts;
- a separate n-gram/spillover/masking sensitivity table or appendix.

## Discussion decisions after rerunning

Do not retain the old ranking or effect-size claims automatically. Rewrite the abstract, results, discussion and conclusion from `results/tables/main_model_table.csv` and the robustness outputs after the full corrected run.
