# Transformer Predictability Beyond Lexical Controls in GECO

Reproducible code for the PACLIC manuscript comparing GPT-2 autoregressive surprisal with BERT and RoBERTa masked-language-model pseudo-surprisal on GECO eye-tracking measures.

This revision replaces the earlier scaffold with an auditable pipeline that:

- preserves the exact stimulus tokens, including punctuation;
- creates a unique sentence-item table before language-model scoring;
- applies reading-time exclusions separately for each target;
- computes rolling-window GPT-2 surprisal and target-centred MLM scores;
- compares token-wise and whole-word masked scoring;
- includes a non-leaky interpolated Kneser-Ney n-gram baseline;
- evaluates identical target-valid samples with fold-local imputation and scaling;
- reports sentence-held-out and participant-held-out out-of-fold metrics;
- adds previous-word lexical and surprisal predictors;
- produces paired cluster-bootstrap confidence intervals for every comparison;
- runs edge/content-word robustness checks, item-level correlations, and optional mixed-effects confirmation;
- writes data, alignment, merge, model, fold, bootstrap, and environment audits.

Read these documents before a paper run:

- `docs/IMPLEMENTATION_REPORT.md` — mapping of all 17 requested changes to code.
- `docs/RUNBOOK.md` — complete setup and execution instructions.
- `docs/RESULTS_GUIDE.md` — expected files, validation checks, and interpretation.
- `docs/PUSH_TO_GITHUB.md` — safe steps for replacing the old repository contents.
- `docs/MANUSCRIPT_REVISION_MAP.md` — claims and methods wording supported by the new code.

## Repository layout

```text
configs/
  analysis.yaml             Explicit feature sets, splits, robustness checks, ridge and bootstrap settings
  geco.yaml                 GECO raw-column mapping and target-specific thresholds
docs/                       Detailed implementation and manuscript reports
paper/                      ACL/PACLIC source shell and bibliography
scripts/
  run_pipeline.sh           Full GECO pipeline
  run_debug.sh              Fast synthetic end-to-end smoke test
  make_synthetic_data.py    Deterministic test corpus and features
src/
  prepare_corpus.py         Generic preprocessing
  annotate_content_words.py POS/content-word annotation for robustness analysis
  prepare_geco.py           GECO-specific preprocessing
  surprisal_ar.py           Rolling-window GPT-2 surprisal
  surprisal_mlm.py          Token-wise and whole-word pseudo-surprisal
  ngram_baseline.py         External-corpus Kneser-Ney baseline
  merge_features.py         Validated feature merge and previous-word features
  evaluation.py             Fold construction, OOF metrics and paired uncertainty
  analyse.py                Ridge CV, OOF predictions, robustness and bootstrap
  mixed_effects.py          Optional confirmatory mixed-effects models
  make_plots.py             Publication-readable figures with uncertainty
  make_tables.py            CSV and LaTeX manuscript tables
tests/                      Unit tests for the high-risk data and evaluation logic
```

## Python environment

Use Python 3.11. The pinned CPU environment is:

```bash
cd /path/to/paclic26_codebase
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
```

On the college Ada GPU system, install the PyTorch 2.5.1 build compatible with the CUDA driver or use the institution's PyTorch module, then install the remaining packages:

```bash
cd /path/to/paclic26_codebase
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-compatible torch==2.5.1 build provided/recommended by Ada first.
python -m pip install -r requirements.txt
```

Verify the GPU before transformer scoring:

```bash
cd /path/to/paclic26_codebase
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

The regression, bootstrap, table and plotting stages are CPU tasks. Use the GPU for GPT-2, BERT and RoBERTa scoring.

## Run the tests and smoke test

```bash
cd /path/to/paclic26_codebase
source .venv/bin/activate
python -m pytest
./scripts/run_debug.sh
```

The debug run does not download language models. It creates deterministic synthetic feature values and verifies preprocessing, merging, both split strategies, OOF metrics, bootstrap intervals, figures and tables.

## Full GECO run

Download `MonolingualReadingData.xlsx` from GECO and obtain an external English corpus with one sentence per line for the n-gram model. Do not train the paper n-gram model on GECO itself.

```bash
cd /path/to/paclic26_codebase
source .venv/bin/activate

export RAW_GECO_XLSX="$PWD/data/raw/geco/MonolingualReadingData.xlsx"
export NGRAM_TRAIN_TEXT="$PWD/data/raw/lm_train.txt"
export DEVICE=cuda
export RUN_MIXED=0
export DOWNLOAD_NLTK=1  # first run only; later use 0

./scripts/run_pipeline.sh
```

For a short scoring test on Ada:

```bash
cd /path/to/paclic26_codebase
source .venv/bin/activate

export RAW_GECO_XLSX="$PWD/data/raw/geco/MonolingualReadingData.xlsx"
export NGRAM_TRAIN_TEXT="$PWD/data/raw/lm_train.txt"
export DEVICE=cuda
export MAX_SENTENCES=10

./scripts/run_pipeline.sh
```

A `MAX_SENTENCES` run is diagnostic only. Do not report its numbers.

## Main outputs

```text
data/processed/data_audit.json
data/processed/content_word_audit.json
data/features/*_alignment_audit.json
data/final/merge_audit.json
results/main/run_manifest.json
results/main/model_metrics.csv
results/main/fold_metrics.csv
results/main/alpha_selections.csv
results/main/bootstrap_intervals.csv
results/main/surprisal_correlations.csv
results/main/oof_predictions/*.parquet
results/main/mixed_effects.csv                 # only when requested
results/figures/*.png and *.pdf
results/tables/main_model_table.csv and .tex
```

The paper's main quantitative claims should be copied only from these generated outputs. The old checked-in result images and CSVs were removed because they were not traceable to the uploaded source code.

## Reproducibility rule

Do not edit result CSVs or figure values manually. Rerun the relevant stage and regenerate tables/figures. Model audits record the resolved Hugging Face revision and package versions used during scoring.
