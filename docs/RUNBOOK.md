# Full runbook

All commands below include the directory in which they should be executed.

## 1. Extract and enter the repository

```bash
cd ~/Downloads
unzip paclic26_codebase_complete.zip
cd ~/Downloads/paclic26_codebase_complete
```

## 2. Create the environment

### CPU workstation

```bash
cd ~/Downloads/paclic26_codebase_complete
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
```

### Ada GPU

```bash
cd /path/on/ada/paclic26_codebase_complete
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the Ada-supported CUDA build of `torch==2.5.1`, then:

```bash
cd /path/on/ada/paclic26_codebase_complete
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Confirm the GPU:

```bash
cd /path/on/ada/paclic26_codebase_complete
source .venv/bin/activate
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
PY
```

## 3. Validate the code before using data

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m compileall -q src scripts
python -m pytest
./scripts/run_debug.sh
```

Expected outcome:

- nine tests pass;
- `results/debug/run_manifest.json` is created;
- `results/debug/model_metrics.csv` contains both sentence and participant splits;
- the combined synthetic transformer model improves more for synthetic total reading time than gaze duration, by construction.

These synthetic values are only a software test.

## 4. Prepare GECO

Place the raw workbook at:

```text
data/raw/geco/MonolingualReadingData.xlsx
```

Inspect it:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.inspect_tabular \
  --input data/raw/geco/MonolingualReadingData.xlsx \
  --sheet DATA \
  --rows 5
```

The configured GECO columns are in `configs/geco.yaml`. Confirm that the workbook includes all of them.

Convert the large workbook to Parquet:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.convert_xlsx_to_parquet \
  --input data/raw/geco/MonolingualReadingData.xlsx \
  --output data/intermediate/geco_raw.parquet \
  --sheet DATA \
  --chunk-size 50000
```

This also writes `data/intermediate/geco_raw.metadata.json` with the raw-file hash, rows and columns.

Create participant, item and sentence tables:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.prepare_geco \
  --input data/intermediate/geco_raw.parquet \
  --out-dir data/processed
```

Inspect:

```bash
cd /path/to/paclic26_codebase_complete
python -m json.tool data/processed/data_audit.json | less
```

Do not continue until:

- participant-item keys have zero duplicates;
- item keys have zero duplicates;
- target counts are reported separately;
- punctuation handling is plausible for the workbook tokenisation (punctuation may be attached to words rather than separate rows);
- the participant and sentence counts match the intended GECO subset.


## 5. Add POS tags for the content-word robustness check

Download the NLTK English tagger once in the environment:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m nltk.downloader averaged_perceptron_tagger_eng
```

Annotate the unique items and merge the tags back to participant rows:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.annotate_content_words \
  --items data/processed/geco_items.parquet \
  --base data/processed/geco_participant_word.parquet \
  --items-output data/processed/geco_items_annotated.parquet \
  --base-output data/processed/geco_participant_word_annotated.parquet
```

The default content-word definition is Penn tags beginning `NN`, `VB`, `JJ` or `RB`. Inspect `data/processed/content_word_audit.json`.

## 6. Train and score the n-gram baseline

Use an external, documented English training corpus, one sentence per line:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.ngram_baseline \
  --input data/processed/geco_items_annotated.parquet \
  --train-text data/raw/lm_train.txt \
  --order 5 \
  --discount 0.1 \
  --output data/features/geco_ngram5_kn.parquet
```

Do not use `--allow-corpus-training` for reported results.

## 7. Compute transformer features on the GPU

### GPT-2 left-to-right surprisal

```bash
cd /path/on/ada/paclic26_codebase_complete
source .venv/bin/activate
python -m src.surprisal_ar \
  --input data/processed/geco_items_annotated.parquet \
  --model gpt2 \
  --device cuda \
  --output data/features/geco_gpt2.parquet
```

### BERT token-wise and whole-word pseudo-surprisal

```bash
cd /path/on/ada/paclic26_codebase_complete
source .venv/bin/activate
python -m src.surprisal_mlm \
  --input data/processed/geco_items_annotated.parquet \
  --model bert-base-uncased \
  --masking both \
  --device cuda \
  --output data/features/geco_bert.parquet
```

### RoBERTa token-wise and whole-word pseudo-surprisal

```bash
cd /path/on/ada/paclic26_codebase_complete
source .venv/bin/activate
python -m src.surprisal_mlm \
  --input data/processed/geco_items_annotated.parquet \
  --model roberta-base \
  --masking both \
  --device cuda \
  --output data/features/geco_roberta.parquet
```

For a ten-sentence smoke test, add `--max-sentences 10`. Delete those partial feature files before the full run.

Inspect all `*_alignment_audit.json` files. Missing item scores should be zero or explained. Long sentences may be windowed, but target items must remain scoreable.

## 8. Merge features and derive spillover predictors

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.merge_features \
  --base data/processed/geco_participant_word_annotated.parquet \
  --feature data/features/geco_ngram5_kn.parquet \
  --feature data/features/geco_gpt2.parquet \
  --feature data/features/geco_bert.parquet \
  --feature data/features/geco_roberta.parquet \
  --output data/final/geco_analysis.parquet
```

This automatically derives `prev_...surprisal` features at the item level. Inspect `data/final/merge_audit.json` and confirm that the base row count did not change.

## 9. Run the main CPU analysis

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.analyse \
  --input data/final/geco_analysis.parquet \
  --config configs/analysis.yaml \
  --output-dir results/main
```

This is the full run: four targets, two split strategies, all configured feature sets, robustness filters, a prespecified ridge alpha of 1.0, and 2,000 cluster-bootstrap and sign-flip resamples.

A faster diagnostic run is:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.analyse \
  --input data/final/geco_analysis.parquet \
  --config configs/analysis.yaml \
  --output-dir results/diagnostic \
  --targets log_gaze_duration log_total_reading_time \
  --splits sentence \
  --robustness full \
  --models lexical_gpt2 lexical_bert lexical_roberta lexical_all_transformers \
  --no-tune-alpha \
  --bootstrap-iterations 100
```

Do not report diagnostic-run uncertainty.

## 10. Run optional mixed-effects confirmation

This is CPU- and memory-intensive:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.mixed_effects \
  --input data/final/geco_analysis.parquet \
  --config configs/analysis.yaml \
  --output results/main/mixed_effects.csv
```

A `--max-rows` run is for debugging only and must not be reported.

## 11. Generate paper figures and tables

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.make_plots \
  --results-dir results/main \
  --output-dir results/figures
python -m src.make_tables \
  --results-dir results/main \
  --output-dir results/tables
```

Use `results/tables/main_model_table.csv` or `.tex` for manuscript values. Never copy values from terminal logs.

## 12. One-command run

After setting the paths:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
export RAW_GECO_XLSX="$PWD/data/raw/geco/MonolingualReadingData.xlsx"
export NGRAM_TRAIN_TEXT="$PWD/data/raw/lm_train.txt"
export DEVICE=cuda
export RUN_MIXED=0
./scripts/run_pipeline.sh
```
