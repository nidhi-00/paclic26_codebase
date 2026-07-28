#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

: "${RAW_GECO_XLSX:?Set RAW_GECO_XLSX to MonolingualReadingData.xlsx}"
: "${NGRAM_TRAIN_TEXT:?Set NGRAM_TRAIN_TEXT to an external one-sentence-per-line English training corpus}"

DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"
RAW_PARQUET="${RAW_PARQUET:-data/intermediate/geco_raw.parquet}"
MAX_SENTENCES="${MAX_SENTENCES:-}"
RUN_MIXED="${RUN_MIXED:-0}"
DOWNLOAD_NLTK="${DOWNLOAD_NLTK:-0}"

LIMIT_ARGS=()
if [[ -n "$MAX_SENTENCES" ]]; then
  LIMIT_ARGS=(--max-sentences "$MAX_SENTENCES")
fi

mkdir -p data/intermediate data/processed data/features data/final results/main results/figures results/tables

$PYTHON -m src.convert_xlsx_to_parquet \
  --input "$RAW_GECO_XLSX" \
  --output "$RAW_PARQUET" \
  --sheet DATA \
  --chunk-size 50000

$PYTHON -m src.prepare_geco \
  --input "$RAW_PARQUET" \
  --config configs/geco.yaml \
  --out-dir data/processed

if [[ "$DOWNLOAD_NLTK" == "1" ]]; then
  $PYTHON -m nltk.downloader averaged_perceptron_tagger_eng
fi

$PYTHON -m src.annotate_content_words \
  --items data/processed/geco_items.parquet \
  --base data/processed/geco_participant_word.parquet \
  --items-output data/processed/geco_items_annotated.parquet \
  --base-output data/processed/geco_participant_word_annotated.parquet

$PYTHON -m src.ngram_baseline \
  --input data/processed/geco_items_annotated.parquet \
  --train-text "$NGRAM_TRAIN_TEXT" \
  --order 5 \
  --output data/features/geco_ngram5_kn.parquet

$PYTHON -m src.surprisal_ar \
  --input data/processed/geco_items_annotated.parquet \
  --model gpt2 \
  --device "$DEVICE" \
  "${LIMIT_ARGS[@]}" \
  --output data/features/geco_gpt2.parquet

$PYTHON -m src.surprisal_mlm \
  --input data/processed/geco_items_annotated.parquet \
  --model bert-base-uncased \
  --masking both \
  --device "$DEVICE" \
  "${LIMIT_ARGS[@]}" \
  --output data/features/geco_bert.parquet

$PYTHON -m src.surprisal_mlm \
  --input data/processed/geco_items_annotated.parquet \
  --model roberta-base \
  --masking both \
  --device "$DEVICE" \
  "${LIMIT_ARGS[@]}" \
  --output data/features/geco_roberta.parquet

if [[ -n "$MAX_SENTENCES" ]]; then
  echo "MAX_SENTENCES was set, so the pipeline stops after scoring audits. Do not analyse partial feature files."
  exit 0
fi

$PYTHON -m src.merge_features \
  --base data/processed/geco_participant_word_annotated.parquet \
  --feature data/features/geco_ngram5_kn.parquet \
  --feature data/features/geco_gpt2.parquet \
  --feature data/features/geco_bert.parquet \
  --feature data/features/geco_roberta.parquet \
  --output data/final/geco_analysis.parquet

$PYTHON -m src.analyse \
  --input data/final/geco_analysis.parquet \
  --config configs/analysis.yaml \
  --output-dir results/main

if [[ "$RUN_MIXED" == "1" ]]; then
  $PYTHON -m src.mixed_effects \
    --input data/final/geco_analysis.parquet \
    --config configs/analysis.yaml \
    --output results/main/mixed_effects.csv
fi

$PYTHON -m src.make_plots --results-dir results/main --output-dir results/figures
$PYTHON -m src.make_tables --results-dir results/main --output-dir results/tables

echo "Full pipeline completed. Start with results/main/run_manifest.json."
