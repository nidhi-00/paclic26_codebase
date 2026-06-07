#!/usr/bin/env bash
set -euo pipefail

python -m src.inspect_tabular --input data/raw/geco/MonolingualReadingData.xlsx --rows 5
python -m src.convert_xlsx_to_parquet --input data/raw/geco/MonolingualReadingData.xlsx --output data/intermediate/geco_raw.parquet --sheet 0 --chunk-size 50000
python -m src.prepare_corpus --config configs/geco.yaml --input data/intermediate/geco_raw.parquet --output data/processed/geco_standardised.parquet
python -m src.ngram_baseline --input data/processed/geco_standardised.parquet --order 5 --output data/features/geco_ngram5.parquet
python -m src.surprisal_ar --input data/processed/geco_standardised.parquet --model distilgpt2 --output data/features/geco_distilgpt2.parquet --max-sentences 100
python -m src.surprisal_mlm --input data/processed/geco_standardised.parquet --model roberta-base --output data/features/geco_roberta.parquet --max-sentences 100
python -m src.merge_features --base data/processed/geco_standardised.parquet --feature data/features/geco_ngram5.parquet --feature data/features/geco_distilgpt2.parquet --feature data/features/geco_roberta.parquet --output data/final/geco_analysis.parquet
python -m src.analyse --input data/final/geco_analysis.parquet --targets log_gaze_duration log_total_reading_time --output-dir results/main
python -m src.make_plots --results-dir results/main --output-dir results/figures
