#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/make_synthetic_data.py --out-dir data/debug
python -m src.merge_features \
  --base data/debug/participant_word.csv \
  --feature data/debug/features.csv \
  --output data/debug/analysis.csv \
  --audit-output results/debug/merge_audit.json
python -m src.analyse \
  --input data/debug/analysis.csv \
  --config configs/analysis.yaml \
  --output-dir results/debug \
  --targets log_gaze_duration log_total_reading_time \
  --splits sentence participant \
  --robustness full \
  --models lexical_gpt2 lexical_bert lexical_roberta lexical_all_transformers \
  --bootstrap-iterations 20 \
  --no-tune-alpha \
  --oof-format csv
python -m src.make_plots --results-dir results/debug --output-dir results/debug/figures
python -m src.make_tables --results-dir results/debug --output-dir results/debug/tables

echo "Debug run completed. Inspect results/debug/run_manifest.json and results/debug/model_metrics.csv."
