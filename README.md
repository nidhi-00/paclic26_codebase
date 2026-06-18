# Surprisal Reading PACLIC Codebase

This is a codebase for the project:

**Online Prediction or Contextual Integration? Comparing Autoregressive Surprisal and Masked Pseudo-Surprisal in Human Reading**

## Which GECO files to download

From the official GECO download page, download only these three files for the main English monolingual analysis:

1. `EnglishMaterial` — English stimulus/material text.
2. `MonolingualReadingData` — eye-tracking data for monolingual English readers. This is the large file.
3. `SubjectInformation` — participant metadata.

Place them here:

```text
data/raw/geco/EnglishMaterial.xlsx
data/raw/geco/MonolingualReadingData.xlsx
data/raw/geco/SubjectInformation.xlsx
```

If the downloaded file has `.xls`, `.csv`, or no extension, keep the original and pass the actual path to the scripts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For transformer surprisal, install PyTorch according to your platform if the default install is slow or fails.

## Pipeline overview

### 1. Inspect GECO columns without opening the huge file in Excel

```bash
python -m src.inspect_tabular --input data/raw/geco/MonolingualReadingData.xlsx --rows 5
```

Copy the printed column names into `configs/geco.yaml`.

### 2. Convert the huge Excel file to Parquet

This avoids loading the full 500 MB workbook into memory.

```bash
python -m src.convert_xlsx_to_parquet \
  --input data/raw/geco/MonolingualReadingData.xlsx \
  --output data/intermediate/geco_raw.parquet \
  --sheet 0 \
  --chunk-size 50000
```

### 3. Standardise corpus columns

Edit `configs/geco.yaml` first. Then run:

```bash
python -m src.prepare_corpus \
  --config configs/geco.yaml \
  --input data/intermediate/geco_raw.parquet \
  --output data/processed/geco_standardised.parquet
```

### 4. Compute n-gram surprisal

Use an external text file if possible. If you do not provide one, the script trains on the corpus sentences as a debug fallback only.

```bash
python -m src.ngram_baseline \
  --input data/processed/geco_standardised.parquet \
  --order 5 \
  --output data/features/geco_ngram5.parquet
```

### 5. Compute autoregressive surprisal

Start with DistilGPT-2 because it is faster.

```bash
python -m src.surprisal_ar \
  --input data/processed/geco_standardised.parquet \
  --model distilgpt2 \
  --output data/features/geco_distilgpt2.parquet \
```

### 6. Compute masked pseudo-surprisal

Start with RoBERTa-base.

```bash
python -m src.surprisal_mlm \
  --input data/processed/geco_standardised.parquet \
  --model roberta-base \
  --output data/features/geco_roberta.parquet \
```

### 7. Merge all features

```bash
python -m src.merge_features \
  --base data/processed/geco_standardised.parquet \
  --feature data/features/geco_ngram5.parquet \
  --feature data/features/geco_distilgpt2.parquet \
  --feature data/features/geco_roberta.parquet \
  --output data/final/geco_analysis.parquet
```

### 8. Run analysis

```bash
python -m src.analyse \
  --input data/final/geco_analysis.parquet \
  --targets log_gaze_duration log_total_reading_time \
  --output-dir results/main
```

### 9. Plot results

```bash
python -m src.make_plots \
  --results-dir results/main \
  --output-dir results/figures
```
