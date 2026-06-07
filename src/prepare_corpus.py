from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from wordfreq import zipf_frequency

from .common import (
    aggregate_sentence_text,
    clean_word,
    is_punctuation_only,
    read_table,
    safe_log_ms,
    write_table,
)

REQUIRED = [
    "participant_id",
    "sentence_id",
    "word_id",
    "word",
    "first_fixation_duration",
    "gaze_duration",
    "go_past_time",
    "total_reading_time",
]

RT_COLS = [
    "first_fixation_duration",
    "gaze_duration",
    "go_past_time",
    "total_reading_time",
]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rename_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    colmap = cfg.get("column_map", {}) or {}
    optmap = cfg.get("optional_column_map", {}) or {}
    rename = {}
    missing = []

    for common, raw in colmap.items():
        if not raw:
            missing.append(common)
        elif raw not in df.columns:
            raise KeyError(f"Configured raw column {raw!r} for {common!r} not found. Available columns: {list(df.columns)}")
        else:
            rename[raw] = common

    if missing:
        raise ValueError(
            "Fill these required fields in the YAML column_map after inspecting the file: " + ", ".join(missing)
        )

    for common, raw in optmap.items():
        if raw and raw in df.columns:
            rename[raw] = common

    return df.rename(columns=rename)


def apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    filters = cfg.get("filters", {}) or {}
    min_rt = filters.get("min_rt", 50)
    out = df.copy()

    for col in RT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out[(out[col].isna()) | (out[col] >= min_rt)]
        max_key = f"max_{col}"
        if max_key in filters and filters[max_key] is not None:
            out = out[(out[col].isna()) | (out[col] <= filters[max_key])]

    return out


def add_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["word"] = out["word"].astype(str)
    out["word_clean"] = out["word"].apply(clean_word)
    out = out[out["word_clean"].str.len() > 0]
    out = out[~out["word"].apply(is_punctuation_only)]

    out["word_lower"] = out["word_clean"].str.lower()
    out["word_length"] = out["word_clean"].str.len()
    out["log_word_frequency"] = out["word_lower"].apply(lambda w: zipf_frequency(w, "en") if w else np.nan)

    for col in RT_COLS:
        out[f"log_{col}"] = safe_log_ms(out[col])

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = read_table(args.input)
    df = rename_columns(df, cfg)

    keep = set(REQUIRED)
    keep.update([c for c in ["sentence_text", "position_in_sentence", "sentence_length"] if c in df.columns])
    df = df[[c for c in df.columns if c in keep]].copy()

    df = apply_filters(df, cfg)
    df = add_lexical_features(df)
    if cfg.get("reconstruct_sentences", True):
        df = aggregate_sentence_text(df)

    df["is_sentence_initial"] = (pd.to_numeric(df["position_in_sentence"], errors="coerce") == 1).astype(int)
    df["is_sentence_final"] = (
        pd.to_numeric(df["position_in_sentence"], errors="coerce")
        == pd.to_numeric(df["sentence_length"], errors="coerce")
    ).astype(int)

    df = df.sort_values(["sentence_id", "position_in_sentence", "participant_id"]).reset_index(drop=True)
    write_table(df, args.output)
    print(f"Wrote {len(df):,} standardised rows to {args.output}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
