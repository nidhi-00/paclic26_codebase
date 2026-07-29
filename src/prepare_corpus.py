from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
try:
    from wordfreq import zipf_frequency
except ImportError:
    def zipf_frequency(word: str, language: str) -> float:
        return float("nan")

from .common import (
    KEY_COLS,
    RT_COLS,
    build_item_table,
    clean_word,
    contains_alnum,
    is_punctuation_only,
    merge_item_features_to_rows,
    normalise_identifier,
    read_table,
    safe_log_ms,
    write_json,
    write_table,
)

REQUIRED = KEY_COLS + ["word"] + RT_COLS
OPTIONAL = [
    "sentence_text",
    "position_in_sentence",
    "sentence_length",
    "pos",
    "is_content_word",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def rename_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Populate canonical columns without creating duplicate column names.

    A raw source column may be reused for multiple canonical fields, such as
    WORD_ID_WITHIN_TRIAL for both word_id and position_in_sentence.
    Existing canonical columns are deliberately overwritten by the configured
    raw source.
    """
    if not df.columns.is_unique:
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Input table contains duplicate raw column names: {duplicates}")

    required_map = config.get("column_map", {}) or {}
    optional_map = config.get("optional_column_map", {}) or {}

    missing_config: list[str] = []
    mappings: list[tuple[str, str]] = []

    for canonical in REQUIRED:
        raw = required_map.get(canonical, "")
        if not raw:
            missing_config.append(canonical)
        elif raw not in df.columns:
            raise KeyError(
                f"Configured raw column {raw!r} for {canonical!r} was not found. "
                f"Available columns: {list(df.columns)}"
            )
        else:
            mappings.append((canonical, raw))

    if missing_config:
        raise ValueError(
            "Fill these required YAML column_map fields: "
            + ", ".join(missing_config)
        )

    for canonical in OPTIONAL:
        raw = optional_map.get(canonical, "")
        if raw:
            if raw not in df.columns:
                raise KeyError(
                    f"Optional configured column {raw!r} for "
                    f"{canonical!r} was not found."
                )
            mappings.append((canonical, raw))

    out = df.copy()

    for canonical, raw in mappings:
        source = df[raw]
        if isinstance(source, pd.DataFrame):
            raise ValueError(
                f"Raw source column {raw!r} is duplicated in the input table."
            )
        out[canonical] = source

    if not out.columns.is_unique:
        duplicates = out.columns[out.columns.duplicated()].tolist()
        raise AssertionError(
            f"Canonical column construction produced duplicates: {duplicates}"
        )

    return out


def apply_raw_row_filters(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for rule in config.get("raw_row_filters", []) or []:
        column = rule["column"]
        if column not in out.columns:
            raise KeyError(f"Raw filter column {column!r} was not found.")
        values = out[column].astype("string")
        if "equals" in rule:
            expected = str(rule["equals"])
            mask = values.str.casefold().eq(expected.casefold())
        elif "contains" in rule:
            mask = values.str.contains(str(rule["contains"]), case=False, regex=False, na=False)
        elif "regex" in rule:
            mask = values.str.contains(str(rule["regex"]), case=False, regex=True, na=False)
        else:
            raise ValueError(f"Unsupported raw filter rule: {rule}")
        out = out[mask].copy()
    return out


def derive_sentence_id(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    spec = config.get("derived_sentence_id")
    if not spec:
        return df
    columns = spec.get("columns", [])
    separator = str(spec.get("separator", "-"))
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot derive sentence_id; missing raw columns: {missing}")
    out = df.copy()
    parts = out[columns].astype("string")
    valid = parts.notna().all(axis=1)
    out["sentence_id"] = parts.fillna("").agg(separator.join, axis=1)
    out.loc[~valid, "sentence_id"] = pd.NA
    return out


def apply_target_specific_filters(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, dict[str, int | float | None]]]:
    """Mark invalid observations missing for that target only; never delete the row."""
    filters = config.get("filters", {}) or {}
    minimum = filters.get("min_rt", 50)
    out = df.copy()
    audit: dict[str, dict[str, int | float | None]] = {}

    for column in RT_COLS:
        values = pd.to_numeric(out[column], errors="coerce")
        maximum = filters.get(f"max_{column}")
        valid = values.notna() & values.gt(0)
        if minimum is not None:
            valid &= values.ge(float(minimum))
        if maximum is not None:
            valid &= values.le(float(maximum))
        invalid_nonmissing = values.notna() & ~valid
        out[f"invalid_{column}"] = invalid_nonmissing.astype(int)
        out[column] = values.where(valid)
        out[f"log_{column}"] = safe_log_ms(out[column])
        audit[column] = {
            "raw_nonmissing": int(values.notna().sum()),
            "valid": int(valid.sum()),
            "invalid_nonmissing": int(invalid_nonmissing.sum()),
            "missing_or_invalid": int((~valid).sum()),
            "minimum_ms": None if minimum is None else float(minimum),
            "maximum_ms": None if maximum is None else float(maximum),
        }
    return out, audit


def add_lexical_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["stimulus_token"] = out["word"].astype("string").fillna("")
    out["lexical_form"] = out["stimulus_token"].map(clean_word)
    out["word_clean"] = out["lexical_form"]
    out["word_lower"] = out["lexical_form"].str.casefold()
    out["word_length"] = out["lexical_form"].str.len().astype(float)
    out["is_punctuation"] = out["stimulus_token"].map(is_punctuation_only).astype(int)
    out["is_analysis_token"] = out["stimulus_token"].map(contains_alnum).astype(int)

    unique_words = out["word_lower"].dropna().unique().tolist()
    frequency_cache = {
        word: (zipf_frequency(word, config.get("language", "en")) if word else np.nan)
        for word in unique_words
    }
    out["zipf_frequency"] = out["word_lower"].map(frequency_cache)

    if "is_content_word" in out.columns:
        numeric = pd.to_numeric(out["is_content_word"], errors="coerce")
        out["is_content_word"] = numeric.where(numeric.isin([0, 1]))
    elif "pos" in out.columns:
        tags = {
            str(tag).upper()
            for tag in config.get(
                "content_pos_tags",
                ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"],
            )
        }
        out["is_content_word"] = out["pos"].astype("string").str.upper().isin(tags).astype(int)
    else:
        out["is_content_word"] = np.nan
    return out


def prepare_dataframe(
    raw: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    filtered_raw = apply_raw_row_filters(raw, config)
    filtered_raw = derive_sentence_id(filtered_raw, config)
    renamed = rename_columns(filtered_raw, config)

    keep = [c for c in REQUIRED + OPTIONAL if c in renamed.columns]
    rows = renamed[keep].copy()
    for column in KEY_COLS:
        rows[column] = normalise_identifier(rows[column])
    rows = rows.dropna(subset=KEY_COLS + ["word"]).copy()

    rows = add_lexical_features(rows, config)
    rows, target_audit = apply_target_specific_filters(rows, config)
    items, sentences = build_item_table(rows)

    item_lexical = items.sort_values(["sentence_id", "position_in_sentence"]).copy()
    item_lexical["prev_word_length"] = item_lexical.groupby("sentence_id", sort=False)["word_length"].shift(1)
    item_lexical["prev_zipf_frequency"] = item_lexical.groupby("sentence_id", sort=False)["zipf_frequency"].shift(1)

    rows = merge_item_features_to_rows(rows, item_lexical)
    rows = rows.sort_values(
        ["sentence_id", "position_in_sentence", "participant_id"]
    ).reset_index(drop=True)

    audit = {
        "corpus_name": config.get("corpus_name", "unknown"),
        "raw_rows": int(len(raw)),
        "rows_after_raw_filters": int(len(filtered_raw)),
        "participant_word_rows": int(len(rows)),
        "participants": int(rows["participant_id"].nunique()),
        "sentence_units": int(rows["sentence_id"].nunique()),
        "unique_items": int(len(items)),
        "unique_lexical_forms": int(rows.loc[rows["is_analysis_token"].eq(1), "word_lower"].nunique()),
        "analysis_token_rows": int(rows["is_analysis_token"].sum()),
        "punctuation_context_rows": int(rows["is_punctuation"].sum()),
        "targets": target_audit,
        "duplicate_participant_item_keys": int(rows.duplicated(KEY_COLS).sum()),
        "duplicate_item_keys": int(items.duplicated(["sentence_id", "word_id"]).sum()),
    }
    if audit["duplicate_participant_item_keys"]:
        raise ValueError("Duplicate participant/sentence/word keys remain after preprocessing.")
    return rows, item_lexical, sentences, audit


def _sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}_{suffix}{output.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a corpus without deleting target-specific observations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Participant-level output table.")
    parser.add_argument("--items-output", default=None)
    parser.add_argument("--sentences-output", default=None)
    parser.add_argument("--audit-output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    raw = read_table(args.input)
    rows, items, sentences, audit = prepare_dataframe(raw, config)

    output = Path(args.output)
    items_output = Path(args.items_output) if args.items_output else _sidecar(output, "items")
    sentences_output = Path(args.sentences_output) if args.sentences_output else _sidecar(output, "sentences")
    audit_output = Path(args.audit_output) if args.audit_output else output.with_name("data_audit.json")

    write_table(rows, output)
    write_table(items, items_output)
    write_table(sentences, sentences_output)
    write_json(audit, audit_output)

    print(f"Participant-level rows: {len(rows):,} -> {output}")
    print(f"Unique items: {len(items):,} -> {items_output}")
    print(f"Sentence units: {len(sentences):,} -> {sentences_output}")
    print(f"Audit: {audit_output}")


if __name__ == "__main__":
    main()
