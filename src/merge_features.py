from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import (
    ITEM_KEY_COLS,
    KEY_COLS,
    add_previous_item_features,
    read_table,
    write_json,
    write_table,
)


def merge_feature_table(base: pd.DataFrame, feature: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    if set(KEY_COLS).issubset(feature.columns):
        keys = KEY_COLS
        validation = "one_to_one"
    elif set(ITEM_KEY_COLS).issubset(feature.columns):
        keys = ITEM_KEY_COLS
        validation = "many_to_one"
    else:
        raise KeyError(
            f"Feature file {source} must contain either {ITEM_KEY_COLS} or {KEY_COLS}."
        )

    if feature.duplicated(keys).any():
        examples = feature.loc[feature.duplicated(keys, keep=False), keys].head(10)
        raise ValueError(f"Duplicate feature keys in {source}:\n{examples.to_string(index=False)}")

    feature_columns = [c for c in feature.columns if c not in keys]
    if not feature_columns:
        raise ValueError(f"No feature columns were found in {source}.")
    overlap = sorted(set(feature_columns) & set(base.columns))
    if overlap:
        raise ValueError(f"Feature columns already exist before merging {source}: {overlap}")

    original_rows = len(base)
    merged = base.merge(
        feature[keys + feature_columns],
        on=keys,
        how="left",
        validate=validation,
    )
    if len(merged) != original_rows:
        raise AssertionError(f"Merging {source} changed row count from {original_rows} to {len(merged)}.")
    return merged, feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge item/model features without multiplying participant rows.")
    parser.add_argument("--base", required=True, help="Participant-level prepared table.")
    parser.add_argument(
        "--feature",
        action="append",
        required=True,
        help="Feature parquet/csv. Repeat this argument for every feature file.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", default=None)
    parser.add_argument(
        "--no-previous-surprisal",
        action="store_true",
        help="Do not derive previous-word versions of surprisal features.",
    )
    args = parser.parse_args()

    data = read_table(args.base)
    missing_base_keys = sorted(set(KEY_COLS + ["position_in_sentence"]) - set(data.columns))
    if missing_base_keys:
        raise KeyError(f"Base table is missing required columns: {missing_base_keys}")
    if data.duplicated(KEY_COLS).any():
        raise ValueError("Base table has duplicate participant/sentence/word keys.")

    merged_columns: list[str] = []
    source_records: list[dict] = []
    for feature_path in args.feature:
        feature = read_table(feature_path)
        data, columns = merge_feature_table(data, feature, feature_path)
        merged_columns.extend(columns)
        source_records.append(
            {
                "path": str(feature_path),
                "rows": int(len(feature)),
                "columns": columns,
            }
        )
        print(f"Merged {feature_path}: {columns}")

    surprisal_columns = [
        column
        for column in merged_columns
        if column.endswith("_surprisal") or column.endswith("_pseudo_surprisal")
    ]
    if not args.no_previous_surprisal:
        data = add_previous_item_features(data, surprisal_columns)

    write_table(data, args.output)
    audit_path = Path(args.audit_output) if args.audit_output else Path(args.output).with_name("merge_audit.json")
    missingness = {
        column: {
            "nonmissing": int(data[column].notna().sum()),
            "missing": int(data[column].isna().sum()),
            "missing_fraction": float(data[column].isna().mean()),
        }
        for column in merged_columns
    }
    audit = {
        "base": str(args.base),
        "base_rows": int(len(data)),
        "feature_sources": source_records,
        "merged_feature_columns": merged_columns,
        "derived_previous_surprisal_columns": [f"prev_{column}" for column in surprisal_columns]
        if not args.no_previous_surprisal
        else [],
        "duplicate_participant_item_keys": int(data.duplicated(KEY_COLS).sum()),
        "feature_missingness": missingness,
    }
    write_json(audit, audit_path)
    print(f"Wrote {len(data):,} rows and {len(data.columns)} columns to {args.output}")
    print(f"Merge audit: {audit_path}")


if __name__ == "__main__":
    main()
