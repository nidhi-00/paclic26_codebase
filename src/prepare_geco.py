from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import write_json, write_table
from .prepare_corpus import load_config, prepare_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the monolingual English GECO analysis tables.")
    parser.add_argument("--input", required=True, help="Raw GECO parquet produced by convert_xlsx_to_parquet.")
    parser.add_argument("--config", default="configs/geco.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(input_path)
    config = load_config(args.config)
    rows, items, sentences, audit = prepare_dataframe(raw, config)
    audit["config"] = str(args.config)

    rows_path = out_dir / "geco_participant_word.parquet"
    items_path = out_dir / "geco_items.parquet"
    sentences_path = out_dir / "geco_sentences.parquet"
    audit_path = out_dir / "data_audit.json"

    write_table(rows, rows_path)
    write_table(items, items_path)
    write_table(sentences, sentences_path)
    write_json(audit, audit_path)

    print(f"Participant-level rows: {len(rows):,} -> {rows_path}")
    print(f"Unique items: {len(items):,} -> {items_path}")
    print(f"Sentence/trial units: {len(sentences):,} -> {sentences_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
