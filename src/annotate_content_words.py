from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import nltk
import pandas as pd

from .common import ITEM_KEY_COLS, read_table, write_json, write_table

CONTENT_PREFIXES = ("NN", "VB", "JJ", "RB")


def annotate_items(
    items: pd.DataFrame,
    tagger: Callable[[Sequence[str]], Sequence[tuple[str, str]]] = nltk.pos_tag,
) -> pd.DataFrame:
    required = set(ITEM_KEY_COLS + ["position_in_sentence", "stimulus_token"])
    missing = sorted(required - set(items.columns))
    if missing:
        raise KeyError(f"Item table is missing columns required for POS annotation: {missing}")
    if items.duplicated(ITEM_KEY_COLS).any():
        raise ValueError("Content-word annotation requires unique item keys.")

    rows: list[pd.DataFrame] = []
    for _, group in items.sort_values(["sentence_id", "position_in_sentence"]).groupby("sentence_id", sort=False):
        if "lexical_form" in group.columns:
            tag_tokens = group["lexical_form"].astype("string").fillna("").copy()
            empty = tag_tokens.str.len().eq(0)
            tag_tokens.loc[empty] = (
                group.loc[empty, "stimulus_token"]
                .astype("string")
                .fillna("")
            )
            tokens = tag_tokens.astype(str).tolist()
        else:
            tokens = group["stimulus_token"].astype(str).tolist()

        tagged = list(tagger(tokens))
        if len(tagged) != len(group):
            raise AssertionError("POS tagger returned a different number of tokens.")
        annotated = group.copy()
        annotated["pos"] = [tag for _, tag in tagged]
        annotated["is_content_word"] = annotated["pos"].str.startswith(CONTENT_PREFIXES).astype(int)
        if "is_analysis_token" in annotated.columns:
            annotated.loc[annotated["is_analysis_token"].ne(1), "is_content_word"] = 0
        rows.append(annotated)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Penn POS tags and a content-word indicator at item level.")
    parser.add_argument("--items", required=True)
    parser.add_argument("--base", required=True, help="Participant-level prepared table.")
    parser.add_argument("--items-output", required=True)
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--audit-output", default=None)
    parser.add_argument(
        "--download-resource",
        action="store_true",
        help="Download NLTK averaged_perceptron_tagger_eng before tagging.",
    )
    args = parser.parse_args()

    if args.download_resource:
        nltk.download("averaged_perceptron_tagger_eng", quiet=False)

    items = read_table(args.items)
    base = read_table(args.base)
    try:
        annotated_items = annotate_items(items)
    except LookupError as error:
        raise RuntimeError(
            "The NLTK English POS tagger is missing. Run: "
            "python -m nltk.downloader averaged_perceptron_tagger_eng"
        ) from error

    replacement = annotated_items[ITEM_KEY_COLS + ["pos", "is_content_word"]]
    base = base.drop(columns=["pos", "is_content_word"], errors="ignore").merge(
        replacement,
        on=ITEM_KEY_COLS,
        how="left",
        validate="many_to_one",
    )
    write_table(annotated_items, args.items_output)
    write_table(base, args.base_output)

    audit_path = Path(args.audit_output) if args.audit_output else Path(args.items_output).with_name(
        "content_word_audit.json"
    )
    analysis_mask = (
        annotated_items["is_analysis_token"].eq(1)
        if "is_analysis_token" in annotated_items.columns
        else pd.Series(True, index=annotated_items.index)
    )
    write_json(
        {
            "items": len(annotated_items),
            "content_items": int(annotated_items["is_content_word"].sum()),
            "content_fraction_among_analysis_tokens": float(
                annotated_items.loc[analysis_mask, "is_content_word"].mean()
            ),
            "pos_counts": annotated_items["pos"].value_counts().to_dict(),
            "definition": "Penn tags beginning NN, VB, JJ or RB",
        },
        audit_path,
    )
    print(f"Annotated item table: {args.items_output}")
    print(f"Annotated participant table: {args.base_output}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
