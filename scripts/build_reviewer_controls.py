from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.common import normalise_identifier


RAW_PATH = Path(
    "data/intermediate/geco_raw_from_old.parquet"
)

FINAL_PATH = Path(
    "data/final/geco_analysis.parquet"
)

OUTPUT_PATH = Path(
    "data/final/geco_analysis_reviewer_controls.parquet"
)

CONFIG_PATH = Path(
    "configs/analysis.yaml"
)


def main() -> None:
    # ------------------------------------------------------------
    # Explicit GECO first-pass skip status
    # ------------------------------------------------------------

    raw = pd.read_parquet(
        RAW_PATH,
        columns=[
            "PP_NR",
            "PART",
            "TRIAL",
            "WORD_ID_WITHIN_TRIAL",
            "WORD_SKIP",
        ],
    )

    raw["WORD_SKIP"] = pd.to_numeric(
        raw["WORD_SKIP"],
        errors="coerce",
    )

    assert raw["WORD_SKIP"].notna().all()
    assert set(
        raw["WORD_SKIP"].unique()
    ).issubset({0, 1})

    raw["participant_id"] = normalise_identifier(
        raw["PP_NR"]
    )

    raw["sentence_id"] = normalise_identifier(
        raw[["PART", "TRIAL"]]
        .astype("string")
        .fillna("")
        .agg("-".join, axis=1)
    )

    raw["word_id"] = normalise_identifier(
        raw["WORD_ID_WITHIN_TRIAL"]
    )

    skip = raw[
        [
            "participant_id",
            "sentence_id",
            "word_id",
            "WORD_SKIP",
        ]
    ].copy()

    duplicate_skip_keys = skip.duplicated(
        [
            "participant_id",
            "sentence_id",
            "word_id",
        ]
    ).sum()

    assert duplicate_skip_keys == 0

    # ------------------------------------------------------------
    # Final analysis table
    # ------------------------------------------------------------

    df = pd.read_parquet(
        FINAL_PATH
    )

    original_rows = len(df)

    # ------------------------------------------------------------
    # Reproduce exactly the lag convention already used by the
    # existing prev_* lexical and surprisal features:
    #
    # previous retained GECO item within the same sentence.
    # ------------------------------------------------------------

    items = (
        df[
            [
                "sentence_id",
                "word_id",
                "position_in_sentence",
                "stimulus_token",
            ]
        ]
        .drop_duplicates(
            ["sentence_id", "word_id"]
        )
        .sort_values(
            [
                "sentence_id",
                "position_in_sentence",
            ]
        )
        .copy()
    )

    items["prev_word_id"] = (
        items.groupby(
            "sentence_id",
            sort=False,
        )["word_id"]
        .shift(1)
    )

    items["prev_position_in_sentence"] = (
        items.groupby(
            "sentence_id",
            sort=False,
        )["position_in_sentence"]
        .shift(1)
    )

    items["lag_position_distance"] = (
        items["position_in_sentence"]
        - items["prev_position_in_sentence"]
    )

    items["retained_sequence_start"] = (
        items["prev_word_id"]
        .isna()
        .astype(int)
    )

    # ------------------------------------------------------------
    # Conservative clause-punctuation control.
    #
    # 1 if the retained item ends in comma, semicolon, or colon,
    # optionally followed by a closing quotation mark.
    # ------------------------------------------------------------

    token = (
        items["stimulus_token"]
        .fillna("")
        .astype(str)
    )

    items["clause_punctuation_current"] = (
        token.str.contains(
            r'[,;:]["”’\']?$',
            regex=True,
        )
        .astype(float)
    )

    items["prev_clause_punctuation"] = (
        items.groupby(
            "sentence_id",
            sort=False,
        )["clause_punctuation_current"]
        .shift(1)
    )

    item_controls = items[
        [
            "sentence_id",
            "word_id",
            "prev_word_id",
            "prev_position_in_sentence",
            "lag_position_distance",
            "retained_sequence_start",
            "clause_punctuation_current",
            "prev_clause_punctuation",
        ]
    ].copy()

    df = df.merge(
        item_controls,
        on=[
            "sentence_id",
            "word_id",
        ],
        how="left",
        validate="many_to_one",
    )

    assert len(df) == original_rows

    # ------------------------------------------------------------
    # Attach WORD_SKIP for precisely the same predecessor used by
    # the existing lagged surprisal features.
    # ------------------------------------------------------------

    skip = skip.rename(
        columns={
            "word_id": "prev_word_id",
            "WORD_SKIP":
                "prev_word_skip_geco",
        }
    )

    df = df.merge(
        skip,
        on=[
            "participant_id",
            "sentence_id",
            "prev_word_id",
        ],
        how="left",
        validate="many_to_one",
    )

    assert len(df) == original_rows

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    has_predecessor = (
        df["prev_word_id"].notna()
    )

    sequence_start = (
        ~has_predecessor
    )

    assert (
        df.loc[
            has_predecessor,
            "prev_clause_punctuation",
        ]
        .notna()
        .all()
    )

    assert (
        df.loc[
            sequence_start,
            "prev_clause_punctuation",
        ]
        .isna()
        .all()
    )

    # Every retained predecessor in this canonical GECO table has
    # participant-level WORD_SKIP information.
    assert (
        df.loc[
            has_predecessor,
            "prev_word_skip_geco",
        ]
        .notna()
        .all()
    )

    print("=" * 80)
    print("REVIEWER CONTROL AUDIT")
    print("=" * 80)

    print(
        "rows =",
        len(df),
    )

    print(
        "unique items =",
        df[
            ["sentence_id", "word_id"]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "retained-sequence-start rows =",
        int(sequence_start.sum()),
    )

    print(
        "rows with retained predecessor =",
        int(has_predecessor.sum()),
    )

    nonconsecutive = (
        has_predecessor
        & df[
            "lag_position_distance"
        ].ne(1)
    )

    print(
        "non-consecutive retained-lag rows =",
        int(nonconsecutive.sum()),
    )

    print(
        "non-consecutive retained-lag unique items =",
        df.loc[
            nonconsecutive,
            ["sentence_id", "word_id"],
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "\nPrevious WORD_SKIP:"
    )

    print(
        df["prev_word_skip_geco"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print(
        "\nPrevious clause punctuation:"
    )

    print(
        df["prev_clause_punctuation"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------
    # Write derived table.
    # ------------------------------------------------------------

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nWROTE |",
        OUTPUT_PATH,
    )

    # ------------------------------------------------------------
    # Generate the three matched-control analysis configs.
    #
    # Controls enter the shared baseline so full_current and
    # full_spillover receive exactly the same controls.
    # ------------------------------------------------------------

    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as handle:
        base = yaml.safe_load(
            handle
        )

    specifications = {
        "reviewer_skip.yaml": [
            "prev_word_skip_geco",
        ],
        "reviewer_punct.yaml": [
            "prev_clause_punctuation",
        ],
        "reviewer_both.yaml": [
            "prev_word_skip_geco",
            "prev_clause_punctuation",
        ],
    }

    for filename, controls in specifications.items():
        config = deepcopy(base)

        config["baseline"] = [
            *base["baseline"],
            *controls,
        ]

        config["robustness"] = {
            "full":
                "is_analysis_token == 1",

            # Requires an explicitly observed immediately adjacent
            # retained predecessor.
            "strict_adjacent":
                "is_analysis_token == 1 "
                "and lag_position_distance == 1",
        }

        path = (
            Path("configs")
            / filename
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                config,
                handle,
                sort_keys=False,
            )

        print(
            "WROTE |",
            path,
            "| controls =",
            controls,
        )

    print(
        "\nCONTROL DATASET BUILD PASSED."
    )


if __name__ == "__main__":
    main()
