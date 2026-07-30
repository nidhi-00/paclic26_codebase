from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_PATH = Path("data/final/geco_analysis.parquet")

PREDICTIONS_PATH = Path(
    "results/main/oof_predictions/"
    "log_total_reading_time__sentence__full.parquet"
)

OUTPUT_DIR = Path("results/qualitative_spillover")

KEYS = [
    "participant_id",
    "sentence_id",
    "word_id",
]

ITEM_KEYS = [
    "sentence_id",
    "word_id",
]

ANALYSIS_COLUMNS = [
    "participant_id",
    "sentence_id",
    "word_id",
    "stimulus_token",
    "lexical_form",
    "word_lower",
    "sentence_text",
    "position_in_sentence",
    "sentence_length",
    "is_sentence_initial",
    "is_sentence_final",
    "is_content_word",
    "pos",
    "word_length",
    "zipf_frequency",
    "ngram5_kn_surprisal",
    "gpt2_surprisal",
    "bert_base_uncased_tokenwise_pseudo_surprisal",
    "roberta_base_tokenwise_pseudo_surprisal",
    "prev_ngram5_kn_surprisal",
    "prev_gpt2_surprisal",
    "prev_bert_base_uncased_tokenwise_pseudo_surprisal",
    "prev_roberta_base_tokenwise_pseudo_surprisal",
]

PREDICTION_COLUMNS = [
    "participant_id",
    "sentence_id",
    "word_id",
    "y_true",
    "pred__full_current",
    "pred__full_spillover",
]


def normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()

    for column in KEYS:
        output[column] = (
            output[column]
            .astype("string")
            .str.strip()
        )

    return output


def assert_unique(
    frame: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    duplicates = int(frame.duplicated(keys).sum())

    if duplicates:
        raise AssertionError(
            f"{label} contains {duplicates:,} duplicate keys: {keys}"
        )


def latex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)

    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def build_item_contexts(
    analysis: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    metadata_columns = [
        "sentence_id",
        "word_id",
        "position_in_sentence",
        "stimulus_token",
    ]

    items = (
        analysis[metadata_columns]
        .drop_duplicates(ITEM_KEYS)
        .copy()
    )

    assert_unique(
        items,
        ITEM_KEYS,
        "Unique item metadata",
    )

    items["position_in_sentence"] = pd.to_numeric(
        items["position_in_sentence"],
        errors="raise",
    )

    records: list[dict[str, object]] = []

    for sentence_id, sentence in items.groupby(
        "sentence_id",
        sort=False,
    ):
        sentence = sentence.sort_values(
            [
                "position_in_sentence",
                "word_id",
            ]
        ).reset_index(drop=True)

        tokens = (
            sentence["stimulus_token"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        for index, row in sentence.iterrows():
            start = max(0, index - window)
            end = min(len(tokens), index + window + 1)

            context_tokens = tokens[start:end].copy()
            target_offset = index - start

            context_tokens[target_offset] = (
                "["
                + context_tokens[target_offset]
                + "]"
            )

            previous_word = (
                tokens[index - 1]
                if index > 0
                else ""
            )

            records.append(
                {
                    "sentence_id": str(sentence_id),
                    "word_id": str(row["word_id"]),
                    "previous_word": previous_word,
                    "short_context": " ".join(
                        context_tokens
                    ),
                }
            )

    contexts = pd.DataFrame(records)

    assert_unique(
        contexts,
        ITEM_KEYS,
        "Generated item contexts",
    )

    return contexts


def select_quantile_examples(
    frame: pd.DataFrame,
    specifications: list[tuple[str, float]],
) -> list[dict[str, object]]:
    if frame.empty:
        raise ValueError(
            "No candidates are available for selection."
        )

    work = frame.copy()

    work["distribution_percentile"] = (
        work["mean_error_reduction"]
        .rank(
            method="average",
            pct=True,
        )
    )

    used_indices: set[int] = set()
    used_sentences: set[str] = set()
    selected: list[dict[str, object]] = []

    for label, quantile in specifications:
        target_value = float(
            work["mean_error_reduction"]
            .quantile(quantile)
        )

        candidates = work.loc[
            ~work.index.isin(used_indices)
        ].copy()

        candidates["selection_distance"] = (
            candidates["mean_error_reduction"]
            .sub(target_value)
            .abs()
        )

        diverse_candidates = candidates.loc[
            ~candidates["sentence_id"]
            .astype(str)
            .isin(used_sentences)
        ]

        if not diverse_candidates.empty:
            candidates = diverse_candidates

        candidates = candidates.sort_values(
            [
                "selection_distance",
                "readers",
                "sentence_id",
                "position_in_sentence",
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
        )

        chosen = candidates.iloc[0]
        chosen_index = int(chosen.name)

        used_indices.add(chosen_index)
        used_sentences.add(
            str(chosen["sentence_id"])
        )

        row = chosen.to_dict()
        row["selection_label"] = label
        row["target_quantile"] = quantile
        row["target_quantile_value"] = target_value

        selected.append(row)

    return selected


def write_latex_table(
    examples: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Illustrative sentence-held-out total-reading-time "
            r"cases selected deterministically at predeclared quantiles "
            r"of item-level squared-error change. Positive error reduction "
            r"means that the spillover model is closer to the observed "
            r"log reading time. These examples are descriptive rather "
            r"than inferential.}"
        ),
        r"\label{tab:qualitative-spillover}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        (
            r"\begin{tabular}{p{1.45cm}p{1.15cm}p{1.15cm}"
            r"p{5.3cm}rrrrr}"
        ),
        r"\toprule",
        (
            r"Case & Previous & Target & Context & Readers & "
            r"Observed log TRT & Current & Spillover & Error reduction \\"
        ),
        r"\midrule",
    ]

    for _, row in examples.iterrows():
        label = (
            str(row["selection_label"])
            .replace("_", " ")
            .title()
        )

        lines.append(
            f"{latex_escape(label)} & "
            f"{latex_escape(row['previous_word'])} & "
            f"{latex_escape(row['stimulus_token'])} & "
            f"{latex_escape(row['short_context'])} & "
            f"{int(row['readers'])} & "
            f"{row['observed_log_trt']:.3f} & "
            f"{row['current_prediction']:.3f} & "
            f"{row['spillover_prediction']:.3f} & "
            f"{row['mean_error_reduction']:.5f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ANALYSIS_PATH.exists():
        raise FileNotFoundError(ANALYSIS_PATH)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    analysis = pd.read_parquet(
        ANALYSIS_PATH,
        columns=ANALYSIS_COLUMNS,
    )

    predictions = pd.read_parquet(
        PREDICTIONS_PATH,
        columns=PREDICTION_COLUMNS,
    )

    analysis = normalise_keys(analysis)
    predictions = normalise_keys(predictions)

    assert_unique(
        analysis,
        KEYS,
        "Analysis table",
    )

    assert_unique(
        predictions,
        KEYS,
        "Prediction table",
    )

    merged = predictions.merge(
        analysis,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(
        merged["_merge"].ne("both").sum()
    )

    if unmatched:
        raise AssertionError(
            f"{unmatched:,} prediction rows did not match analysis data."
        )

    merged = merged.drop(
        columns="_merge"
    )

    required_nonmissing = [
        "y_true",
        "pred__full_current",
        "pred__full_spillover",
        "stimulus_token",
        "position_in_sentence",
        "is_sentence_initial",
        "is_sentence_final",
        "is_content_word",
    ]

    for column in required_nonmissing:
        missing = int(
            merged[column].isna().sum()
        )

        if missing:
            raise AssertionError(
                f"{column} has {missing:,} missing values."
            )

    metadata_columns = [
        "stimulus_token",
        "lexical_form",
        "word_lower",
        "sentence_text",
        "position_in_sentence",
        "sentence_length",
        "is_sentence_initial",
        "is_sentence_final",
        "is_content_word",
        "pos",
        "word_length",
        "zipf_frequency",
        "ngram5_kn_surprisal",
        "gpt2_surprisal",
        "bert_base_uncased_tokenwise_pseudo_surprisal",
        "roberta_base_tokenwise_pseudo_surprisal",
        "prev_ngram5_kn_surprisal",
        "prev_gpt2_surprisal",
        "prev_bert_base_uncased_tokenwise_pseudo_surprisal",
        "prev_roberta_base_tokenwise_pseudo_surprisal",
    ]

    consistency = (
        merged.groupby(
            ITEM_KEYS,
            sort=False,
        )[metadata_columns]
        .nunique(
            dropna=False,
        )
    )

    inconsistent = consistency.gt(1)

    if inconsistent.any().any():
        bad_columns = (
            inconsistent.any(axis=0)
            .loc[lambda values: values]
            .index
            .tolist()
        )

        raise AssertionError(
            "Item metadata varies across participants for: "
            + ", ".join(bad_columns)
        )

    merged["current_squared_error"] = (
        merged["y_true"]
        - merged["pred__full_current"]
    ) ** 2

    merged["spillover_squared_error"] = (
        merged["y_true"]
        - merged["pred__full_spillover"]
    ) ** 2

    merged["error_reduction"] = (
        merged["current_squared_error"]
        - merged["spillover_squared_error"]
    )

    aggregations: dict[str, tuple[str, str]] = {
        "readers": (
            "participant_id",
            "nunique",
        ),
        "participant_rows": (
            "participant_id",
            "size",
        ),
        "observed_log_trt": (
            "y_true",
            "mean",
        ),
        "current_prediction": (
            "pred__full_current",
            "mean",
        ),
        "spillover_prediction": (
            "pred__full_spillover",
            "mean",
        ),
        "current_mse": (
            "current_squared_error",
            "mean",
        ),
        "spillover_mse": (
            "spillover_squared_error",
            "mean",
        ),
        "mean_error_reduction": (
            "error_reduction",
            "mean",
        ),
    }

    for column in metadata_columns:
        aggregations[column] = (
            column,
            "first",
        )

    items = (
        merged.groupby(
            ITEM_KEYS,
            as_index=False,
            sort=False,
        )
        .agg(**aggregations)
    )

    contexts = build_item_contexts(
        analysis,
        window=5,
    )

    items = items.merge(
        contexts,
        on=ITEM_KEYS,
        how="left",
        validate="one_to_one",
    )

    if items["short_context"].isna().any():
        raise AssertionError(
            "Some item contexts could not be generated."
        )

    items["observed_geometric_mean_ms"] = np.exp(
        items["observed_log_trt"]
    )

    items["current_geometric_mean_ms"] = np.exp(
        items["current_prediction"]
    )

    items["spillover_geometric_mean_ms"] = np.exp(
        items["spillover_prediction"]
    )

    eligible = items.loc[
        items["is_content_word"].eq(1)
        & items["is_sentence_initial"].eq(0)
        & items["is_sentence_final"].eq(0)
        & items["previous_word"].ne("")
        & items["readers"].ge(10)
    ].copy()

    helpful = eligible.loc[
        eligible["mean_error_reduction"].gt(0)
    ].copy()

    harmful = eligible.loc[
        eligible["mean_error_reduction"].lt(0)
    ].copy()

    if len(helpful) < 100:
        raise AssertionError(
            f"Only {len(helpful)} helpful candidates."
        )

    if len(harmful) < 100:
        raise AssertionError(
            f"Only {len(harmful)} harmful candidates."
        )

    helpful_examples = select_quantile_examples(
        helpful,
        [
            ("helpful_median", 0.50),
            ("helpful_upper_quartile", 0.75),
            ("helpful_high", 0.95),
        ],
    )

    harmful_examples = select_quantile_examples(
        harmful,
        [
            ("harmful_lower_quartile", 0.25),
            ("harmful_extreme", 0.05),
        ],
    )

    examples = pd.DataFrame(
        helpful_examples
        + harmful_examples
    )

    examples["case_type"] = np.where(
        examples["mean_error_reduction"].gt(0),
        "helpful",
        "harmful",
    )

    case_order = {
        "helpful_median": 0,
        "helpful_upper_quartile": 1,
        "helpful_high": 2,
        "harmful_lower_quartile": 3,
        "harmful_extreme": 4,
    }

    examples["case_order"] = (
        examples["selection_label"]
        .map(case_order)
    )

    examples = examples.sort_values(
        "case_order"
    ).reset_index(drop=True)

    examples_path = (
        OUTPUT_DIR
        / "qualitative_examples.csv"
    )

    item_path = (
        OUTPUT_DIR
        / "item_error_changes.parquet"
    )

    latex_path = (
        OUTPUT_DIR
        / "qualitative_examples.tex"
    )

    audit_path = (
        OUTPUT_DIR
        / "qualitative_selection_audit.json"
    )

    claim_path = (
        OUTPUT_DIR
        / "approved_qualitative_claim.md"
    )

    examples.to_csv(
        examples_path,
        index=False,
    )

    items.to_parquet(
        item_path,
        index=False,
    )

    write_latex_table(
        examples,
        latex_path,
    )

    audit = {
        "analysis_input": str(ANALYSIS_PATH),
        "prediction_input": str(PREDICTIONS_PATH),
        "prediction_target": "log_total_reading_time",
        "prediction_split": "sentence",
        "row_level_metric": (
            "(y_true - pred_full_current)^2 - "
            "(y_true - pred_full_spillover)^2"
        ),
        "positive_metric_interpretation": (
            "Positive values mean lower squared error "
            "for the spillover model."
        ),
        "item_aggregation": (
            "Mean row-level metric across valid readers "
            "for each sentence_id and word_id."
        ),
        "eligibility": {
            "content_word": True,
            "sentence_initial": False,
            "sentence_final": False,
            "minimum_valid_readers": 10,
        },
        "eligible_items": int(len(eligible)),
        "helpful_candidates": int(len(helpful)),
        "harmful_candidates": int(len(harmful)),
        "helpful_quantiles": [
            0.50,
            0.75,
            0.95,
        ],
        "harmful_quantiles": [
            0.25,
            0.05,
        ],
        "selection": (
            "Nearest item to each predeclared quantile, "
            "without replacement and preferring distinct sentences."
        ),
        "selected_sentence_count": int(
            examples["sentence_id"].nunique()
        ),
        "selected_item_count": int(len(examples)),
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    claim_text = """# Approved qualitative interpretation

The examples were selected deterministically at predeclared
quantiles of item-level out-of-fold squared-error change after
restricting targets to non-edge content words with at least ten
valid readers.

They illustrate that adding previous-word surprisal can move
predictions either closer to or farther from the observed item-level
reading time. They are descriptive examples and do not establish
that any particular lexical item, syntactic construction, or
cognitive mechanism causes the global spillover effect.

The examples must not be described as representative garden paths,
relative clauses, syntactic reanalysis cases, or causal evidence
unless an independent construction-level annotation is performed.
"""

    claim_path.write_text(
        claim_text,
        encoding="utf-8",
    )

    print("prediction rows =", len(predictions))
    print("merged rows =", len(merged))
    print("unique TRT items =", len(items))
    print("eligible items =", len(eligible))
    print("helpful candidates =", len(helpful))
    print("harmful candidates =", len(harmful))
    print(
        "selected distinct sentences =",
        examples["sentence_id"].nunique(),
    )

    print("\nSelected examples:")

    display_columns = [
        "selection_label",
        "sentence_id",
        "position_in_sentence",
        "previous_word",
        "stimulus_token",
        "pos",
        "readers",
        "observed_log_trt",
        "current_prediction",
        "spillover_prediction",
        "mean_error_reduction",
        "short_context",
    ]

    print(
        examples[display_columns]
        .to_string(index=False)
    )

    print("\nWrote:")
    print(examples_path)
    print(item_path)
    print(latex_path)
    print(audit_path)
    print(claim_path)


if __name__ == "__main__":
    main()
