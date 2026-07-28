from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from nltk.lm import KneserNeyInterpolated, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
from tqdm import tqdm

from .common import ITEM_KEY_COLS, environment_versions, read_table, write_json, write_table


def tokenize_line(text: str, lowercase: bool = True) -> list[str]:
    tokens = str(text).strip().split()
    return [token.casefold() for token in tokens] if lowercase else tokens


def load_external_sentences(path: str | Path, lowercase: bool = True) -> list[list[str]]:
    sentences: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens = tokenize_line(line, lowercase=lowercase)
            if tokens:
                sentences.append(tokens)
    if not sentences:
        raise ValueError(f"No non-empty training sentences were found in {path}")
    return sentences


def corpus_sentences(items: pd.DataFrame, lowercase: bool = True) -> list[list[str]]:
    return [
        [token.casefold() if lowercase else token for token in group["stimulus_token"].astype(str)]
        for _, group in items.sort_values(["sentence_id", "position_in_sentence"]).groupby("sentence_id", sort=False)
    ]


def train_model(
    sentences: Sequence[Sequence[str]],
    order: int,
    discount: float,
    vocabulary_cutoff: int,
) -> KneserNeyInterpolated:
    flat = [token for sentence in sentences for token in sentence]
    vocabulary = Vocabulary(flat, unk_cutoff=vocabulary_cutoff)
    train_data, _ = padded_everygram_pipeline(order, sentences)
    model = KneserNeyInterpolated(
        order=order,
        discount=discount,
        vocabulary=vocabulary,
    )
    model.fit(train_data)
    return model


def score_items(
    model: KneserNeyInterpolated,
    items: pd.DataFrame,
    order: int,
    lowercase: bool = True,
    probability_floor: float = 1e-12,
) -> pd.DataFrame:
    feature = f"ngram{order}_kn_surprisal"
    rows: list[dict] = []
    grouped = items.sort_values(["sentence_id", "position_in_sentence"]).groupby("sentence_id", sort=False)
    for _, group in tqdm(grouped, desc=f"{order}-gram Kneser-Ney surprisal"):
        context: list[str] = []
        for _, item in group.iterrows():
            token = str(item["stimulus_token"])
            token = token.casefold() if lowercase else token
            lookup_token = model.vocab.lookup(token)
            lookup_context = [model.vocab.lookup(value) for value in context[-(order - 1) :]]
            probability = float(model.score(lookup_token, lookup_context))
            surprisal = -math.log2(max(probability, probability_floor))
            rows.append(
                {
                    "sentence_id": item["sentence_id"],
                    "word_id": item["word_id"],
                    feature: surprisal,
                }
            )
            context.append(token)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a non-leaky interpolated Kneser-Ney n-gram baseline.")
    parser.add_argument("--input", required=True, help="Unique item table.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-text", default=None, help="External one-sentence-per-line training text.")
    parser.add_argument(
        "--allow-corpus-training",
        action="store_true",
        help="Debug only: train on the evaluation corpus. Never use this for paper results.",
    )
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.1)
    parser.add_argument("--vocabulary-cutoff", type=int, default=1)
    parser.add_argument("--preserve-case", action="store_true")
    parser.add_argument("--audit-output", default=None)
    args = parser.parse_args()

    if args.order < 1:
        raise ValueError("--order must be at least 1.")
    if not args.train_text and not args.allow_corpus_training:
        raise ValueError(
            "A non-leaky paper run requires --train-text. Use --allow-corpus-training only for debugging."
        )

    items = read_table(args.input)
    required = set(ITEM_KEY_COLS + ["position_in_sentence", "stimulus_token"])
    missing = sorted(required - set(items.columns))
    if missing:
        raise KeyError(f"Item table is missing required columns: {missing}")
    if items.duplicated(ITEM_KEY_COLS).any():
        raise ValueError("Input to ngram_baseline must contain unique sentence/word items.")

    lowercase = not args.preserve_case
    if args.train_text:
        sentences = load_external_sentences(args.train_text, lowercase=lowercase)
        training_source = str(Path(args.train_text))
        leaky_debug = False
    else:
        sentences = corpus_sentences(items, lowercase=lowercase)
        training_source = "evaluation_corpus_debug_only"
        leaky_debug = True

    model = train_model(
        sentences=sentences,
        order=args.order,
        discount=args.discount,
        vocabulary_cutoff=args.vocabulary_cutoff,
    )
    features = score_items(model, items, args.order, lowercase=lowercase)
    if features.duplicated(ITEM_KEY_COLS).any():
        raise AssertionError("N-gram scorer produced duplicate item keys.")
    write_table(features, args.output)

    feature_name = f"ngram{args.order}_kn_surprisal"
    audit_path = Path(args.audit_output) if args.audit_output else Path(args.output).with_name(
        f"{Path(args.output).stem}_audit.json"
    )
    audit = {
        "order": args.order,
        "smoothing": "interpolated_kneser_ney",
        "discount": args.discount,
        "vocabulary_cutoff": args.vocabulary_cutoff,
        "lowercase": lowercase,
        "training_source": training_source,
        "leaky_debug_run": leaky_debug,
        "training_sentences": len(sentences),
        "vocabulary_size": len(model.vocab),
        "items": len(features),
        "feature": feature_name,
        "unit": "bits",
        "missing_items": int(features[feature_name].isna().sum()),
        "environment_versions": environment_versions(),
    }
    write_json(audit, audit_path)
    print(f"Wrote {len(features):,} item features to {args.output}")
    print(f"Audit: {audit_path}")
    if leaky_debug:
        print("WARNING: this is a leaky debug-only n-gram run and must not be reported.")


if __name__ == "__main__":
    main()
