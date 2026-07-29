from __future__ import annotations

import pandas as pd

from src.ngram_baseline import score_items, train_model


def test_kneser_ney_baseline_scores_unique_items() -> None:
    sentences = [["the", "cat", "sat"], ["the", "dog", "sat"]]
    model = train_model(sentences, order=3, discount=0.1, vocabulary_cutoff=1)
    items = pd.DataFrame(
        {
            "sentence_id": ["s1", "s1", "s1"],
            "word_id": ["1", "2", "3"],
            "position_in_sentence": [1, 2, 3],
            "stimulus_token": ["the", "cat", "sat"],
        }
    )
    scored = score_items(model, items, order=3)
    assert len(scored) == 3
    assert scored["ngram3_kn_surprisal"].notna().all()
    assert not scored.duplicated(["sentence_id", "word_id"]).any()


def test_ngram_tokenisation_handles_attached_punctuation() -> None:
    from src.ngram_baseline import tokenize_line

    assert tokenize_line("Medicine, and lived.") == [
        "medicine",
        ",",
        "and",
        "lived",
        ".",
    ]

    assert tokenize_line("role @-@ playing") == [
        "role-playing",
    ]

    assert tokenize_line("one @,@ two") == [
        "one",
        ",",
        "two",
    ]

    assert tokenize_line("one @.@ Two") == [
        "one",
        ".",
        "two",
    ]


def test_ngram_aggregates_subtokens_to_word_level() -> None:
    from src.ngram_baseline import (
        score_items,
        tokenize_line,
        train_model,
    )

    sentences = [
        tokenize_line("medicine, and lived."),
        tokenize_line("medicine, and worked."),
    ]

    model = train_model(
        sentences,
        order=3,
        discount=0.1,
        vocabulary_cutoff=2,
    )

    items = pd.DataFrame(
        {
            "sentence_id": ["s1", "s1"],
            "word_id": ["1", "2"],
            "position_in_sentence": [1, 2],
            "stimulus_token": ["medicine,", "and"],
        }
    )

    scored = score_items(
        model,
        items,
        order=3,
    )

    assert scored["ngram3_kn_surprisal"].notna().all()
    assert scored["ngram3_kn_subtoken_count"].tolist() == [2, 1]
    assert scored["ngram3_kn_oov_subtoken_count"].tolist() == [0, 0]
