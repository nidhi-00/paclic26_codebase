from __future__ import annotations

import pandas as pd

from src.annotate_content_words import annotate_items


def test_content_word_annotation_uses_penn_prefixes() -> None:
    items = pd.DataFrame(
        {
            "sentence_id": ["s1"] * 4,
            "word_id": ["1", "2", "3", "4"],
            "position_in_sentence": [1, 2, 3, 4],
            "stimulus_token": ["Readers", "quickly", "understand", "."],
            "is_analysis_token": [1, 1, 1, 0],
        }
    )

    def fake_tagger(tokens):
        return list(zip(tokens, ["NNS", "RB", "VB", "."]))

    out = annotate_items(items, tagger=fake_tagger)
    assert out["is_content_word"].tolist() == [1, 1, 1, 0]
