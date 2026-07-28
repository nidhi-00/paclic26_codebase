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
