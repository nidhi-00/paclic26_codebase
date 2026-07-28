from __future__ import annotations

import pandas as pd

from src.common import add_previous_item_features
from src.merge_features import merge_feature_table


def test_item_feature_merge_does_not_multiply_rows() -> None:
    base = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p1", "p2"],
            "sentence_id": ["s1"] * 4,
            "word_id": ["1", "1", "2", "2"],
            "position_in_sentence": [1, 1, 2, 2],
        }
    )
    feature = pd.DataFrame(
        {
            "sentence_id": ["s1", "s1"],
            "word_id": ["1", "2"],
            "gpt2_surprisal": [3.0, 4.0],
        }
    )
    merged, columns = merge_feature_table(base, feature, "feature.csv")
    assert len(merged) == len(base)
    assert columns == ["gpt2_surprisal"]
    assert merged["gpt2_surprisal"].tolist() == [3.0, 3.0, 4.0, 4.0]


def test_previous_item_feature_is_shared_across_participants() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p1", "p2"],
            "sentence_id": ["s1"] * 4,
            "word_id": ["1", "1", "2", "2"],
            "position_in_sentence": [1, 1, 2, 2],
            "gpt2_surprisal": [3.0, 3.0, 4.0, 4.0],
        }
    )
    out = add_previous_item_features(frame, ["gpt2_surprisal"])
    assert out.loc[out["word_id"].eq("1"), "prev_gpt2_surprisal"].isna().all()
    assert out.loc[out["word_id"].eq("2"), "prev_gpt2_surprisal"].tolist() == [3.0, 3.0]
