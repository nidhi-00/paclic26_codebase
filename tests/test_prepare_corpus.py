from __future__ import annotations

import numpy as np
import pandas as pd

from src.prepare_corpus import apply_target_specific_filters, prepare_dataframe


def test_target_specific_filtering_does_not_delete_other_targets() -> None:
    frame = pd.DataFrame(
        {
            "first_fixation_duration": [100, 900],
            "gaze_duration": [200, 250],
            "go_past_time": [300, 350],
            "total_reading_time": [400, 450],
        }
    )
    config = {
        "filters": {
            "min_rt": 50,
            "max_first_fixation_duration": 800,
            "max_gaze_duration": 1500,
            "max_go_past_time": 5000,
            "max_total_reading_time": 8000,
        }
    }
    filtered, audit = apply_target_specific_filters(frame, config)
    assert len(filtered) == 2
    assert np.isnan(filtered.loc[1, "first_fixation_duration"])
    assert filtered.loc[1, "gaze_duration"] == 250
    assert audit["first_fixation_duration"]["invalid_nonmissing"] == 1
    assert audit["gaze_duration"]["invalid_nonmissing"] == 0


def test_prepare_dataframe_uses_unique_items_and_preserves_punctuation() -> None:
    rows = []
    for participant in ["p1", "p2"]:
        for position, token in enumerate(["Hello", ",", "world", "!"], start=1):
            rows.append(
                {
                    "pid": participant,
                    "sid": "s1",
                    "wid": str(position),
                    "token": token,
                    "position": position,
                    "ffd": 100,
                    "gd": 150,
                    "gpt": 200,
                    "trt": 250,
                }
            )
    raw = pd.DataFrame(rows)
    config = {
        "corpus_name": "test",
        "column_map": {
            "participant_id": "pid",
            "sentence_id": "sid",
            "word_id": "wid",
            "word": "token",
            "first_fixation_duration": "ffd",
            "gaze_duration": "gd",
            "go_past_time": "gpt",
            "total_reading_time": "trt",
        },
        "optional_column_map": {"position_in_sentence": "position"},
        "filters": {"min_rt": 50},
    }
    participant_rows, items, sentences, audit = prepare_dataframe(raw, config)
    assert len(participant_rows) == 8
    assert len(items) == 4
    assert items["stimulus_token"].tolist() == ["Hello", ",", "world", "!"]
    assert sentences.loc[0, "reconstructed_sentence_text"] == "Hello , world !"
    assert items["position_in_sentence"].tolist() == [1, 2, 3, 4]
    assert items["sentence_length"].tolist() == [4, 4, 4, 4]
    assert items["is_punctuation"].tolist() == [0, 1, 0, 1]
    assert audit["duplicate_item_keys"] == 0
    assert audit["punctuation_context_rows"] == 4



def test_rename_columns_overwrites_existing_canonical_columns() -> None:
    from src.prepare_corpus import rename_columns

    raw = pd.DataFrame(
        {
            "PP_NR": ["p1"],
            "participant_id": ["stale-participant"],
            "SENT": ["s1"],
            "sentence_id": ["stale-sentence"],
            "WORD_ID_WITHIN_TRIAL": [1],
            "word_id": ["stale-word-id"],
            "WORD": ["Hello,"],
            "word": ["stale-word"],
            "FFD": [100],
            "GD": [150],
            "GPT": [200],
            "TRT": [250],
        }
    )

    config = {
        "column_map": {
            "participant_id": "PP_NR",
            "sentence_id": "SENT",
            "word_id": "WORD_ID_WITHIN_TRIAL",
            "word": "WORD",
            "first_fixation_duration": "FFD",
            "gaze_duration": "GD",
            "go_past_time": "GPT",
            "total_reading_time": "TRT",
        },
        "optional_column_map": {
            "position_in_sentence": "WORD_ID_WITHIN_TRIAL",
        },
    }

    renamed = rename_columns(raw, config)

    assert renamed.columns.is_unique
    assert renamed.loc[0, "participant_id"] == "p1"
    assert renamed.loc[0, "sentence_id"] == "s1"
    assert renamed.loc[0, "word_id"] == 1
    assert renamed.loc[0, "position_in_sentence"] == 1
    assert renamed.loc[0, "word"] == "Hello,"
