from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import write_table
from src.prepare_corpus import prepare_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small deterministic corpus for pipeline smoke tests.")
    parser.add_argument("--out-dir", default="data/debug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260801)
    participants = [f"p{i:02d}" for i in range(1, 9)]
    vocabulary = [
        "language",
        "reader",
        "context",
        "model",
        "predicts",
        "human",
        "processing",
        "during",
        "natural",
        "reading",
        "today",
    ]

    raw_rows: list[dict] = []
    item_latent: dict[tuple[str, str], float] = {}
    for sentence_number in range(1, 31):
        sentence_id = f"s{sentence_number:03d}"
        tokens = [vocabulary[(sentence_number + position) % len(vocabulary)] for position in range(1, 12)] + ["."]
        for position, token in enumerate(tokens, start=1):
            word_id = str(position)
            item_latent[(sentence_id, word_id)] = float(rng.normal())
            for participant_index, participant_id in enumerate(participants):
                lexical = 0.03 * len(token) + 0.02 * position
                contextual = 0.06 * item_latent[(sentence_id, word_id)]
                reader = 0.03 * participant_index
                noise = rng.normal(0, 0.07, size=4)
                logs = np.array([5.25, 5.50, 5.70, 5.90]) + lexical + reader + contextual * np.array([0.1, 0.5, 0.8, 1.2]) + noise
                durations = np.exp(logs)
                raw_rows.append(
                    {
                        "pid": participant_id,
                        "sid": sentence_id,
                        "wid": word_id,
                        "token": token,
                        "position": position,
                        "ffd": durations[0],
                        "gd": durations[1],
                        "gpt": durations[2],
                        "trt": durations[3],
                        "content": int(token != "."),
                    }
                )

    raw = pd.DataFrame(raw_rows)
    config = {
        "corpus_name": "synthetic_debug",
        "language": "en",
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
        "optional_column_map": {
            "position_in_sentence": "position",
            "is_content_word": "content",
        },
        "filters": {
            "min_rt": 50,
            "max_first_fixation_duration": 1200,
            "max_gaze_duration": 2000,
            "max_go_past_time": 5000,
            "max_total_reading_time": 8000,
        },
    }
    rows, items, sentences, _ = prepare_dataframe(raw, config)
    write_table(rows, out_dir / "participant_word.csv")
    write_table(items, out_dir / "items.csv")
    write_table(sentences, out_dir / "sentences.csv")

    features = items[["sentence_id", "word_id"]].copy()
    latent = np.array([item_latent[(row.sentence_id, str(row.word_id))] for row in items.itertuples()])
    shared = latent + rng.normal(0, 0.35, size=len(items))
    features["ngram5_kn_surprisal"] = 7.0 + 0.7 * shared + rng.normal(0, 0.4, size=len(items))
    features["gpt2_surprisal"] = 6.5 + 0.9 * shared + rng.normal(0, 0.25, size=len(items))
    features["bert_base_uncased_tokenwise_pseudo_surprisal"] = 5.9 + 0.75 * shared + rng.normal(0, 0.3, size=len(items))
    features["roberta_base_tokenwise_pseudo_surprisal"] = 5.7 + 0.82 * shared + rng.normal(0, 0.28, size=len(items))
    features["bert_base_uncased_whole_word_pseudo_surprisal"] = features["bert_base_uncased_tokenwise_pseudo_surprisal"] + rng.normal(0.15, 0.1, size=len(items))
    features["roberta_base_whole_word_pseudo_surprisal"] = features["roberta_base_tokenwise_pseudo_surprisal"] + rng.normal(0.12, 0.1, size=len(items))
    write_table(features, out_dir / "features.csv")
    print(f"Synthetic debug data written to {out_dir}")


if __name__ == "__main__":
    main()
