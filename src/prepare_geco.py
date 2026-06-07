import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None


READING_COLUMNS = {
    "ffd": "WORD_FIRST_FIXATION_DURATION",
    "gd": "WORD_GAZE_DURATION",
    "gpt": "WORD_GO_PAST_TIME",
    "trt": "WORD_TOTAL_READING_TIME",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def to_num(series):
    return pd.to_numeric(series.replace(".", np.nan), errors="coerce")


def has_alnum(word):
    if pd.isna(word):
        return False
    return bool(re.search(r"[A-Za-z0-9]", str(word)))


def reconstruct_sentence(group):
    group = group.sort_values("word_pos")
    words = group["word"].astype(str).tolist()
    positions = group["word_pos"].astype(int).tolist()

    parts = []
    spans = []
    cursor = 0

    for i, (pos, word) in enumerate(zip(positions, words)):
        if i > 0:
            parts.append(" ")
            cursor += 1

        start = cursor
        parts.append(word)
        cursor += len(word)
        end = cursor

        spans.append({
            "word_pos": int(pos),
            "word": word,
            "start": int(start),
            "end": int(end),
        })

    return pd.Series({
        "sentence_text": "".join(parts),
        "word_spans_json": json.dumps(spans, ensure_ascii=False),
        "sent_len": len(words),
    })


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw parquet: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Raw shape: {df.shape}")

    # Filter to English monolingual rows.
    df = df[
        df["GROUP"].astype(str).str.lower().eq("monolingual")
        & df["LANGUAGE_RANK"].astype(str).str.upper().eq("L1")
        & df["LANGUAGE"].astype(str).str.lower().eq("english")
    ].copy()

    print(f"After monolingual English filter: {df.shape}")

    # Core identifiers.
    df["participant_id"] = df["PP_NR"].astype(str)
    df["part"] = to_num(df["PART"]).astype("Int64")
    df["trial"] = to_num(df["TRIAL"]).astype("Int64")
    df["word_pos"] = to_num(df["WORD_ID_WITHIN_TRIAL"]).astype("Int64")
    df["word_id"] = df["WORD_ID"].astype(str)
    df["word"] = df["WORD"].astype(str)

    # Drop broken rows.
    df = df.dropna(subset=["part", "trial", "word_pos", "word"]).copy()
    df = df[df["word"].apply(has_alnum)].copy()

    # Treat each PART-TRIAL as one sentence/trial.
    df["sent_id"] = df["part"].astype(str) + "-" + df["trial"].astype(str)

    # Reading-time measures.
    for short_name, raw_col in READING_COLUMNS.items():
        df[short_name] = to_num(df[raw_col])
        df[f"log_{short_name}"] = np.where(df[short_name] > 0, np.log(df[short_name]), np.nan)

    # Lexical features.
    df["word_lower"] = df["word"].str.lower()
    df["word_clean"] = df["word_lower"].str.replace(r"[^a-z0-9']", "", regex=True)
    df["word_len"] = df["word_clean"].str.len()

    print("Computing word frequency features...")
    if zipf_frequency is not None:
        unique_words = df["word_clean"].dropna().unique()
        freq_cache = {w: zipf_frequency(w, "en") if w else np.nan for w in unique_words}
        df["zipf_freq"] = df["word_clean"].map(freq_cache)
    else:
        df["zipf_freq"] = np.nan

    # Unique word items: this is what we will score with GPT/BERT later.
    print("Creating unique word-item table...")
    items = (
        df[["sent_id", "word_pos", "word", "word_clean", "word_len", "zipf_freq"]]
        .drop_duplicates(["sent_id", "word_pos"])
        .sort_values(["sent_id", "word_pos"])
        .copy()
    )

    print("Reconstructing sentence/trial text...")
    sent_info = (
        items.groupby("sent_id", group_keys=False)
        .apply(reconstruct_sentence)
        .reset_index()
    )

    items = items.merge(sent_info[["sent_id", "sent_len"]], on="sent_id", how="left")
    items["is_sent_initial"] = (items["word_pos"].astype(int) == 1).astype(int)
    items["is_sent_final"] = (items["word_pos"].astype(int) == items["sent_len"].astype(int)).astype(int)

    # Merge item-level sentence features back to participant-level data.
    df = df.merge(
        items[["sent_id", "word_pos", "sent_len", "is_sent_initial", "is_sent_final"]],
        on=["sent_id", "word_pos"],
        how="left",
    )

    df = df.sort_values(["participant_id", "sent_id", "word_pos"])
    df["prev_word_len"] = df.groupby(["participant_id", "sent_id"])["word_len"].shift(1)
    df["prev_zipf_freq"] = df.groupby(["participant_id", "sent_id"])["zipf_freq"].shift(1)

    clean_out = out_dir / "geco_clean_wordlevel.parquet"
    items_out = out_dir / "geco_items.parquet"
    sent_out = out_dir / "geco_sentences.parquet"

    df.to_parquet(clean_out, index=False)
    items.to_parquet(items_out, index=False)
    sent_info.to_parquet(sent_out, index=False)

    print("\nDone.")
    print(f"Saved participant-level word data: {clean_out}")
    print(f"Saved unique word-item data:       {items_out}")
    print(f"Saved sentence/trial data:         {sent_out}")
    print(f"Final participant-level rows:      {len(df):,}")
    print(f"Unique word items to score:        {len(items):,}")
    print(f"Unique sentence/trial texts:       {len(sent_info):,}")


if __name__ == "__main__":
    main()