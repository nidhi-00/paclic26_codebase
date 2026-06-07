from __future__ import annotations

import argparse
import math
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .common import KEY_COLS, read_table, word_char_spans, write_table


def overlapping_token_indices(offsets: List[tuple[int, int]], span: tuple[int, int]) -> List[int]:
    start, end = span
    idxs = []
    for i, (a, b) in enumerate(offsets):
        if a == b == 0:
            continue
        if max(a, start) < min(b, end):
            idxs.append(i)
    return idxs


def compute_sentence_mlm(model, tokenizer, words: List[str], device: str, max_length: int = 512) -> List[float | None]:
    text, spans = word_char_spans(words)
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=True, max_length=max_length)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    offsets = encoded["offset_mapping"]
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask token.")

    values: List[float | None] = []
    seq_len = input_ids.shape[1]
    for span in spans:
        tok_idxs = [i for i in overlapping_token_indices(offsets, span) if i < seq_len]
        if not tok_idxs:
            values.append(None)
            continue
        total = 0.0
        for idx in tok_idxs:
            original_id = input_ids[0, idx].item()
            masked = input_ids.clone()
            masked[0, idx] = mask_id
            with torch.no_grad():
                logits = model(input_ids=masked).logits[0, idx]
                log_probs = F.log_softmax(logits, dim=-1)
            total += -float(log_probs[original_id].item()) / math.log(2)
        values.append(total)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-sentences", type=int, default=None, help="Debug limit.")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    df = read_table(args.input)
    sentence_df = df.drop_duplicates(["sentence_id", "position_in_sentence"]).copy()
    sentence_ids = sentence_df["sentence_id"].drop_duplicates().tolist()
    if args.max_sentences is not None:
        sentence_ids = sentence_ids[: args.max_sentences]
        sentence_df = sentence_df[sentence_df["sentence_id"].isin(sentence_ids)]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(args.device)
    model.eval()

    feature_name = args.model.replace("/", "_").replace("-", "_") + "_pseudo_surprisal"
    rows = []

    for sid, g in tqdm(sentence_df.groupby("sentence_id"), desc=f"MLM pseudo-surprisal: {args.model}"):
        g = g.sort_values("position_in_sentence")
        words = [str(w) for w in g["word_clean"].tolist()]
        values = compute_sentence_mlm(model, tokenizer, words, args.device, args.max_length)
        for (_, row), val in zip(g.iterrows(), values):
            rows.append({"sentence_id": row["sentence_id"], "word_id": row["word_id"], feature_name: val})

    feat = pd.DataFrame(rows)
    out = df[KEY_COLS].merge(feat, on=["sentence_id", "word_id"], how="left")
    write_table(out, args.output)
    print(f"Wrote {len(out):,} rows to {args.output} with column {feature_name}")


if __name__ == "__main__":
    main()
