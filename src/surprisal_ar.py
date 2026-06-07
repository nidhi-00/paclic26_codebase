from __future__ import annotations

import argparse
import math
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import KEY_COLS, read_table, write_table


def token_surprisal_bits(model, input_ids: torch.Tensor, target_start: int) -> float:
    """Sum -log2 p(token_t | previous tokens) for positions >= target_start."""
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[0]
    total = 0.0
    ids = input_ids[0]
    for pos in range(target_start, len(ids)):
        # logits[pos-1] predicts ids[pos]
        log_probs = F.log_softmax(logits[pos - 1], dim=-1)
        total += -float(log_probs[ids[pos]].item()) / math.log(2)
    return total


def compute_sentence_ar(model, tokenizer, words: List[str], device: str) -> List[float | None]:
    surprisals: List[float | None] = []
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    for i, word in enumerate(words):
        prefix = " ".join(words[:i])
        target = (" " if prefix else "") + word
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids if prefix else []
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        if not target_ids:
            surprisals.append(None)
            continue
        context_ids = ([bos_id] if bos_id is not None else []) + prefix_ids
        ids = context_ids + target_ids
        target_start = len(context_ids)
        if target_start == 0:
            # No BOS available; first-token probability is undefined for GPT-style models.
            surprisals.append(None)
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        surprisals.append(token_surprisal_bits(model, input_ids, target_start))
    return surprisals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-sentences", type=int, default=None, help="Debug limit.")
    args = parser.parse_args()

    df = read_table(args.input)
    sentence_df = df.drop_duplicates(["sentence_id", "position_in_sentence"]).copy()
    sentence_ids = sentence_df["sentence_id"].drop_duplicates().tolist()
    if args.max_sentences is not None:
        sentence_ids = sentence_ids[: args.max_sentences]
        sentence_df = sentence_df[sentence_df["sentence_id"].isin(sentence_ids)]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    feature_name = args.model.replace("/", "_").replace("-", "_") + "_surprisal"
    rows = []

    for sid, g in tqdm(sentence_df.groupby("sentence_id"), desc=f"AR surprisal: {args.model}"):
        g = g.sort_values("position_in_sentence")
        words = [str(w) for w in g["word_clean"].tolist()]
        values = compute_sentence_ar(model, tokenizer, words, args.device)
        for (_, row), val in zip(g.iterrows(), values):
            rows.append({"sentence_id": row["sentence_id"], "word_id": row["word_id"], feature_name: val})

    feat = pd.DataFrame(rows)
    out = df[KEY_COLS].merge(feat, on=["sentence_id", "word_id"], how="left")
    write_table(out, args.output)
    print(f"Wrote {len(out):,} rows to {args.output} with column {feature_name}")


if __name__ == "__main__":
    main()
