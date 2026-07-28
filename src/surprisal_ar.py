from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import ITEM_KEY_COLS, environment_versions, model_slug, read_table, word_char_spans, write_json, write_table
from .scoring import aggregate_token_scores_to_words, effective_context_limit


def score_causal_tokens(
    model,
    token_ids: Sequence[int],
    bos_id: int,
    device: str,
    max_context: int,
    stride: int,
) -> tuple[np.ndarray, int]:
    """Score every content token once with a rolling left-context window."""
    if max_context < 2:
        raise ValueError("The causal model context must contain at least two tokens.")
    if stride <= 0 or stride >= max_context:
        raise ValueError("--stride must be positive and smaller than the context limit.")

    full_ids = [int(bos_id)] + [int(value) for value in token_ids]
    total = len(full_ids)
    scores = np.full(len(token_ids), np.nan, dtype=float)
    previous_end = 1
    end = min(max_context, total)
    windows = 0

    while previous_end < total:
        begin = max(0, end - max_context)
        score_start = max(previous_end, begin + 1)
        input_ids = torch.tensor([full_ids[begin:end]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0]
        log_probs = F.log_softmax(logits, dim=-1)
        for global_position in range(score_start, end):
            local_position = global_position - begin
            gold_id = full_ids[global_position]
            score = -float(log_probs[local_position - 1, gold_id].item()) / math.log(2)
            scores[global_position - 1] = score
        previous_end = end
        windows += 1
        if end >= total:
            break
        end = min(total, end + stride)

    if np.isnan(scores).any():
        missing = int(np.isnan(scores).sum())
        raise RuntimeError(f"Causal scoring left {missing} content tokens unscored.")
    return scores, windows


def score_sentence(
    model,
    tokenizer,
    words: Sequence[str],
    device: str,
    max_context: int,
    stride: int,
) -> tuple[list[float | None], list[int], dict[str, int | float]]:
    text, spans = word_char_spans(words)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    token_ids = encoded["input_ids"]
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("The tokenizer has neither a BOS nor EOS token for initial context.")

    token_scores, windows = score_causal_tokens(
        model=model,
        token_ids=token_ids,
        bos_id=bos_id,
        device=device,
        max_context=max_context,
        stride=stride,
    )
    values, counts = aggregate_token_scores_to_words(token_scores, offsets, spans)
    diagnostics = {
        "characters": len(text),
        "model_tokens": len(token_ids),
        "words": len(words),
        "scored_words": sum(value is not None for value in values),
        "missing_words": sum(value is None for value in values),
        "multi_subword_words": sum(count > 1 for count in counts),
        "max_subwords_per_word": max(counts, default=0),
        "windows": windows,
    }
    return values, counts, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute rolling-window autoregressive word surprisal in bits.")
    parser.add_argument("--input", required=True, help="Unique item table from prepare_corpus/prepare_geco.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-sentences", type=int, default=None, help="Debug-only sentence limit.")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--audit-output", default=None)
    args = parser.parse_args()

    items = read_table(args.input)
    required = set(ITEM_KEY_COLS + ["position_in_sentence", "stimulus_token"])
    missing = sorted(required - set(items.columns))
    if missing:
        raise KeyError(f"Item table is missing required columns: {missing}")
    if items.duplicated(ITEM_KEY_COLS).any():
        raise ValueError("Input to surprisal_ar must contain unique sentence/word items.")

    sentence_ids = items["sentence_id"].drop_duplicates().tolist()
    if args.max_sentences is not None:
        sentence_ids = sentence_ids[: args.max_sentences]
        items = items[items["sentence_id"].isin(sentence_ids)].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offset mappings is required.")
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision).to(args.device)
    model.eval()

    context_limit = effective_context_limit(tokenizer, model, args.max_length)
    stride = args.stride or max(1, context_limit // 2)
    if stride >= context_limit:
        stride = context_limit - 1

    slug = model_slug(args.model)
    feature_name = f"{slug}_surprisal"
    subword_name = f"{slug}_ar_subword_count"
    rows: list[dict] = []
    sentence_audit: list[dict] = []

    grouped = items.groupby("sentence_id", sort=False)
    for sentence_id, group in tqdm(grouped, desc=f"AR surprisal: {args.model}"):
        group = group.sort_values("position_in_sentence")
        words = group["stimulus_token"].astype(str).tolist()
        values, counts, diagnostics = score_sentence(
            model=model,
            tokenizer=tokenizer,
            words=words,
            device=args.device,
            max_context=context_limit,
            stride=stride,
        )
        for (_, item), value, count in zip(group.iterrows(), values, counts):
            rows.append(
                {
                    "sentence_id": item["sentence_id"],
                    "word_id": item["word_id"],
                    feature_name: value,
                    subword_name: count,
                }
            )
        sentence_audit.append({"sentence_id": sentence_id, **diagnostics})

    features = pd.DataFrame(rows)
    if features.duplicated(ITEM_KEY_COLS).any():
        raise AssertionError("Autoregressive scorer produced duplicate item keys.")
    write_table(features, args.output)

    sentence_audit_df = pd.DataFrame(sentence_audit)
    audit_path = Path(args.audit_output) if args.audit_output else Path(args.output).with_name(
        f"{Path(args.output).stem}_alignment_audit.json"
    )
    audit = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "device": args.device,
        "feature": feature_name,
        "unit": "bits",
        "context_limit": context_limit,
        "stride": stride,
        "sentences": int(len(sentence_audit_df)),
        "items": int(len(features)),
        "scored_items": int(features[feature_name].notna().sum()),
        "missing_items": int(features[feature_name].isna().sum()),
        "maximum_model_tokens": int(sentence_audit_df["model_tokens"].max()) if len(sentence_audit_df) else 0,
        "sentences_over_context_limit": int((sentence_audit_df["model_tokens"] + 1 > context_limit).sum()) if len(sentence_audit_df) else 0,
        "multi_subword_items": int((features[subword_name] > 1).sum()),
        "mean_subwords_per_item": float(features[subword_name].mean()) if len(features) else None,
        "environment_versions": environment_versions(),
    }
    write_json(audit, audit_path)
    sentence_audit_path = audit_path.with_name(audit_path.stem.replace("alignment_audit", "sentence_audit") + ".csv")
    write_table(sentence_audit_df, sentence_audit_path)

    print(f"Wrote {len(features):,} item features to {args.output}")
    print(f"Feature column: {feature_name}")
    print(f"Alignment audit: {audit_path}")


if __name__ == "__main__":
    main()
