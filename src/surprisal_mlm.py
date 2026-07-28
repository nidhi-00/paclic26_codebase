from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .common import (
    ITEM_KEY_COLS,
    batched,
    environment_versions,
    model_slug,
    read_table,
    word_char_spans,
    write_json,
    write_table,
)
from .scoring import (
    content_positions_from_special_mask,
    effective_context_limit,
    target_centred_window,
    word_token_indices,
)


def prepare_window(
    tokenizer,
    content_ids: Sequence[int],
    start: int,
    end: int,
) -> tuple[torch.Tensor, list[int]]:
    prepared = tokenizer.prepare_for_model(
        list(content_ids[start:end]),
        add_special_tokens=True,
        return_special_tokens_mask=True,
        return_attention_mask=False,
        truncation=False,
    )
    input_ids = torch.tensor(prepared["input_ids"], dtype=torch.long)
    content_positions = content_positions_from_special_mask(prepared["special_tokens_mask"])
    if len(content_positions) != end - start:
        raise AssertionError("Tokenizer special-token mapping does not match content length.")
    return input_ids, content_positions


def run_masked_tasks(
    model,
    tasks: list[dict],
    pad_id: int,
    device: str,
    batch_size: int,
    values: dict[str, list[float | None]],
) -> int:
    model_batches = 0
    for task_batch in batched(tasks, batch_size):
        maximum_length = max(len(task["input_ids"]) for task in task_batch)
        input_ids = torch.full(
            (len(task_batch), maximum_length),
            fill_value=pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, task in enumerate(task_batch):
            length = len(task["input_ids"])
            input_ids[row, :length] = task["input_ids"]
            attention_mask[row, :length] = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
        model_batches += 1

        for row, task in enumerate(task_batch):
            score = 0.0
            for position, gold_id in zip(task["positions"], task["gold_ids"]):
                log_probs = F.log_softmax(logits[row, position], dim=-1)
                score += -float(log_probs[int(gold_id)].item()) / math.log(2)
            current = values[task["mode"]][task["word_index"]]
            values[task["mode"]][task["word_index"]] = float(current or 0.0) + score
    return model_batches


def score_sentence(
    model,
    tokenizer,
    words: Sequence[str],
    device: str,
    max_length: int,
    modes: Sequence[str],
    batch_size: int = 32,
) -> tuple[dict[str, list[float | None]], list[int], dict[str, int | float]]:
    text, spans = word_char_spans(words)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    content_ids = encoded["input_ids"]
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    mapping = word_token_indices(offsets, spans)
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = max_length - special_count
    if capacity <= 0:
        raise ValueError("Model maximum length leaves no room for content tokens.")
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    if mask_id is None:
        raise ValueError("The tokenizer has no mask token.")
    if pad_id is None:
        raise ValueError("The tokenizer has no pad token, which is required for batched MLM scoring.")
    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    values: dict[str, list[float | None]] = {
        mode: [None] * len(mapping) for mode in modes
    }
    tasks: list[dict] = []
    windowed_words = 0
    missing_words = 0

    for word_index, indices in enumerate(mapping):
        if not indices:
            missing_words += 1
            continue
        window = target_centred_window(
            total_tokens=len(content_ids),
            target_start=min(indices),
            target_end=max(indices) + 1,
            capacity=capacity,
        )
        if window.start > 0 or window.end < len(content_ids):
            windowed_words += 1
        base_ids, content_positions = prepare_window(
            tokenizer, content_ids, window.start, window.end
        )
        target_positions = [
            content_positions[index - window.start] for index in indices
        ]

        if "tokenwise" in modes:
            values["tokenwise"][word_index] = 0.0
            for position in target_positions:
                masked = base_ids.clone()
                gold_id = int(masked[position].item())
                masked[position] = mask_id
                tasks.append(
                    {
                        "input_ids": masked,
                        "positions": [position],
                        "gold_ids": [gold_id],
                        "mode": "tokenwise",
                        "word_index": word_index,
                    }
                )

        if "whole_word" in modes:
            values["whole_word"][word_index] = 0.0
            masked = base_ids.clone()
            gold_ids = [int(masked[position].item()) for position in target_positions]
            for position in target_positions:
                masked[position] = mask_id
            tasks.append(
                {
                    "input_ids": masked,
                    "positions": target_positions,
                    "gold_ids": gold_ids,
                    "mode": "whole_word",
                    "word_index": word_index,
                }
            )

    model_batches = run_masked_tasks(
        model=model,
        tasks=tasks,
        pad_id=pad_id,
        device=device,
        batch_size=batch_size,
        values=values,
    )
    counts = [len(indices) for indices in mapping]
    diagnostics = {
        "characters": len(text),
        "model_tokens": len(content_ids),
        "words": len(words),
        "scored_words": len(words) - missing_words,
        "missing_words": missing_words,
        "multi_subword_words": sum(count > 1 for count in counts),
        "max_subwords_per_word": max(counts, default=0),
        "windowed_words": windowed_words,
        "masked_sequences": len(tasks),
        "model_batches": model_batches,
    }
    return values, counts, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute token-wise and whole-word masked pseudo-surprisal in bits."
    )
    parser.add_argument(
        "--input", required=True, help="Unique item table from prepare_corpus/prepare_geco."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--max-sentences", type=int, default=None, help="Debug-only sentence limit."
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--masking",
        choices=["tokenwise", "whole_word", "both"],
        default="both",
    )
    parser.add_argument("--audit-output", default=None)
    args = parser.parse_args()

    items = read_table(args.input)
    required = set(ITEM_KEY_COLS + ["position_in_sentence", "stimulus_token"])
    missing = sorted(required - set(items.columns))
    if missing:
        raise KeyError(f"Item table is missing required columns: {missing}")
    if items.duplicated(ITEM_KEY_COLS).any():
        raise ValueError(
            "Input to surprisal_mlm must contain unique sentence/word items."
        )

    sentence_ids = items["sentence_id"].drop_duplicates().tolist()
    if args.max_sentences is not None:
        sentence_ids = sentence_ids[: args.max_sentences]
        items = items[items["sentence_id"].isin(sentence_ids)].copy()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offset mappings is required.")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, revision=args.revision
    ).to(args.device)
    model.eval()
    context_limit = effective_context_limit(tokenizer, model, args.max_length)
    modes = ["tokenwise", "whole_word"] if args.masking == "both" else [args.masking]

    slug = model_slug(args.model)
    feature_names = {
        "tokenwise": f"{slug}_tokenwise_pseudo_surprisal",
        "whole_word": f"{slug}_whole_word_pseudo_surprisal",
    }
    subword_name = f"{slug}_mlm_subword_count"
    rows: list[dict] = []
    sentence_audit: list[dict] = []

    for sentence_id, group in tqdm(
        items.groupby("sentence_id", sort=False),
        desc=f"MLM pseudo-surprisal: {args.model}",
    ):
        group = group.sort_values("position_in_sentence")
        words = group["stimulus_token"].astype(str).tolist()
        mode_values, counts, diagnostics = score_sentence(
            model=model,
            tokenizer=tokenizer,
            words=words,
            device=args.device,
            max_length=context_limit,
            modes=modes,
            batch_size=args.batch_size,
        )
        for row_index, (_, item) in enumerate(group.iterrows()):
            record = {
                "sentence_id": item["sentence_id"],
                "word_id": item["word_id"],
                subword_name: counts[row_index],
            }
            for mode in modes:
                record[feature_names[mode]] = mode_values[mode][row_index]
            rows.append(record)
        sentence_audit.append({"sentence_id": sentence_id, **diagnostics})

    features = pd.DataFrame(rows)
    if features.duplicated(ITEM_KEY_COLS).any():
        raise AssertionError("Masked scorer produced duplicate item keys.")
    write_table(features, args.output)

    sentence_audit_df = pd.DataFrame(sentence_audit)
    audit_path = (
        Path(args.audit_output)
        if args.audit_output
        else Path(args.output).with_name(
            f"{Path(args.output).stem}_alignment_audit.json"
        )
    )
    feature_summary = {
        name: {
            "scored_items": int(features[name].notna().sum()),
            "missing_items": int(features[name].isna().sum()),
        }
        for mode, name in feature_names.items()
        if mode in modes
    }
    audit = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "device": args.device,
        "batch_size": args.batch_size,
        "masking_modes": modes,
        "features": feature_summary,
        "unit": "bits",
        "context_limit": context_limit,
        "sentences": int(len(sentence_audit_df)),
        "items": int(len(features)),
        "maximum_model_tokens": int(sentence_audit_df["model_tokens"].max())
        if len(sentence_audit_df)
        else 0,
        "sentences_over_context_limit": int(
            (
                sentence_audit_df["model_tokens"]
                + tokenizer.num_special_tokens_to_add(False)
                > context_limit
            ).sum()
        )
        if len(sentence_audit_df)
        else 0,
        "windowed_item_scores": int(sentence_audit_df["windowed_words"].sum())
        if len(sentence_audit_df)
        else 0,
        "masked_sequences": int(sentence_audit_df["masked_sequences"].sum())
        if len(sentence_audit_df)
        else 0,
        "model_batches": int(sentence_audit_df["model_batches"].sum())
        if len(sentence_audit_df)
        else 0,
        "multi_subword_items": int((features[subword_name] > 1).sum()),
        "mean_subwords_per_item": float(features[subword_name].mean())
        if len(features)
        else None,
        "environment_versions": environment_versions(),
    }
    write_json(audit, audit_path)
    sentence_audit_path = audit_path.with_name(
        audit_path.stem.replace("alignment_audit", "sentence_audit") + ".csv"
    )
    write_table(sentence_audit_df, sentence_audit_path)

    print(f"Wrote {len(features):,} item features to {args.output}")
    print("Feature columns:", [feature_names[mode] for mode in modes])
    print(f"Alignment audit: {audit_path}")


if __name__ == "__main__":
    main()
