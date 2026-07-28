from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .common import overlapping_token_indices


@dataclass(frozen=True)
class Window:
    start: int
    end: int


def effective_context_limit(tokenizer, model, requested: int | None = None) -> int:
    candidates: list[int] = []
    if requested and requested > 0:
        candidates.append(int(requested))
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)
    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(getattr(model, "config", object()), attr, None)
        if isinstance(value, int) and value > 0:
            candidates.append(value)
    if not candidates:
        raise ValueError("Could not determine the model context limit; pass --max-length.")
    return min(candidates)


def target_centred_window(
    total_tokens: int,
    target_start: int,
    target_end: int,
    capacity: int,
) -> Window:
    """Choose a content-token window that contains the complete target span."""
    if not (0 <= target_start < target_end <= total_tokens):
        raise ValueError("Invalid target token span.")
    target_width = target_end - target_start
    if target_width > capacity:
        raise ValueError(
            f"Target has {target_width} tokens, exceeding the available window capacity {capacity}."
        )
    spare = capacity - target_width
    left_budget = spare // 2
    start = max(0, target_start - left_budget)
    end = min(total_tokens, start + capacity)
    start = max(0, end - capacity)
    if start > target_start or end < target_end:
        raise AssertionError("Window construction failed to retain the complete target.")
    return Window(start=start, end=end)


def word_token_indices(
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
) -> list[list[int]]:
    return [overlapping_token_indices(offsets, span) for span in spans]


def aggregate_token_scores_to_words(
    token_scores: Sequence[float],
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
) -> tuple[list[float | None], list[int]]:
    mapping = word_token_indices(offsets, spans)
    values: list[float | None] = []
    counts: list[int] = []
    scores = np.asarray(token_scores, dtype=float)
    for indices in mapping:
        counts.append(len(indices))
        if not indices or np.isnan(scores[indices]).any():
            values.append(None)
        else:
            values.append(float(scores[indices].sum()))
    return values, counts


def content_positions_from_special_mask(mask: Sequence[int]) -> list[int]:
    return [index for index, is_special in enumerate(mask) if not int(is_special)]
