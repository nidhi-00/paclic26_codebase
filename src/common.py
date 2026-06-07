from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

KEY_COLS = ["participant_id", "sentence_id", "word_id"]


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, **kwargs)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported file type: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    else:
        raise ValueError(f"Unsupported output type: {path}")


def clean_word(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    # Remove isolated punctuation while preserving contractions/hyphens.
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text, flags=re.UNICODE)
    return text


def is_punctuation_only(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and re.fullmatch(r"[^\w]+", text, flags=re.UNICODE) is not None


def safe_log_ms(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric > 0)
    return np.log(numeric)


def sentence_key(df: pd.DataFrame) -> pd.Series:
    return df["sentence_id"].astype(str)


def aggregate_sentence_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create sentence_text, position_in_sentence, and sentence_length if missing."""
    out = df.copy()
    out["_sort_word_id"] = pd.to_numeric(out["word_id"], errors="coerce")
    out["_sort_word_id"] = out["_sort_word_id"].fillna(np.arange(len(out)))
    out = out.sort_values(["sentence_id", "_sort_word_id"]).reset_index(drop=True)

    if "position_in_sentence" not in out.columns or out["position_in_sentence"].isna().all():
        out["position_in_sentence"] = out.groupby("sentence_id").cumcount() + 1

    if "sentence_length" not in out.columns or out["sentence_length"].isna().all():
        out["sentence_length"] = out.groupby("sentence_id")["word"].transform("size")

    if "sentence_text" not in out.columns or out["sentence_text"].isna().all():
        sentence_texts = (
            out.groupby("sentence_id")["word"]
            .apply(lambda words: " ".join(str(w) for w in words))
            .rename("sentence_text")
        )
        out = out.drop(columns=["sentence_text"], errors="ignore").merge(
            sentence_texts, left_on="sentence_id", right_index=True, how="left"
        )

    return out.drop(columns=["_sort_word_id"], errors="ignore")


def word_char_spans(words: List[str]) -> tuple[str, List[tuple[int, int]]]:
    """Join words with spaces and return char spans for each word."""
    parts = []
    spans: List[tuple[int, int]] = []
    pos = 0
    for i, word in enumerate(words):
        if i > 0:
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(word)
        pos += len(word)
        spans.append((start, pos))
    return "".join(parts), spans


def batched(iterable: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]
