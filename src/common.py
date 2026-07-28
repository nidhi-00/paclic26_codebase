from __future__ import annotations

import hashlib
import json
import importlib.metadata as metadata
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

KEY_COLS = ["participant_id", "sentence_id", "word_id"]
ITEM_KEY_COLS = ["sentence_id", "word_id"]
RT_COLS = [
    "first_fixation_duration",
    "gaze_duration",
    "go_past_time",
    "total_reading_time",
]


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


def write_json(value: object, path: str | Path) -> None:
    path = Path(path)
    ensure_parent(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")




def environment_versions(packages: Sequence[str] | None = None) -> dict[str, str]:
    names = list(packages or [
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "transformers",
        "torch",
        "nltk",
        "wordfreq",
    ])
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_word(value: object) -> str:
    """Return a lexical lookup form while leaving the stimulus token untouched."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", text, flags=re.UNICODE)


def contains_alnum(value: object) -> bool:
    if pd.isna(value):
        return False
    return re.search(r"\w", str(value), flags=re.UNICODE) is not None


def is_punctuation_only(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and re.fullmatch(r"[^\w]+", text, flags=re.UNICODE) is not None


def safe_log_ms(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.log(numeric.where(numeric > 0))


def normalise_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _stable_sort_values(df: pd.DataFrame) -> pd.Series:
    if "position_in_sentence" in df.columns:
        pos = pd.to_numeric(df["position_in_sentence"], errors="coerce")
    else:
        pos = pd.Series(np.nan, index=df.index, dtype=float)
    wid = pd.to_numeric(df["word_id"], errors="coerce")
    fallback = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    return pos.fillna(wid).fillna(fallback)


def build_item_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one item per sentence/word and a sentence table without participant duplication."""
    required = {"sentence_id", "word_id", "stimulus_token"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns required to construct items: {missing}")

    work = df.copy()
    work["sentence_id"] = normalise_identifier(work["sentence_id"])
    work["word_id"] = normalise_identifier(work["word_id"])
    work["stimulus_token"] = work["stimulus_token"].astype("string").fillna("")
    work["_sort_key"] = _stable_sort_values(work)

    conflict_counts = (
        work.groupby(ITEM_KEY_COLS, dropna=False)["stimulus_token"]
        .nunique(dropna=False)
    )
    conflicts = conflict_counts[conflict_counts > 1]
    if not conflicts.empty:
        examples = conflicts.head(10).index.tolist()
        raise ValueError(f"Conflicting stimulus tokens for the same item keys: {examples}")

    preferred = [
        "sentence_id",
        "word_id",
        "stimulus_token",
        "word",
        "lexical_form",
        "word_clean",
        "word_lower",
        "word_length",
        "zipf_frequency",
        "position_in_sentence",
        "sentence_length",
        "sentence_text",
        "pos",
        "is_content_word",
        "is_punctuation",
        "is_analysis_token",
    ]
    item_cols = [c for c in preferred if c in work.columns]
    items = (
        work.sort_values(["sentence_id", "_sort_key", "word_id"])
        .drop_duplicates(ITEM_KEY_COLS, keep="first")[item_cols + ["_sort_key"]]
        .copy()
    )

    if items.duplicated(ITEM_KEY_COLS).any():
        raise AssertionError("Item keys are not unique after deduplication.")

    items = items.sort_values(["sentence_id", "_sort_key", "word_id"]).reset_index(drop=True)
    computed_position = items.groupby("sentence_id", sort=False).cumcount() + 1
    if "position_in_sentence" not in items.columns:
        items["position_in_sentence"] = computed_position
    else:
        supplied = pd.to_numeric(items["position_in_sentence"], errors="coerce")
        items["position_in_sentence"] = supplied.fillna(computed_position).astype(int)

    duplicate_positions = items.duplicated(["sentence_id", "position_in_sentence"], keep=False)
    if duplicate_positions.any():
        bad = items.loc[duplicate_positions, ["sentence_id", "word_id", "position_in_sentence"]].head(10)
        raise ValueError(f"Duplicate positions within a sentence:\n{bad.to_string(index=False)}")

    lengths = items.groupby("sentence_id")["word_id"].transform("size").astype(int)
    items["sentence_length"] = lengths
    items["is_sentence_initial"] = (items["position_in_sentence"] == 1).astype(int)
    items["is_sentence_final"] = (items["position_in_sentence"] == items["sentence_length"]).astype(int)

    reconstructed = (
        items.sort_values(["sentence_id", "position_in_sentence"])
        .groupby("sentence_id", sort=False)["stimulus_token"]
        .apply(lambda values: " ".join(str(value) for value in values))
        .rename("reconstructed_sentence_text")
    )

    if "sentence_text" in work.columns and work["sentence_text"].notna().any():
        provided = (
            work.dropna(subset=["sentence_text"])
            .groupby("sentence_id", sort=False)["sentence_text"]
            .first()
            .astype(str)
            .rename("sentence_text")
        )
        sentence_table = pd.concat([provided, reconstructed], axis=1).reset_index()
        sentence_table["sentence_text"] = sentence_table["sentence_text"].fillna(
            sentence_table["reconstructed_sentence_text"]
        )
    else:
        sentence_table = reconstructed.reset_index().rename(
            columns={"reconstructed_sentence_text": "sentence_text"}
        )
        sentence_table["reconstructed_sentence_text"] = sentence_table["sentence_text"]

    sentence_lengths = items.groupby("sentence_id").size().rename("sentence_length")
    sentence_table = sentence_table.merge(sentence_lengths, on="sentence_id", how="left", validate="one_to_one")
    items = items.drop(columns=["sentence_text", "_sort_key"], errors="ignore").merge(
        sentence_table[["sentence_id", "sentence_text"]],
        on="sentence_id",
        how="left",
        validate="many_to_one",
    )
    return items, sentence_table


def merge_item_features_to_rows(rows: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    if items.duplicated(ITEM_KEY_COLS).any():
        raise ValueError("Item table contains duplicate item keys.")
    original_n = len(rows)
    overlap = [c for c in items.columns if c in rows.columns and c not in ITEM_KEY_COLS]
    base = rows.drop(columns=overlap, errors="ignore")
    merged = base.merge(items, on=ITEM_KEY_COLS, how="left", validate="many_to_one")
    if len(merged) != original_n:
        raise AssertionError("Merging item features changed the participant-row count.")
    return merged


def add_previous_item_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    prefix: str = "prev_",
) -> pd.DataFrame:
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        return df.copy()
    item_cols = ITEM_KEY_COLS + ["position_in_sentence"] + available
    items = (
        df[item_cols]
        .drop_duplicates(ITEM_KEY_COLS)
        .sort_values(["sentence_id", "position_in_sentence"])
        .copy()
    )
    for col in available:
        items[f"{prefix}{col}"] = items.groupby("sentence_id", sort=False)[col].shift(1)
    lag_cols = [f"{prefix}{c}" for c in available]
    return df.merge(
        items[ITEM_KEY_COLS + lag_cols],
        on=ITEM_KEY_COLS,
        how="left",
        validate="many_to_one",
    )


def word_char_spans(words: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for index, word in enumerate(words):
        word = str(word)
        if index:
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(word)
        pos += len(word)
        spans.append((start, pos))
    return "".join(parts), spans


def overlapping_token_indices(
    offsets: Sequence[tuple[int, int]], span: tuple[int, int]
) -> list[int]:
    start, end = span
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if not (left == right == 0) and max(left, start) < min(right, end)
    ]


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()


def batched(values: Sequence, batch_size: int) -> Iterator[Sequence]:
    for index in range(0, len(values), batch_size):
        yield values[index : index + batch_size]


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values))
