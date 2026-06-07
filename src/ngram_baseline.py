from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

from .common import KEY_COLS, read_table, write_table

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


def tokenize_sentence(text: str) -> List[str]:
    return str(text).strip().split()


class AddKNGram:
    def __init__(self, order: int = 5, alpha: float = 0.1, min_count: int = 1):
        self.order = order
        self.alpha = alpha
        self.min_count = min_count
        self.counts = [Counter() for _ in range(order + 1)]
        self.context_counts = [Counter() for _ in range(order + 1)]
        self.vocab = set()

    def fit(self, sentences: Iterable[List[str]]) -> None:
        raw_counts = Counter(tok.lower() for sent in sentences for tok in sent)
        self.vocab = {w for w, c in raw_counts.items() if c >= self.min_count}
        self.vocab.update({BOS, EOS, UNK})
        for sent in sentences:
            toks = [w.lower() if w.lower() in self.vocab else UNK for w in sent]
            padded = [BOS] * (self.order - 1) + toks + [EOS]
            for n in range(1, self.order + 1):
                for i in range(len(padded) - n + 1):
                    ng = tuple(padded[i : i + n])
                    self.counts[n][ng] += 1
                    if n > 1:
                        self.context_counts[n][ng[:-1]] += 1

    def prob(self, word: str, context: List[str]) -> float:
        word = word.lower() if word.lower() in self.vocab else UNK
        ctx = [w.lower() if w.lower() in self.vocab else UNK for w in context]
        vocab_size = len(self.vocab)
        # Try highest-order available context with add-alpha smoothing.
        for n in range(min(self.order, len(ctx) + 1), 0, -1):
            hist = tuple(ctx[-(n - 1) :]) if n > 1 else tuple()
            ng = hist + (word,)
            numerator = self.counts[n][ng] + self.alpha
            denominator = (self.context_counts[n][hist] if n > 1 else sum(self.counts[1].values())) + self.alpha * vocab_size
            if denominator > 0:
                return numerator / denominator
        return 1.0 / vocab_size

    def surprisal(self, word: str, context: List[str]) -> float:
        return -math.log2(max(self.prob(word, context), 1e-12))


def load_training_sentences(path: str | None, fallback_df: pd.DataFrame) -> List[List[str]]:
    if path:
        sentences = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                toks = tokenize_sentence(line)
                if toks:
                    sentences.append(toks)
        return sentences
    print("WARNING: No --train-text supplied. Training n-gram model on corpus sentences as debug fallback only.")
    return [tokenize_sentence(s) for s in fallback_df.drop_duplicates("sentence_id")["sentence_text"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-text", default=None)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    df = read_table(args.input)
    sentence_df = df.drop_duplicates(["sentence_id", "position_in_sentence"])
    train_sents = load_training_sentences(args.train_text, sentence_df)
    model = AddKNGram(order=args.order, alpha=args.alpha)
    model.fit(train_sents)

    rows = []
    for sid, g in tqdm(sentence_df.groupby("sentence_id"), desc="Computing n-gram surprisal"):
        g = g.sort_values("position_in_sentence")
        context: List[str] = []
        for _, row in g.iterrows():
            word = str(row["word_clean"] if "word_clean" in g.columns else row["word"])
            surprisal = model.surprisal(word, context)
            rows.append({"sentence_id": row["sentence_id"], "word_id": row["word_id"], f"ngram{args.order}_surprisal": surprisal})
            context.append(word)

    feat = pd.DataFrame(rows)
    # Expand to participant rows via merge keys sentence_id + word_id.
    out = df[KEY_COLS].merge(feat, on=["sentence_id", "word_id"], how="left")
    write_table(out, args.output)
    print(f"Wrote {len(out):,} rows to {args.output}")


if __name__ == "__main__":
    main()
