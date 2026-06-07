from __future__ import annotations

import argparse
from pathlib import Path

from .common import KEY_COLS, read_table, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--feature", action="append", required=True, help="Feature parquet/csv path. Repeat this argument.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = read_table(args.base)
    for feature_path in args.feature:
        feat = read_table(feature_path)
        feature_cols = [c for c in feat.columns if c not in KEY_COLS]
        if not feature_cols:
            raise ValueError(f"No feature columns found in {feature_path}")
        keep_cols = KEY_COLS + feature_cols
        df = df.merge(feat[keep_cols], on=KEY_COLS, how="left")
        print(f"Merged {feature_path}: {feature_cols}")

    write_table(df, args.output)
    print(f"Wrote merged table with {len(df):,} rows and {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
