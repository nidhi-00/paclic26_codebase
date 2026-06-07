from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import ensure_parent, read_table

BASE_CONTROLS = [
    "word_length",
    "log_word_frequency",
    "position_in_sentence",
    "sentence_length",
]
BASE_BINARY = ["is_sentence_initial", "is_sentence_final"]


def available_predictors(df: pd.DataFrame) -> Dict[str, List[str]]:
    ngram = [c for c in df.columns if c.startswith("ngram") and c.endswith("_surprisal")]
    ar = [c for c in df.columns if c.endswith("_surprisal") and not c.startswith("ngram") and "pseudo" not in c]
    mlm = [c for c in df.columns if c.endswith("_pseudo_surprisal")]
    return {"ngram": ngram, "ar": ar, "mlm": mlm}


def model_specs(df: pd.DataFrame) -> Dict[str, List[str]]:
    pred = available_predictors(df)
    ngram_main = pred["ngram"][:1]
    ar_main = pred["ar"][:1]
    mlm_main = pred["mlm"][:1]
    base = [c for c in BASE_CONTROLS + BASE_BINARY if c in df.columns]
    return {
        "lexical": base,
        "lexical_plus_ngram": base + ngram_main,
        "lexical_plus_ar": base + ar_main,
        "lexical_plus_mlm": base + mlm_main,
        "lexical_plus_ngram_ar": base + ngram_main + ar_main,
        "full_ngram_ar_mlm": base + ngram_main + ar_main + mlm_main,
    }


def fit_ols(df: pd.DataFrame, target: str, predictors: List[str]) -> dict:
    cols = [target] + predictors
    sub = df[cols].dropna().copy()
    if len(sub) < 100:
        raise ValueError(f"Too few rows for {target} with predictors {predictors}: {len(sub)}")
    formula = target + " ~ " + " + ".join(predictors)
    model = smf.ols(formula, data=sub).fit()
    return {
        "n": int(len(sub)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "params": {k: float(v) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
    }


def cv_metrics(df: pd.DataFrame, target: str, predictors: List[str], group_col: str = "sentence_id") -> dict:
    cols = [target, group_col] + predictors
    sub = df[cols].dropna().copy()
    if sub[group_col].nunique() < 5:
        return {"cv_rmse": np.nan, "cv_mae": np.nan}
    X = sub[predictors].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    groups = sub[group_col].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=5)
    preds = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
    return {
        "cv_rmse": float(mean_squared_error(y, preds, squared=False)),
        "cv_mae": float(mean_absolute_error(y, preds)),
    }


def run_analysis(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    specs = model_specs(df)
    records = []
    detailed = {}
    lexical_r2 = None

    for name, predictors in specs.items():
        predictors = [p for p in predictors if p in df.columns]
        if not predictors:
            continue
        result = fit_ols(df, target, predictors)
        cv = cv_metrics(df, target, predictors)
        if name == "lexical":
            lexical_r2 = result["r2"]
        delta = result["r2"] - lexical_r2 if lexical_r2 is not None else np.nan
        records.append({
            "target": target,
            "model": name,
            "n_predictors": len(predictors),
            "n": result["n"],
            "r2": result["r2"],
            "adj_r2": result["adj_r2"],
            "delta_r2_vs_lexical": delta,
            "aic": result["aic"],
            "bic": result["bic"],
            **cv,
            "predictors": ",".join(predictors),
        })
        detailed[name] = result

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / f"{target}_model_comparison.csv", index=False)
    with open(output_dir / f"{target}_ols_details.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    print(f"Wrote results for {target} to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--targets", nargs="+", default=["log_gaze_duration", "log_total_reading_time"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = read_table(args.input)
    outdir = Path(args.output_dir)
    for target in args.targets:
        if target not in df.columns:
            print(f"Skipping missing target: {target}")
            continue
        run_analysis(df, target, outdir)


if __name__ == "__main__":
    main()
