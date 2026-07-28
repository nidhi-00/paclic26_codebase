from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml

from .common import read_table, write_json, write_table
from .evaluation import resolve_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm ridge findings with participant and item random-intercept mixed models."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lexical_gpt2", "lexical_bert", "lexical_roberta", "lexical_all_transformers"],
    )
    parser.add_argument("--robustness-expression", default="is_analysis_token == 1")
    parser.add_argument("--random-slope", default=None, help="Optional feature alias for a participant random slope.")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only; do not use a sampled run in the paper.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    data = read_table(args.input).query(args.robustness_expression, engine="python").copy()
    if args.max_rows is not None and len(data) > args.max_rows:
        data = data.sample(args.max_rows, random_state=20260801)

    models, skipped = resolve_models(
        data.columns,
        list(config["baseline"]),
        dict(config["features"]),
        {name: aliases for name, aliases in config["models"].items() if name in args.models},
        strict_core=False,
    )
    targets = args.targets or list(config.get("targets", []))
    random_slope_column = None
    if args.random_slope:
        random_slope_column = config["features"].get(args.random_slope)
        if random_slope_column not in data.columns:
            raise KeyError(f"Random-slope feature alias {args.random_slope!r} is unavailable.")

    rows: list[dict] = []
    failures: list[dict] = []
    for target in targets:
        if target not in data.columns:
            failures.append({"target": target, "reason": "target missing"})
            continue
        for model_spec in models:
            columns = [target, "participant_id", "sentence_id", "word_id", *model_spec.predictors]
            sample = data[columns].copy()
            for predictor in model_spec.predictors:
                sample[predictor] = pd.to_numeric(sample[predictor], errors="coerce")
            sample[target] = pd.to_numeric(sample[target], errors="coerce")
            sample = sample.dropna().copy()
            if len(sample) < 100:
                failures.append(
                    {"target": target, "model": model_spec.name, "reason": f"only {len(sample)} complete rows"}
                )
                continue
            sample["item_id"] = sample["sentence_id"].astype(str) + "::" + sample["word_id"].astype(str)
            for predictor in model_spec.predictors:
                standard_deviation = sample[predictor].std(ddof=0)
                if standard_deviation > 0:
                    sample[predictor] = (sample[predictor] - sample[predictor].mean()) / standard_deviation
                else:
                    sample[predictor] = 0.0

            formula = target + " ~ " + " + ".join(model_spec.predictors)
            re_formula = "1"
            if random_slope_column and random_slope_column in model_spec.predictors:
                re_formula = f"1 + {random_slope_column}"
            try:
                model = smf.mixedlm(
                    formula,
                    sample,
                    groups=sample["participant_id"].astype(str),
                    re_formula=re_formula,
                    vc_formula={"item": "0 + C(item_id)"},
                )
                result = model.fit(method="lbfgs", reml=False, maxiter=500, disp=False)
            except Exception as error:
                failures.append(
                    {"target": target, "model": model_spec.name, "reason": repr(error)}
                )
                continue

            confidence = result.conf_int()
            for term, coefficient in result.fe_params.items():
                rows.append(
                    {
                        "target": target,
                        "model": model_spec.name,
                        "term": term,
                        "coefficient": float(coefficient),
                        "standard_error": float(result.bse_fe[term]),
                        "z": float(result.tvalues[term]),
                        "p_value": float(result.pvalues[term]),
                        "ci_low": float(confidence.loc[term, 0]),
                        "ci_high": float(confidence.loc[term, 1]),
                        "rows": len(sample),
                        "participants": sample["participant_id"].nunique(),
                        "items": sample["item_id"].nunique(),
                        "converged": bool(result.converged),
                        "log_likelihood": float(result.llf),
                        "aic": float(result.aic),
                        "bic": float(result.bic),
                        "random_effects": "participant intercept; item variance component"
                        + (f"; participant slope for {random_slope_column}" if re_formula != "1" else ""),
                    }
                )

    output = pd.DataFrame(rows)
    write_table(output, args.output)
    write_json(
        {
            "input": args.input,
            "models_requested": args.models,
            "models_skipped_at_resolution": skipped,
            "failures": failures,
            "debug_max_rows": args.max_rows,
            "warning": "A run with --max-rows is diagnostic only and must not be reported.",
        },
        Path(args.output).with_suffix(".json"),
    )
    print(f"Wrote {len(output):,} mixed-effects coefficient rows to {args.output}")


if __name__ == "__main__":
    main()
