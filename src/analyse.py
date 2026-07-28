from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .common import environment_versions, read_table, write_json, write_table
from .evaluation import (
    bootstrap_oof_differences,
    evaluate_models,
    resolve_models,
    surprisal_correlations,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def select_robustness_sample(data: pd.DataFrame, name: str, expression: str) -> pd.DataFrame:
    try:
        sample = data.query(expression, engine="python").copy()
    except Exception as error:
        raise ValueError(f"Could not apply robustness filter {name!r}: {expression!r}") from error
    return sample




def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(float("nan"), index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    count = len(ordered)
    adjusted = ordered.to_numpy() * count / (pd.Series(range(1, count + 1), index=ordered.index).to_numpy())
    adjusted = pd.Series(adjusted, index=ordered.index)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    result.loc[adjusted.index] = adjusted
    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run like-for-like ridge evaluation with sentence- and participant-held-out predictions."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--robustness", nargs="+", default=None)
    parser.add_argument("--allow-missing-core", action="store_true")
    parser.add_argument("--no-tune-alpha", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--oof-format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    config = load_config(args.config)
    data = read_table(args.input)
    output_dir = Path(args.output_dir)
    oof_dir = output_dir / "oof_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    baseline = list(config["baseline"])
    feature_map = dict(config["features"])
    model_map = {name: list(aliases or []) for name, aliases in config["models"].items()}
    if args.models:
        requested_models = list(dict.fromkeys(["lexical", *args.models]))
        unknown_models = [name for name in requested_models if name not in model_map]
        if unknown_models:
            raise KeyError(f"Unknown model names: {unknown_models}")
        model_map = {name: model_map[name] for name in requested_models}
    models, skipped_models = resolve_models(
        columns=data.columns,
        baseline=baseline,
        feature_map=feature_map,
        model_map=model_map,
        core_models=config.get("core_models", []),
        strict_core=not args.allow_missing_core,
    )

    targets = args.targets or list(config.get("targets", []))
    split_config = dict(config.get("splits", {}))
    split_names = args.splits or list(split_config)
    robustness_config = dict(config.get("robustness", {"full": "is_analysis_token == 1"}))
    robustness_names = args.robustness or list(robustness_config)

    ridge_config = config.get("ridge", {})
    alphas = [float(value) for value in ridge_config.get("alphas", [1.0])]
    tune_alpha = bool(ridge_config.get("tune_alpha", True)) and not args.no_tune_alpha
    inner_folds = int(ridge_config.get("inner_folds", 3))
    bootstrap_config = config.get("bootstrap", {})
    bootstrap_iterations = args.bootstrap_iterations or int(bootstrap_config.get("iterations", 2000))
    seed = args.seed if args.seed is not None else int(bootstrap_config.get("seed", 20260801))

    all_metrics: list[pd.DataFrame] = []
    all_fold_metrics: list[pd.DataFrame] = []
    all_bootstrap: list[pd.DataFrame] = []
    all_alphas: list[pd.DataFrame] = []
    skipped_runs: list[dict[str, Any]] = []
    oof_manifest: list[dict[str, Any]] = []

    for robustness_name in robustness_names:
        if robustness_name not in robustness_config:
            raise KeyError(f"Unknown robustness filter: {robustness_name}")
        expression = robustness_config[robustness_name]
        try:
            sample = select_robustness_sample(data, robustness_name, expression)
        except ValueError as error:
            skipped_runs.append(
                {"robustness": robustness_name, "reason": str(error)}
            )
            continue
        if len(sample) < 100:
            skipped_runs.append(
                {
                    "robustness": robustness_name,
                    "reason": f"filter retained only {len(sample)} rows",
                }
            )
            continue

        for target in targets:
            if target not in sample.columns:
                skipped_runs.append(
                    {"target": target, "robustness": robustness_name, "reason": "target column missing"}
                )
                continue
            for split_name in split_names:
                if split_name not in split_config:
                    raise KeyError(f"Unknown split: {split_name}")
                split = split_config[split_name]
                group_column = split["group_column"]
                if group_column not in sample.columns:
                    skipped_runs.append(
                        {
                            "target": target,
                            "split": split_name,
                            "robustness": robustness_name,
                            "reason": f"group column {group_column!r} missing",
                        }
                    )
                    continue
                try:
                    metrics, fold_metrics, oof, alpha_rows = evaluate_models(
                        sample=sample,
                        target=target,
                        group_column=group_column,
                        split_name=split_name,
                        split_method=split["method"],
                        folds=split.get("folds"),
                        models=models,
                        alphas=alphas,
                        tune=tune_alpha,
                        inner_folds=inner_folds,
                        robustness_name=robustness_name,
                    )
                except ValueError as error:
                    skipped_runs.append(
                        {
                            "target": target,
                            "split": split_name,
                            "robustness": robustness_name,
                            "reason": str(error),
                        }
                    )
                    continue

                bootstrap = bootstrap_oof_differences(
                    oof=oof,
                    group_column=group_column,
                    model_names=[model.name for model in models],
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
                bootstrap.insert(0, "robustness", robustness_name)
                bootstrap.insert(0, "split", split_name)
                bootstrap.insert(0, "target", target)

                safe_target = target.replace("/", "_")
                oof_path = oof_dir / f"{safe_target}__{split_name}__{robustness_name}.{args.oof_format}"
                write_table(oof, oof_path)
                oof_manifest.append(
                    {
                        "target": target,
                        "split": split_name,
                        "robustness": robustness_name,
                        "path": str(oof_path.relative_to(output_dir)),
                        "rows": len(oof),
                        "models": [model.name for model in models],
                    }
                )
                all_metrics.append(metrics)
                all_fold_metrics.append(fold_metrics)
                all_bootstrap.append(bootstrap)
                all_alphas.append(alpha_rows)
                print(
                    f"Completed target={target}, split={split_name}, robustness={robustness_name}, rows={len(oof):,}"
                )

    if not all_metrics:
        raise RuntimeError("No analysis run completed. Check the run_manifest.json diagnostics.")

    metrics_output = pd.concat(all_metrics, ignore_index=True)
    fold_output = pd.concat(all_fold_metrics, ignore_index=True)
    bootstrap_output = pd.concat(all_bootstrap, ignore_index=True)
    alpha_output = pd.concat(all_alphas, ignore_index=True)
    bootstrap_output["cluster_signflip_q_bh_global"] = benjamini_hochberg(
        bootstrap_output["cluster_signflip_p_two_sided"]
    )
    bootstrap_output["cluster_signflip_q_bh_within_run"] = (
        bootstrap_output.groupby(["target", "split", "robustness"], group_keys=False)[
            "cluster_signflip_p_two_sided"
        ].apply(benjamini_hochberg)
    )
    write_table(metrics_output, output_dir / "model_metrics.csv")
    write_table(fold_output, output_dir / "fold_metrics.csv")
    write_table(bootstrap_output, output_dir / "bootstrap_intervals.csv")
    write_table(alpha_output, output_dir / "alpha_selections.csv")

    correlation_columns = list(
        dict.fromkeys(
            column
            for alias, column in feature_map.items()
            if "prev_" not in alias and column in data.columns and "surprisal" in column
        )
    )
    correlations = surprisal_correlations(data, correlation_columns)
    write_table(correlations, output_dir / "surprisal_correlations.csv")

    manifest = {
        "input": str(args.input),
        "config": str(args.config),
        "python": sys.version,
        "platform": platform.platform(),
        "environment_versions": environment_versions(),
        "input_rows": len(data),
        "targets": targets,
        "splits": split_names,
        "robustness_filters": robustness_names,
        "resolved_models": [
            {
                "name": model.name,
                "aliases": list(model.aliases),
                "predictors": list(model.predictors),
            }
            for model in models
        ],
        "skipped_models": skipped_models,
        "skipped_runs": skipped_runs,
        "ridge_alphas": alphas,
        "nested_alpha_tuning": tune_alpha,
        "inner_folds": inner_folds,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "oof_files": oof_manifest,
        "metric_definition": {
            "r2": "pooled out-of-fold R2 on a common target-valid sample",
            "delta_r2": "model pooled out-of-fold R2 minus lexical pooled out-of-fold R2",
            "bootstrap": "paired cluster bootstrap over the held-out grouping unit using saved OOF predictions",
        },
    }
    write_json(manifest, output_dir / "run_manifest.json")
    print(f"Analysis outputs written to {output_dir}")


if __name__ == "__main__":
    main()
