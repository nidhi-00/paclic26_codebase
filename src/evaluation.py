from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ResolvedModel:
    name: str
    predictors: tuple[str, ...]
    aliases: tuple[str, ...]


def build_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def resolve_models(
    columns: Iterable[str],
    baseline: list[str],
    feature_map: dict[str, str],
    model_map: dict[str, list[str]],
    core_models: Iterable[str] = (),
    strict_core: bool = True,
) -> tuple[list[ResolvedModel], list[dict[str, Any]]]:
    available = set(columns)
    missing_baseline = [column for column in baseline if column not in available]
    if missing_baseline:
        raise KeyError(f"Missing required lexical baseline columns: {missing_baseline}")

    core = set(core_models)
    resolved: list[ResolvedModel] = []
    skipped: list[dict[str, Any]] = []
    for name, aliases in model_map.items():
        feature_columns: list[str] = []
        missing_aliases: list[str] = []
        for alias in aliases:
            column = feature_map.get(alias)
            if not column or column not in available:
                missing_aliases.append(alias)
            else:
                feature_columns.append(column)
        if missing_aliases:
            record = {
                "model": name,
                "missing_aliases": missing_aliases,
                "reason": "feature columns not present",
            }
            if strict_core and name in core:
                raise KeyError(f"Core model {name!r} cannot be resolved: {record}")
            skipped.append(record)
            continue
        predictors = tuple(dict.fromkeys([*baseline, *feature_columns]))
        resolved.append(ResolvedModel(name=name, predictors=predictors, aliases=tuple(aliases)))
    if not any(model.name == "lexical" for model in resolved):
        raise ValueError("The analysis configuration must define a resolvable 'lexical' model.")
    return resolved, skipped


def make_outer_splits(
    groups: np.ndarray,
    method: str,
    folds: int | None,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two groups are required for held-out evaluation.")
    dummy = np.zeros((len(groups), 1), dtype=float)
    y = np.zeros(len(groups), dtype=float)
    splits: list[tuple[np.ndarray, np.ndarray, str]] = []
    if method == "group_kfold":
        n_splits = min(int(folds or 5), len(unique_groups))
        if n_splits < 2:
            raise ValueError("GroupKFold requires at least two folds.")
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(dummy, y, groups)
        for index, (train, test) in enumerate(iterator):
            splits.append((train, test, f"fold_{index + 1}"))
    elif method == "leave_one_group_out":
        splitter = LeaveOneGroupOut()
        iterator = splitter.split(dummy, y, groups)
        for train, test in iterator:
            held_out = str(groups[test][0])
            splits.append((train, test, f"heldout_{held_out}"))
    else:
        raise ValueError(f"Unsupported split method: {method}")
    return splits


def tune_alpha(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    inner_folds: int,
) -> tuple[float, dict[float, float]]:
    unique_groups = pd.unique(groups)
    if len(alphas) == 1 or len(unique_groups) < 2:
        return float(alphas[0]), {float(alphas[0]): np.nan}
    n_splits = min(int(inner_folds), len(unique_groups))
    if n_splits < 2:
        return float(alphas[0]), {float(alphas[0]): np.nan}

    splitter = GroupKFold(n_splits=n_splits)
    losses: dict[float, float] = {}
    for alpha in alphas:
        squared_error = 0.0
        observations = 0
        for train_index, test_index in splitter.split(x, y, groups):
            pipeline = build_pipeline(alpha)
            pipeline.fit(x.iloc[train_index], y[train_index])
            predictions = pipeline.predict(x.iloc[test_index])
            residuals = y[test_index] - predictions
            squared_error += float(np.dot(residuals, residuals))
            observations += len(test_index)
        losses[float(alpha)] = squared_error / max(observations, 1)
    best = min(losses, key=lambda value: (losses[value], value))
    return float(best), losses


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_models(
    sample: pd.DataFrame,
    target: str,
    group_column: str,
    split_name: str,
    split_method: str,
    folds: int | None,
    models: list[ResolvedModel],
    alphas: list[float],
    tune: bool,
    inner_folds: int,
    robustness_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {target, group_column, "participant_id", "sentence_id", "word_id"}
    missing = sorted(required - set(sample.columns))
    if missing:
        raise KeyError(f"Evaluation sample is missing columns: {missing}")

    work = sample.loc[sample[target].notna() & sample[group_column].notna()].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.loc[np.isfinite(work[target])].copy()
    work["_source_row_index"] = work.index.astype(str)
    work = work.reset_index(drop=True)
    if len(work) < 100:
        raise ValueError(
            f"Only {len(work)} rows remain for {target}/{split_name}/{robustness_name}."
        )

    y = work[target].to_numpy(dtype=float)
    groups = work[group_column].astype(str).to_numpy()
    outer_splits = make_outer_splits(groups, split_method, folds)

    identifiers = [
        "_source_row_index",
        "participant_id",
        "sentence_id",
        "word_id",
        target,
        group_column,
    ]
    identifiers = list(dict.fromkeys(identifiers))
    oof = work[identifiers].copy().rename(columns={target: "y_true"})
    oof["target"] = target
    oof["split"] = split_name
    oof["robustness"] = robustness_name
    oof["outer_fold"] = ""

    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []

    for model_spec in models:
        x = work[list(model_spec.predictors)].apply(pd.to_numeric, errors="coerce")
        predictions = np.full(len(work), np.nan, dtype=float)
        fold_assignment = np.full(len(work), "", dtype=object)

        for train_index, test_index, fold_label in outer_splits:
            if tune:
                selected_alpha, inner_losses = tune_alpha(
                    x=x.iloc[train_index].reset_index(drop=True),
                    y=y[train_index],
                    groups=groups[train_index],
                    alphas=alphas,
                    inner_folds=inner_folds,
                )
            else:
                selected_alpha = float(alphas[0])
                inner_losses = {selected_alpha: np.nan}

            pipeline = build_pipeline(selected_alpha)
            pipeline.fit(x.iloc[train_index], y[train_index])
            fold_predictions = pipeline.predict(x.iloc[test_index])
            predictions[test_index] = fold_predictions
            fold_assignment[test_index] = fold_label

            fold_metric = regression_metrics(y[test_index], fold_predictions)
            fold_rows.append(
                {
                    "target": target,
                    "split": split_name,
                    "robustness": robustness_name,
                    "model": model_spec.name,
                    "outer_fold": fold_label,
                    "held_out_groups": int(pd.unique(groups[test_index]).size),
                    "train_rows": int(len(train_index)),
                    "test_rows": int(len(test_index)),
                    "alpha": selected_alpha,
                    **fold_metric,
                }
            )
            alpha_rows.append(
                {
                    "target": target,
                    "split": split_name,
                    "robustness": robustness_name,
                    "model": model_spec.name,
                    "outer_fold": fold_label,
                    "selected_alpha": selected_alpha,
                    "inner_mse_by_alpha": ";".join(
                        f"{alpha:g}:{loss:.12g}" for alpha, loss in sorted(inner_losses.items())
                    ),
                }
            )

        if np.isnan(predictions).any() or (fold_assignment == "").any():
            raise AssertionError(f"Model {model_spec.name} did not produce exactly one OOF prediction per row.")
        oof[f"pred__{model_spec.name}"] = predictions
        if model_spec.name == "lexical":
            oof["outer_fold"] = fold_assignment
        metric_rows.append(
            {
                "target": target,
                "split": split_name,
                "robustness": robustness_name,
                "model": model_spec.name,
                "rows": int(len(work)),
                "groups": int(pd.unique(groups).size),
                "predictor_count": len(model_spec.predictors),
                "predictors": ",".join(model_spec.predictors),
                **regression_metrics(y, predictions),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    lexical_r2 = float(metrics.loc[metrics["model"].eq("lexical"), "r2"].iloc[0])
    lexical_rmse = float(metrics.loc[metrics["model"].eq("lexical"), "rmse"].iloc[0])
    lexical_mae = float(metrics.loc[metrics["model"].eq("lexical"), "mae"].iloc[0])
    metrics["delta_r2_vs_lexical"] = metrics["r2"] - lexical_r2
    metrics["delta_rmse_vs_lexical"] = metrics["rmse"] - lexical_rmse
    metrics["delta_mae_vs_lexical"] = metrics["mae"] - lexical_mae
    return metrics, pd.DataFrame(fold_rows), oof, pd.DataFrame(alpha_rows)


def _group_sufficient_statistics(
    oof: pd.DataFrame,
    group_column: str,
    model_names: list[str],
) -> tuple[list[str], dict[str, np.ndarray]]:
    frame = oof.copy()
    frame[group_column] = frame[group_column].astype(str)
    groups = sorted(frame[group_column].unique().tolist())
    grouped = frame.groupby(group_column, sort=False)
    base = grouped["y_true"].agg(["size", "sum"])
    base["sum_y2"] = grouped["y_true"].apply(lambda values: float(np.dot(values, values)))
    base = base.reindex(groups)
    stats: dict[str, np.ndarray] = {
        "n": base["size"].to_numpy(dtype=float),
        "sum_y": base["sum"].to_numpy(dtype=float),
        "sum_y2": base["sum_y2"].to_numpy(dtype=float),
    }
    for model in model_names:
        pred_col = f"pred__{model}"
        residual = frame["y_true"].to_numpy(dtype=float) - frame[pred_col].to_numpy(dtype=float)
        frame[f"_sse_{model}"] = residual**2
        frame[f"_sae_{model}"] = np.abs(residual)
        aggregate = frame.groupby(group_column, sort=False)[[f"_sse_{model}", f"_sae_{model}"]].sum().reindex(groups)
        stats[f"sse_{model}"] = aggregate[f"_sse_{model}"].to_numpy(dtype=float)
        stats[f"sae_{model}"] = aggregate[f"_sae_{model}"].to_numpy(dtype=float)
    return groups, stats


def bootstrap_oof_differences(
    oof: pd.DataFrame,
    group_column: str,
    model_names: list[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    groups, stats = _group_sufficient_statistics(oof, group_column, model_names)
    group_count = len(groups)
    if group_count < 2:
        raise ValueError("Cluster bootstrap requires at least two groups.")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(group_count, np.repeat(1 / group_count, group_count), size=iterations)
    n = counts @ stats["n"]
    sum_y = counts @ stats["sum_y"]
    sum_y2 = counts @ stats["sum_y2"]
    denominator = sum_y2 - (sum_y**2 / n)
    denominator = np.where(denominator > 0, denominator, np.nan)

    r2_values: dict[str, np.ndarray] = {}
    rmse_values: dict[str, np.ndarray] = {}
    mae_values: dict[str, np.ndarray] = {}
    for model in model_names:
        sse = counts @ stats[f"sse_{model}"]
        sae = counts @ stats[f"sae_{model}"]
        r2_values[model] = 1 - sse / denominator
        rmse_values[model] = np.sqrt(sse / n)
        mae_values[model] = sae / n

    lexical_r2 = r2_values["lexical"]
    lexical_rmse = rmse_values["lexical"]
    lexical_mae = mae_values["lexical"]
    rows: list[dict[str, Any]] = []
    for model in model_names:
        delta_r2 = r2_values[model] - lexical_r2
        delta_rmse = rmse_values[model] - lexical_rmse
        delta_mae = mae_values[model] - lexical_mae
        bootstrap_sign_p = np.nan
        cluster_signflip_p = np.nan
        if model != "lexical":
            left = (np.sum(delta_r2 <= 0) + 1) / (iterations + 1)
            right = (np.sum(delta_r2 >= 0) + 1) / (iterations + 1)
            bootstrap_sign_p = min(1.0, 2 * min(left, right))
            improvement_by_group = stats["sse_lexical"] - stats[f"sse_{model}"]
            observed = abs(float(improvement_by_group.sum()))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, group_count))
            permuted = np.abs(signs @ improvement_by_group)
            cluster_signflip_p = float((np.sum(permuted >= observed) + 1) / (iterations + 1))
        rows.append(
            {
                "model": model,
                "bootstrap_groups": group_count,
                "bootstrap_iterations": iterations,
                "r2_mean": float(np.nanmean(r2_values[model])),
                "r2_ci_low": float(np.nanquantile(r2_values[model], 0.025)),
                "r2_ci_high": float(np.nanquantile(r2_values[model], 0.975)),
                "delta_r2_mean": float(np.nanmean(delta_r2)),
                "delta_r2_ci_low": float(np.nanquantile(delta_r2, 0.025)),
                "delta_r2_ci_high": float(np.nanquantile(delta_r2, 0.975)),
                "bootstrap_sign_p_two_sided": bootstrap_sign_p,
                "cluster_signflip_p_two_sided": cluster_signflip_p,
                "delta_rmse_mean": float(np.nanmean(delta_rmse)),
                "delta_rmse_ci_low": float(np.nanquantile(delta_rmse, 0.025)),
                "delta_rmse_ci_high": float(np.nanquantile(delta_rmse, 0.975)),
                "delta_mae_mean": float(np.nanmean(delta_mae)),
                "delta_mae_ci_low": float(np.nanquantile(delta_mae, 0.025)),
                "delta_mae_ci_high": float(np.nanquantile(delta_mae, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def surprisal_correlations(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    present = [column for column in columns if column in data.columns]
    if len(present) < 2:
        return pd.DataFrame()
    item = data.drop_duplicates(["sentence_id", "word_id"])
    rows: list[dict[str, Any]] = []
    for index, left in enumerate(present):
        for right in present[index + 1 :]:
            pair = item[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3:
                continue
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "items": len(pair),
                    "pearson_r": float(pair[left].corr(pair[right], method="pearson")),
                    "spearman_rho": float(pair[left].corr(pair[right], method="spearman")),
                }
            )
    return pd.DataFrame(rows)
