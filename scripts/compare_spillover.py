from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_LABELS = {
    "log_first_fixation_duration": "First fixation duration",
    "log_gaze_duration": "Gaze duration",
    "log_go_past_time": "Go-past time",
    "log_total_reading_time": "Total reading time",
}

COMPARISONS = {
    "all_lm_spillover": {
        "label": "Full spillover vs. full current",
        "baseline": "pred__full_current",
        "augmented": "pred__full_spillover",
    },
    "gpt2_spillover": {
        "label": "GPT-2 spillover vs. current GPT-2",
        "baseline": "pred__lexical_gpt2",
        "augmented": "pred__lexical_gpt2_spillover",
    },
}


def metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    residual = y - prediction
    sse = float(np.dot(residual, residual))
    centred = y - y.mean()
    sst = float(np.dot(centred, centred))

    r2 = 1.0 - sse / sst
    rmse = math.sqrt(sse / len(y))
    mae = float(np.abs(residual).mean())

    return r2, rmse, mae


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()

    if valid.empty:
        return output

    ordered = valid.sort_values()
    count = len(ordered)
    ranks = np.arange(1, count + 1, dtype=float)

    adjusted = ordered.to_numpy(dtype=float) * count / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    output.loc[ordered.index] = adjusted
    return output


def aggregate_clusters(
    frame: pd.DataFrame,
    group_column: str,
    baseline_column: str,
    augmented_column: str,
) -> pd.DataFrame:
    work = frame[
        [
            group_column,
            "y_true",
            baseline_column,
            augmented_column,
        ]
    ].copy()

    work["y_sq"] = work["y_true"] ** 2
    work["baseline_sq_error"] = (
        work["y_true"] - work[baseline_column]
    ) ** 2
    work["augmented_sq_error"] = (
        work["y_true"] - work[augmented_column]
    ) ** 2
    work["baseline_abs_error"] = (
        work["y_true"] - work[baseline_column]
    ).abs()
    work["augmented_abs_error"] = (
        work["y_true"] - work[augmented_column]
    ).abs()

    return (
        work.groupby(group_column, sort=False)
        .agg(
            rows=("y_true", "size"),
            y_sum=("y_true", "sum"),
            y_sq_sum=("y_sq", "sum"),
            baseline_sse=("baseline_sq_error", "sum"),
            augmented_sse=("augmented_sq_error", "sum"),
            baseline_sae=("baseline_abs_error", "sum"),
            augmented_sae=("augmented_abs_error", "sum"),
        )
        .reset_index()
    )


def bootstrap_differences(
    clusters: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    rows = clusters["rows"].to_numpy(dtype=float)
    y_sum = clusters["y_sum"].to_numpy(dtype=float)
    y_sq_sum = clusters["y_sq_sum"].to_numpy(dtype=float)
    baseline_sse = clusters["baseline_sse"].to_numpy(dtype=float)
    augmented_sse = clusters["augmented_sse"].to_numpy(dtype=float)
    baseline_sae = clusters["baseline_sae"].to_numpy(dtype=float)
    augmented_sae = clusters["augmented_sae"].to_numpy(dtype=float)

    group_count = len(clusters)

    delta_r2 = np.empty(iterations, dtype=float)
    delta_rmse = np.empty(iterations, dtype=float)
    delta_mae = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        sampled = rng.integers(
            0,
            group_count,
            size=group_count,
        )

        sample_rows = rows[sampled].sum()
        sample_y_sum = y_sum[sampled].sum()
        sample_y_sq_sum = y_sq_sum[sampled].sum()

        sample_sst = (
            sample_y_sq_sum
            - sample_y_sum**2 / sample_rows
        )

        baseline_sample_sse = baseline_sse[sampled].sum()
        augmented_sample_sse = augmented_sse[sampled].sum()

        baseline_r2 = 1.0 - baseline_sample_sse / sample_sst
        augmented_r2 = 1.0 - augmented_sample_sse / sample_sst

        baseline_rmse = math.sqrt(
            baseline_sample_sse / sample_rows
        )
        augmented_rmse = math.sqrt(
            augmented_sample_sse / sample_rows
        )

        baseline_mae = (
            baseline_sae[sampled].sum() / sample_rows
        )
        augmented_mae = (
            augmented_sae[sampled].sum() / sample_rows
        )

        delta_r2[iteration] = augmented_r2 - baseline_r2
        delta_rmse[iteration] = augmented_rmse - baseline_rmse
        delta_mae[iteration] = augmented_mae - baseline_mae

    left = float(np.mean(delta_r2 <= 0))
    right = float(np.mean(delta_r2 >= 0))
    bootstrap_p = min(1.0, 2.0 * min(left, right))

    return {
        "bootstrap_delta_r2_mean": float(delta_r2.mean()),
        "delta_r2_ci_low": float(np.quantile(delta_r2, 0.025)),
        "delta_r2_ci_high": float(np.quantile(delta_r2, 0.975)),
        "bootstrap_delta_rmse_mean": float(delta_rmse.mean()),
        "delta_rmse_ci_low": float(np.quantile(delta_rmse, 0.025)),
        "delta_rmse_ci_high": float(np.quantile(delta_rmse, 0.975)),
        "bootstrap_delta_mae_mean": float(delta_mae.mean()),
        "delta_mae_ci_low": float(np.quantile(delta_mae, 0.025)),
        "delta_mae_ci_high": float(np.quantile(delta_mae, 0.975)),
        "bootstrap_p_two_sided": bootstrap_p,
    }


def signflip_p_value(
    clusters: pd.DataFrame,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, str]:
    improvement = (
        clusters["baseline_sse"].to_numpy(dtype=float)
        - clusters["augmented_sse"].to_numpy(dtype=float)
    ) / clusters["rows"].to_numpy(dtype=float)

    observed = float(improvement.mean())
    group_count = len(improvement)

    if group_count <= 20:
        statistics = []

        for signs in itertools.product(
            (-1.0, 1.0),
            repeat=group_count,
        ):
            sign_array = np.asarray(signs, dtype=float)
            statistics.append(
                float(np.mean(improvement * sign_array))
            )

        statistics_array = np.asarray(statistics)
        p_value = float(
            np.mean(
                np.abs(statistics_array)
                >= abs(observed) - 1e-15
            )
        )
        method = f"exact_{2 ** group_count}_sign_patterns"
    else:
        exceedances = 0

        for _ in range(iterations):
            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=group_count,
            )
            statistic = float(np.mean(improvement * signs))

            if abs(statistic) >= abs(observed):
                exceedances += 1

        p_value = float(
            (exceedances + 1) / (iterations + 1)
        )
        method = f"monte_carlo_{iterations}"

    return p_value, method


def comparison_result(
    frame: pd.DataFrame,
    target: str,
    split: str,
    comparison_name: str,
    bootstrap_iterations: int,
    signflip_iterations: int,
    seed: int,
) -> dict[str, object]:
    comparison = COMPARISONS[comparison_name]
    baseline_column = comparison["baseline"]
    augmented_column = comparison["augmented"]

    required = {
        "y_true",
        baseline_column,
        augmented_column,
        "sentence_id",
        "participant_id",
    }

    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    group_column = (
        "sentence_id"
        if split == "sentence"
        else "participant_id"
    )

    clean = frame[
        [
            "y_true",
            baseline_column,
            augmented_column,
            group_column,
        ]
    ].dropna()

    y = clean["y_true"].to_numpy(dtype=float)
    baseline_prediction = clean[
        baseline_column
    ].to_numpy(dtype=float)
    augmented_prediction = clean[
        augmented_column
    ].to_numpy(dtype=float)

    baseline_r2, baseline_rmse, baseline_mae = metrics(
        y,
        baseline_prediction,
    )
    augmented_r2, augmented_rmse, augmented_mae = metrics(
        y,
        augmented_prediction,
    )

    clusters = aggregate_clusters(
        clean,
        group_column,
        baseline_column,
        augmented_column,
    )

    rng = np.random.default_rng(seed)

    bootstrap = bootstrap_differences(
        clusters,
        bootstrap_iterations,
        rng,
    )

    signflip_p, signflip_method = signflip_p_value(
        clusters,
        rng,
        signflip_iterations,
    )

    return {
        "target": target,
        "target_label": TARGET_LABELS[target],
        "split": split,
        "comparison": comparison_name,
        "comparison_label": comparison["label"],
        "baseline_model": baseline_column.removeprefix("pred__"),
        "augmented_model": augmented_column.removeprefix("pred__"),
        "rows": len(clean),
        "clusters": len(clusters),
        "bootstrap_iterations": bootstrap_iterations,
        "signflip_method": signflip_method,
        "baseline_r2": baseline_r2,
        "augmented_r2": augmented_r2,
        "delta_r2": augmented_r2 - baseline_r2,
        "baseline_rmse": baseline_rmse,
        "augmented_rmse": augmented_rmse,
        "delta_rmse": augmented_rmse - baseline_rmse,
        "baseline_mae": baseline_mae,
        "augmented_mae": augmented_mae,
        "delta_mae": augmented_mae - baseline_mae,
        "cluster_signflip_p_two_sided": signflip_p,
        **bootstrap,
    }


def latex_number(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def write_latex_table(results: pd.DataFrame, path: Path) -> None:
    full = results[
        results["comparison"].eq("all_lm_spillover")
    ].copy()

    full["split_order"] = full["split"].map(
        {"sentence": 0, "participant": 1}
    )
    full["target_order"] = full["target"].map(
        {
            "log_first_fixation_duration": 0,
            "log_gaze_duration": 1,
            "log_go_past_time": 2,
            "log_total_reading_time": 3,
        }
    )

    full = full.sort_values(
        ["split_order", "target_order"]
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Direct paired comparison of the full spillover "
            r"model against the otherwise identical full current-word "
            r"model. Positive $\Delta R^2$ favors spillover. Confidence "
            r"intervals use cluster bootstrap resampling over the held-out "
            r"unit. $q$ is the global Benjamini--Hochberg-adjusted "
            r"cluster sign-flip value.}"
        ),
        r"\label{tab:direct-spillover}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"Split & Measure & Current $R^2$ & Spillover $R^2$ "
            r"& Direct $\Delta R^2$ [95\% CI] & $q$ \\"
        ),
        r"\midrule",
    ]

    for _, row in full.iterrows():
        split_label = (
            "Sentence"
            if row["split"] == "sentence"
            else "Participant"
        )

        interval = (
            f"{latex_number(row['delta_r2'])} "
            f"[{latex_number(row['delta_r2_ci_low'])}, "
            f"{latex_number(row['delta_r2_ci_high'])}]"
        )

        lines.append(
            f"{split_label} & {row['target_label']} & "
            f"{latex_number(row['baseline_r2'])} & "
            f"{latex_number(row['augmented_r2'])} & "
            f"{interval} & "
            f"{row['cluster_signflip_q_bh_global']:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-dir",
        default="results/main/oof_predictions",
    )
    parser.add_argument(
        "--output-dir",
        default="results/direct_spillover",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--signflip-iterations",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260801,
    )
    args = parser.parse_args()

    oof_dir = Path(args.oof_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(oof_dir.glob("*__full.parquet"))

    if len(paths) != 8:
        raise RuntimeError(
            f"Expected 8 OOF files, found {len(paths)}."
        )

    rows: list[dict[str, object]] = []

    for file_index, path in enumerate(paths):
        frame = pd.read_parquet(path)

        target = str(frame["target"].iloc[0])
        split = str(frame["split"].iloc[0])

        print(
            f"Processing target={target}, split={split}, "
            f"rows={len(frame):,}"
        )

        for comparison_index, comparison_name in enumerate(
            COMPARISONS
        ):
            result = comparison_result(
                frame=frame,
                target=target,
                split=split,
                comparison_name=comparison_name,
                bootstrap_iterations=args.bootstrap_iterations,
                signflip_iterations=args.signflip_iterations,
                seed=(
                    args.seed
                    + file_index * 100
                    + comparison_index
                ),
            )
            rows.append(result)

    results = pd.DataFrame(rows)

    results["cluster_signflip_q_bh_global"] = (
        benjamini_hochberg(
            results["cluster_signflip_p_two_sided"]
        )
    )

    results["cluster_signflip_q_bh_within_comparison"] = (
        results.groupby(
            "comparison",
            group_keys=False,
        )["cluster_signflip_p_two_sided"]
        .apply(benjamini_hochberg)
    )

    results["bootstrap_q_bh_global"] = (
        benjamini_hochberg(
            results["bootstrap_p_two_sided"]
        )
    )

    results = results.sort_values(
        ["comparison", "split", "target"]
    ).reset_index(drop=True)

    csv_path = output_dir / "direct_spillover_comparisons.csv"
    results.to_csv(csv_path, index=False)

    table_columns = [
        "target",
        "target_label",
        "split",
        "comparison",
        "baseline_r2",
        "augmented_r2",
        "delta_r2",
        "delta_r2_ci_low",
        "delta_r2_ci_high",
        "delta_rmse",
        "delta_rmse_ci_low",
        "delta_rmse_ci_high",
        "delta_mae",
        "delta_mae_ci_low",
        "delta_mae_ci_high",
        "cluster_signflip_p_two_sided",
        "cluster_signflip_q_bh_global",
        "cluster_signflip_q_bh_within_comparison",
    ]

    summary_path = output_dir / "direct_spillover_summary.csv"
    results[table_columns].to_csv(
        summary_path,
        index=False,
    )

    latex_path = output_dir / "direct_spillover_table.tex"
    write_latex_table(results, latex_path)

    print("\nDirect comparison results:")
    print(
        results[
            [
                "target_label",
                "split",
                "comparison",
                "delta_r2",
                "delta_r2_ci_low",
                "delta_r2_ci_high",
                "cluster_signflip_p_two_sided",
                "cluster_signflip_q_bh_global",
            ]
        ].to_string(index=False)
    )

    print("\nWrote:")
    print(csv_path)
    print(summary_path)
    print(latex_path)


if __name__ == "__main__":
    main()
