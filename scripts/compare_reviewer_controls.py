from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


RUNS = {
    "skip": Path("results/reviewer_skip_full/oof_predictions"),
    "punct": Path("results/reviewer_punct_full/oof_predictions"),
    "both": Path("results/reviewer_both_full/oof_predictions"),
}

TARGETS = [
    "log_first_fixation_duration",
    "log_gaze_duration",
    "log_go_past_time",
    "log_total_reading_time",
]

TARGET_LABELS = {
    "log_first_fixation_duration": "First fixation duration",
    "log_gaze_duration": "Gaze duration",
    "log_go_past_time": "Go-past time",
    "log_total_reading_time": "Total reading time",
}

SPLITS = [
    "sentence",
    "participant",
]

ROBUSTNESS = [
    "full",
    "strict_adjacent",
]


def metrics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:

    residual = y - prediction

    sse = float(
        np.dot(residual, residual)
    )

    centred = y - y.mean()

    sst = float(
        np.dot(centred, centred)
    )

    r2 = 1.0 - sse / sst

    rmse = math.sqrt(
        sse / len(y)
    )

    mae = float(
        np.abs(residual).mean()
    )

    return r2, rmse, mae


def benjamini_hochberg(
    values: pd.Series,
) -> pd.Series:

    out = pd.Series(
        np.nan,
        index=values.index,
        dtype=float,
    )

    valid = values.dropna()

    if valid.empty:
        return out

    ordered = valid.sort_values()

    n = len(ordered)

    ranks = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    adjusted = (
        ordered.to_numpy(dtype=float)
        * n
        / ranks
    )

    adjusted = (
        np.minimum.accumulate(
            adjusted[::-1]
        )[::-1]
    )

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    out.loc[
        ordered.index
    ] = adjusted

    return out


def aggregate_clusters(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    work = frame[
        [
            group_column,
            "y_true",
            "pred__full_current",
            "pred__full_spillover",
        ]
    ].copy()

    work["y_sq"] = (
        work["y_true"] ** 2
    )

    work["current_sq_error"] = (
        work["y_true"]
        - work["pred__full_current"]
    ) ** 2

    work["spillover_sq_error"] = (
        work["y_true"]
        - work["pred__full_spillover"]
    ) ** 2

    work["current_abs_error"] = (
        work["y_true"]
        - work["pred__full_current"]
    ).abs()

    work["spillover_abs_error"] = (
        work["y_true"]
        - work["pred__full_spillover"]
    ).abs()

    return (
        work.groupby(
            group_column,
            sort=False,
        )
        .agg(
            rows=("y_true", "size"),
            y_sum=("y_true", "sum"),
            y_sq_sum=("y_sq", "sum"),
            current_sse=(
                "current_sq_error",
                "sum",
            ),
            spillover_sse=(
                "spillover_sq_error",
                "sum",
            ),
            current_sae=(
                "current_abs_error",
                "sum",
            ),
            spillover_sae=(
                "spillover_abs_error",
                "sum",
            ),
        )
        .reset_index()
    )


def bootstrap_differences(
    clusters: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float]:

    rows = clusters[
        "rows"
    ].to_numpy(dtype=float)

    y_sum = clusters[
        "y_sum"
    ].to_numpy(dtype=float)

    y_sq_sum = clusters[
        "y_sq_sum"
    ].to_numpy(dtype=float)

    current_sse = clusters[
        "current_sse"
    ].to_numpy(dtype=float)

    spillover_sse = clusters[
        "spillover_sse"
    ].to_numpy(dtype=float)

    current_sae = clusters[
        "current_sae"
    ].to_numpy(dtype=float)

    spillover_sae = clusters[
        "spillover_sae"
    ].to_numpy(dtype=float)

    n_clusters = len(clusters)

    delta_r2 = np.empty(
        iterations,
        dtype=float,
    )

    delta_rmse = np.empty(
        iterations,
        dtype=float,
    )

    delta_mae = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(iterations):

        sampled = rng.integers(
            0,
            n_clusters,
            size=n_clusters,
        )

        sample_rows = (
            rows[sampled].sum()
        )

        sample_y_sum = (
            y_sum[sampled].sum()
        )

        sample_y_sq_sum = (
            y_sq_sum[sampled].sum()
        )

        sample_sst = (
            sample_y_sq_sum
            - sample_y_sum**2
            / sample_rows
        )

        current_sample_sse = (
            current_sse[sampled].sum()
        )

        spillover_sample_sse = (
            spillover_sse[sampled].sum()
        )

        current_r2 = (
            1.0
            - current_sample_sse
            / sample_sst
        )

        spillover_r2 = (
            1.0
            - spillover_sample_sse
            / sample_sst
        )

        current_rmse = math.sqrt(
            current_sample_sse
            / sample_rows
        )

        spillover_rmse = math.sqrt(
            spillover_sample_sse
            / sample_rows
        )

        current_mae = (
            current_sae[sampled].sum()
            / sample_rows
        )

        spillover_mae = (
            spillover_sae[sampled].sum()
            / sample_rows
        )

        delta_r2[i] = (
            spillover_r2
            - current_r2
        )

        delta_rmse[i] = (
            spillover_rmse
            - current_rmse
        )

        delta_mae[i] = (
            spillover_mae
            - current_mae
        )

    left = float(
        np.mean(delta_r2 <= 0)
    )

    right = float(
        np.mean(delta_r2 >= 0)
    )

    bootstrap_p = min(
        1.0,
        2.0 * min(left, right),
    )

    return {
        "bootstrap_delta_r2_mean":
            float(delta_r2.mean()),

        "delta_r2_ci_low":
            float(
                np.quantile(
                    delta_r2,
                    0.025,
                )
            ),

        "delta_r2_ci_high":
            float(
                np.quantile(
                    delta_r2,
                    0.975,
                )
            ),

        "bootstrap_delta_rmse_mean":
            float(delta_rmse.mean()),

        "bootstrap_delta_mae_mean":
            float(delta_mae.mean()),

        "bootstrap_p_two_sided":
            bootstrap_p,
    }


def signflip_p_value(
    clusters: pd.DataFrame,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, str]:

    improvement = (
        clusters[
            "current_sse"
        ].to_numpy(dtype=float)
        - clusters[
            "spillover_sse"
        ].to_numpy(dtype=float)
    ) / clusters[
        "rows"
    ].to_numpy(dtype=float)

    observed = float(
        improvement.mean()
    )

    n_clusters = len(improvement)

    if n_clusters <= 20:

        statistics = np.empty(
            2 ** n_clusters,
            dtype=float,
        )

        for i, signs in enumerate(
            itertools.product(
                (-1.0, 1.0),
                repeat=n_clusters,
            )
        ):

            sign_array = np.asarray(
                signs,
                dtype=float,
            )

            statistics[i] = float(
                np.mean(
                    improvement
                    * sign_array
                )
            )

        p = float(
            np.mean(
                np.abs(statistics)
                >= abs(observed) - 1e-15
            )
        )

        method = (
            f"exact_"
            f"{2 ** n_clusters}"
            f"_sign_patterns"
        )

    else:

        exceedances = 0

        for _ in range(iterations):

            signs = rng.choice(
                np.array(
                    [-1.0, 1.0]
                ),
                size=n_clusters,
            )

            statistic = float(
                np.mean(
                    improvement
                    * signs
                )
            )

            if (
                abs(statistic)
                >= abs(observed)
            ):
                exceedances += 1

        p = float(
            (exceedances + 1)
            / (iterations + 1)
        )

        method = (
            f"monte_carlo_"
            f"{iterations}"
        )

    return p, method


def evaluate_one(
    path: Path,
    condition: str,
    target: str,
    split: str,
    robustness: str,
    bootstrap_iterations: int,
    signflip_iterations: int,
    seed: int,
) -> dict[str, object]:

    frame = pd.read_parquet(
        path
    )

    required = {
        "y_true",
        "sentence_id",
        "participant_id",
        "pred__full_current",
        "pred__full_spillover",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{path}: missing "
            f"{sorted(missing)}"
        )

    group_column = (
        "sentence_id"
        if split == "sentence"
        else "participant_id"
    )

    clean = frame[
        [
            "y_true",
            "pred__full_current",
            "pred__full_spillover",
            group_column,
        ]
    ].dropna()

    y = clean[
        "y_true"
    ].to_numpy(dtype=float)

    current = clean[
        "pred__full_current"
    ].to_numpy(dtype=float)

    spillover = clean[
        "pred__full_spillover"
    ].to_numpy(dtype=float)

    current_r2, current_rmse, current_mae = (
        metrics(
            y,
            current,
        )
    )

    spill_r2, spill_rmse, spill_mae = (
        metrics(
            y,
            spillover,
        )
    )

    clusters = aggregate_clusters(
        clean,
        group_column,
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap = bootstrap_differences(
        clusters,
        bootstrap_iterations,
        rng,
    )

    signflip_p, signflip_method = (
        signflip_p_value(
            clusters,
            rng,
            signflip_iterations,
        )
    )

    return {
        "condition": condition,
        "target": target,
        "target_label":
            TARGET_LABELS[target],
        "split": split,
        "robustness": robustness,
        "rows": len(clean),
        "clusters": len(clusters),
        "current_r2": current_r2,
        "spillover_r2": spill_r2,
        "delta_r2":
            spill_r2 - current_r2,
        "current_rmse": current_rmse,
        "spillover_rmse": spill_rmse,
        "delta_rmse":
            spill_rmse - current_rmse,
        "current_mae": current_mae,
        "spillover_mae": spill_mae,
        "delta_mae":
            spill_mae - current_mae,
        "cluster_signflip_p_two_sided":
            signflip_p,
        "signflip_method":
            signflip_method,
        "bootstrap_iterations":
            bootstrap_iterations,
        **bootstrap,
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "reviewer_controls_inference"
        ),
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
        default=20260809,
    )

    args = parser.parse_args()

    rows = []

    test_index = 0

    for condition, root in RUNS.items():

        for robustness in ROBUSTNESS:

            for target in TARGETS:

                for split in SPLITS:

                    path = (
                        root
                        / (
                            f"{target}"
                            f"__{split}"
                            f"__{robustness}"
                            f".parquet"
                        )
                    )

                    if not path.exists():
                        raise FileNotFoundError(
                            path
                        )

                    print(
                        "Processing",
                        f"condition={condition}",
                        f"target={target}",
                        f"split={split}",
                        f"robustness={robustness}",
                    )

                    rows.append(
                        evaluate_one(
                            path=path,
                            condition=condition,
                            target=target,
                            split=split,
                            robustness=robustness,
                            bootstrap_iterations=(
                                args.bootstrap_iterations
                            ),
                            signflip_iterations=(
                                args.signflip_iterations
                            ),
                            seed=(
                                args.seed
                                + test_index * 1009
                            ),
                        )
                    )

                    test_index += 1

    results = pd.DataFrame(
        rows
    )

    # Conservative correction over every reviewer-control
    # test produced by this script: 3 conditions x
    # 2 samples x 4 outcomes x 2 split strategies = 48.
    results[
        "signflip_q_bh_global_48"
    ] = benjamini_hochberg(
        results[
            "cluster_signflip_p_two_sided"
        ]
    )

    # Also report correction within each logically separate
    # control/sample family (8 tests: 4 outcomes x 2 splits).
    results[
        "signflip_q_bh_within_family"
    ] = (
        results.groupby(
            [
                "condition",
                "robustness",
            ],
            group_keys=False,
        )[
            "cluster_signflip_p_two_sided"
        ]
        .apply(
            benjamini_hochberg
        )
    )

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out_dir
        / "reviewer_control_direct_tests.csv"
    )

    results.to_csv(
        path,
        index=False,
    )

    summary = results[
        [
            "condition",
            "target_label",
            "split",
            "robustness",
            "rows",
            "clusters",
            "current_r2",
            "spillover_r2",
            "delta_r2",
            "delta_r2_ci_low",
            "delta_r2_ci_high",
            "cluster_signflip_p_two_sided",
            "signflip_q_bh_within_family",
            "signflip_q_bh_global_48",
            "signflip_method",
        ]
    ].copy()

    summary_path = (
        out_dir
        / "reviewer_control_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("RESULTS")
    print("=" * 110)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print(
        "\nWROTE |",
        path,
    )

    print(
        "WROTE |",
        summary_path,
    )


if __name__ == "__main__":
    main()
