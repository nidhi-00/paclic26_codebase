from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from compare_spillover import (
    COMPARISONS,
    benjamini_hochberg,
    comparison_result,
)


def apply_bh_by_group(
    frame: pd.DataFrame,
    group_column: str,
    p_column: str,
) -> pd.Series:
    output = pd.Series(
        float("nan"),
        index=frame.index,
        dtype=float,
    )

    for _, indices in frame.groupby(
        group_column
    ).groups.items():
        output.loc[indices] = benjamini_hochberg(
            frame.loc[indices, p_column]
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--oof-dir",
        default=(
            "results/robustness_spillover/"
            "oof_predictions"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="results/direct_spillover_robustness",
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=3000,
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

    paths = sorted(oof_dir.glob("*.parquet"))

    if len(paths) != 32:
        raise RuntimeError(
            f"Expected 32 OOF files, found {len(paths)}."
        )

    rows: list[dict[str, object]] = []

    for file_index, path in enumerate(paths):
        frame = pd.read_parquet(path)

        target = str(frame["target"].iloc[0])
        split = str(frame["split"].iloc[0])
        robustness = str(frame["robustness"].iloc[0])

        print(
            f"Processing target={target}, "
            f"split={split}, "
            f"robustness={robustness}, "
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
                bootstrap_iterations=(
                    args.bootstrap_iterations
                ),
                signflip_iterations=(
                    args.signflip_iterations
                ),
                seed=(
                    args.seed
                    + file_index * 100
                    + comparison_index
                ),
            )

            result["robustness"] = robustness
            rows.append(result)

    results = pd.DataFrame(rows)

    p_column = "cluster_signflip_p_two_sided"

    results["cluster_signflip_q_bh_global"] = (
        benjamini_hochberg(results[p_column])
    )

    results[
        "cluster_signflip_q_bh_within_robustness"
    ] = apply_bh_by_group(
        results,
        "robustness",
        p_column,
    )

    results[
        "cluster_signflip_q_bh_within_comparison"
    ] = apply_bh_by_group(
        results,
        "comparison",
        p_column,
    )

    results = results.sort_values(
        [
            "comparison",
            "robustness",
            "split",
            "target",
        ]
    ).reset_index(drop=True)

    all_path = (
        output_dir
        / "direct_spillover_robustness.csv"
    )

    results.to_csv(all_path, index=False)

    headline = results[
        results["comparison"].eq(
            "all_lm_spillover"
        )
    ][
        [
            "target_label",
            "split",
            "robustness",
            "rows",
            "clusters",
            "baseline_r2",
            "augmented_r2",
            "delta_r2",
            "delta_r2_ci_low",
            "delta_r2_ci_high",
            "delta_rmse",
            "delta_mae",
            "cluster_signflip_p_two_sided",
            "cluster_signflip_q_bh_global",
            "cluster_signflip_q_bh_within_robustness",
        ]
    ]

    headline_path = (
        output_dir
        / "full_model_robustness_summary.csv"
    )

    headline.to_csv(headline_path, index=False)

    print("\nFull-model robustness results:")
    print(headline.to_string(index=False))

    print("\nWrote:")
    print(all_path)
    print(headline_path)


if __name__ == "__main__":
    main()
