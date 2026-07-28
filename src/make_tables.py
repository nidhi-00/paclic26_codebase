from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_LABELS = {
    "log_first_fixation_duration": "First fixation duration",
    "log_gaze_duration": "Gaze duration",
    "log_go_past_time": "Go-past time",
    "log_total_reading_time": "Total reading time",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manuscript tables from model and bootstrap outputs.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(results_dir / "model_metrics.csv")
    intervals = pd.read_csv(results_dir / "bootstrap_intervals.csv")

    main_models = [
        "lexical",
        "lexical_gpt2",
        "lexical_bert",
        "lexical_roberta",
        "lexical_all_transformers",
    ]
    table = metrics[
        metrics["robustness"].eq("full") & metrics["model"].isin(main_models)
    ].merge(
        intervals[
            [
                "target",
                "split",
                "robustness",
                "model",
                "delta_r2_ci_low",
                "delta_r2_ci_high",
                "cluster_signflip_p_two_sided",
                "cluster_signflip_q_bh_within_run",
            ]
        ],
        on=["target", "split", "robustness", "model"],
        how="left",
        validate="one_to_one",
    )
    table["measure"] = table["target"].map(TARGET_LABELS).fillna(table["target"])
    table = table[
        [
            "split",
            "measure",
            "model",
            "rows",
            "r2",
            "rmse",
            "mae",
            "delta_r2_vs_lexical",
            "delta_r2_ci_low",
            "delta_r2_ci_high",
            "cluster_signflip_p_two_sided",
            "cluster_signflip_q_bh_within_run",
        ]
    ]
    table.to_csv(output_dir / "main_model_table.csv", index=False)
    (output_dir / "main_model_table.tex").write_text(
        table.to_latex(index=False, float_format=lambda value: f"{value:.5f}"),
        encoding="utf-8",
    )

    robustness = metrics[
        metrics["model"].eq("lexical_all_transformers")
    ][
        ["target", "split", "robustness", "rows", "r2", "delta_r2_vs_lexical"]
    ].copy()
    robustness["measure"] = robustness["target"].map(TARGET_LABELS).fillna(robustness["target"])
    robustness.to_csv(output_dir / "robustness_table.csv", index=False)

    print(f"Tables written to {output_dir}")


if __name__ == "__main__":
    main()
