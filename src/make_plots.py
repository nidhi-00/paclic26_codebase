from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_LABELS = {
    "log_first_fixation_duration": "First fixation",
    "log_gaze_duration": "Gaze duration",
    "log_go_past_time": "Go-past time",
    "log_total_reading_time": "Total reading time",
}

CORE_MODELS = [
    "lexical_gpt2",
    "lexical_bert",
    "lexical_roberta",
    "lexical_all_transformers",
]


def save_figure(figure: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_r2(metrics: pd.DataFrame, output_dir: Path) -> None:
    subset = metrics[
        metrics["robustness"].eq("full")
        & metrics["model"].isin(["lexical", *CORE_MODELS])
    ].copy()
    if subset.empty:
        return
    for split, frame in subset.groupby("split"):
        pivot = frame.pivot(index="target", columns="model", values="r2")
        order = [target for target in TARGET_LABELS if target in pivot.index]
        pivot = pivot.reindex(order)
        figure, axis = plt.subplots(figsize=(10.5, 5.5))
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel("Pooled out-of-fold $R^2$")
        axis.set_xlabel("")
        axis.set_xticklabels([TARGET_LABELS.get(value, value) for value in pivot.index], rotation=0)
        axis.set_title(f"Held-out predictive performance: {split}")
        axis.axhline(0, linewidth=0.8)
        axis.legend(title="Model", fontsize=8, ncol=2)
        figure.tight_layout()
        save_figure(figure, output_dir / f"r2_{split}")


def plot_delta_with_intervals(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output_dir: Path,
) -> None:
    subset = metrics[
        metrics["robustness"].eq("full") & metrics["model"].isin(CORE_MODELS)
    ].copy()
    intervals = bootstrap[
        bootstrap["robustness"].eq("full") & bootstrap["model"].isin(CORE_MODELS)
    ].copy()
    if subset.empty or intervals.empty:
        return

    merged = subset.merge(
        intervals[
            [
                "target",
                "split",
                "model",
                "delta_r2_ci_low",
                "delta_r2_ci_high",
            ]
        ],
        on=["target", "split", "model"],
        how="left",
        validate="one_to_one",
    )

    for split, frame in merged.groupby("split"):
        target_order = [target for target in TARGET_LABELS if target in frame["target"].unique()]
        figure, axis = plt.subplots(figsize=(11, 5.8))
        x = np.arange(len(target_order), dtype=float)
        width = 0.18
        for model_index, model in enumerate(CORE_MODELS):
            model_frame = frame[frame["model"].eq(model)].set_index("target").reindex(target_order)
            values = model_frame["delta_r2_vs_lexical"].to_numpy(dtype=float)
            lower = values - model_frame["delta_r2_ci_low"].to_numpy(dtype=float)
            upper = model_frame["delta_r2_ci_high"].to_numpy(dtype=float) - values
            offsets = x + (model_index - (len(CORE_MODELS) - 1) / 2) * width
            axis.bar(offsets, values, width=width, label=model)
            axis.errorbar(offsets, values, yerr=np.vstack([lower, upper]), fmt="none", capsize=3)
        axis.axhline(0, linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels([TARGET_LABELS.get(value, value) for value in target_order])
        axis.set_ylabel(r"$\Delta R^2$ over lexical baseline")
        axis.set_title(f"Incremental predictability with cluster-bootstrap 95% CIs: {split}")
        axis.legend(title="Model", fontsize=8, ncol=2)
        figure.tight_layout()
        save_figure(figure, output_dir / f"delta_r2_{split}")


def plot_robustness(metrics: pd.DataFrame, output_dir: Path) -> None:
    subset = metrics[
        metrics["model"].eq("lexical_all_transformers")
        & metrics["split"].eq("sentence")
    ].copy()
    if subset.empty:
        return
    pivot = subset.pivot(index="target", columns="robustness", values="delta_r2_vs_lexical")
    order = [target for target in TARGET_LABELS if target in pivot.index]
    pivot = pivot.reindex(order)
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    pivot.plot(kind="bar", ax=axis)
    axis.axhline(0, linewidth=0.8)
    axis.set_ylabel(r"$\Delta R^2$ over lexical baseline")
    axis.set_xlabel("")
    axis.set_xticklabels([TARGET_LABELS.get(value, value) for value in pivot.index], rotation=0)
    axis.set_title("Robustness of the combined-transformer result")
    axis.legend(title="Filter", fontsize=8, ncol=2)
    figure.tight_layout()
    save_figure(figure, output_dir / "robustness_combined_transformers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-readable figures from auditable analysis outputs.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    metrics = pd.read_csv(results_dir / "model_metrics.csv")
    bootstrap = pd.read_csv(results_dir / "bootstrap_intervals.csv")
    plot_r2(metrics, output_dir)
    plot_delta_with_intervals(metrics, bootstrap, output_dir)
    plot_robustness(metrics, output_dir)
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
