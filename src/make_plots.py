from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_model_comparison(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    target = df["target"].iloc[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.8))
    plt.bar(df["model"], df["delta_r2_vs_lexical"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Delta R² over lexical baseline")
    plt.title(f"Incremental variance: {target}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{target}_delta_r2.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.bar(df["model"], df["cv_rmse"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("5-fold sentence-level CV RMSE")
    plt.title(f"Predictive error: {target}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{target}_cv_rmse.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    for csv_path in results_dir.glob("*_model_comparison.csv"):
        plot_model_comparison(csv_path, output_dir)
        print(f"Plotted {csv_path}")


if __name__ == "__main__":
    main()
