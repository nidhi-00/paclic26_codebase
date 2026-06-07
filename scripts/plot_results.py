# scripts/plot_results.py
import matplotlib.pyplot as plt
import pandas as pd

summary = pd.read_csv("results/stats/model_summary.csv")
pivot = summary.pivot(index="target", columns="model", values="r2")

ax = pivot.plot(kind="bar", figsize=(8, 4))
ax.set_title("Model comparison by cross-validated R²")
ax.set_ylabel("R²")
ax.figure.tight_layout()
ax.figure.savefig("paper/figures/r2_comparison.png", dpi=200)
