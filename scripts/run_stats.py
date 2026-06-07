# scripts/run_stats.py
import pandas as pd
import statsmodels.formula.api as smf
from src.stats_utils import evaluate_cv

df = pd.read_csv("data/processed/analysis_table.csv")

lexical = ["word_len", "log_freq", "position", "sent_len",
           "is_sent_initial", "is_sent_final", "prev_word_len", "prev_log_freq"]
candidate = [c for c in df.columns if c.startswith("surprisal_") or c.startswith("pseudo_")]

for target in ["log_ffd", "log_gd", "log_gpt", "log_trt"]:
    work = df.dropna(subset=[target]).copy()

    formula = (
        f"{target} ~ " +
        " + ".join(lexical + candidate)
    )
    ols = smf.ols(formula, data=work.dropna(subset=lexical + candidate)).fit()
    print(target, ols.rsquared, ols.aic, ols.bic)

    cv = evaluate_cv(
        work.dropna(subset=lexical + candidate),
        x_cols=lexical + candidate,
        y_col=target,
        group_col="sentence_id",
        n_splits=5,
    )
    print(cv)
