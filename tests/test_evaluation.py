from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import (
    ResolvedModel,
    bootstrap_oof_differences,
    evaluate_models,
)


def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for participant in range(6):
        for sentence in range(12):
            for word in range(4):
                lexical = rng.normal()
                surprisal = rng.normal()
                target = 0.4 * lexical + 0.3 * surprisal + 0.1 * participant + rng.normal(0, 0.2)
                rows.append(
                    {
                        "participant_id": f"p{participant}",
                        "sentence_id": f"s{sentence}",
                        "word_id": str(word),
                        "word_length": lexical,
                        "gpt2_surprisal": surprisal,
                        "target": target,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.loc[frame.index[::17], "gpt2_surprisal"] = np.nan
    return frame


def test_oof_predictions_cover_common_sample_with_imputation() -> None:
    data = synthetic_frame()
    models = [
        ResolvedModel("lexical", ("word_length",), ()),
        ResolvedModel("lexical_gpt2", ("word_length", "gpt2_surprisal"), ("gpt2",)),
    ]
    metrics, folds, oof, alphas = evaluate_models(
        sample=data,
        target="target",
        group_column="sentence_id",
        split_name="sentence",
        split_method="group_kfold",
        folds=4,
        models=models,
        alphas=[1.0],
        tune=False,
        inner_folds=3,
        robustness_name="full",
    )
    assert len(oof) == len(data)
    assert oof["pred__lexical"].notna().all()
    assert oof["pred__lexical_gpt2"].notna().all()
    assert metrics["rows"].nunique() == 1
    assert len(folds) == 8
    assert len(alphas) == 8
    assert metrics.set_index("model").loc["lexical_gpt2", "r2"] > metrics.set_index("model").loc["lexical", "r2"]


def test_participant_leave_one_out_and_bootstrap() -> None:
    data = synthetic_frame()
    models = [
        ResolvedModel("lexical", ("word_length",), ()),
        ResolvedModel("lexical_gpt2", ("word_length", "gpt2_surprisal"), ("gpt2",)),
    ]
    _, folds, oof, _ = evaluate_models(
        sample=data,
        target="target",
        group_column="participant_id",
        split_name="participant",
        split_method="leave_one_group_out",
        folds=None,
        models=models,
        alphas=[1.0],
        tune=False,
        inner_folds=3,
        robustness_name="full",
    )
    assert folds["outer_fold"].nunique() == data["participant_id"].nunique()
    intervals = bootstrap_oof_differences(
        oof,
        group_column="participant_id",
        model_names=["lexical", "lexical_gpt2"],
        iterations=100,
        seed=11,
    )
    assert set(intervals["model"]) == {"lexical", "lexical_gpt2"}
    assert intervals.loc[intervals["model"].eq("lexical_gpt2"), "delta_r2_ci_low"].notna().all()
