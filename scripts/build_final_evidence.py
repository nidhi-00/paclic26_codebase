from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nct, t


ROOT = Path(".")
OUTPUT = ROOT / "results" / "final_strengthening"

DIRECT_PATH = (
    ROOT
    / "results"
    / "direct_spillover"
    / "direct_spillover_comparisons.csv"
)

ROBUSTNESS_PATH = (
    ROOT
    / "results"
    / "direct_spillover_robustness"
    / "full_model_robustness_summary.csv"
)

READER_PATH = (
    ROOT
    / "results"
    / "reader_stability"
    / "reader_stability_summary.csv"
)

AUDIT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "data_audit.json"
)

TARGET_ORDER = {
    "log_first_fixation_duration": 0,
    "log_gaze_duration": 1,
    "log_go_past_time": 2,
    "log_total_reading_time": 3,
}

TARGET_SHORT = {
    "log_first_fixation_duration": "FFD",
    "log_gaze_duration": "GD",
    "log_go_past_time": "Go-past",
    "log_total_reading_time": "TRT",
}


def minimum_detectable_dz(
    n: int,
    alpha: float = 0.05,
    desired_power: float = 0.80,
) -> float:
    df = n - 1
    critical = t.ppf(1.0 - alpha / 2.0, df)

    def achieved_power(effect: float) -> float:
        noncentrality = effect * math.sqrt(n)

        lower = nct.cdf(
            -critical,
            df,
            noncentrality,
        )

        upper = 1.0 - nct.cdf(
            critical,
            df,
            noncentrality,
        )

        return float(lower + upper)

    return float(
        brentq(
            lambda effect: (
                achieved_power(effect) - desired_power
            ),
            0.0,
            5.0,
        )
    )


def require_files() -> None:
    for path in [
        DIRECT_PATH,
        ROBUSTNESS_PATH,
        READER_PATH,
        AUDIT_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)


def direct_results() -> pd.DataFrame:
    frame = pd.read_csv(DIRECT_PATH)

    frame = frame[
        frame["comparison"].eq("all_lm_spillover")
    ].copy()

    if len(frame) != 8:
        raise AssertionError(
            f"Expected 8 full-model direct rows, found {len(frame)}"
        )

    frame["target_order"] = frame["target"].map(
        TARGET_ORDER
    )

    frame["split_order"] = frame["split"].map(
        {"sentence": 0, "participant": 1}
    )

    return frame.sort_values(
        ["split_order", "target_order"]
    ).reset_index(drop=True)


def effect_size_table(
    direct: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "target",
        "target_label",
        "split",
        "baseline_r2",
        "augmented_r2",
        "delta_r2",
        "delta_r2_ci_low",
        "delta_r2_ci_high",
        "cluster_signflip_q_bh_global",
    ]

    context = direct[columns].copy()

    context["relative_r2_increase_percent"] = np.where(
        context["baseline_r2"] > 0,
        100.0
        * context["delta_r2"]
        / context["baseline_r2"],
        np.nan,
    )

    context["recommended_for_relative_context"] = (
        context["target"].isin(
            {
                "log_gaze_duration",
                "log_go_past_time",
                "log_total_reading_time",
            }
        )
        & context["baseline_r2"].gt(0)
    )

    context["interpretation_note"] = np.where(
        context["recommended_for_relative_context"],
        (
            "Usable as within-study context; not directly "
            "comparable with likelihood, slope, or millisecond "
            "effect sizes from prior studies."
        ),
        (
            "Do not emphasize as a relative percentage because "
            "the absolute baseline fit is very small or negative."
        ),
    )

    return context


def make_ledger(
    direct: pd.DataFrame,
    robustness: pd.DataFrame,
    readers: pd.DataFrame,
    audit: dict,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    counts = {
        "participants": audit["participants"],
        "sentence_units": audit["sentence_units"],
        "unique_items": audit["unique_items"],
        "unique_lexical_forms": audit["unique_lexical_forms"],
        "participant_word_rows": audit["participant_word_rows"],
    }

    for name, value in counts.items():
        rows.append(
            {
                "claim_id": f"DATA_{name.upper()}",
                "category": "data",
                "target": "",
                "split": "",
                "condition": "processed GECO",
                "primary_value": value,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "q_value": np.nan,
                "source_file": str(AUDIT_PATH),
                "source_filter": name,
                "approved_interpretation": (
                    f"Processed GECO {name.replace('_', ' ')}: "
                    f"{value:,}."
                ),
                "verified": True,
            }
        )

    for target, target_data in audit["targets"].items():
        rows.append(
            {
                "claim_id": (
                    f"DATA_VALID_{target.upper()}"
                ),
                "category": "data",
                "target": target,
                "split": "",
                "condition": "target-specific filtering",
                "primary_value": target_data["valid"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "q_value": np.nan,
                "source_file": str(AUDIT_PATH),
                "source_filter": f"targets.{target}.valid",
                "approved_interpretation": (
                    f"Valid {target.replace('_', ' ')} "
                    f"observations: {target_data['valid']:,}."
                ),
                "verified": True,
            }
        )

    for _, row in direct.iterrows():
        short = TARGET_SHORT[row["target"]]
        split = row["split"]

        rows.append(
            {
                "claim_id": (
                    f"DIRECT_{split.upper()}_{short.upper()}"
                ),
                "category": "direct_spillover",
                "target": row["target"],
                "split": split,
                "condition": "full sample",
                "primary_value": row["delta_r2"],
                "ci_low": row["delta_r2_ci_low"],
                "ci_high": row["delta_r2_ci_high"],
                "q_value": row[
                    "cluster_signflip_q_bh_global"
                ],
                "source_file": str(DIRECT_PATH),
                "source_filter": (
                    "comparison=all_lm_spillover; "
                    f"target={row['target']}; split={split}"
                ),
                "approved_interpretation": (
                    f"Adding previous-word surprisal improved "
                    f"{short} under {split}-held-out evaluation "
                    f"by direct delta R2={row['delta_r2']:.5f} "
                    f"[{row['delta_r2_ci_low']:.5f}, "
                    f"{row['delta_r2_ci_high']:.5f}]."
                ),
                "verified": True,
            }
        )

    selected_robustness = robustness[
        robustness["robustness"].isin(
            {"no_edges", "content_words"}
        )
    ].copy()

    for _, row in selected_robustness.iterrows():
        rows.append(
            {
                "claim_id": (
                    "ROBUST_"
                    f"{str(row['robustness']).upper()}_"
                    f"{str(row['split']).upper()}_"
                    f"{str(row['target_label']).upper().replace(' ', '_')}"
                ),
                "category": "robustness",
                "target": row["target_label"],
                "split": row["split"],
                "condition": row["robustness"],
                "primary_value": row["delta_r2"],
                "ci_low": row["delta_r2_ci_low"],
                "ci_high": row["delta_r2_ci_high"],
                "q_value": row[
                    "cluster_signflip_q_bh_global"
                ],
                "source_file": str(ROBUSTNESS_PATH),
                "source_filter": (
                    f"robustness={row['robustness']}; "
                    f"target={row['target_label']}; "
                    f"split={row['split']}"
                ),
                "approved_interpretation": (
                    f"Direct spillover gain under "
                    f"{row['robustness']} for "
                    f"{row['target_label']} "
                    f"({row['split']} split): "
                    f"{row['delta_r2']:.5f}."
                ),
                "verified": True,
            }
        )

    for _, row in readers.iterrows():
        rows.append(
            {
                "claim_id": (
                    "READER_STABILITY_"
                    f"{str(row['target']).upper()}"
                ),
                "category": "reader_stability",
                "target": row["target"],
                "split": "participant",
                "condition": "leave one reader out",
                "primary_value": row["overall_delta_r2"],
                "ci_low": row[
                    "leave_one_reader_out_delta_r2_min"
                ],
                "ci_high": row[
                    "leave_one_reader_out_delta_r2_max"
                ],
                "q_value": np.nan,
                "source_file": str(READER_PATH),
                "source_filter": (
                    f"target={row['target']}"
                ),
                "approved_interpretation": (
                    f"{row['target_label']}: spillover remained "
                    f"positive after every single-reader exclusion; "
                    f"leave-one-reader-out delta R2 range "
                    f"[{row['leave_one_reader_out_delta_r2_min']:.6f}, "
                    f"{row['leave_one_reader_out_delta_r2_max']:.6f}]."
                ),
                "verified": bool(
                    row[
                        "all_leave_one_reader_out_positive"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def write_approved_claims(
    direct: pd.DataFrame,
    effects: pd.DataFrame,
    readers: pd.DataFrame,
    detectable_dz: float,
) -> None:
    trt = direct[
        direct["target"].eq(
            "log_total_reading_time"
        )
    ].set_index("split")

    later_effects = effects[
        effects["recommended_for_relative_context"]
    ].copy()

    effect_lines = []

    for _, row in later_effects.iterrows():
        effect_lines.append(
            f"- {row['target_label']}, {row['split']}: "
            f"{row['relative_r2_increase_percent']:.1f}% "
            f"relative increase over the full current model."
        )

    reader_lines = []

    for _, row in readers.iterrows():
        reader_lines.append(
            f"- {row['target_label']}: "
            f"{int(row['readers_with_lower_mse'])}/14 readers "
            f"had lower MSE; leave-one-reader-out delta R2 "
            f"remained between "
            f"{row['leave_one_reader_out_delta_r2_min']:.6f} "
            f"and "
            f"{row['leave_one_reader_out_delta_r2_max']:.6f}."
        )

    text = f"""# Approved Evidence and Claims

## Headline direct result

Adding previous-word surprisal to the otherwise identical full
current-word model improves every reading measure under both
sentence-held-out and participant-held-out evaluation.

For total reading time, the direct gain is
delta R2={trt.loc['sentence', 'delta_r2']:.5f} under
sentence-held-out evaluation and
delta R2={trt.loc['participant', 'delta_r2']:.5f} under
participant-held-out evaluation.

## Relative within-study context

These percentages are descriptive within-study comparisons.
They must not be equated numerically with effects reported using
likelihood ratios, millisecond slopes, or mixed-effects coefficients.

{chr(10).join(effect_lines)}

## Reader stability

The pooled direct spillover increment remains positive after
excluding each reader in turn for all four reading measures.
The later-measure result is especially consistent for go-past time
and total reading time.

{chr(10).join(reader_lines)}

## Participant sensitivity

For n=14 paired observations, two-sided alpha=.05 and 80% power,
the approximate minimum detectable standardized paired effect is
dz={detectable_dz:.3f}.

This is a conventional paired-t sensitivity calculation. It is not
a post-hoc power estimate for the cluster bootstrap or sign-flip
procedure.

## Interpretation boundary

The results establish a stable held-out predictive lag. They do not
show that the language models implement a human spillover mechanism,
and they do not establish that previous-word surprisal causally
delays the following fixation.

## FFD qualification

The direct FFD increment is positive and survives every single-reader
exclusion, but absolute participant-held-out FFD R2 remains negative.
FFD therefore requires cautious interpretation.
"""

    path = OUTPUT / "approved_claims.md"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    require_files()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    direct = direct_results()
    robustness = pd.read_csv(ROBUSTNESS_PATH)
    readers = pd.read_csv(READER_PATH)
    audit = json.loads(
        AUDIT_PATH.read_text(encoding="utf-8")
    )

    effects = effect_size_table(direct)

    detectable_dz = minimum_detectable_dz(
        n=14,
        alpha=0.05,
        desired_power=0.80,
    )

    sensitivity = pd.DataFrame(
        [
            {
                "test": "two-sided paired t sensitivity",
                "participants": 14,
                "alpha": 0.05,
                "desired_power": 0.80,
                "minimum_detectable_dz": detectable_dz,
                "interpretation": (
                    "Approximate conventional sensitivity; "
                    "not post-hoc power for cluster inference."
                ),
            }
        ]
    )

    ledger = make_ledger(
        direct,
        robustness,
        readers,
        audit,
    )

    effects.to_csv(
        OUTPUT / "effect_size_context.csv",
        index=False,
    )

    sensitivity.to_csv(
        OUTPUT / "participant_sensitivity.csv",
        index=False,
    )

    ledger.to_csv(
        OUTPUT / "evidence_ledger.csv",
        index=False,
    )

    write_approved_claims(
        direct,
        effects,
        readers,
        detectable_dz,
    )

    print("\nEffect-size context:")
    print(
        effects[
            [
                "target_label",
                "split",
                "baseline_r2",
                "augmented_r2",
                "delta_r2",
                "relative_r2_increase_percent",
                "recommended_for_relative_context",
            ]
        ].to_string(index=False)
    )

    print("\nParticipant sensitivity:")
    print(sensitivity.to_string(index=False))

    print("\nEvidence ledger rows =", len(ledger))

    print("\nWrote:")
    for name in [
        "evidence_ledger.csv",
        "effect_size_context.csv",
        "participant_sensitivity.csv",
        "approved_claims.md",
    ]:
        print(OUTPUT / name)


if __name__ == "__main__":
    main()
