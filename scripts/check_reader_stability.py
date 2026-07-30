from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "log_first_fixation_duration": "First fixation duration",
    "log_gaze_duration": "Gaze duration",
    "log_go_past_time": "Go-past time",
    "log_total_reading_time": "Total reading time",
}


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    current = frame["pred__full_current"].to_numpy(dtype=float)
    spillover = frame["pred__full_spillover"].to_numpy(dtype=float)

    current_error = y - current
    spillover_error = y - spillover

    centred = y - y.mean()
    sst = float(np.dot(centred, centred))

    current_sse = float(np.dot(current_error, current_error))
    spillover_sse = float(np.dot(spillover_error, spillover_error))

    current_r2 = 1.0 - current_sse / sst
    spillover_r2 = 1.0 - spillover_sse / sst

    current_mse = float(np.mean(current_error**2))
    spillover_mse = float(np.mean(spillover_error**2))

    current_mae = float(np.mean(np.abs(current_error)))
    spillover_mae = float(np.mean(np.abs(spillover_error)))

    return {
        "current_r2": current_r2,
        "spillover_r2": spillover_r2,
        "delta_r2": spillover_r2 - current_r2,
        "current_mse": current_mse,
        "spillover_mse": spillover_mse,
        "mse_reduction": current_mse - spillover_mse,
        "current_mae": current_mae,
        "spillover_mae": spillover_mae,
        "mae_reduction": current_mae - spillover_mae,
    }


def main() -> None:
    root = Path("results/main/oof_predictions")
    output = Path("results/reader_stability")
    output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    reader_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []

    for target, target_label in TARGETS.items():
        path = root / f"{target}__participant__full.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_parquet(
            path,
            columns=[
                "participant_id",
                "y_true",
                "pred__full_current",
                "pred__full_spillover",
            ],
        ).dropna()

        if frame.empty:
            raise ValueError(f"No valid rows in {path}")

        participants = sorted(
            frame["participant_id"].astype(str).unique()
        )

        if len(participants) != 14:
            raise AssertionError(
                f"{target}: expected 14 participants, found {len(participants)}"
            )

        overall = metrics(frame)
        target_reader_rows: list[dict[str, object]] = []
        exclusion_delta_r2: list[float] = []

        for participant in participants:
            reader_frame = frame[
                frame["participant_id"].astype(str).eq(participant)
            ]

            result = metrics(reader_frame)

            row = {
                "target": target,
                "target_label": target_label,
                "participant_id": participant,
                "rows": len(reader_frame),
                **result,
            }

            reader_rows.append(row)
            target_reader_rows.append(row)

        for excluded_participant in participants:
            remaining = frame[
                ~frame["participant_id"]
                .astype(str)
                .eq(excluded_participant)
            ]

            result = metrics(remaining)
            exclusion_delta_r2.append(result["delta_r2"])

            exclusion_rows.append(
                {
                    "target": target,
                    "target_label": target_label,
                    "excluded_participant": excluded_participant,
                    "remaining_rows": len(remaining),
                    **result,
                }
            )

        reader_results = pd.DataFrame(target_reader_rows)

        summary_rows.append(
            {
                "target": target,
                "target_label": target_label,
                "rows": len(frame),
                "participants": len(participants),
                "overall_current_r2": overall["current_r2"],
                "overall_spillover_r2": overall["spillover_r2"],
                "overall_delta_r2": overall["delta_r2"],
                "readers_with_lower_mse": int(
                    reader_results["mse_reduction"].gt(0).sum()
                ),
                "readers_with_lower_mae": int(
                    reader_results["mae_reduction"].gt(0).sum()
                ),
                "median_reader_mse_reduction": float(
                    reader_results["mse_reduction"].median()
                ),
                "median_reader_mae_reduction": float(
                    reader_results["mae_reduction"].median()
                ),
                "leave_one_reader_out_delta_r2_min": float(
                    np.min(exclusion_delta_r2)
                ),
                "leave_one_reader_out_delta_r2_max": float(
                    np.max(exclusion_delta_r2)
                ),
                "all_leave_one_reader_out_positive": bool(
                    np.min(exclusion_delta_r2) > 0
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    by_reader = pd.DataFrame(reader_rows)
    exclusions = pd.DataFrame(exclusion_rows)

    summary.to_csv(
        output / "reader_stability_summary.csv",
        index=False,
    )

    by_reader.to_csv(
        output / "reader_stability_by_reader.csv",
        index=False,
    )

    exclusions.to_csv(
        output / "reader_exclusion_sensitivity.csv",
        index=False,
    )

    print(
        summary[
            [
                "target_label",
                "overall_delta_r2",
                "readers_with_lower_mse",
                "readers_with_lower_mae",
                "leave_one_reader_out_delta_r2_min",
                "leave_one_reader_out_delta_r2_max",
                "all_leave_one_reader_out_positive",
            ]
        ].to_string(index=False)
    )

    print("\nWrote:")
    print(output / "reader_stability_summary.csv")
    print(output / "reader_stability_by_reader.csv")
    print(output / "reader_exclusion_sensitivity.csv")


if __name__ == "__main__":
    main()
