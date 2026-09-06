#!/usr/bin/env python3
"""GT-referenced static/dynamic diagnostic evaluation for STARSS predictions.

This is deliberately a diagnostic complement to, not a replacement for, the
official DCASE ER/F/LE/LR evaluator.  Predictions and all GT events are first
matched globally per (file, frame, class) using Hungarian angular assignment.
Matched errors and misses are then attributed to the GT source's motion group.
Unmatched predictions remain a global false-positive count because assigning
them to static or dynamic sources would be arbitrary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def polar_to_unit(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return (
        cos_elevation * math.cos(azimuth),
        cos_elevation * math.sin(azimuth),
        math.sin(elevation),
    )


def angular_distance_deg(first: Sequence[float], second: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(first, second))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def load_gt(source_frames_csv: Path, include_groups: set[str]) -> tuple[dict, list[dict]]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    raw_rows: list[dict] = []
    with source_frames_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            motion_group = row["motion_group"]
            if motion_group not in include_groups:
                continue
            filename = Path(row["metadata_file"]).stem
            item = {
                "filename": filename,
                "frame": int(row["frame"]),
                "class_id": int(row["class_id"]),
                "source_id": int(row["source_id"]),
                "segment_id": row["segment_id"],
                "motion_group": motion_group,
                "azimuth_deg": float(row["azimuth_deg"]),
                "elevation_deg": float(row["elevation_deg"]),
                "vector": polar_to_unit(float(row["azimuth_deg"]), float(row["elevation_deg"])),
            }
            grouped[(filename, item["frame"], item["class_id"])].append(item)
            raw_rows.append(item)
    return grouped, raw_rows


def load_predictions(prediction_dir: Path) -> tuple[dict, int, int]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    file_count = 0
    row_count = 0
    for path in sorted(prediction_dir.glob("*.csv")):
        file_count += 1
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, values in enumerate(csv.reader(handle), start=1):
                if len(values) < 4:
                    raise ValueError(f"{path}:{line_number}: expected >=4 columns")
                frame = int(values[0])
                class_id = int(values[1])
                # DCASE submission: frame,class,azimuth,elevation.  If a track
                # column is present, use the final two columns as polar DOA.
                azimuth = float(values[-2])
                elevation = float(values[-1])
                grouped[(path.stem, frame, class_id)].append(
                    {
                        "azimuth_deg": azimuth,
                        "elevation_deg": elevation,
                        "vector": polar_to_unit(azimuth, elevation),
                    }
                )
                row_count += 1
    if file_count == 0:
        raise RuntimeError(f"No prediction CSV files under {prediction_dir}")
    return grouped, file_count, row_count


def summarize(rows: list[dict], groups: Sequence[str], threshold_deg: float) -> dict:
    result = {}
    for motion_group in groups:
        group_rows = [row for row in rows if row["motion_group"] == motion_group]
        matched_rows = [row for row in group_rows if row["matched"]]
        within_rows = [row for row in matched_rows if row["angular_error_deg"] <= threshold_deg]
        errors = [row["angular_error_deg"] for row in matched_rows]

        per_class = {}
        for class_id in sorted({row["class_id"] for row in group_rows}):
            class_rows = [row for row in group_rows if row["class_id"] == class_id]
            class_matched = [row for row in class_rows if row["matched"]]
            class_within = [row for row in class_matched if row["angular_error_deg"] <= threshold_deg]
            class_errors = [row["angular_error_deg"] for row in class_matched]
            per_class[str(class_id)] = {
                "references": len(class_rows),
                "matched_recall": len(class_matched) / len(class_rows),
                "recall_at_threshold": len(class_within) / len(class_rows),
                "mean_localization_error_deg": mean(class_errors) if class_errors else None,
            }
        macro_localization_errors = [
            value["mean_localization_error_deg"]
            for value in per_class.values()
            if value["mean_localization_error_deg"] is not None
        ]
        result[motion_group] = {
            "references": len(group_rows),
            "matched": len(matched_rows),
            "within_threshold": len(within_rows),
            "matched_recall_micro": len(matched_rows) / len(group_rows) if group_rows else None,
            "recall_at_threshold_micro": len(within_rows) / len(group_rows) if group_rows else None,
            "mean_localization_error_deg_micro": mean(errors) if errors else None,
            "median_localization_error_deg_micro": median(errors) if errors else None,
            "p90_localization_error_deg_micro": percentile(errors, 0.90),
            "matched_recall_macro_class": (
                mean(value["matched_recall"] for value in per_class.values()) if per_class else None
            ),
            "recall_at_threshold_macro_class": (
                mean(value["recall_at_threshold"] for value in per_class.values()) if per_class else None
            ),
            "mean_localization_error_deg_macro_class": (
                mean(macro_localization_errors) if macro_localization_errors else None
            ),
            "per_class": per_class,
        }
    return result


def evaluate(args: argparse.Namespace) -> dict:
    groups = [group.strip() for group in args.motion_groups.split(",") if group.strip()]
    gt_grouped, gt_rows = load_gt(args.source_frames_csv, set(groups))
    pred_grouped, prediction_files, prediction_rows = load_predictions(args.prediction_dir)

    matched_output: list[dict] = []
    unmatched_predictions = 0
    all_keys = set(gt_grouped) | set(pred_grouped)
    for key in sorted(all_keys):
        references = gt_grouped.get(key, [])
        predictions = pred_grouped.get(key, [])
        assignments: dict[int, tuple[int, float]] = {}
        if references and predictions:
            cost = np.asarray(
                [
                    [angular_distance_deg(reference["vector"], prediction["vector"]) for prediction in predictions]
                    for reference in references
                ],
                dtype=np.float64,
            )
            reference_indices, prediction_indices = linear_sum_assignment(cost)
            assignments = {
                int(reference_index): (int(prediction_index), float(cost[reference_index, prediction_index]))
                for reference_index, prediction_index in zip(reference_indices, prediction_indices)
            }
        unmatched_predictions += max(0, len(predictions) - len(assignments))

        for reference_index, reference in enumerate(references):
            if reference_index in assignments:
                prediction_index, error = assignments[reference_index]
                prediction = predictions[prediction_index]
                matched = True
                predicted_azimuth = prediction["azimuth_deg"]
                predicted_elevation = prediction["elevation_deg"]
            else:
                error = None
                matched = False
                predicted_azimuth = None
                predicted_elevation = None
            matched_output.append(
                {
                    "model": args.model_name,
                    "filename": reference["filename"],
                    "frame": reference["frame"],
                    "class_id": reference["class_id"],
                    "source_id": reference["source_id"],
                    "segment_id": reference["segment_id"],
                    "motion_group": reference["motion_group"],
                    "matched": int(matched),
                    "within_threshold": int(matched and error <= args.threshold_deg),
                    "angular_error_deg": "" if error is None else error,
                    "gt_azimuth_deg": reference["azimuth_deg"],
                    "gt_elevation_deg": reference["elevation_deg"],
                    "pred_azimuth_deg": "" if predicted_azimuth is None else predicted_azimuth,
                    "pred_elevation_deg": "" if predicted_elevation is None else predicted_elevation,
                }
            )

    numeric_rows = []
    for row in matched_output:
        numeric = dict(row)
        numeric["matched"] = bool(row["matched"])
        numeric["angular_error_deg"] = (
            float(row["angular_error_deg"]) if row["angular_error_deg"] != "" else None
        )
        numeric_rows.append(numeric)

    summary = {
        "model": args.model_name,
        "prediction_dir": str(args.prediction_dir),
        "source_frames_csv": str(args.source_frames_csv),
        "threshold_deg": args.threshold_deg,
        "diagnostic_scope": "GT-referenced frame-level class-conditional Hungarian matching",
        "official_metric_replacement": False,
        "prediction_files": prediction_files,
        "prediction_rows": prediction_rows,
        "gt_rows_included": len(gt_rows),
        "unmatched_predictions_global": unmatched_predictions,
        "motion_groups": summarize(numeric_rows, groups, args.threshold_deg),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "motion_stratified_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (args.output_dir / "motion_stratified_matches.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matched_output[0].keys()))
        writer.writeheader()
        writer.writerows(matched_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-frames-csv", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--motion-groups", default="static,dynamic")
    parser.add_argument("--threshold-deg", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    print(json.dumps(evaluate(parsed_args), ensure_ascii=False, indent=2))
