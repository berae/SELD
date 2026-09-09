#!/usr/bin/env python3
"""Audit source-motion coverage in DCASE 2020 SELD metadata.

The metadata rows are expected to be:
    frame_index, class_index, source_index, azimuth_deg, elevation_deg

Motion is measured only between adjacent 100 ms label frames belonging to the
same contiguous (class, source) segment.  Gaps split a reused source key into
separate segments, which avoids inventing motion across inactive intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Row = Tuple[int, int, int, float, float]


def angular_distance_deg(az1: float, el1: float, az2: float, el2: float) -> float:
    """Great-circle angular distance between two azimuth/elevation points."""
    az1_r, el1_r, az2_r, el2_r = map(math.radians, (az1, el1, az2, el2))
    dot = (
        math.cos(el1_r) * math.cos(el2_r) * math.cos(az1_r - az2_r)
        + math.sin(el1_r) * math.sin(el2_r)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def parse_metadata(path: Path) -> Tuple[List[Row], int]:
    rows: List[Row] = []
    malformed = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle):
            if not fields:
                continue
            if len(fields) < 5:
                malformed += 1
                continue
            try:
                rows.append(
                    (
                        int(fields[0]),
                        int(fields[1]),
                        int(fields[2]),
                        float(fields[3]),
                        float(fields[4]),
                    )
                )
            except ValueError:
                malformed += 1
    return rows, malformed


def contiguous_segments(rows: Iterable[Row]) -> Iterable[List[Row]]:
    grouped: Dict[Tuple[int, int], List[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row[1], row[2])].append(row)

    for source_rows in grouped.values():
        source_rows.sort(key=lambda item: item[0])
        segment = [source_rows[0]]
        for row in source_rows[1:]:
            if row[0] == segment[-1][0] + 1:
                segment.append(row)
            else:
                yield segment
                segment = [row]
        yield segment


def empty_counts() -> Dict[str, int]:
    return {
        "files": 0,
        "files_with_motion": 0,
        "annotated_frames": 0,
        "frames_touching_motion": 0,
        "active_rows": 0,
        "segments": 0,
        "dynamic_segments": 0,
        "observed_static_segments": 0,
        "unobservable_singleton_segments": 0,
        "rows_in_dynamic_segments": 0,
        "rows_in_observed_static_segments": 0,
        "rows_in_unobservable_segments": 0,
        "adjacent_pairs": 0,
        "moving_pairs": 0,
        "zero_motion_pairs": 0,
        "duplicate_source_frames": 0,
        "malformed_rows": 0,
    }


def empty_class_counts() -> Dict[str, int]:
    return {
        "files": 0,
        "files_with_motion": 0,
        "active_rows": 0,
        "segments": 0,
        "dynamic_segments": 0,
        "observed_static_segments": 0,
        "unobservable_singleton_segments": 0,
        "rows_in_dynamic_segments": 0,
        "rows_in_observed_static_segments": 0,
        "rows_in_unobservable_segments": 0,
        "adjacent_pairs": 0,
        "moving_pairs": 0,
        "zero_motion_pairs": 0,
    }


def audit_split(
    name: str, paths: Sequence[Path], motion_epsilon: float
) -> Tuple[dict, List[dict], List[dict]]:
    counts = empty_counts()
    class_counts: Dict[int, Dict[str, int]] = defaultdict(empty_class_counts)
    nonzero_steps: List[float] = []
    source_frame_rows: List[dict] = []

    for path in paths:
        rows, malformed = parse_metadata(path)
        counts["files"] += 1
        counts["malformed_rows"] += malformed
        counts["active_rows"] += len(rows)
        annotated_frames = {row[0] for row in rows}
        counts["annotated_frames"] += len(annotated_frames)

        seen_keys = set()
        for row in rows:
            key = (row[0], row[1], row[2])
            if key in seen_keys:
                counts["duplicate_source_frames"] += 1
            seen_keys.add(key)

        file_has_motion = False
        file_motion_classes = set()
        motion_frames = set()
        for segment_index, segment in enumerate(contiguous_segments(rows)):
            class_id = segment[0][1]
            source_id = segment[0][2]
            cc = class_counts[class_id]
            counts["segments"] += 1
            cc["segments"] += 1
            cc["active_rows"] += len(segment)

            steps = [
                angular_distance_deg(left[3], left[4], right[3], right[4])
                for left, right in zip(segment, segment[1:])
            ]
            moving_steps = [step for step in steps if step > motion_epsilon]
            counts["adjacent_pairs"] += len(steps)
            counts["moving_pairs"] += len(moving_steps)
            counts["zero_motion_pairs"] += len(steps) - len(moving_steps)
            cc["adjacent_pairs"] += len(steps)
            cc["moving_pairs"] += len(moving_steps)
            cc["zero_motion_pairs"] += len(steps) - len(moving_steps)
            nonzero_steps.extend(moving_steps)

            if moving_steps:
                file_has_motion = True
                file_motion_classes.add(class_id)
                counts["dynamic_segments"] += 1
                counts["rows_in_dynamic_segments"] += len(segment)
                cc["dynamic_segments"] += 1
                cc["rows_in_dynamic_segments"] += len(segment)
                for left, right, step in zip(segment, segment[1:], steps):
                    if step > motion_epsilon:
                        motion_frames.add(left[0])
                        motion_frames.add(right[0])
            elif steps:
                counts["observed_static_segments"] += 1
                counts["rows_in_observed_static_segments"] += len(segment)
                cc["observed_static_segments"] += 1
                cc["rows_in_observed_static_segments"] += len(segment)
            else:
                counts["unobservable_singleton_segments"] += 1
                counts["rows_in_unobservable_segments"] += len(segment)
                cc["unobservable_singleton_segments"] += 1
                cc["rows_in_unobservable_segments"] += len(segment)

            motion_group = (
                "dynamic" if moving_steps else "static" if steps else "unobservable"
            )
            segment_id = (
                f"{path.name}::c{class_id}::s{source_id}::seg{segment_index:04d}"
            )
            for row_index, row in enumerate(segment):
                step = 0.0 if row_index == 0 else steps[row_index - 1]
                source_frame_rows.append(
                    {
                        "split": name,
                        "metadata_file": path.name,
                        "segment_id": segment_id,
                        "frame": row[0],
                        "class_id": class_id,
                        "source_id": source_id,
                        "azimuth_deg": row[3],
                        "elevation_deg": row[4],
                        "motion_group": motion_group,
                        "velocity_valid": int(row_index > 0),
                        "moving_frame": int(row_index > 0 and step > motion_epsilon),
                        "angular_step_deg": step,
                    }
                )

        if file_has_motion:
            counts["files_with_motion"] += 1
        counts["frames_touching_motion"] += len(motion_frames)

        for class_id in {row[1] for row in rows}:
            class_counts[class_id]["files"] += 1
        for class_id in file_motion_classes:
            class_counts[class_id]["files_with_motion"] += 1

    summary = {
        "split": name,
        **counts,
        "ratios": {
            "files_with_motion": ratio(counts["files_with_motion"], counts["files"]),
            "dynamic_segments": ratio(counts["dynamic_segments"], counts["segments"]),
            "rows_in_dynamic_segments": ratio(
                counts["rows_in_dynamic_segments"], counts["active_rows"]
            ),
            "moving_pairs": ratio(counts["moving_pairs"], counts["adjacent_pairs"]),
            "frames_touching_motion": ratio(
                counts["frames_touching_motion"], counts["annotated_frames"]
            ),
        },
        "nonzero_angular_step_deg_per_100ms": {
            "count": len(nonzero_steps),
            "mean": sum(nonzero_steps) / len(nonzero_steps) if nonzero_steps else None,
            "p25": percentile(nonzero_steps, 0.25),
            "p50": percentile(nonzero_steps, 0.50),
            "p75": percentile(nonzero_steps, 0.75),
            "p90": percentile(nonzero_steps, 0.90),
            "p95": percentile(nonzero_steps, 0.95),
            "p99": percentile(nonzero_steps, 0.99),
            "max": max(nonzero_steps) if nonzero_steps else None,
        },
        "moving_pair_threshold_sensitivity": {
            f">{threshold:g}deg": {
                "count": sum(step > max(motion_epsilon, threshold) for step in nonzero_steps),
                "share_of_all_adjacent_pairs": ratio(
                    sum(step > max(motion_epsilon, threshold) for step in nonzero_steps),
                    counts["adjacent_pairs"],
                ),
            }
            for threshold in (0.0, 1.0, 2.0, 4.0, 5.0, 10.0)
        },
    }

    by_class = []
    for class_id, cc in sorted(class_counts.items()):
        by_class.append(
            {
                "split": name,
                "class_id": class_id,
                **cc,
                "files_with_motion_ratio": ratio(cc["files_with_motion"], cc["files"]),
                "dynamic_segment_ratio": ratio(cc["dynamic_segments"], cc["segments"]),
                "dynamic_row_ratio": ratio(cc["rows_in_dynamic_segments"], cc["active_rows"]),
                "moving_pair_ratio": ratio(cc["moving_pairs"], cc["adjacent_pairs"]),
            }
        )
    return summary, by_class, source_frame_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/TAU2020_SELD_dataset"),
        help="Dataset root containing metadata_dev/source and metadata_eval/source",
    )
    parser.add_argument("--output-json", type=Path, default=Path("runs/analysis/motion_coverage.json"))
    parser.add_argument(
        "--output-class-csv",
        type=Path,
        default=Path("runs/analysis/motion_coverage_by_class.csv"),
    )
    parser.add_argument(
        "--output-validation-source-frames-csv",
        type=Path,
        default=Path("runs/analysis/motion_source_frames_validation_fold1.csv"),
        help="Frame-level fold-1 validation GT with static/dynamic segment labels",
    )
    parser.add_argument(
        "--output-evaluation-source-frames-csv",
        type=Path,
        default=Path("runs/analysis/motion_source_frames_evaluation.csv"),
        help="Frame-level evaluation GT with static/dynamic segment labels",
    )
    parser.add_argument(
        "--motion-epsilon",
        type=float,
        default=1e-6,
        help="Angular change in degrees above which a pair is counted as moving",
    )
    args = parser.parse_args()

    dev_paths = sorted((args.data_root / "metadata_dev" / "source").glob("*.csv"))
    eval_paths = sorted((args.data_root / "metadata_eval" / "source").glob("*.csv"))
    split_paths = {
        "train_fold2-6": [path for path in dev_paths if path.name.startswith(tuple(f"fold{i}_" for i in range(2, 7)))],
        "validation_fold1": [path for path in dev_paths if path.name.startswith("fold1_")],
        "evaluation": eval_paths,
    }
    for name, paths in split_paths.items():
        if not paths:
            raise FileNotFoundError(f"No metadata CSV files found for {name} under {args.data_root}")

    summaries = []
    class_rows = []
    source_rows_by_split = {}
    for name, paths in split_paths.items():
        summary, rows, source_rows = audit_split(name, paths, args.motion_epsilon)
        summaries.append(summary)
        class_rows.extend(rows)
        source_rows_by_split[name] = source_rows

    payload = {
        "definition": {
            "source_segment": "same (class_id, source_id) within one file, split at frame gaps",
            "dynamic_segment": "contiguous segment with at least one adjacent-frame angular change",
            "moving_pair": "adjacent 100 ms frames with angular change > motion_epsilon_deg",
            "frames_touching_motion": "annotated frames that are either endpoint of a moving pair",
            "motion_epsilon_deg": args.motion_epsilon,
        },
        "splits": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    args.output_class_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(class_rows[0].keys())
    with args.output_class_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(class_rows)

    frame_outputs = {
        args.output_validation_source_frames_csv: source_rows_by_split["validation_fold1"],
        args.output_evaluation_source_frames_csv: source_rows_by_split["evaluation"],
    }
    for output_path, source_rows in frame_outputs.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
            writer.writeheader()
            writer.writerows(source_rows)

    for summary in summaries:
        ratios = summary["ratios"]
        print(
            f"{summary['split']}: files={summary['files']}, "
            f"files_with_motion={summary['files_with_motion']} ({ratios['files_with_motion']:.2%}), "
            f"dynamic_segments={summary['dynamic_segments']}/{summary['segments']} "
            f"({ratios['dynamic_segments']:.2%}), dynamic_rows={summary['rows_in_dynamic_segments']}/"
            f"{summary['active_rows']} ({ratios['rows_in_dynamic_segments']:.2%}), "
            f"moving_pairs={summary['moving_pairs']}/{summary['adjacent_pairs']} "
            f"({ratios['moving_pairs']:.2%}), frames_touching_motion="
            f"{summary['frames_touching_motion']}/{summary['annotated_frames']} "
            f"({ratios['frames_touching_motion']:.2%})"
        )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_class_csv}")
    print(f"wrote {args.output_validation_source_frames_csv}")
    print(f"wrote {args.output_evaluation_source_frames_csv}")


if __name__ == "__main__":
    main()
