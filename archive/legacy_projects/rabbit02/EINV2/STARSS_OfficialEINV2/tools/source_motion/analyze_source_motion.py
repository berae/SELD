#!/usr/bin/env python3
"""Derive directional static/dynamic source statistics from STARSS metadata.

The STARSS metadata does not provide an official static/dynamic label.  This
script therefore derives a reproducible *directional-motion* label from the
annotated azimuth/elevation trajectory.  It intentionally ignores radial-only
motion because the current velocity auxiliary target is the derivative of the
unit Cartesian DOA vector, not physical 3-D velocity.

Outputs are intended both for dataset auditing and for later GT-referenced
static/dynamic evaluation.  The primary analysis should compare only the
``static`` and ``dynamic`` groups; ``borderline`` and ``insufficient`` are kept
explicitly rather than silently forced into either group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


FRAME_HOP_SECONDS = 0.1


@dataclass
class EventRow:
    frame: int
    class_id: int
    source_id: int
    azimuth_deg: float
    elevation_deg: float
    distance_cm: float | None
    vector: tuple[float, float, float]


def polar_to_unit(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return (
        cos_elevation * math.cos(azimuth),
        cos_elevation * math.sin(azimuth),
        math.sin(elevation),
    )


def angular_distance_deg(
    first: Sequence[float], second: Sequence[float]
) -> float:
    dot = sum(a * b for a, b in zip(first, second))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean_direction(vectors: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    summed = [sum(vector[index] for vector in vectors) for index in range(3)]
    norm = math.sqrt(sum(value * value for value in summed))
    if norm < 1e-12:
        return tuple(vectors[0])  # Extremely dispersed trajectory; any anchor is fine.
    return tuple(value / norm for value in summed)


def read_metadata(path: Path) -> list[EventRow]:
    rows: list[EventRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle), start=1):
            if len(values) < 5:
                raise ValueError(f"{path}:{line_number}: expected >=5 columns, got {len(values)}")
            frame = int(values[0])
            class_id = int(values[1])
            source_id = int(values[2])
            azimuth = float(values[3])
            elevation = float(values[4])
            distance = float(values[5]) if len(values) >= 6 and values[5] != "" else None
            rows.append(
                EventRow(
                    frame=frame,
                    class_id=class_id,
                    source_id=source_id,
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    distance_cm=distance,
                    vector=polar_to_unit(azimuth, elevation),
                )
            )
    return rows


def build_contiguous_trajectories(rows: Iterable[EventRow]) -> tuple[list[list[EventRow]], int]:
    """Build trajectories per (class, source), splitting gaps and duplicates.

    Source identifiers are recording-local.  A greedy nearest-direction link is
    used only when duplicate rows with the same class/source occur in one frame.
    """

    grouped: dict[tuple[int, int], list[EventRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.class_id, row.source_id)].append(row)

    trajectories: list[list[EventRow]] = []
    duplicate_key_rows = 0
    for group_rows in grouped.values():
        by_frame: dict[int, list[EventRow]] = defaultdict(list)
        for row in group_rows:
            by_frame[row.frame].append(row)
        duplicate_key_rows += sum(max(0, len(frame_rows) - 1) for frame_rows in by_frame.values())

        active: list[list[EventRow]] = []
        for frame in sorted(by_frame):
            frame_rows = by_frame[frame]
            candidates = [index for index, chain in enumerate(active) if chain[-1].frame == frame - 1]
            unused_candidates = set(candidates)
            for row in frame_rows:
                if unused_candidates:
                    best = min(
                        unused_candidates,
                        key=lambda index: angular_distance_deg(active[index][-1].vector, row.vector),
                    )
                    active[best].append(row)
                    unused_candidates.remove(best)
                else:
                    active.append([row])
        trajectories.extend(active)
    return trajectories, duplicate_key_rows


def classify_trajectory(
    trajectory: Sequence[EventRow],
    static_radius_deg: float,
    dynamic_radius_deg: float,
    moving_step_deg: float,
    min_moving_steps: int,
) -> tuple[str, dict[str, float | int]]:
    vectors = [row.vector for row in trajectory]
    steps = [
        angular_distance_deg(previous.vector, current.vector)
        for previous, current in zip(trajectory, trajectory[1:])
        if current.frame == previous.frame + 1
    ]
    speeds = [step / FRAME_HOP_SECONDS for step in steps]
    center = mean_direction(vectors)
    deviations = [angular_distance_deg(center, vector) for vector in vectors]
    moving_steps = sum(step >= moving_step_deg for step in steps)
    stats: dict[str, float | int] = {
        "frames": len(trajectory),
        "duration_s": len(trajectory) * FRAME_HOP_SECONDS,
        "consecutive_transitions": len(steps),
        "moving_transitions": moving_steps,
        "max_deviation_from_mean_deg": max(deviations, default=0.0),
        "path_angle_deg": sum(steps),
        "net_angle_deg": angular_distance_deg(vectors[0], vectors[-1]),
        "mean_speed_deg_s": mean(speeds) if speeds else 0.0,
        "p95_speed_deg_s": percentile(speeds, 0.95),
        "max_speed_deg_s": max(speeds, default=0.0),
    }

    if len(trajectory) < 3 or len(steps) < 2:
        category = "insufficient"
    elif (
        stats["max_deviation_from_mean_deg"] >= dynamic_radius_deg
        and (
            moving_steps >= min_moving_steps
            or stats["max_speed_deg_s"] >= 50.0
        )
    ):
        category = "dynamic"
    elif (
        stats["max_deviation_from_mean_deg"] <= static_radius_deg
        and stats["p95_speed_deg_s"] <= moving_step_deg / FRAME_HOP_SECONDS
    ):
        category = "static"
    else:
        category = "borderline"
    return category, stats


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict:
    metadata_root = args.dataset_root / "raw" / "metadata_dev"
    if not metadata_root.is_dir():
        raise FileNotFoundError(metadata_root)
    metadata_files = sorted(metadata_root.glob("**/*.csv"))
    if not metadata_files:
        raise RuntimeError(f"No metadata CSV files under {metadata_root}")

    trajectory_rows: list[dict] = []
    frame_rows: list[dict] = []
    duplicate_key_rows_total = 0
    class_ids: set[int] = set()
    overlap_histogram: Counter[int] = Counter()

    for metadata_path in metadata_files:
        relative = metadata_path.relative_to(metadata_root)
        split = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        file_id = metadata_path.stem
        rows = read_metadata(metadata_path)
        class_ids.update(row.class_id for row in rows)
        frame_counts = Counter(row.frame for row in rows)
        overlap_histogram.update(frame_counts.values())
        trajectories, duplicate_key_rows = build_contiguous_trajectories(rows)
        duplicate_key_rows_total += duplicate_key_rows

        for trajectory_index, trajectory in enumerate(trajectories):
            category, stats = classify_trajectory(
                trajectory,
                static_radius_deg=args.static_radius_deg,
                dynamic_radius_deg=args.dynamic_radius_deg,
                moving_step_deg=args.moving_step_deg,
                min_moving_steps=args.min_moving_steps,
            )
            first = trajectory[0]
            segment_id = (
                f"{relative.as_posix()}::c{first.class_id}::s{first.source_id}::"
                f"seg{trajectory_index:04d}"
            )
            trajectory_rows.append(
                {
                    "dataset": args.dataset_name,
                    "split": split,
                    "metadata_file": relative.as_posix(),
                    "segment_id": segment_id,
                    "class_id": first.class_id,
                    "source_id": first.source_id,
                    "start_frame": first.frame,
                    "end_frame": trajectory[-1].frame,
                    "motion_group": category,
                    **stats,
                }
            )

            previous: EventRow | None = None
            for event in trajectory:
                velocity_valid = int(previous is not None and event.frame == previous.frame + 1)
                if velocity_valid:
                    velocity = tuple(
                        (current_value - previous_value) / FRAME_HOP_SECONDS
                        for current_value, previous_value in zip(event.vector, previous.vector)
                    )
                    angular_step = angular_distance_deg(previous.vector, event.vector)
                else:
                    velocity = (0.0, 0.0, 0.0)
                    angular_step = 0.0
                frame_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "split": split,
                        "metadata_file": relative.as_posix(),
                        "segment_id": segment_id,
                        "frame": event.frame,
                        "class_id": event.class_id,
                        "source_id": event.source_id,
                        "azimuth_deg": event.azimuth_deg,
                        "elevation_deg": event.elevation_deg,
                        "distance_cm": "" if event.distance_cm is None else event.distance_cm,
                        "motion_group": category,
                        "velocity_valid": velocity_valid,
                        "moving_frame": int(velocity_valid and angular_step >= args.moving_step_deg),
                        "angular_step_deg": angular_step,
                        "angular_speed_deg_s": angular_step / FRAME_HOP_SECONDS,
                        "x": event.vector[0],
                        "y": event.vector[1],
                        "z": event.vector[2],
                        "vx": velocity[0],
                        "vy": velocity[1],
                        "vz": velocity[2],
                    }
                )
                previous = event

    trajectory_fields = list(trajectory_rows[0].keys())
    frame_fields = list(frame_rows[0].keys())
    write_csv(args.output_dir / "source_trajectories.csv", trajectory_rows, trajectory_fields)
    write_csv(args.output_dir / "source_frames.csv", frame_rows, frame_fields)

    category_counts = Counter(row["motion_group"] for row in trajectory_rows)
    category_frames = Counter()
    for row in trajectory_rows:
        category_frames[row["motion_group"]] += int(row["frames"])

    by_split_class: dict[tuple[str, int, str], dict[str, float | int | str]] = {}
    for row in trajectory_rows:
        key = (str(row["split"]), int(row["class_id"]), str(row["motion_group"]))
        entry = by_split_class.setdefault(
            key,
            {
                "dataset": args.dataset_name,
                "split": key[0],
                "class_id": key[1],
                "motion_group": key[2],
                "source_trajectories": 0,
                "active_source_frames": 0,
                "duration_s": 0.0,
            },
        )
        entry["source_trajectories"] = int(entry["source_trajectories"]) + 1
        entry["active_source_frames"] = int(entry["active_source_frames"]) + int(row["frames"])
        entry["duration_s"] = float(entry["duration_s"]) + float(row["duration_s"])
    class_summary_rows = [by_split_class[key] for key in sorted(by_split_class)]
    write_csv(
        args.output_dir / "motion_summary_by_split_class.csv",
        class_summary_rows,
        list(class_summary_rows[0].keys()),
    )

    summary = {
        "dataset": args.dataset_name,
        "metadata_root": str(metadata_root),
        "metadata_files": len(metadata_files),
        "metadata_rows": len(frame_rows),
        "class_ids": sorted(class_ids),
        "source_trajectory_definition": "recording-local contiguous (class_id, source_id) segment",
        "motion_definition": "directional motion on unit Cartesian DOA; radial-only motion ignored",
        "thresholds": {
            "frame_hop_seconds": FRAME_HOP_SECONDS,
            "static_radius_deg": args.static_radius_deg,
            "dynamic_radius_deg": args.dynamic_radius_deg,
            "moving_step_deg": args.moving_step_deg,
            "min_moving_steps": args.min_moving_steps,
        },
        "trajectory_counts": dict(sorted(category_counts.items())),
        "active_source_frame_counts": dict(sorted(category_frames.items())),
        "overlap_frame_histogram": {str(key): value for key, value in sorted(overlap_histogram.items())},
        "duplicate_same_frame_class_source_rows": duplicate_key_rows_total,
        "primary_comparison_groups": ["static", "dynamic"],
        "excluded_from_primary_comparison": ["borderline", "insufficient"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "motion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, choices=["STARSS22", "STARSS23"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-radius-deg", type=float, default=1.0)
    parser.add_argument("--dynamic-radius-deg", type=float, default=2.5)
    parser.add_argument("--moving-step-deg", type=float, default=1.0)
    parser.add_argument("--min-moving-steps", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    result = analyze(parsed_args)
    print(json.dumps(result, ensure_ascii=False, indent=2))

