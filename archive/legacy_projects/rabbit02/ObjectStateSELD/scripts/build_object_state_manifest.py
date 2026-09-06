#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


FOLD_RE = re.compile(r"fold(\d+)")
ROOM_RE = re.compile(r"room(\d+)")


@dataclass(frozen=True)
class LabelRow:
    frame: int
    class_id: int
    track_id: int
    azimuth: float
    elevation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Object-State SELD v0 continuous-track manifests."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--require-audio",
        action="store_true",
        help="Fail when a metadata recording has no matching WAV file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def direction_xyz(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def split_continuous(rows: list[LabelRow]) -> Iterable[list[LabelRow]]:
    if not rows:
        return
    start = 0
    for index in range(1, len(rows)):
        if rows[index].frame != rows[index - 1].frame + 1:
            yield rows[start:index]
            start = index
    yield rows[start:]


def compute_velocity(
    positions: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(positions)
    raw = np.zeros_like(positions)
    speed = np.zeros(count, dtype=np.float64)
    mask = np.zeros(count, dtype=np.uint8)
    if count < 2:
        return raw, raw.copy(), speed, mask

    raw[0] = (positions[1] - positions[0]) / dt
    raw[-1] = (positions[-1] - positions[-2]) / dt
    speed[0] = angular_distance_deg(positions[0], positions[1]) / dt
    speed[-1] = angular_distance_deg(positions[-2], positions[-1]) / dt
    mask[[0, -1]] = 1
    if count > 2:
        raw[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
        speed[1:-1] = np.asarray(
            [
                angular_distance_deg(positions[index - 1], positions[index + 1])
                / (2.0 * dt)
                for index in range(1, count - 1)
            ],
            dtype=np.float64,
        )
        mask[1:-1] = 1

    radial = np.sum(raw * positions, axis=1, keepdims=True) * positions
    tangent = raw - radial
    return raw, tangent, speed, mask


def motion_bin(speed: float, config: dict[str, Any]) -> str:
    if speed <= float(config["static_max_deg_s"]):
        return "static"
    if speed <= float(config["slow_max_deg_s"]):
        return "slow"
    if speed <= float(config["medium_max_deg_s"]):
        return "medium"
    return "fast"


def compact_vector(values: np.ndarray | list[int]) -> str:
    if isinstance(values, np.ndarray):
        serializable = [round(float(value), 8) for value in values.tolist()]
    else:
        serializable = values
    return json.dumps(serializable, ensure_ascii=True, separators=(",", ":"))


def parse_recording(path: Path) -> tuple[int, str, list[LabelRow]]:
    fold_match = FOLD_RE.search(path.stem)
    room_match = ROOM_RE.search(path.stem)
    if fold_match is None or room_match is None:
        raise ValueError(f"Cannot parse fold/room from {path.name}")
    rows: list[LabelRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 5:
                raise ValueError(f"{path.name}:{line_number}: expected 5 columns")
            label = LabelRow(
                frame=int(row[0]),
                class_id=int(row[1]),
                track_id=int(row[2]),
                azimuth=float(row[3]),
                elevation=float(row[4]),
            )
            if not 0 <= label.class_id <= 13:
                raise ValueError(f"{path.name}:{line_number}: invalid class_id")
            if not -180 <= label.azimuth <= 180 or not -90 <= label.elevation <= 90:
                raise ValueError(f"{path.name}:{line_number}: invalid angle")
            rows.append(label)
    return int(fold_match.group(1)), room_match.group(1), rows


def split_for_fold(fold: int, split_by_fold: dict[str, list[int]]) -> str:
    matches = [name for name, folds in split_by_fold.items() if fold in folds]
    if len(matches) != 1:
        raise ValueError(f"Fold {fold} maps to {matches}, expected exactly one split")
    return matches[0]


def build(config: dict[str, Any], require_audio: bool) -> dict[str, Any]:
    project = config["project"]
    dataset = config["dataset"]
    window = config["window"]
    output_config = config["output"]
    raw_root = Path(project["raw_root"])
    metadata_root = raw_root / dataset["metadata_subdir"]
    audio_root = raw_root / dataset["audio_subdir"]
    intermediate_root = Path(project["intermediate_root"])
    manifest_root = intermediate_root / "manifests"
    track_root = intermediate_root / "parsed_tracks"
    velocity_root = intermediate_root / "velocity_labels"
    for path in (manifest_root, track_root, velocity_root):
        path.mkdir(parents=True, exist_ok=True)

    label_rate = int(dataset["label_rate_hz"])
    sample_rate = int(dataset["sample_rate"])
    dt = 1.0 / label_rate
    samples_per_label_frame = sample_rate // label_rate
    context_frames = int(round(float(window["context_duration_s"]) * label_rate))
    minimum_track_frames = int(
        round(float(window["minimum_track_duration_s"]) * label_rate)
    )
    horizon_frames = [
        int(round(float(horizon) * label_rate))
        for horizon in window["future_horizons_s"]
    ]
    horizon_labels = [int(round(float(horizon) * 1000)) for horizon in window["future_horizons_s"]]
    max_future = max(horizon_frames)
    hop_frames = int(window["hop_frames"])
    require_polyphony_one = bool(window["require_scene_polyphony_one"])

    manifests: dict[str, list[dict[str, Any]]] = {
        split: [] for split in dataset["split_by_fold"]
    }
    tracks: list[dict[str, Any]] = []
    missing_audio: list[str] = []
    metadata_paths = sorted(
        path
        for path in metadata_root.glob("*.csv")
        if not path.name.startswith("._")
    )
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata CSV files found under {metadata_root}")

    for metadata_path in metadata_paths:
        recording_id = metadata_path.stem
        fold, room_id, rows = parse_recording(metadata_path)
        split = split_for_fold(fold, dataset["split_by_fold"])
        audio_path = audio_root / f"{recording_id}.wav"
        if not audio_path.is_file():
            missing_audio.append(str(audio_path))
            if require_audio:
                raise FileNotFoundError(audio_path)

        polyphony = Counter(row.frame for row in rows)
        grouped: dict[tuple[int, int], list[LabelRow]] = defaultdict(list)
        for row in rows:
            grouped[(row.class_id, row.track_id)].append(row)

        for (class_id, track_id), track_rows in sorted(grouped.items()):
            ordered = sorted(track_rows, key=lambda item: item.frame)
            frames = [row.frame for row in ordered]
            if len(frames) != len(set(frames)):
                raise ValueError(
                    f"Duplicate track frame: {recording_id}, class={class_id}, track={track_id}"
                )
            for segment_index, segment in enumerate(split_continuous(ordered)):
                positions = np.vstack(
                    [direction_xyz(row.azimuth, row.elevation) for row in segment]
                )
                raw_velocity, tangent_velocity, speeds, velocity_mask = compute_velocity(
                    positions, dt
                )
                segment_id = (
                    f"{recording_id}_c{class_id:02d}_t{track_id:02d}_s{segment_index:03d}"
                )
                tracks.append(
                    {
                        "segment_id": segment_id,
                        "recording_id": recording_id,
                        "split": split,
                        "fold": fold,
                        "room_id": room_id,
                        "class_id": class_id,
                        "track_id": track_id,
                        "start_frame": segment[0].frame,
                        "end_frame": segment[-1].frame,
                        "frame_count": len(segment),
                        "duration_s": round(len(segment) / label_rate, 3),
                        "mean_speed_deg_s": round(float(np.mean(speeds)), 6),
                        "max_speed_deg_s": round(float(np.max(speeds)), 6),
                    }
                )
                if len(segment) < minimum_track_frames:
                    continue

                first_anchor = context_frames - 1
                final_anchor = len(segment) - max_future - 1
                for anchor_index in range(first_anchor, final_anchor + 1, hop_frames):
                    context_start_index = anchor_index - context_frames + 1
                    sample_end_index = anchor_index + max_future
                    sample_rows = segment[context_start_index : sample_end_index + 1]
                    if require_polyphony_one and any(
                        polyphony[row.frame] != 1 for row in sample_rows
                    ):
                        continue

                    anchor = segment[anchor_index]
                    future_positions = {
                        label: positions[anchor_index + frame_offset]
                        for label, frame_offset in zip(horizon_labels, horizon_frames)
                    }
                    sample_key = (
                        f"{recording_id}|{class_id}|{track_id}|{anchor.frame}|"
                        f"{output_config['preprocess_version']}"
                    )
                    digest = hashlib.sha1(sample_key.encode("utf-8")).hexdigest()[:12]
                    sample_id = f"osse_{recording_id}_a{anchor.frame:03d}_{digest}"
                    context_start_frame = segment[context_start_index].frame
                    manifest_row: dict[str, Any] = {
                        "sample_id": sample_id,
                        "recording_id": recording_id,
                        "audio_path": str(audio_path),
                        "split": split,
                        "fold": fold,
                        "room_id": room_id,
                        "start_sample": context_start_frame * samples_per_label_frame,
                        "end_sample": (anchor.frame + 1) * samples_per_label_frame,
                        "context_start_frame": context_start_frame,
                        "anchor_frame": anchor.frame,
                        "class_id": class_id,
                        "track_id": track_id,
                        "activity_mask": compact_vector([1] * context_frames),
                        "continuity_mask": compact_vector([1] * context_frames),
                        "azimuth": anchor.azimuth,
                        "elevation": anchor.elevation,
                        "position_xyz": compact_vector(positions[anchor_index]),
                        "velocity_raw_xyz": compact_vector(raw_velocity[anchor_index]),
                        "velocity_tangent_xyz": compact_vector(
                            tangent_velocity[anchor_index]
                        ),
                        "speed_deg_s": round(float(speeds[anchor_index]), 8),
                        "velocity_mask": int(velocity_mask[anchor_index]),
                        "motion_bin": motion_bin(
                            float(speeds[anchor_index]), config["motion"]
                        ),
                        "source_dataset": dataset["name"],
                        "preprocess_version": output_config["preprocess_version"],
                        "label_version": output_config["label_version"],
                    }
                    for label in horizon_labels:
                        manifest_row[f"future_position_{label}ms"] = compact_vector(
                            future_positions[label]
                        )
                        manifest_row[f"future_valid_mask_{label}ms"] = 1
                    manifests[split].append(manifest_row)

    track_fieldnames = list(tracks[0]) if tracks else []
    with (track_root / "tracks_v0.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=track_fieldnames)
        writer.writeheader()
        writer.writerows(tracks)

    split_summaries: dict[str, Any] = {}
    for split, rows in manifests.items():
        if not rows:
            split_summaries[split] = {"samples": 0}
            continue
        fieldnames = list(rows[0])
        csv_path = manifest_root / f"{split}_manifest_v0.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if output_config.get("write_npz", True):
            positions = np.asarray(
                [json.loads(row["position_xyz"]) for row in rows], dtype=np.float32
            )
            velocity_raw = np.asarray(
                [json.loads(row["velocity_raw_xyz"]) for row in rows], dtype=np.float32
            )
            velocity_tangent = np.asarray(
                [json.loads(row["velocity_tangent_xyz"]) for row in rows],
                dtype=np.float32,
            )
            future_position = np.asarray(
                [
                    [json.loads(row[f"future_position_{label}ms"]) for label in horizon_labels]
                    for row in rows
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                manifest_root / f"{split}_manifest_v0.npz",
                sample_id=np.asarray([row["sample_id"] for row in rows]),
                recording_id=np.asarray([row["recording_id"] for row in rows]),
                audio_path=np.asarray([row["audio_path"] for row in rows]),
                fold=np.asarray([row["fold"] for row in rows], dtype=np.int16),
                start_sample=np.asarray(
                    [row["start_sample"] for row in rows], dtype=np.int64
                ),
                end_sample=np.asarray([row["end_sample"] for row in rows], dtype=np.int64),
                anchor_frame=np.asarray(
                    [row["anchor_frame"] for row in rows], dtype=np.int16
                ),
                class_id=np.asarray([row["class_id"] for row in rows], dtype=np.int16),
                track_id=np.asarray([row["track_id"] for row in rows], dtype=np.int16),
                position_xyz=positions,
                velocity_raw_xyz=velocity_raw,
                velocity_tangent_xyz=velocity_tangent,
                speed_deg_s=np.asarray(
                    [row["speed_deg_s"] for row in rows], dtype=np.float32
                ),
                velocity_mask=np.asarray(
                    [row["velocity_mask"] for row in rows], dtype=np.uint8
                ),
                future_position=future_position,
                future_valid_mask=np.ones(
                    (len(rows), len(horizon_labels)), dtype=np.uint8
                ),
                future_horizons_ms=np.asarray(horizon_labels, dtype=np.int16),
                motion_bin=np.asarray([row["motion_bin"] for row in rows]),
            )

        split_summaries[split] = {
            "samples": len(rows),
            "recordings": len({row["recording_id"] for row in rows}),
            "tracks": len(
                {
                    (row["recording_id"], row["class_id"], row["track_id"])
                    for row in rows
                }
            ),
            "motion_bins": dict(Counter(row["motion_bin"] for row in rows)),
            "classes": dict(Counter(str(row["class_id"]) for row in rows)),
            "csv": str(csv_path),
        }

    summary = {
        "metadata_files": len(metadata_paths),
        "track_segments": len(tracks),
        "missing_audio_count": len(missing_audio),
        "missing_audio_examples": missing_audio[:20],
        "context_frames": context_frames,
        "future_horizon_frames": horizon_frames,
        "minimum_track_frames": minimum_track_frames,
        "require_scene_polyphony_one": require_polyphony_one,
        "splits": split_summaries,
    }
    with (manifest_root / "manifest_build_summary_v0.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = build(config, require_audio=args.require_audio)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
