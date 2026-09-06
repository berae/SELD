#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "sample_id",
    "recording_id",
    "audio_path",
    "split",
    "fold",
    "room_id",
    "start_sample",
    "end_sample",
    "context_start_frame",
    "anchor_frame",
    "class_id",
    "track_id",
    "position_xyz",
    "velocity_raw_xyz",
    "velocity_tangent_xyz",
    "speed_deg_s",
    "velocity_mask",
    "motion_bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Object-State SELD manifests.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--require-audio",
        action="store_true",
        help="Treat missing WAV files as a hard failure.",
    )
    return parser.parse_args()


def vector_column(series: pd.Series) -> np.ndarray:
    return np.asarray([json.loads(value) for value in series], dtype=np.float64)


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    intermediate_root = Path(config["project"]["intermediate_root"])
    docs_root = Path(config["project"]["docs_root"])
    manifest_root = intermediate_root / "manifests"
    docs_root.mkdir(parents=True, exist_ok=True)
    expected_context_samples = int(
        config["dataset"]["sample_rate"] * config["window"]["context_duration_s"]
    )

    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    split_summary: dict[str, dict[str, Any]] = {}
    recording_sets: dict[str, set[str]] = {}
    all_sample_ids: list[str] = []

    for split in config["dataset"]["split_by_fold"]:
        path = manifest_root / f"{split}_manifest_v0.csv"
        if not path.is_file():
            errors.append(f"Missing manifest: {path}")
            continue
        frame = pd.read_csv(path)
        frames[split] = frame
        missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            errors.append(f"{split}: missing columns {missing_columns}")
            continue
        if frame["sample_id"].duplicated().any():
            errors.append(f"{split}: duplicate sample_id")
        all_sample_ids.extend(frame["sample_id"].astype(str).tolist())
        recording_sets[split] = set(frame["recording_id"].astype(str))

        position = vector_column(frame["position_xyz"])
        tangent = vector_column(frame["velocity_tangent_xyz"])
        position_norm_error = np.abs(np.linalg.norm(position, axis=1) - 1.0)
        tangent_dot = np.abs(np.sum(position * tangent, axis=1))
        bad_context_length = int(
            ((frame["end_sample"] - frame["start_sample"]) != expected_context_samples).sum()
        )
        missing_audio = sorted(
            {path for path in frame["audio_path"].astype(str) if not Path(path).is_file()}
        )
        if args.require_audio and missing_audio:
            errors.append(f"{split}: {len(missing_audio)} missing audio files")
        if float(position_norm_error.max(initial=0.0)) > 1e-5:
            errors.append(f"{split}: position norm error exceeds 1e-5")
        if float(tangent_dot.max(initial=0.0)) > 1e-5:
            errors.append(f"{split}: p dot v_tangent exceeds 1e-5")
        if bad_context_length:
            errors.append(f"{split}: {bad_context_length} invalid context sample spans")

        future_norm_max = 0.0
        for horizon in [100, 300, 500]:
            column = f"future_position_{horizon}ms"
            mask_column = f"future_valid_mask_{horizon}ms"
            if column not in frame or mask_column not in frame:
                errors.append(f"{split}: missing {column} or {mask_column}")
                continue
            future = vector_column(frame[column])
            future_norm_max = max(
                future_norm_max,
                float(np.abs(np.linalg.norm(future, axis=1) - 1.0).max(initial=0.0)),
            )
            if not frame[mask_column].isin([0, 1]).all():
                errors.append(f"{split}: invalid values in {mask_column}")

        split_summary[split] = {
            "samples": int(len(frame)),
            "recordings": int(frame["recording_id"].nunique()),
            "tracks": int(
                frame[["recording_id", "class_id", "track_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "classes": {
                str(key): int(value)
                for key, value in frame["class_id"].value_counts().sort_index().items()
            },
            "motion_bins": {
                str(key): int(value)
                for key, value in frame["motion_bin"].value_counts().items()
            },
            "speed_deg_s": {
                "min": float(frame["speed_deg_s"].min()),
                "median": float(frame["speed_deg_s"].median()),
                "p95": float(frame["speed_deg_s"].quantile(0.95)),
                "max": float(frame["speed_deg_s"].max()),
            },
            "position_norm_error_max": float(position_norm_error.max(initial=0.0)),
            "tangent_dot_abs_max": float(tangent_dot.max(initial=0.0)),
            "future_norm_error_max": future_norm_max,
            "bad_context_length": bad_context_length,
            "missing_audio_count": len(missing_audio),
            "missing_audio_examples": missing_audio[:10],
        }

    if len(all_sample_ids) != len(set(all_sample_ids)):
        errors.append("sample_id collision across splits")
    split_names = sorted(recording_sets)
    split_overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(recording_sets[left] & recording_sets[right])
            split_overlaps[f"{left}-{right}"] = overlap
            if overlap:
                errors.append(f"recording leakage between {left} and {right}")

    track_path = intermediate_root / "parsed_tracks" / "tracks_v0.csv"
    track_summary: dict[str, Any] = {}
    if track_path.is_file():
        tracks = pd.read_csv(track_path)
        track_summary = {
            "segments": int(len(tracks)),
            "continuous_segments_ge_2_5s": int((tracks["duration_s"] >= 2.5).sum()),
            "duration_s": {
                "median": float(tracks["duration_s"].median()),
                "p95": float(tracks["duration_s"].quantile(0.95)),
                "max": float(tracks["duration_s"].max()),
            },
        }
    else:
        errors.append(f"Missing track table: {track_path}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "split_recording_overlap": split_overlaps,
        "tracks": track_summary,
        "splits": split_summary,
    }
    json_path = manifest_root / "data_audit_v0.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    rows = [
        [
            split,
            values["samples"],
            values["recordings"],
            values["tracks"],
            values["missing_audio_count"],
        ]
        for split, values in split_summary.items()
    ]
    report = [
        "# Object-State SELD v0 Data Audit",
        "",
        f"Status: **{result['status']}**",
        "",
        markdown_table(
            rows,
            ["Split", "Samples", "Recordings", "Tracks", "Missing audio"],
        ),
        "",
        "## Track summary",
        "",
        f"- Continuous segments: {track_summary.get('segments', 0)}",
        f"- Segments >= 2.5 s: {track_summary.get('continuous_segments_ge_2_5s', 0)}",
        "",
        "## Geometry and leakage",
        "",
    ]
    for split, values in split_summary.items():
        report.append(
            f"- {split}: max |norm(p)-1|={values['position_norm_error_max']:.3e}; "
            f"max |p dot v_tan|={values['tangent_dot_abs_max']:.3e}; "
            f"max future norm error={values['future_norm_error_max']:.3e}."
        )
    report.append(f"- Recording overlaps: {split_overlaps}")
    report.extend(["", "## Errors", ""])
    report.extend([f"- {error}" for error in errors] or ["- None"])
    report.append("")
    (docs_root / "data_audit_v0.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
