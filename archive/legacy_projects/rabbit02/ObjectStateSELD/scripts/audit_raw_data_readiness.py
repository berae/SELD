from __future__ import annotations

import csv
import json
from pathlib import Path

import soundfile as sf


ROOT = Path("/work/zhanghc/Myllm/SELD/ObjectStateSELD/data_raw/dcase2020")


def audit_split(name: str) -> dict[str, object]:
    audio_dir = ROOT / f"foa_{name}"
    metadata_dir = ROOT / f"metadata_{name}"
    audio_files = sorted(audio_dir.rglob("*.wav"))
    metadata_files = sorted(
        path for path in metadata_dir.rglob("*.csv") if not path.name.startswith("._")
    )

    invalid_rows: list[dict[str, object]] = []
    row_count = 0
    class_ids: set[int] = set()
    min_frame = None
    max_frame = None
    for path in metadata_files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                row_count += 1
                try:
                    if len(row) != 5:
                        raise ValueError(f"expected 5 columns, got {len(row)}")
                    frame, class_id, track_id = map(int, row[:3])
                    azimuth, elevation = map(float, row[3:])
                    if not 0 <= class_id <= 13:
                        raise ValueError(f"class_id={class_id}")
                    if not -180 <= azimuth <= 180:
                        raise ValueError(f"azimuth={azimuth}")
                    if not -90 <= elevation <= 90:
                        raise ValueError(f"elevation={elevation}")
                    if track_id < 0 or frame < 0:
                        raise ValueError("negative frame or track_id")
                    class_ids.add(class_id)
                    min_frame = frame if min_frame is None else min(min_frame, frame)
                    max_frame = frame if max_frame is None else max(max_frame, frame)
                except Exception as exc:  # audit should report all source issues
                    if len(invalid_rows) < 20:
                        invalid_rows.append(
                            {"file": path.name, "line": line_number, "error": str(exc)}
                        )

    invalid_audio: list[dict[str, object]] = []
    for path in audio_files:
        try:
            info = sf.info(str(path))
            if info.channels != 4 or info.samplerate != 24000 or abs(info.duration - 60.0) > 1e-3:
                invalid_audio.append(
                    {
                        "file": path.name,
                        "channels": info.channels,
                        "samplerate": info.samplerate,
                        "duration": info.duration,
                    }
                )
        except Exception as exc:
            invalid_audio.append({"file": path.name, "error": str(exc)})

    audio_stems = {path.stem for path in audio_files}
    metadata_stems = {path.stem for path in metadata_files}
    if name == "dev":
        # Development filenames share the full fold/room/mix stem.
        missing_audio = sorted(metadata_stems - audio_stems)
        missing_metadata = sorted(audio_stems - metadata_stems)
    else:
        # Evaluation audio and released labels both use mixNNN stems.
        missing_audio = sorted(metadata_stems - audio_stems)
        missing_metadata = sorted(audio_stems - metadata_stems)

    return {
        "split": name,
        "audio_files": len(audio_files),
        "metadata_files": len(metadata_files),
        "metadata_rows": row_count,
        "class_ids": sorted(class_ids),
        "frame_range": [min_frame, max_frame],
        "invalid_metadata_rows": invalid_rows,
        "invalid_audio_count": len(invalid_audio),
        "invalid_audio_examples": invalid_audio[:20],
        "missing_audio_count": len(missing_audio),
        "missing_audio_examples": missing_audio[:20],
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_examples": missing_metadata[:20],
    }


print(json.dumps({"dev": audit_split("dev"), "eval": audit_split("eval")}, indent=2))
