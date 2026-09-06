from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset[dict[str, Any]]):
    """Read fixed-length FOA excerpts and Object-State targets from a manifest."""

    def __init__(self, manifest_path: str | Path, limit: int | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"Manifest has no rows: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        start = int(row["start_sample"])
        stop = int(row["end_sample"])
        waveform, sample_rate = sf.read(
            row["audio_path"],
            start=start,
            stop=stop,
            dtype="float32",
            always_2d=True,
        )
        if sample_rate != 24000:
            raise ValueError(f"Expected 24 kHz audio, got {sample_rate}")
        if waveform.shape != (stop - start, 4):
            raise ValueError(
                f"Bad excerpt shape {waveform.shape} for {row['sample_id']}"
            )
        position = np.asarray(json.loads(row["position_xyz"]), dtype=np.float32)
        return {
            "sample_id": row["sample_id"],
            "waveform": torch.from_numpy(waveform.T.copy()),
            "class_id": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "activity": torch.tensor(1.0, dtype=torch.float32),
            "position_xyz": torch.from_numpy(position),
        }
