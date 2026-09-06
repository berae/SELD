#!/usr/bin/env python3
"""Create verified five-column STARSS metadata for EINV2 preprocessing.

STARSS22 already has five columns.  STARSS23 adds distance as column six, but
the DCASE2022 EINV2 utility interprets six-column files as Cartesian output.
Canonicalizing both datasets to the first five polar columns prevents that
silent format collision while preserving the original metadata unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(source_root: Path, destination_root: Path, dataset_name: str) -> dict:
    source_files = sorted(source_root.glob("**/*.csv"))
    if not source_files:
        raise RuntimeError(f"No CSV files under {source_root}")
    if destination_root.exists() and any(destination_root.rglob("*.csv")):
        raise FileExistsError(
            f"Destination already contains CSV files: {destination_root}. "
            "Use a new directory; this script never overwrites prepared labels."
        )

    column_histogram: Counter[int] = Counter()
    class_ids: set[int] = set()
    source_ids: set[int] = set()
    total_rows = 0
    output_manifest = []
    for source_path in source_files:
        relative = source_path.relative_to(source_root)
        destination_path = destination_root / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with source_path.open("r", encoding="utf-8", newline="") as source_handle, destination_path.open(
            "x", encoding="utf-8", newline=""
        ) as destination_handle:
            reader = csv.reader(source_handle)
            writer = csv.writer(destination_handle, lineterminator="\n")
            for line_number, values in enumerate(reader, start=1):
                if len(values) < 5:
                    raise ValueError(
                        f"{source_path}:{line_number}: expected >=5 columns, got {len(values)}"
                    )
                column_histogram[len(values)] += 1
                canonical = values[:5]
                int(canonical[0])
                class_ids.add(int(canonical[1]))
                source_ids.add(int(canonical[2]))
                float(canonical[3])
                float(canonical[4])
                writer.writerow(canonical)
                row_count += 1
        total_rows += row_count
        output_manifest.append(
            {
                "relative_path": relative.as_posix(),
                "rows": row_count,
                "sha256": sha256(destination_path),
            }
        )

    manifest = {
        "dataset": dataset_name,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "files": len(source_files),
        "rows": total_rows,
        "source_column_histogram": {str(key): value for key, value in sorted(column_histogram.items())},
        "canonical_columns": ["frame", "class_id", "source_id", "azimuth_deg", "elevation_deg"],
        "class_ids": sorted(class_ids),
        "source_ids": sorted(source_ids),
        "outputs": output_manifest,
    }
    manifest_path = destination_root.parent / f"{destination_root.name}_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, choices=["STARSS22", "STARSS23"])
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = canonicalize(args.source_root, args.destination_root, args.dataset_name)
    print(json.dumps({key: value for key, value in result.items() if key != "outputs"}, ensure_ascii=False, indent=2))

