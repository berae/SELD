#!/usr/bin/env python3
"""GT-referenced static/dynamic diagnostics for SELD prediction directories.

The evaluator accepts both DCASE polar prediction rows used by EINV2 and the
seven-column Cartesian rows emitted by the Multi-ACCDOA baseline.  References
and predictions are matched jointly per (recording, frame, class) with a
Hungarian angular assignment before each matched/missed reference is attributed
to its source segment's motion group.

Unmatched predictions are intentionally kept global: a false positive has no
ground-truth source identity, so assigning it to static or dynamic would be an
unsupported choice.  These diagnostics therefore complement, rather than
replace, official ER/F/LE/LR/SELD metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


Vector = Tuple[float, float, float]


@dataclass(frozen=True)
class PredictionSpec:
    family: str
    context: str
    variant: str
    seed: int
    prediction_dir: Path
    prediction_format: str
    evidence_level: str
    official_metrics: dict

    @property
    def model_name(self) -> str:
        return f"{self.family}_{self.variant}_seed{self.seed}"


def polar_to_unit(azimuth_deg: float, elevation_deg: float) -> Vector:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return (
        cos_elevation * math.cos(azimuth),
        cos_elevation * math.sin(azimuth),
        math.sin(elevation),
    )


def normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("Prediction contains a zero Cartesian direction vector")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def unit_to_polar(vector: Sequence[float]) -> Tuple[float, float]:
    x, y, z = normalize(vector)
    return math.degrees(math.atan2(y, x)), math.degrees(math.asin(max(-1.0, min(1.0, z))))


def angular_distance_deg(first: Sequence[float], second: Sequence[float]) -> float:
    dot = sum(left * right for left, right in zip(first, second))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def load_gt(source_frames_csv: Path, groups: set[str]) -> Tuple[dict, int]:
    grouped: Dict[Tuple[str, int, int], List[dict]] = defaultdict(list)
    row_count = 0
    with source_frames_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["motion_group"] not in groups:
                continue
            item = {
                "filename": Path(row["metadata_file"]).stem,
                "frame": int(row["frame"]),
                "class_id": int(row["class_id"]),
                "source_id": int(row["source_id"]),
                "segment_id": row["segment_id"],
                "motion_group": row["motion_group"],
                "azimuth_deg": float(row["azimuth_deg"]),
                "elevation_deg": float(row["elevation_deg"]),
            }
            item["vector"] = polar_to_unit(item["azimuth_deg"], item["elevation_deg"])
            grouped[(item["filename"], item["frame"], item["class_id"])].append(item)
            row_count += 1
    if not grouped:
        raise RuntimeError(f"No GT rows for groups {sorted(groups)} in {source_frames_csv}")
    return grouped, row_count


def parse_prediction_row(values: Sequence[str], prediction_format: str) -> Tuple[int, int, dict]:
    frame = int(values[0])
    class_id = int(values[1])
    if prediction_format == "cartesian":
        if len(values) < 6:
            raise ValueError(f"Expected Cartesian row with >=6 columns, got {len(values)}")
        vector = normalize(tuple(float(value) for value in values[3:6]))
        azimuth, elevation = unit_to_polar(vector)
    elif prediction_format == "polar":
        if len(values) == 4:
            azimuth, elevation = float(values[2]), float(values[3])
        elif len(values) >= 5:
            azimuth, elevation = float(values[-2]), float(values[-1])
        else:
            raise ValueError(f"Expected polar row with >=4 columns, got {len(values)}")
        vector = polar_to_unit(azimuth, elevation)
    else:
        raise ValueError(f"Unsupported prediction format: {prediction_format}")
    return frame, class_id, {
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "vector": vector,
    }


def load_predictions(prediction_dir: Path, prediction_format: str) -> Tuple[dict, int, int]:
    grouped: Dict[Tuple[str, int, int], List[dict]] = defaultdict(list)
    file_count = 0
    row_count = 0
    for path in sorted(prediction_dir.glob("*.csv")):
        file_count += 1
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, values in enumerate(csv.reader(handle), start=1):
                try:
                    frame, class_id, item = parse_prediction_row(values, prediction_format)
                except (ValueError, IndexError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                grouped[(path.stem, frame, class_id)].append(item)
                row_count += 1
    if file_count == 0:
        raise RuntimeError(f"No prediction CSV files under {prediction_dir}")
    return grouped, file_count, row_count


def new_bucket() -> dict:
    return {
        "references": 0,
        "matched": 0,
        "within_threshold": 0,
        "errors": [],
    }


def summarize_bucket(bucket: dict) -> dict:
    references = bucket["references"]
    errors = bucket["errors"]
    return {
        "references": references,
        "matched": bucket["matched"],
        "within_threshold": bucket["within_threshold"],
        "matched_recall_micro": bucket["matched"] / references if references else None,
        "recall_at_threshold_micro": (
            bucket["within_threshold"] / references if references else None
        ),
        "mean_localization_error_deg_micro": mean(errors) if errors else None,
        "median_localization_error_deg_micro": median(errors) if errors else None,
        "p90_localization_error_deg_micro": percentile(errors, 0.90),
    }


def evaluate_spec(
    spec: PredictionSpec,
    gt_grouped: dict,
    gt_row_count: int,
    motion_groups: Sequence[str],
    threshold_deg: float,
) -> dict:
    pred_grouped, prediction_files, prediction_rows = load_predictions(
        spec.prediction_dir, spec.prediction_format
    )
    group_buckets = {group: new_bucket() for group in motion_groups}
    class_buckets: Dict[Tuple[str, int], dict] = defaultdict(new_bucket)
    unmatched_predictions = 0

    for key in sorted(set(gt_grouped) | set(pred_grouped)):
        references = gt_grouped.get(key, [])
        predictions = pred_grouped.get(key, [])
        assignments: Dict[int, float] = {}
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
                int(reference_index): float(cost[reference_index, prediction_index])
                for reference_index, prediction_index in zip(reference_indices, prediction_indices)
            }
        unmatched_predictions += max(0, len(predictions) - len(assignments))

        for reference_index, reference in enumerate(references):
            group = reference["motion_group"]
            buckets = (group_buckets[group], class_buckets[(group, reference["class_id"])])
            for bucket in buckets:
                bucket["references"] += 1
            if reference_index not in assignments:
                continue
            error = assignments[reference_index]
            for bucket in buckets:
                bucket["matched"] += 1
                bucket["errors"].append(error)
                if error <= threshold_deg:
                    bucket["within_threshold"] += 1

    motion_summary = {}
    for group in motion_groups:
        summary = summarize_bucket(group_buckets[group])
        per_class = {
            str(class_id): summarize_bucket(bucket)
            for (bucket_group, class_id), bucket in sorted(class_buckets.items())
            if bucket_group == group
        }
        summary["matched_recall_macro_class"] = (
            mean(value["matched_recall_micro"] for value in per_class.values())
            if per_class
            else None
        )
        summary["recall_at_threshold_macro_class"] = (
            mean(value["recall_at_threshold_micro"] for value in per_class.values())
            if per_class
            else None
        )
        class_errors = [
            value["mean_localization_error_deg_micro"]
            for value in per_class.values()
            if value["mean_localization_error_deg_micro"] is not None
        ]
        summary["mean_localization_error_deg_macro_class"] = (
            mean(class_errors) if class_errors else None
        )
        summary["per_class"] = per_class
        motion_summary[group] = summary

    return {
        "model": spec.model_name,
        "family": spec.family,
        "context": spec.context,
        "variant": spec.variant,
        "seed": spec.seed,
        "evidence_level": spec.evidence_level,
        "prediction_dir": str(spec.prediction_dir),
        "prediction_format": spec.prediction_format,
        "prediction_files": prediction_files,
        "prediction_rows": prediction_rows,
        "gt_rows_included": gt_row_count,
        "unmatched_predictions_global": unmatched_predictions,
        "threshold_deg": threshold_deg,
        "official_metrics": spec.official_metrics,
        "motion_groups": motion_summary,
    }


def read_einv2_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        key: float(row[key])
        for key in ("ER20", "F20", "LE20", "LR20", "seld20")
        if row.get(key) not in (None, "")
    }


def read_multiaccdoa_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        key: float(row[key])
        for key in ("ER", "F", "LE", "LR", "SELD")
        if row.get(key) not in (None, "")
    }


def discover_specs(config: dict, multi_root: Path, einv2_root: Path) -> List[PredictionSpec]:
    specs = []
    manifest_glob = config.get("multiaccdoa_manifest_glob")
    if manifest_glob:
        for manifest_path in sorted(multi_root.glob(manifest_glob)):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                continue
            prediction_dir = Path(manifest["prediction_dir"])
            if not prediction_dir.is_dir():
                raise FileNotFoundError(prediction_dir)
            evaluation = manifest.get("evaluation", {})
            official = {
                key: value[0] if isinstance(value, list) else value
                for key, value in evaluation.items()
            }
            variant = manifest["variant"]
            specs.append(
                PredictionSpec(
                    family="Multi-ACCDOA",
                    context="noncausal" if variant.startswith("A") else "causal",
                    variant=variant,
                    seed=int(manifest["seed"]),
                    prediction_dir=prediction_dir,
                    prediction_format="cartesian",
                    evidence_level="formal_three_seed",
                    official_metrics=official,
                )
            )

    for item in config.get("multiaccdoa_predictions", []):
        prediction_dir = multi_root / item["prediction_dir"]
        if not prediction_dir.is_dir():
            raise FileNotFoundError(prediction_dir)
        metrics_path = item.get("official_metrics_csv")
        specs.append(
            PredictionSpec(
                family="Multi-ACCDOA",
                context=item.get("context", "causal"),
                variant=item["variant"],
                seed=int(item["seed"]),
                prediction_dir=prediction_dir,
                prediction_format=item.get("prediction_format", "cartesian"),
                evidence_level=item.get("evidence_level", "validation_pilot"),
                official_metrics=(
                    read_multiaccdoa_metrics(multi_root / metrics_path)
                    if metrics_path else item.get("official_metrics", {})
                ),
            )
        )

    for item in config.get("einv2_predictions", []):
        prediction_dir = einv2_root / item["prediction_dir"]
        metrics_path = einv2_root / item["official_metrics_csv"]
        if not prediction_dir.is_dir():
            raise FileNotFoundError(prediction_dir)
        specs.append(
            PredictionSpec(
                family="EINV2",
                context=item["context"],
                variant=item["variant"],
                seed=int(item["seed"]),
                prediction_dir=prediction_dir,
                prediction_format=item.get("prediction_format", "polar"),
                evidence_level=item.get("evidence_level", "historical_pilot_single_seed"),
                official_metrics=read_einv2_metrics(metrics_path),
            )
        )
    return specs


RUN_METRICS = (
    "matched_recall_micro",
    "recall_at_threshold_micro",
    "mean_localization_error_deg_micro",
    "median_localization_error_deg_micro",
    "p90_localization_error_deg_micro",
    "matched_recall_macro_class",
    "recall_at_threshold_macro_class",
    "mean_localization_error_deg_macro_class",
)


def flatten_runs(results: Sequence[dict], motion_groups: Sequence[str]) -> List[dict]:
    rows = []
    for result in results:
        for group in motion_groups:
            metrics = result["motion_groups"][group]
            rows.append(
                {
                    "family": result["family"],
                    "context": result["context"],
                    "variant": result["variant"],
                    "seed": result["seed"],
                    "evidence_level": result["evidence_level"],
                    "motion_group": group,
                    "prediction_files": result["prediction_files"],
                    "prediction_rows": result["prediction_rows"],
                    "unmatched_predictions_global": result["unmatched_predictions_global"],
                    "references": metrics["references"],
                    **{metric: metrics[metric] for metric in RUN_METRICS},
                    "prediction_dir": result["prediction_dir"],
                }
            )
    return rows


def aggregate_rows(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["context"], row["variant"], row["motion_group"])].append(row)
    output = []
    for key, group_rows in sorted(grouped.items()):
        item = {
            "family": key[0],
            "context": key[1],
            "variant": key[2],
            "motion_group": key[3],
            "n_seeds": len(group_rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group_rows, key=lambda row: row["seed"])),
        }
        for metric in RUN_METRICS:
            values = [float(row[metric]) for row in group_rows if row[metric] is not None]
            item[f"{metric}_mean"] = mean(values) if values else None
            item[f"{metric}_sample_std"] = stdev(values) if len(values) > 1 else None
        output.append(item)
    return output


BASELINES = {
    ("Multi-ACCDOA", "noncausal"): "A0",
    ("Multi-ACCDOA", "causal"): "C0",
    ("EINV2", "causal"): "C0.1",
    ("EINV2", "noncausal"): "E0",
}


def paired_deltas(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    lookup = {
        (row["family"], row["context"], row["variant"], row["seed"], row["motion_group"]): row
        for row in rows
    }
    deltas = []
    for row in rows:
        baseline_variant = BASELINES[(row["family"], row["context"])]
        if row["variant"] == baseline_variant:
            continue
        baseline = lookup.get(
            (row["family"], row["context"], baseline_variant, row["seed"], row["motion_group"])
        )
        if baseline is None:
            continue
        deltas.append(
            {
                "family": row["family"],
                "context": row["context"],
                "variant": row["variant"],
                "baseline": baseline_variant,
                "seed": row["seed"],
                "motion_group": row["motion_group"],
                **{
                    f"delta_{metric}": float(row[metric]) - float(baseline[metric])
                    for metric in RUN_METRICS
                    if row[metric] is not None and baseline[metric] is not None
                },
            }
        )

    grouped: Dict[Tuple[str, str, str, str, str], List[dict]] = defaultdict(list)
    for row in deltas:
        grouped[
            (row["family"], row["context"], row["variant"], row["baseline"], row["motion_group"])
        ].append(row)
    aggregate = []
    for key, group_rows in sorted(grouped.items()):
        item = {
            "family": key[0],
            "context": key[1],
            "variant": key[2],
            "baseline": key[3],
            "motion_group": key[4],
            "n_seed_pairs": len(group_rows),
        }
        for metric in RUN_METRICS:
            delta_key = f"delta_{metric}"
            values = [float(row[delta_key]) for row in group_rows if delta_key in row]
            item[f"{delta_key}_mean"] = mean(values) if values else None
            item[f"{delta_key}_sample_std"] = stdev(values) if len(values) > 1 else None
        aggregate.append(item)
    return deltas, aggregate


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/motion_stratified_evaluation.json"))
    parser.add_argument("--multi-root", type=Path, default=Path.cwd())
    parser.add_argument("--einv2-root", type=Path, required=True)
    parser.add_argument(
        "--source-frames-csv",
        type=Path,
        default=Path("runs/analysis/motion_source_frames_evaluation.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/analysis/motion_stratified"))
    parser.add_argument("--motion-groups", default="static,dynamic")
    parser.add_argument("--threshold-deg", type=float, default=20.0)
    parser.add_argument(
        "--expected-prediction-files",
        type=int,
        default=200,
        help="Expected files per prediction set; use 100 for TAU2020 fold-1 validation",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    motion_groups = [group.strip() for group in args.motion_groups.split(",") if group.strip()]
    gt_grouped, gt_row_count = load_gt(args.source_frames_csv, set(motion_groups))
    specs = discover_specs(config, args.multi_root, args.einv2_root)
    if not specs:
        raise RuntimeError("No prediction specifications found")

    results = []
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.model_name}", flush=True)
        result = evaluate_spec(spec, gt_grouped, gt_row_count, motion_groups, args.threshold_deg)
        if result["prediction_files"] != args.expected_prediction_files:
            raise RuntimeError(
                f"{spec.model_name}: expected {args.expected_prediction_files} prediction files, "
                f"found {result['prediction_files']}"
            )
        results.append(result)

    run_rows = flatten_runs(results, motion_groups)
    aggregate = aggregate_rows(run_rows)
    delta_rows, delta_aggregate = paired_deltas(run_rows)
    payload = {
        "definition": {
            "matching": "global Hungarian angular assignment per (file, frame, class)",
            "motion_attribution": "matched and missed references inherit their contiguous source segment group",
            "false_positives": "reported globally because unmatched predictions have no GT motion identity",
            "official_metric_replacement": False,
            "threshold_deg": args.threshold_deg,
            "motion_groups": motion_groups,
        },
        "source_frames_csv": str(args.source_frames_csv),
        "run_count": len(results),
        "runs": results,
        "aggregate": aggregate,
        "paired_delta_aggregate": delta_aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "motion_stratified_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "motion_stratified_runs.csv", run_rows)
    write_csv(args.output_dir / "motion_stratified_aggregate.csv", aggregate)
    write_csv(args.output_dir / "motion_stratified_paired_deltas.csv", delta_rows)
    write_csv(args.output_dir / "motion_stratified_paired_delta_aggregate.csv", delta_aggregate)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
