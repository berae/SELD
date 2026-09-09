#!/usr/bin/env python3
"""Summarize completed paper_v1 manifests without reading training logs."""

import csv
import glob
import json
import os
import statistics


PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_DIR = os.path.join(PROJECT, "runs", "manifests")
OUTPUT_CSV = os.path.join(PROJECT, "runs", "paper_v1_results.csv")
VARIANTS = ("A0", "A1", "A2", "A3", "C0", "C1", "C2", "C3")
SEEDS = (2026, 2027, 2028)
METRICS = ("ER", "F", "LE", "LR", "SELD")


def point_estimate(value):
    """DCASE jackknife values are [point, [low, high]]."""
    if isinstance(value, list):
        return float(value[0])
    return float(value)


def load_runs():
    runs = {}
    for path in glob.glob(os.path.join(MANIFEST_DIR, "*.json")):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        job_id = record.get("job_id", "")
        if record.get("status") != "completed" or not job_id.startswith("paper_v1_"):
            continue
        key = (record["variant"], int(record["seed"]))
        if key in runs:
            raise RuntimeError("duplicate completed manifest for {}".format(key))
        record["manifest_path"] = path
        runs[key] = record
    return runs


def main():
    runs = load_runs()
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            record = runs.get((variant, seed))
            row = {"variant": variant, "seed": seed, "status": "missing"}
            if record:
                row.update({
                    "status": "completed",
                    "git_commit": record.get("git", {}).get("commit"),
                    "checkpoint": record.get("checkpoint"),
                    "manifest": record["manifest_path"],
                })
                row.update({metric: point_estimate(record["evaluation"][metric]) for metric in METRICS})
            rows.append(row)

    fieldnames = [
        "variant", "seed", "status", *METRICS,
        "git_commit", "checkpoint", "manifest",
    ]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("per-seed results: {}".format(OUTPUT_CSV))
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant and row["status"] == "completed"]
        if not variant_rows:
            print("{}: 0/3 complete".format(variant))
            continue
        summaries = []
        for metric in METRICS:
            values = [row[metric] for row in variant_rows]
            std = statistics.stdev(values) if len(values) > 1 else float("nan")
            summaries.append("{}={:.4f}±{:.4f}".format(metric, statistics.mean(values), std))
        print("{}: {}/3 {}".format(variant, len(variant_rows), " ".join(summaries)))


if __name__ == "__main__":
    main()
