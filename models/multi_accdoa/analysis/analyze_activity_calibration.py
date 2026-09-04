#!/usr/bin/env python3
"""Diagnose whether auxiliary variants mainly shift ACCDOA activity calibration.

This is a validation-only diagnostic. It computes class activity after collapsing
the three output tracks and the six ADPIT target tracks; it does not replace the
official location-aware SELD evaluation.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

import cls_data_generator
import parameters
import seldnet_model
from experiment_matrix import TASK_IDS, VARIANTS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT, "runs", "analysis", "paper_v1_activity_thresholds.csv"),
    )
    return parser.parse_args()


def manifest_for(variant, seed):
    pattern = os.path.join(
        PROJECT, "runs", "manifests", "*_paper_v1_{}_seed{}_*.json".format(variant, seed)
    )
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise RuntimeError("expected one manifest for {} seed {}, found {}".format(variant, seed, matches))
    with open(matches[0], encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "completed":
        raise RuntimeError("run is not completed: {}".format(matches[0]))
    return manifest


def evaluate_run(variant, seed, thresholds, device):
    task_id = TASK_IDS[variant]["full"]
    params = parameters.get_params(task_id)
    params["seed"] = seed
    data = cls_data_generator.DataGenerator(params, split=1, shuffle=False, per_file=True)
    input_shape, output_shape = data.get_data_sizes()
    model = seldnet_model.SeldModel(input_shape, output_shape, params).to(device)
    manifest = manifest_for(variant, seed)
    model.load_state_dict(torch.load(manifest["checkpoint"], map_location=device))
    model.eval()

    counts = {threshold: {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "ref": 0} for threshold in thresholds}
    with torch.no_grad():
        for features, labels in data.generate():
            features = torch.as_tensor(features, dtype=torch.float32, device=device)
            output = model(features)
            doa = output["doa"] if isinstance(output, dict) else output
            batch, frames, _ = doa.shape
            classes = params["unique_classes"]
            track_xyz = doa.reshape(batch, frames, 3, 3, classes)
            magnitudes = torch.linalg.vector_norm(track_xyz, dim=3)
            target = torch.as_tensor(labels, device=device)
            reference_active = target[:, :, :, 0, :].gt(0.5).any(dim=2)

            for threshold in thresholds:
                predicted_active = magnitudes.gt(threshold).any(dim=2)
                tp = (predicted_active & reference_active).sum().item()
                fp = (predicted_active & ~reference_active).sum().item()
                fn = (~predicted_active & reference_active).sum().item()
                counts[threshold]["tp"] += tp
                counts[threshold]["fp"] += fp
                counts[threshold]["fn"] += fn
                counts[threshold]["pred"] += predicted_active.sum().item()
                counts[threshold]["ref"] += reference_active.sum().item()

    rows = []
    for threshold in thresholds:
        item = counts[threshold]
        precision = item["tp"] / max(1, item["tp"] + item["fp"])
        recall = item["tp"] / max(1, item["tp"] + item["fn"])
        f_score = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append({
            "variant": variant,
            "seed": seed,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            **item,
        })
    return rows


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            run_rows = evaluate_run(variant, seed, args.thresholds, device)
            rows.extend(run_rows)
            best = max(run_rows, key=lambda row: row["f_score"])
            print(
                "{} seed {}: best threshold={:.2f}, F={:.4f}, P={:.4f}, R={:.4f}".format(
                    variant, seed, best["threshold"], best["f_score"],
                    best["precision"], best["recall"],
                )
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
