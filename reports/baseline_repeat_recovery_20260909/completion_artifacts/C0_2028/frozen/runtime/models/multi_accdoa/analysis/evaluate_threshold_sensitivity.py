#!/usr/bin/env python3
"""Run official SELD scoring across ACCDOA activity thresholds.

The checkpoint is evaluated once per split. Its raw Multi-ACCDOA output is then
decoded at every requested threshold using the same track-unification logic as
the training entry point. Threshold selection must use validation results only.
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
import cls_feature_class
import parameters
import seldnet_model
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from experiment_matrix import TASK_IDS, VARIANTS
from train_seldnet import determine_similar_location


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument(
        "--manifest-paths", nargs="+",
        help="Evaluate explicit completed manifests, preserving each job_id in output paths",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--split", choices=("validation", "evaluation"), default="validation")
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    if bool(args.variant) == bool(args.manifest_paths):
        parser.error("specify exactly one of --variant or --manifest-paths")
    if args.manifest_paths and not args.output_csv:
        parser.error("--output-csv is required with --manifest-paths")
    return args


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


def get_tracks(doa_output, classes, threshold):
    x0 = doa_output[:, :, :classes]
    y0 = doa_output[:, :, classes:2 * classes]
    z0 = doa_output[:, :, 2 * classes:3 * classes]
    x1 = doa_output[:, :, 3 * classes:4 * classes]
    y1 = doa_output[:, :, 4 * classes:5 * classes]
    z1 = doa_output[:, :, 5 * classes:6 * classes]
    x2 = doa_output[:, :, 6 * classes:7 * classes]
    y2 = doa_output[:, :, 7 * classes:8 * classes]
    z2 = doa_output[:, :, 8 * classes:]
    tracks = []
    for x, y, z, start, end in (
        (x0, y0, z0, 0, 3 * classes),
        (x1, y1, z1, 3 * classes, 6 * classes),
        (x2, y2, z2, 6 * classes, 9 * classes),
    ):
        sed = reshape_3Dto2D(np.sqrt(x ** 2 + y ** 2 + z ** 2) > threshold)
        doa = reshape_3Dto2D(doa_output[:, :, start:end])
        tracks.append((sed, doa))
    return tracks


def append_track(output, frame, class_index, doa, classes):
    output.setdefault(frame, []).append([
        class_index,
        doa[class_index],
        doa[class_index + classes],
        doa[class_index + 2 * classes],
    ])


def decode(doa_output, params, threshold):
    classes = params["unique_classes"]
    (sed0, doa0), (sed1, doa1), (sed2, doa2) = get_tracks(doa_output, classes, threshold)
    output = {}
    for frame in range(sed0.shape[0]):
        for class_index in range(classes):
            similar01 = determine_similar_location(
                sed0[frame][class_index], sed1[frame][class_index],
                doa0[frame], doa1[frame], class_index, params["thresh_unify"], classes,
            )
            similar12 = determine_similar_location(
                sed1[frame][class_index], sed2[frame][class_index],
                doa1[frame], doa2[frame], class_index, params["thresh_unify"], classes,
            )
            similar20 = determine_similar_location(
                sed2[frame][class_index], sed0[frame][class_index],
                doa2[frame], doa0[frame], class_index, params["thresh_unify"], classes,
            )
            similar_count = similar01 + similar12 + similar20
            if similar_count == 0:
                for sed, doa in ((sed0, doa0), (sed1, doa1), (sed2, doa2)):
                    if sed[frame][class_index] > 0.5:
                        append_track(output, frame, class_index, doa[frame], classes)
            elif similar_count == 1:
                if similar01:
                    if sed2[frame][class_index] > 0.5:
                        append_track(output, frame, class_index, doa2[frame], classes)
                    append_track(output, frame, class_index, (doa0[frame] + doa1[frame]) / 2, classes)
                elif similar12:
                    if sed0[frame][class_index] > 0.5:
                        append_track(output, frame, class_index, doa0[frame], classes)
                    append_track(output, frame, class_index, (doa1[frame] + doa2[frame]) / 2, classes)
                else:
                    if sed1[frame][class_index] > 0.5:
                        append_track(output, frame, class_index, doa1[frame], classes)
                    append_track(output, frame, class_index, (doa2[frame] + doa0[frame]) / 2, classes)
            else:
                append_track(
                    output, frame, class_index,
                    (doa0[frame] + doa1[frame] + doa2[frame]) / 3, classes,
                )
    return output


def prediction_dir(run_label, seed, split, threshold):
    threshold_name = "{:03d}".format(round(threshold * 100))
    seed_suffix = "_seed{}".format(seed)
    if not run_label.endswith(seed_suffix):
        run_label = "{}{}".format(run_label, seed_suffix)
    return os.path.join(
        PROJECT, "runs", "analysis", "threshold_predictions",
        "{}_{}_t{}".format(run_label, split, threshold_name),
    )


def evaluate_seed(variant, seed, split, thresholds, device, manifest=None, run_label=None, output_root=None):
    if manifest is None:
        manifest = manifest_for(variant, seed)
        params = parameters.get_params(TASK_IDS[variant]["full"])
    else:
        if manifest.get("status") not in ("completed", "validation_completed"):
            raise RuntimeError("run is not completed: {}".format(manifest.get("job_id", "unknown")))
        params = dict(manifest["params"])
    params["seed"] = seed
    is_eval = split == "evaluation"
    split_id = [0] if is_eval else [1]
    data = cls_data_generator.DataGenerator(
        params, split=split_id, shuffle=False, per_file=True, is_eval=is_eval
    )
    input_shape, output_shape = data.get_data_sizes()
    if output_shape is None:
        output_shape = (params["batch_size"], params["label_sequence_length"], 9 * params["unique_classes"])
    model = seldnet_model.SeldModel(input_shape, output_shape, params).to(device)
    model.load_state_dict(torch.load(manifest["checkpoint"], map_location=device))
    model.eval()

    run_label = run_label or variant
    directories = {threshold: (os.path.join(str(output_root), 'predictions_t{:03d}'.format(round(threshold * 100)))
                               if output_root is not None else prediction_dir(run_label, seed, split, threshold))
                   for threshold in thresholds}
    for path in directories.values():
        os.makedirs(path, exist_ok=False)

    file_names = data.get_filelist()
    file_index = 0
    with torch.no_grad():
        for batch in data.generate():
            features = batch if is_eval else batch[0]
            features = torch.as_tensor(features, dtype=torch.float32, device=device)
            output = model(features)
            doa = output["doa"] if isinstance(output, dict) else output
            doa = doa.detach().cpu().numpy()
            output_name = file_names[file_index].replace(".npy", ".csv")
            file_index += 1
            for threshold, path in directories.items():
                data.write_output_format_file(
                    os.path.join(path, output_name), decode(doa, params, threshold)
                )

    reference_dir = os.path.join(params["dataset_dir"], "metadata_eval") if is_eval else None
    scorer = ComputeSELDResults(params, ref_files_folder=reference_dir)
    rows = []
    for threshold, path in directories.items():
        er, f_score, le, lr, seld, _ = scorer.get_SELD_Results(path)
        rows.append({
            "variant": variant,
            "run_label": run_label,
            "seed": seed,
            "split": split,
            "threshold": threshold,
            "ER": er,
            "F": f_score,
            "LE": le,
            "LR": lr,
            "SELD": seld,
            "prediction_dir": path,
        })
        print(
            "{} seed {} {} threshold {:.2f}: SELD={:.6f} ER={:.6f} F={:.6f} LE={:.4f} LR={:.6f}".format(
                variant, seed, split, threshold, seld, er, f_score, le, lr,
            )
        )
    return rows


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    if args.manifest_paths:
        for manifest_path in args.manifest_paths:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            rows.extend(evaluate_seed(
                manifest["variant"], int(manifest["seed"]), args.split,
                args.thresholds, device, manifest=manifest,
                run_label=manifest["job_id"],
            ))
        output_path = args.output_csv
    else:
        for seed in args.seeds:
            rows.extend(evaluate_seed(args.variant, seed, args.split, args.thresholds, device))
        output_path = args.output_csv or os.path.join(
            PROJECT, "runs", "analysis",
            "threshold_official_{}_{}.csv".format(args.variant, args.split),
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(output_path))


if __name__ == "__main__":
    main()
