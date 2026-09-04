#!/usr/bin/env python3
"""Measure main/auxiliary gradient alignment on frozen paper-v1 checkpoints."""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

import cls_data_generator
import parameters
import seldnet_model
from experiment_matrix import TASK_IDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["C1", "C2", "C3"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def manifest_for(variant, seed):
    paths = glob.glob(os.path.join(
        PROJECT, "runs", "manifests", "*_paper_v1_{}_seed{}_*.json".format(variant, seed)
    ))
    if len(paths) != 1:
        raise RuntimeError("expected one manifest, found {}".format(paths))
    with open(paths[0], encoding="utf-8") as handle:
        return json.load(handle)


def shared_parameters(model):
    prefixes = ("conv_block_list", "gru", "mhsa_block_list", "layer_norm_list")
    return [parameter for name, parameter in model.named_parameters() if name.startswith(prefixes)]


def gradient_vector(loss, shared, retain_graph=True):
    gradients = torch.autograd.grad(
        loss, shared, retain_graph=retain_graph, allow_unused=True
    )
    return [gradient.detach() for gradient in gradients if gradient is not None]


def compare(left, right):
    if len(left) != len(right):
        raise RuntimeError("gradient lists are not aligned")
    dot = sum((a * b).sum() for a, b in zip(left, right))
    left_norm = torch.sqrt(sum((a * a).sum() for a in left))
    right_norm = torch.sqrt(sum((b * b).sum() for b in right))
    cosine = dot / (left_norm * right_norm).clamp_min(1e-12)
    return cosine.item(), left_norm.item(), right_norm.item()


def evaluate_variant(variant, seed, batches, batch_size, device):
    params = parameters.get_params(TASK_IDS[variant]["full"])
    params["seed"] = seed
    params["batch_size"] = batch_size
    params["dropout_rate"] = 0.0
    data = cls_data_generator.DataGenerator(params, split=[2, 3, 4, 5, 6], shuffle=False)
    input_shape, output_shape = data.get_data_sizes()
    model = seldnet_model.SeldModel(input_shape, output_shape, params).to(device)
    manifest = manifest_for(variant, seed)
    model.load_state_dict(torch.load(manifest["checkpoint"], map_location=device))
    # cuDNN requires recurrent modules to be in training mode for backward.
    # Disable stochastic dropout above and keep BatchNorm statistics frozen.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    target_model = None
    if params["use_jepa"]:
        target_model = seldnet_model.SeldModel(input_shape, output_shape, params).to(device)
        target_path = manifest["checkpoint"].replace("_model.h5", "_ema_target.h5")
        target_model.load_state_dict(torch.load(target_path, map_location=device))
        target_model.eval()
        for parameter in target_model.parameters():
            parameter.requires_grad = False

    criterion = seldnet_model.DynamicMSELoss_ADPIT(params)
    shared = shared_parameters(model)
    rows = []
    for batch_index, (features, target) in enumerate(data.generate()):
        if batch_index >= batches:
            break
        features = torch.as_tensor(features, dtype=torch.float32, device=device)
        target = torch.as_tensor(target, dtype=torch.float32, device=device)
        output = model(features)
        with torch.no_grad():
            target_latent = target_model(features, latent_only=True) if target_model else None
        losses = criterion(output, target, target_latent)
        doa_gradient = gradient_vector(losses["doa"], shared)
        for auxiliary in ("velocity", "jepa"):
            if not params["use_{}".format(auxiliary)]:
                continue
            aux_gradient = gradient_vector(losses[auxiliary], shared)
            cosine, doa_norm, aux_norm = compare(doa_gradient, aux_gradient)
            row = {
                "variant": variant,
                "seed": seed,
                "batch": batch_index,
                "auxiliary": auxiliary,
                "gradient_cosine": cosine,
                "doa_gradient_norm": doa_norm,
                "aux_gradient_norm": aux_norm,
                "aux_to_doa_norm_ratio": aux_norm / max(doa_norm, 1e-12),
                "doa_loss": losses["doa"].item(),
                "aux_loss": losses[auxiliary].item(),
            }
            rows.append(row)
            print(row)
    return rows


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for variant in args.variants:
        rows.extend(evaluate_variant(
            variant, args.seed, args.batches, args.batch_size, device
        ))

    output_path = os.path.join(
        PROJECT, "runs", "analysis", "paper_v1_gradient_conflict.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(output_path))
    for variant in args.variants:
        for auxiliary in ("velocity", "jepa"):
            values = [
                row["gradient_cosine"] for row in rows
                if row["variant"] == variant and row["auxiliary"] == auxiliary
            ]
            if values:
                print(
                    "{} {} cosine mean={:.6f} std={:.6f}".format(
                        variant, auxiliary, statistics.mean(values),
                        statistics.stdev(values) if len(values) > 1 else float("nan"),
                    )
                )


if __name__ == "__main__":
    main()
