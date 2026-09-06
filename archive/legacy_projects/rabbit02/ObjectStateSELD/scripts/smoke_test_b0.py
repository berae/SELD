#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from object_state_seld import ManifestDataset, ObjectStateSELD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real-audio B0 forward/backward step.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data_intermediate/manifests/train_manifest_v0.csv",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(2026)
    device = torch.device(args.device)
    dataset = ManifestDataset(args.manifest, limit=args.batch_size)
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
    model = ObjectStateSELD().to(device)
    waveform = batch["waveform"].to(device)
    outputs = model(waveform)
    class_id = batch["class_id"].to(device)
    activity = batch["activity"].to(device)
    position = batch["position_xyz"].to(device)
    losses = {
        "class": F.cross_entropy(outputs["class_logits"], class_id),
        "activity": F.binary_cross_entropy_with_logits(outputs["activity_logits"], activity),
        "doa": F.mse_loss(outputs["position_xyz"], position),
    }
    total = sum(losses.values())
    total.backward()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "status": "PASS",
        "device": str(device),
        "sample_ids": list(batch["sample_id"]),
        "waveform_shape": list(waveform.shape),
        "class_logits_shape": list(outputs["class_logits"].shape),
        "position_shape": list(outputs["position_xyz"].shape),
        "parameter_count": parameter_count,
        "loss": round(float(total.detach().cpu()), 6),
        "gradient_tensors": sum(
            parameter.grad is not None for parameter in model.parameters()
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
