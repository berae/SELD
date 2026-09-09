#!/usr/bin/env python3
"""Summarize validation-only pilot-v2 manifests."""

import glob
import json
import os


PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    paths = sorted(glob.glob(os.path.join(
        PROJECT, "runs", "manifests", "*_pilot_v2_*.json"
    )))
    if not paths:
        print("No pilot-v2 manifests found")
        return
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            run = json.load(handle)
        validation = run.get("best_validation", {})
        params = run.get("params", {})
        print(
            "{} status={} seed={} lambda_velocity={} lambda_jepa={} "
            "best_epoch={} val_SELD={}".format(
                run.get("job_id"), run.get("status"), run.get("seed"),
                params.get("lambda_velocity"), params.get("lambda_jepa"),
                validation.get("epoch"), validation.get("SELD"),
            )
        )


if __name__ == "__main__":
    main()
