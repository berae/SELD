#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/zhanghc/Myllm/SELD/EINV2/STARSS_OfficialEINV2
PYTHON=/work/zhanghc/Myllm/SELD/EINV2/.venvs/einv2-cu121/bin/python
RAW=/work/zhanghc/Myllm/SELD/datasets
LOG="$ROOT/logs/starss_metadata_motion_prepare.log"
EXIT_FILE="$ROOT/logs/starss_metadata_motion_prepare.exit_code"
PID_FILE="$ROOT/logs/starss_metadata_motion_prepare.pid"

printf '%s\n' "$$" > "$PID_FILE"

record_exit() {
    status=$?
    printf '%s\n' "$status" > "$EXIT_FILE"
    exit "$status"
}
trap record_exit EXIT

mkdir -p "$ROOT/logs" "$ROOT/analysis/source_motion/STARSS22" "$ROOT/analysis/source_motion/STARSS23"

printf '[%s] STARSS22 canonical metadata\n' "$(date -Is)" >> "$LOG"
nice -n 15 ionice -c 2 -n 7 "$PYTHON" "$ROOT/tools/source_motion/prepare_canonical_metadata.py" \
    --dataset-name STARSS22 \
    --source-root "$RAW/STARSS22/raw/metadata_dev" \
    --destination-root "$ROOT/dataset/STARSS22/metadata_dev" >> "$LOG" 2>&1

printf '[%s] STARSS23 canonical metadata (drop distance only at model input)\n' "$(date -Is)" >> "$LOG"
nice -n 15 ionice -c 2 -n 7 "$PYTHON" "$ROOT/tools/source_motion/prepare_canonical_metadata.py" \
    --dataset-name STARSS23 \
    --source-root "$RAW/STARSS23/raw/metadata_dev" \
    --destination-root "$ROOT/dataset/STARSS23/metadata_dev" >> "$LOG" 2>&1

printf '[%s] STARSS22 directional motion analysis\n' "$(date -Is)" >> "$LOG"
nice -n 15 ionice -c 2 -n 7 "$PYTHON" "$ROOT/tools/source_motion/analyze_source_motion.py" \
    --dataset-name STARSS22 \
    --dataset-root "$RAW/STARSS22" \
    --output-dir "$ROOT/analysis/source_motion/STARSS22" >> "$LOG" 2>&1

printf '[%s] STARSS23 directional motion analysis\n' "$(date -Is)" >> "$LOG"
nice -n 15 ionice -c 2 -n 7 "$PYTHON" "$ROOT/tools/source_motion/analyze_source_motion.py" \
    --dataset-name STARSS23 \
    --dataset-root "$RAW/STARSS23" \
    --output-dir "$ROOT/analysis/source_motion/STARSS23" >> "$LOG" 2>&1

printf '[%s] COMPLETE\n' "$(date -Is)" >> "$LOG"
