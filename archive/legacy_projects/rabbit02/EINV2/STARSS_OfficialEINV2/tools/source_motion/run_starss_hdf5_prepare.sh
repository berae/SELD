#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/zhanghc/Myllm/SELD/EINV2/STARSS_OfficialEINV2
PYTHON=/work/zhanghc/Myllm/SELD/EINV2/.venvs/einv2-cu121/bin/python
LOG="$ROOT/logs/starss_hdf5_prepare.log"
EXIT_FILE="$ROOT/logs/starss_hdf5_prepare.exit_code"
PID_FILE="$ROOT/logs/starss_hdf5_prepare.pid"

printf '%s\n' "$$" > "$PID_FILE"
record_exit() {
    status=$?
    printf '%s\n' "$status" > "$EXIT_FILE"
    exit "$status"
}
trap record_exit EXIT

if [ "$(cat "$ROOT/logs/starss_metadata_motion_prepare.exit_code")" != "0" ]; then
    printf 'Canonical metadata/motion preparation did not pass.\n' >> "$LOG"
    exit 10
fi
if [ -e "$ROOT/_hdf5/STARSS22" ] || [ -e "$ROOT/_hdf5/STARSS23" ]; then
    printf 'Target HDF5 directory already exists; refusing to overwrite or resume implicitly.\n' >> "$LOG"
    exit 11
fi

cd "$ROOT"
for dataset in STARSS22 STARSS23; do
    config="configs/ein_seld/seld_${dataset,,}.yaml"
    printf '[%s] %s dev waveform HDF5\n' "$(date -Is)" "$dataset" >> "$LOG"
    nice -n 19 ionice -c 3 "$PYTHON" code/main.py -c "$config" --dataset "$dataset" \
        preprocess --preproc_mode extract_data --dataset_type dev --no_cuda >> "$LOG" 2>&1

    printf '[%s] %s three-track PIT labels\n' "$(date -Is)" "$dataset" >> "$LOG"
    nice -n 19 ionice -c 3 "$PYTHON" code/main.py -c "$config" --dataset "$dataset" \
        preprocess --preproc_mode extract_pit_label --dataset_type dev --no_cuda >> "$LOG" 2>&1

    printf '[%s] %s dev indexes\n' "$(date -Is)" "$dataset" >> "$LOG"
    nice -n 19 ionice -c 3 "$PYTHON" code/main.py -c "$config" --dataset "$dataset" \
        preprocess --preproc_mode extract_indexes --dataset_type dev --no_cuda >> "$LOG" 2>&1

    frame_dir="$ROOT/_hdf5/$dataset/dcase2022task3/label/frame/$dataset"
    mkdir -p "$frame_dir"
    find "$ROOT/dataset/$dataset/metadata_dev" -type f -name '*.csv' -exec cp -t "$frame_dir" {} +

    printf '[%s] %s evaluation waveform HDF5 and indexes\n' "$(date -Is)" "$dataset" >> "$LOG"
    nice -n 19 ionice -c 3 "$PYTHON" code/main.py -c "$config" --dataset "$dataset" \
        preprocess --preproc_mode extract_data --dataset_type eval --no_cuda >> "$LOG" 2>&1
    nice -n 19 ionice -c 3 "$PYTHON" code/main.py -c "$config" --dataset "$dataset" \
        preprocess --preproc_mode extract_indexes --dataset_type eval --no_cuda >> "$LOG" 2>&1
done

printf '[%s] COMPLETE (scalar intentionally deferred until GPU04 is free)\n' "$(date -Is)" >> "$LOG"
