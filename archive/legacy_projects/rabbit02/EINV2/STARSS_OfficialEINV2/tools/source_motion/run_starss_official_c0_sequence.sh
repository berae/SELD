#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 STARSS22|STARSS23 GPU_ID DDP_PORT" >&2
    exit 64
fi

DATASET="$1"
GPU_ID="$2"
DDP_PORT="$3"
ROOT="/work/zhanghc/Myllm/SELD/EINV2/STARSS_OfficialEINV2"
PYTHON="/work/zhanghc/Myllm/SELD/EINV2/.venvs/einv2-cu121/bin/python"
SEED=2026

case "${DATASET}" in
    STARSS22)
        CONFIG="configs/ein_seld/seld_starss22.yaml"
        TRAIN_ID="STARSS22_C0_OfficialEINV2_seed2026"
        ;;
    STARSS23)
        CONFIG="configs/ein_seld/seld_starss23.yaml"
        TRAIN_ID="STARSS23_C0_OfficialEINV2_seed2026"
        ;;
    *)
        echo "unsupported dataset: ${DATASET}" >&2
        exit 65
        ;;
esac

LOG="${ROOT}/logs/${TRAIN_ID}.log"
EXIT_FILE="${ROOT}/logs/${TRAIN_ID}.exit_code"
PID_FILE="${ROOT}/logs/${TRAIN_ID}.pid"
OUT_DIR="${ROOT}/results/out_train/ein_seld/${TRAIN_ID}"
SCALAR="${ROOT}/_hdf5/${DATASET}/dcase2022task3/data/24000fs/scalar/logmelIV_nfft1024_hop300_mel128___none__.h5"

printf '%s\n' "$$" > "${PID_FILE}"
record_exit() {
    status=$?
    printf '%s\n' "${status}" > "${EXIT_FILE}"
    exit "${status}"
}
trap record_exit EXIT

if [[ "$(cat "${ROOT}/logs/starss_hdf5_prepare.exit_code")" != "0" ]]; then
    echo "ERROR: STARSS HDF5 preparation did not pass." >&2
    exit 10
fi
if [[ -e "${OUT_DIR}" ]]; then
    echo "ERROR: output already exists: ${OUT_DIR}" >&2
    exit 11
fi

cd "${ROOT}"
echo "[$(date -Is)] ${DATASET} scalar/train sequence starts on GPU${GPU_ID}" 
echo "config=${CONFIG} train_id=${TRAIN_ID} seed=${SEED} port=${DDP_PORT}"

if [[ ! -s "${SCALAR}" ]]; then
    echo "[$(date -Is)] extracting ${DATASET} scalar"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
        "${PYTHON}" code/main.py -c "${CONFIG}" --dataset "${DATASET}" \
        preprocess --preproc_mode extract_scalar --dataset_type dev --num_workers 4
fi

"${PYTHON}" - "${SCALAR}" <<'PY'
import sys
import h5py
import numpy as np

path = sys.argv[1]
with h5py.File(path, "r") as handle:
    mean = handle["mean"][:]
    std = handle["std"][:]
if mean.shape != (1, 7, 1, 128) or std.shape != (1, 7, 1, 128):
    raise SystemExit(f"invalid scalar shape: mean={mean.shape}, std={std.shape}")
if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
    raise SystemExit("scalar contains non-finite or non-positive values")
print(f"scalar_audit=PASS mean_shape={mean.shape} std_min={std.min():.8f}")
PY

echo "[$(date -Is)] starting ${DATASET} official C0 training"
timeout --signal=TERM --kill-after=5m 18h \
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
    "${PYTHON}" code/main.py -c "${CONFIG}" --dataset "${DATASET}" \
    train --seed "${SEED}" --num_workers 4 --port "${DDP_PORT}"

METRICS="${OUT_DIR}/checkpoints/metrics_statistics.csv"
if [[ ! -s "${METRICS}" ]]; then
    echo "ERROR: missing metrics file: ${METRICS}" >&2
    exit 20
fi
if ! find "${OUT_DIR}/checkpoints" -maxdepth 1 -type f -name '*latest*.pth' -size +0c | grep -q .; then
    echo "ERROR: latest checkpoint missing" >&2
    exit 21
fi
if ! find "${OUT_DIR}/checkpoints" -maxdepth 1 -type f -name '*best*.pth' -size +0c | grep -q .; then
    echo "ERROR: best checkpoint missing" >&2
    exit 22
fi

echo "[$(date -Is)] ${DATASET} official C0 training COMPLETE"
