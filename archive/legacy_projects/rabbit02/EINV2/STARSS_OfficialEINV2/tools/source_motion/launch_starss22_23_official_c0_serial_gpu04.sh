#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/zhanghc/Myllm/SELD/EINV2/STARSS_OfficialEINV2"
WORKER="${ROOT}/tools/source_motion/run_starss_official_c0_sequence.sh"
SEQUENCE_LOG="${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.log"
SEQUENCE_EXIT="${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.exit_code"
SEQUENCE_PID="${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.pid"

printf '%s\n' "$$" > "${SEQUENCE_PID}"
record_exit() {
    status=$?
    printf '%s\n' "${status}" > "${SEQUENCE_EXIT}"
    exit "${status}"
}
trap record_exit EXIT

if nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
    echo "ERROR: physical GPU04 already has a compute process" >&2
    exit 40
fi

echo "[$(date -Is)] serial sequence starts; physical GPU04 only"
echo "[$(date -Is)] stage 1/2: STARSS22 C0"
"${WORKER}" STARSS22 4 12424 \
    > "${ROOT}/logs/STARSS22_C0_OfficialEINV2_seed2026.log" 2>&1

if [[ "$(cat "${ROOT}/logs/STARSS22_C0_OfficialEINV2_seed2026.exit_code")" != "0" ]]; then
    echo "ERROR: STARSS22 failed; STARSS23 will not be started" >&2
    exit 41
fi

echo "[$(date -Is)] stage 2/2: STARSS23 C0"
"${WORKER}" STARSS23 4 12425 \
    > "${ROOT}/logs/STARSS23_C0_OfficialEINV2_seed2026.log" 2>&1

if [[ "$(cat "${ROOT}/logs/STARSS23_C0_OfficialEINV2_seed2026.exit_code")" != "0" ]]; then
    echo "ERROR: STARSS23 failed" >&2
    exit 42
fi

echo "[$(date -Is)] serial sequence COMPLETE"
