#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/zhanghc/Myllm/SELD/EINV2/STARSS_OfficialEINV2"
WORKER="${ROOT}/tools/source_motion/run_starss_official_c0_sequence.sh"
ARCHIVE="${ROOT}/logs/retry_archive_$(date +%Y%m%d_%H%M%S)"

if pgrep -af 'run_starss_official_c0_sequence|launch_starss22_23_official_c0_serial_gpu04' | grep -v pgrep >/dev/null; then
    echo "ERROR: a STARSS C0 worker is already running" >&2
    exit 60
fi
for gpu in 3 4; do
    if nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
        echo "ERROR: physical GPU0${gpu} already has a compute process" >&2
        exit 61
    fi
done
if [[ -e "${ROOT}/results/out_train/ein_seld/STARSS22_C0_OfficialEINV2_seed2026" || \
      -e "${ROOT}/results/out_train/ein_seld/STARSS23_C0_OfficialEINV2_seed2026" ]]; then
    echo "ERROR: a target output directory already exists" >&2
    exit 62
fi

mkdir -p "${ARCHIVE}"
for dataset in STARSS22 STARSS23; do
    prefix="${ROOT}/logs/${dataset}_C0_OfficialEINV2_seed2026"
    for suffix in log exit_code pid launcher_pid; do
        path="${prefix}.${suffix}"
        if [[ -e "${path}" ]]; then
            mv "${path}" "${ARCHIVE}/"
        fi
    done
done
for suffix in log exit_code pid launcher_pid; do
    path="${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.${suffix}"
    if [[ -e "${path}" ]]; then
        mv "${path}" "${ARCHIVE}/"
    fi
done

setsid "${WORKER}" STARSS22 4 12444 \
    > "${ROOT}/logs/STARSS22_C0_OfficialEINV2_seed2026.log" 2>&1 < /dev/null &
pid22=$!
printf '%s\n' "${pid22}" > "${ROOT}/logs/STARSS22_C0_OfficialEINV2_seed2026.launcher_pid"

setsid "${WORKER}" STARSS23 3 12433 \
    > "${ROOT}/logs/STARSS23_C0_OfficialEINV2_seed2026.log" 2>&1 < /dev/null &
pid23=$!
printf '%s\n' "${pid23}" > "${ROOT}/logs/STARSS23_C0_OfficialEINV2_seed2026.launcher_pid"

echo "STARSS22_GPU04_launcher_pid=${pid22}"
echo "STARSS23_GPU03_launcher_pid=${pid23}"
echo "previous_failed_logs=${ARCHIVE}"
