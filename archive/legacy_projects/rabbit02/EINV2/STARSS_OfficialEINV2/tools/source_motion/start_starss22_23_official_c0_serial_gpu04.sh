#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/zhanghc/Myllm/SELD/EINV2/STARSS_OfficialEINV2"
SEQUENCE="${ROOT}/tools/source_motion/launch_starss22_23_official_c0_serial_gpu04.sh"
ARCHIVE="${ROOT}/logs/retry_archive_$(date +%Y%m%d_%H%M%S)"

if pgrep -af 'run_starss_official_c0_sequence|launch_starss22_23_official_c0_serial_gpu04' | grep -v pgrep >/dev/null; then
    echo "ERROR: a STARSS C0 sequence is already running" >&2
    exit 50
fi
if [[ -e "${ROOT}/results/out_train/ein_seld/STARSS22_C0_OfficialEINV2_seed2026" || \
      -e "${ROOT}/results/out_train/ein_seld/STARSS23_C0_OfficialEINV2_seed2026" ]]; then
    echo "ERROR: a target output directory already exists" >&2
    exit 51
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

for path in \
    "${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.log" \
    "${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.exit_code" \
    "${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.pid" \
    "${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.launcher_pid"; do
    if [[ -e "${path}" ]]; then
        mv "${path}" "${ARCHIVE}/"
    fi
done

setsid "${SEQUENCE}" \
    > "${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.log" 2>&1 < /dev/null &
launcher=$!
printf '%s\n' "${launcher}" > "${ROOT}/logs/STARSS22_then_STARSS23_C0_serial_gpu04.launcher_pid"
echo "launcher_pid=${launcher}"
echo "previous_failed_logs=${ARCHIVE}"
