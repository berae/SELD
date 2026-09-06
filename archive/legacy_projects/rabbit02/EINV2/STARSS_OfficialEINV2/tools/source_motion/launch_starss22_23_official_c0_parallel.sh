#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/zhanghc/Myllm/SELD/EINV2/STARSS_OfficialEINV2"
WORKER="${ROOT}/tools/source_motion/run_starss_official_c0_sequence.sh"

for dataset in STARSS22 STARSS23; do
    if pgrep -af "run_starss_official_c0_sequence.sh ${dataset}" | grep -v pgrep >/dev/null; then
        echo "ERROR: ${dataset} sequence is already running" >&2
        exit 30
    fi
done

setsid "${WORKER}" STARSS22 0 12422 \
    > "${ROOT}/logs/STARSS22_C0_OfficialEINV2_seed2026.log" 2>&1 < /dev/null &
pid22=$!

setsid "${WORKER}" STARSS23 1 12423 \
    > "${ROOT}/logs/STARSS23_C0_OfficialEINV2_seed2026.log" 2>&1 < /dev/null &
pid23=$!

printf '%s\n' "${pid22}" > "${ROOT}/logs/STARSS22_C0_OfficialEINV2_seed2026.launcher_pid"
printf '%s\n' "${pid23}" > "${ROOT}/logs/STARSS23_C0_OfficialEINV2_seed2026.launcher_pid"
echo "STARSS22_launcher_pid=${pid22}"
echo "STARSS23_launcher_pid=${pid23}"
