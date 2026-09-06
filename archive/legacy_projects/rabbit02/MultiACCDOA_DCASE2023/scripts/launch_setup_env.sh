#!/usr/bin/env bash
set -uo pipefail

project=/work/zhanghc/Myllm/SELD/MultiACCDOA_DCASE2023
mkdir -p "$project/logs"
log="$project/logs/setup_env.log"
exit_file="$project/logs/setup_env.exit_code"
rm -f "$exit_file"
bash "$project/scripts/setup_env.sh" >>"$log" 2>&1
code=$?
printf '%s\n' "$code" >"$exit_file"
exit "$code"
