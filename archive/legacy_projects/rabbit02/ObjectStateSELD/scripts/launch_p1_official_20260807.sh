#!/usr/bin/env bash
set -euo pipefail
host_name=$(hostname)
if [[ "$host_name" == rabbit02* ]]; then
  ROOT=/work/zhanghc/Myllm/SELD/ObjectStateSELD
else
  ROOT=/root/autodl-tmp/SELD/ObjectStateSELD
fi
cd "$ROOT"
mkdir -p logs scripts
SCRIPT=scripts/download_p1_official_20260807.sh
LOG=logs/p1_official_download_20260807.log
EXIT=logs/p1_official_download_20260807.exit_code
PID=logs/p1_official_download_20260807.pid
if pgrep -af 'download_p1_official_20260807.sh' | grep -v pgrep; then
  echo "Official downloader already running" >&2
  exit 2
fi
rm -f "$EXIT"
setsid bash -c "bash '$SCRIPT' > '$LOG' 2>&1; code=\$?; printf '%s\n' \"\$code\" > '$EXIT'; exit \"\$code\"" < /dev/null > /dev/null 2>&1 &
echo $! > "$PID"
sleep 3
echo "host=$host_name pid=$(cat "$PID")"
cat logs/p1_official_download_20260807.state 2>/dev/null || true
tail -n 12 "$LOG" 2>/dev/null || true
ps -p "$(cat "$PID")" -o pid,ppid,etime,stat,cmd
