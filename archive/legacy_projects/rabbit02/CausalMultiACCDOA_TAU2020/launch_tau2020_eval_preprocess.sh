#!/usr/bin/env bash
set +e

root=/work/zhanghc/Myllm/SELD/MultiACCDOA_TAU2020
python_bin="$root/.venvs/dcase2023-official-py38/bin/python"
log="$root/logs/preprocess_eval_task32.log"
exit_file="$root/logs/preprocess_eval_task32.exit_code"

cd "$root" || exit 98
rm -f "$exit_file"
timeout --signal=TERM --kill-after=60s 8h \
  "$python_bin" -u batch_feature_extraction_eval.py 32 \
  >"$log" 2>&1
code=$?
printf '%s\n' "$code" >"$exit_file"
exit "$code"
