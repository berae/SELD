#!/usr/bin/env bash
set +e

root=/work/zhanghc/Myllm/SELD/CausalMultiACCDOA_TAU2020
python_bin="$root/.venvs/dcase2023-official-py38/bin/python"
run_id=causal_tau2020_eval_multiaccdoa_seed2026
log="$root/logs/${run_id}.log"
exit_file="$root/logs/${run_id}.exit_code"
pid_file="$root/logs/${run_id}.pid"

cd "$root" || exit 98
rm -f "$exit_file"
printf '%s\n' "$$" >"$pid_file"
export CUDA_VISIBLE_DEVICES=4
export PYTHONHASHSEED=2026
timeout --signal=TERM --kill-after=60s 12h \
  "$python_bin" -u train_seldnet.py 35 "$run_id" \
  >"$log" 2>&1
code=$?
printf '%s\n' "$code" >"$exit_file"
exit "$code"
