#!/usr/bin/env bash
set -u

project_dir=/work/zhanghc/Myllm/SELD/EINV2/STARSS_OfficialEINV2
run_id=STARSS23_C0_RealOnly_CkptFix_IOtuned_seed2026
python_bin=/work/zhanghc/Myllm/SELD/EINV2/.venvs/einv2-cu121/bin/python
log_path="$project_dir/logs/$run_id.log"
exit_path="$project_dir/logs/$run_id.exit_code"

cd "$project_dir" || exit 98
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

timeout 12h "$python_bin" code/main.py \
  -c configs/ein_seld/seld_starss23_c0_realonly_ckptfix_iotuned.yaml \
  --dataset STARSS23 train --seed 2026 --num_workers 2 --port 12437 \
  >"$log_path" 2>&1
status=$?
printf '%s\n' "$status" >"$exit_path"
exit "$status"
