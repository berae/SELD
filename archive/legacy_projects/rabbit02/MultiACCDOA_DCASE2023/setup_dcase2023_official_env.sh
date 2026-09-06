#!/usr/bin/env bash
set -euo pipefail

project=/work/zhanghc/Myllm/SELD/MultiACCDOA_DCASE2023
env_dir="$project/.venvs/dcase2023-official-py38"
conda_bin=/work/zhanghc/miniconda3/bin/conda

if [ ! -x "$env_dir/bin/python" ]; then
  "$conda_bin" create -y -p "$env_dir" python=3.8.11 pip
fi

"$env_dir/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu111 \
  torch==1.10.0+cu111
"$env_dir/bin/python" -m pip install \
  numpy==1.22.4 scipy==1.7.3 librosa==0.8.1 scikit-learn==1.0.2 \
  joblib==1.1.1 matplotlib==3.5.3 ipython==8.12.3 soundfile==0.12.1

"$env_dir/bin/python" - <<'PY'
import librosa
import numpy
import scipy
import sklearn
import torch

print("python-ready")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("librosa", librosa.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
PY
