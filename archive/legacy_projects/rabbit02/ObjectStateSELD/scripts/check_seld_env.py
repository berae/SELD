from __future__ import annotations

import sys
from pathlib import Path

import h5py
import librosa
import numpy as np
import pandas as pd
import ruamel.yaml
import scipy
import sklearn
import soundfile as sf
import torch
import yaml


ROOT = Path("/work/zhanghc/Myllm/SELD/ObjectStateSELD")
WAV = ROOT / "data_raw/dcase2020/foa_eval/mix001.wav"

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
print(
    "versions="
    f"numpy:{np.__version__},scipy:{scipy.__version__},pandas:{pd.__version__},"
    f"librosa:{librosa.__version__},soundfile:{sf.__version__},"
    f"sklearn:{sklearn.__version__},h5py:{h5py.__version__},pyyaml:{yaml.__version__},"
    f"ruamel:{ruamel.yaml.__version__}"
)

info = sf.info(str(WAV))
print(
    f"wav={WAV.name} samplerate={info.samplerate} channels={info.channels} "
    f"frames={info.frames} duration={info.duration:.3f} format={info.format}"
)

assert info.samplerate == 24000
assert info.channels == 4
assert abs(info.duration - 60.0) < 1e-3

sys.path.insert(0, str(ROOT / "external/EIN-SELD/seld"))
from methods.ein_seld.models.seld import EINV2  # noqa: E402,F401

sys.path.insert(0, str(ROOT / "external/seld-dcase2022"))
import parameters as dcase2022_parameters  # noqa: E402,F401
import seldnet_model as dcase2022_model  # noqa: E402,F401

print("ein_seld_import=ok")
print("dcase2022_import=ok")
print("environment_smoke_test=PASS")
