import sys

import librosa
import numba
import numpy
import scipy
import sklearn
import torch


print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("numba", numba.__version__)
print("librosa", librosa.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_devices", torch.cuda.device_count())
