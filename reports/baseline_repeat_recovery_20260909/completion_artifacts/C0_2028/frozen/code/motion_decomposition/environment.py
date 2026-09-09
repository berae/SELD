import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys

versions={}
for name in ('numpy','scipy','h5py','torch','librosa','matplotlib'):
    try:versions[name]=importlib.import_module(name).__version__
    except Exception as e:versions[name]=repr(e)
with Path(sys.argv[1]).open('x') as f:json.dump(dict(python=sys.version,platform=platform.platform(),versions=versions,
    nvidia_smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout),f,indent=2)
