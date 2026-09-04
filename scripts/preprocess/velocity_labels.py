"""Create auxiliary velocity targets; metadata/HDF5 paths are explicit CLI arguments."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'models/einv2/variants/C1/tools/build_velocity_labels.py'), run_name='__main__')
