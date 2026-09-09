"""Create continuity masks for JEPA supervision (training targets only)."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'models/einv2/variants/C3/tools/build_jepa_masks.py'), run_name='__main__')
