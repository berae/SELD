"""Custom static/dynamic diagnostics; not official LR_CD."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'models/multi_accdoa/analysis/evaluate_motion_strata.py'), run_name='__main__')
