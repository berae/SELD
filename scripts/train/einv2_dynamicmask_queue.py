"""Explicit name for the frozen D1/D3 queue; the old filename is provenance-bound."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('einv2_weightprobe_queue.py')), run_name='__main__')
