"""Shared model-independent TAU2020 official-protocol scoring."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evaluation'))
from score_directory import main
if __name__ == '__main__':
    main()
