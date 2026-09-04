import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models/einv2/audited'))
from runner import eval_main
if __name__ == '__main__':
    eval_main()
