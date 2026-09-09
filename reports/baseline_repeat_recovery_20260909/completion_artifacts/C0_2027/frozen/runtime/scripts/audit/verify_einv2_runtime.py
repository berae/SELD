"""Persist regression and real-artifact preflight evidence for audited EINV2."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'models/einv2/audited'))
from runtime import sha256
from runner import source_manifest
import h5py
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = {'captured_at': datetime.now(timezone.utc).isoformat(), 'tests': []}
    for rel in ('tests/test_einv2_audited.py', 'tests/test_repository.py', 'tests/test_experiment_catalog.py',
                'tests/test_entrypoint_config.py', 'evaluation/test_alignment.py'):
        completed = subprocess.run([sys.executable, str(ROOT / rel)], cwd=ROOT, text=True, capture_output=True)
        evidence['tests'].append(dict(script=rel, exit_code=completed.returncode,
                                     output=(completed.stdout + completed.stderr)[-12000:]))
        print(json.dumps({'script': rel, 'exit_code': completed.returncode}), flush=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
    scalar = args.experiment_root / 'scalar_trainfolds2-6.h5'
    manifest = json.loads(scalar.with_suffix('.json').read_text())
    assert len(manifest['files']) == 500
    assert all(Path(p).name.startswith(tuple('fold' + str(f) + '_' for f in range(2, 7))) for p in manifest['files'])
    assert manifest['sha256'] == sha256(scalar)
    with h5py.File(scalar) as hf:
        assert hf.attrs['folds'] == '2,3,4,5,6'
        assert hf['mean'].shape == hf['std'].shape == (1, 7, 1, 256)
        assert np.isfinite(hf['mean'][:]).all() and np.isfinite(hf['std'][:]).all() and (hf['std'][:] > 0).all()
    smoke = args.experiment_root / 'runs/smoke_C3_seed2026_framecausal_v020'
    status = json.loads((smoke / 'status.json').read_text())
    validation = json.loads((smoke / 'validation/metrics.json').read_text())
    checkpoint = torch.load(smoke / 'best.pth', map_location='cpu')
    assert status['status'] == 'completed'
    assert checkpoint['epoch'] == 1 and checkpoint['teacher'] is not None
    assert checkpoint['config']['audit']['source_hashes'] == source_manifest()
    assert validation['files'] == 100 and validation['training_validation_max_abs_diff'] == 0
    evidence.update(status='PASS', scalar=dict(path=str(scalar), sha256=sha256(scalar), train_files=500, chunks=7500),
                    smoke=dict(path=str(smoke), training_batches=2, epoch=1, validation_files=100,
                               independent_validation_max_abs_diff=0, checkpoint_bytes=(smoke / 'best.pth').stat().st_size,
                               limitation='Empty smoke predictions; not a performance or nonempty-prediction consistency claim.'),
                    runtime_source_hashes=source_manifest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
