"""Read-only asset integrity manifest: all label bytes, sampled audio bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    p = args.project / 'EINV2'
    paths = dict(audio_dev=args.hdf5 / 'dcase2020task3/data/24000fs/dev/foa',
                 audio_eval=args.hdf5 / 'dcase2020task3/data/24000fs/eval/foa',
                 meta_dev=args.hdf5 / 'dcase2020task3/meta/dev',
                 velocity=p / 'C1_CausalEINV2_Velocity_seed2026/velocity_hdf5/dcase2020task3/velocity/dev',
                 jepa=p / 'C2_CausalEINV2_JEPA_seed2026/jepa_hdf5/dcase2020task3/jepa/dev',
                 csv_dev=p / 'dataset_root/metadata_dev', csv_eval=p / 'dataset_root/metadata_eval')
    shape_map = dict(meta_dev={'sed_label': (600, 2, 14), 'doa_label': (600, 2, 3)},
                     velocity={'velocity_label': (600, 2, 3), 'velocity_mask': (600, 2)},
                     jepa={'identity_label': (600, 2, 2), 'jepa_valid_mask': (600, 2, 3)})
    result = dict(host=socket.gethostname(), project=str(args.project), groups={})
    stems = {}
    for key, path in paths.items():
        files = sorted(f for f in path.glob('*.csv' if key.startswith('csv') else '*.h5') if not f.name.startswith('.'))
        assert len(files) == (200 if key.endswith('eval') else 600), (key, len(files))
        stems[key] = {f.stem for f in files}
        hashes = {}
        inventory = []
        samples = set(files[:2] + files[-2:])
        if key == 'audio_dev':
            for fold in range(1, 7):
                samples.add(next(f for f in files if f.name.startswith('fold{}_'.format(fold))))
        for f in files:
            inventory.append((f.name, f.stat().st_size))
            if f.suffix == '.h5':
                with h5py.File(f, 'r') as hf:
                    if key.startswith('audio'):
                        assert hf['waveform'].shape == (4, 1440000) and hf['waveform'].dtype == np.int16, str(f)
                        hf['waveform'][:, :2400]
                        hf['waveform'][:, -2400:]
                    else:
                        for name, shape in shape_map[key].items():
                            data = hf[name][:]
                            assert data.shape == shape and np.isfinite(data).all(), (str(f), name)
            if not key.startswith('audio') or f in samples:
                hashes[f.name] = digest(f)
        result['groups'][key] = dict(path=str(path), count=len(files),
                                    inventory_sha256=hashlib.sha256(json.dumps(inventory).encode()).hexdigest(),
                                    hashes=hashes)
    for key in ('meta_dev', 'velocity', 'jepa', 'csv_dev'):
        assert stems[key] == stems['audio_dev'], key
    assert stems['csv_eval'] == stems['audio_eval']
    result['status'] = 'PASS'
    result['limitation'] = 'All headers and all labels checked; audio payload byte hashes sampled, not exhaustive.'
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
        print(json.dumps({'status': 'PASS', 'output': str(args.output), 'counts': {k: v['count'] for k, v in result['groups'].items()}}))
    else:
        print(rendered)


if __name__ == '__main__':
    main()
