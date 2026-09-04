"""New, train-fold-only EINV2 scaler; never overwrites the historical scaler."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'models/einv2/audited'))
from runtime import CausalLogmelIntensity_Extractor, seed_all, sha256
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ruamel.yaml import YAML


class AudioChunks(Dataset):
    def __init__(self, hdf5):
        self.files = sorted(p for p in hdf5.glob('*.h5') if p.name.startswith(tuple('fold' + str(f) + '_' for f in range(2, 7))))
        assert len(self.files) == 500, len(self.files)

    def __len__(self):
        return len(self.files) * 15

    def __getitem__(self, index):
        clip, segment = divmod(index, 15)
        with h5py.File(self.files[clip], 'r') as hf:
            return torch.from_numpy(hf['waveform'][:, segment * 96000:(segment + 1) * 96000].astype(np.float32) / 32767.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seed_all(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = YAML(typ='safe').load((ROOT / 'configs/einv2/C0.yaml').read_text())
    feature = CausalLogmelIntensity_Extractor(cfg).to(device).eval()
    data = AudioChunks(args.project_root / 'EINV2/C0_CausalEINV2_seed2026/_hdf5/dcase2020task3/data/24000fs/dev/foa')
    loader = DataLoader(data, batch_size=32, num_workers=4, shuffle=False)
    total = torch.zeros(7, 256, dtype=torch.float64, device=device)
    second = torch.zeros_like(total)
    count = 0
    with torch.no_grad():
        for step, waveform in enumerate(loader):
            x = feature(waveform.to(device)).double()
            total += x.sum((0, 2))
            second += x.square().sum((0, 2))
            count += x.shape[0] * x.shape[2]
            if step % 25 == 0:
                print(json.dumps({'batches': step + 1, 'total': len(loader)}), flush=True)
    mean = total / count
    std = (second / count - mean.square()).clamp_min(1e-12).sqrt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, 'x') as hf:
        hf.create_dataset('mean', data=mean.cpu().numpy()[None, :, None, :].astype(np.float32))
        hf.create_dataset('std', data=std.cpu().numpy()[None, :, None, :].astype(np.float32))
        hf.attrs.update(folds='2,3,4,5,6', channel_order='logmel_WYZX_IV_XYZ', files=500, chunks=7500, chunk_seconds=4)
    manifest = {'scalar': str(args.output), 'sha256': sha256(args.output), 'files': [str(p) for p in data.files],
                'count_per_channel_and_mel': count, 'excluded_folds': [1], 'evaluation_used': False}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'completed', 'sha256': manifest['sha256'], 'files': 500, 'chunks': 7500}))


if __name__ == '__main__':
    main()
