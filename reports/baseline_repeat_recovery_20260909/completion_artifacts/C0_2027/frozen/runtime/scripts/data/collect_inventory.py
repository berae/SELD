"""Read-only dataset/cache inventory. Reads audio headers, never audio samples.

Example: python collect_inventory.py --host rabbit02 --root /work/zhanghc/Myllm/SELD
Output is JSON on stdout; progress on stderr. Optional soundfile/numpy/h5py
enable header/shape inspection. File-list/size digests are NOT content checksums.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

try:
    import soundfile as sf
except ImportError:
    sf = None

RABBIT = {
    'TAU2020_raw': 'ObjectStateSELD/data_raw/dcase2020',
    'TAU2021_raw': 'datasets/TAU2021/raw',
    'STARSS22_raw': 'datasets/STARSS22/raw',
    'STARSS23_raw': 'datasets/STARSS23/raw',
    'DCASE2024_synthetic_raw': 'datasets/DCASE2024_Synthetic/raw',
    'DCASE2025_stereo_raw': 'datasets/DCASE2025_StereoSELD/raw',
    'Multi_TAU2020_data': 'MultiACCDOA_TAU2020/data/TAU2020_SELD_dataset',
    'Multi_strict_features': 'MultiACCDOA_TAU2020/features/tau2020_foa_multiaccdoa_strictcausal_trainfolds2-6',
    'Multi_legacy_features': 'MultiACCDOA_TAU2020/features/tau2020_foa_multiaccdoa',
    'EINV2_TAU2020_hdf5': 'EINV2/C0_CausalEINV2_seed2026/_hdf5',
    'EINV2_velocity': 'EINV2/C1_CausalEINV2_Velocity_seed2026/velocity_hdf5',
    'EINV2_jepa': 'EINV2/C2_CausalEINV2_JEPA_seed2026/jepa_hdf5',
    'EINV2_STARSS_hdf5': 'EINV2/STARSS_OfficialEINV2/_hdf5',
    'EINV2_STARSS_data': 'EINV2/STARSS_OfficialEINV2/dataset',
    'PSELDNets_data': 'PSELDNets/datasets',
    'PSELDNets_hdf5': 'PSELDNets/_hdf5',
    'Multi_DCASE2023_data': 'MultiACCDOA_DCASE2023/data',
    'ObjectState_intermediate': 'ObjectStateSELD/data_intermediate',
    'ObjectState_processed': 'ObjectStateSELD/data_processed',
    'ObjectState_legacy_TAU2021': 'ObjectStateSELD/data_raw/dcase2021',
    'p1_archives': 'ObjectStateSELD/data_raw/p1_downloads',
    'TAU2021_downloads': 'datasets/TAU2021/downloads',
    'STARSS22_downloads': 'datasets/STARSS22/downloads',
    'STARSS23_downloads': 'datasets/STARSS23/downloads',
    'DCASE2025_downloads': 'datasets/DCASE2025_StereoSELD/downloads',
}
RB05 = {
    'Multi_TAU2020_data': 'DynamicCausalMultiACCDOA_TAU2020/data/TAU2020_SELD_dataset',
    'Multi_strict_features': 'DynamicCausalMultiACCDOA_TAU2020/features/tau2020_foa_multiaccdoa_strictcausal_trainfolds2-6',
    'EINV2_TAU2020_hdf5': 'EINV2/shared/_hdf5',
    'EINV2_velocity': 'EINV2/C1_CausalEINV2_Velocity_seed2026/velocity_hdf5',
    'EINV2_jepa': 'EINV2/C2_CausalEINV2_JEPA_seed2026/jepa_hdf5',
    'EINV2_TAU2020_metadata': 'EINV2/dataset_root',
    'other_datasets_root': 'datasets',
}


def digest(values):
    return hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()


def content_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sample_shape(path):
    try:
        if path.suffix == '.npy':
            import numpy as np
            x = np.load(path, mmap_mode='r', allow_pickle=False)
            return dict(shape=list(x.shape), dtype=str(x.dtype))
        if path.suffix in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(path, 'r') as f:
                return {k: dict(shape=list(f[k].shape), dtype=str(f[k].dtype))
                        for k in list(f.keys())[:8] if isinstance(f[k], h5py.Dataset)}
    except Exception as e:
        return dict(error=str(e))
    return None


def profile(asset, root):
    out = dict(asset=asset, path=str(root), resolved_path=str(root.resolve()), exists=root.exists(),
               groups=[], symlinks=[], broken_symlinks=[], errors=[])
    if not root.exists():
        return out
    parents = defaultdict(list)
    visited = set()
    for folder, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(folder)
        if real in visited:
            dirs[:] = []
            continue
        visited.add(real)
        for name in dirs + files:
            p = Path(folder) / name
            if p.is_symlink():
                row = dict(path=str(p), target=os.readlink(p), resolved=str(p.resolve()))
                out['symlinks'].append(row)
                if not p.exists():
                    out['broken_symlinks'].append(row)
        for name in files:
            p = Path(folder) / name
            try:
                if p.is_file():
                    parents[str(p.parent.relative_to(root))].append((p, p.stat()))
            except OSError as e:
                out['errors'].append(dict(path=str(p), error=str(e)))
    for parent, rows in sorted(parents.items()):
        rows.sort(key=lambda row: row[0].name)
        ext = Counter(p.suffix.lower() or '<none>' for p, _ in rows)
        g = dict(directory=parent, files=len(rows), bytes=sum(s.st_size for _, s in rows),
                 extensions=dict(ext), zero_bytes=sum(s.st_size == 0 for _, s in rows),
                 file_size_digest=digest(p.name + '\t' + str(s.st_size) for p, s in rows),
                 stem_digests={}, samples=[], content_samples=[], audio_header_errors=[])
        for suffix in ('.wav', '.csv', '.npy', '.h5'):
            stems = [p.stem for p, _ in rows if p.suffix.lower() == suffix]
            if stems:
                g['stem_digests'][suffix] = dict(count=len(stems), unique=len(set(stems)), sha256=digest(stems))
        audio = [(p, s) for p, s in rows if p.suffix.lower() in ('.wav', '.flac')]
        rates, channels, formats = Counter(), Counter(), Counter()
        seconds = 0.0
        lengths = []
        if sf:
            for p, _ in audio:
                try:
                    info = sf.info(str(p))
                    rates[info.samplerate] += 1
                    channels[info.channels] += 1
                    formats[info.subtype] += 1
                    seconds += info.frames / info.samplerate
                    lengths.append(info.frames / info.samplerate)
                except Exception as e:
                    g['audio_header_errors'].append(dict(file=p.name, error=str(e)))
        g['audio'] = dict(files=len(audio), headers_read=len(lengths), seconds=seconds,
                          duration_min=min(lengths) if lengths else None,
                          duration_max=max(lengths) if lengths else None,
                          sample_rates=dict(rates), channels=dict(channels), subtypes=dict(formats))
        # Deterministic first/middle/last file content samples, bounded to 32 MiB.
        for index in sorted({0, len(rows)//2, len(rows)-1}):
            p, stat = rows[index]
            if stat.st_size <= 32 * 1024 * 1024 and p.suffix.lower() in ('.wav', '.csv', '.npy', '.h5', '.hdf5'):
                g['content_samples'].append(dict(file=p.name, bytes=stat.st_size, sha256=content_hash(p)))
        for suffix in ('.csv', '.npy', '.h5', '.hdf5'):
            match = next((p for p, _ in rows if p.suffix.lower() == suffix), None)
            if match:
                sample = dict(file=match.name, shape=sample_shape(match))
                if suffix == '.csv':
                    with match.open('rb') as f:
                        first = f.readline().strip()
                    sample['first_row_columns'] = len(first.split(b',')) if first else 0
                g['samples'].append(sample)
        # Fully checksum main TAU2020 label and small scaler groups only.
        if (('TAU2020' in asset and ext.get('.csv', 0) == len(rows)) or
            ('scalar' in parent.lower() and g['bytes'] < 32 * 1024 * 1024)):
            g['full_content_digest'] = digest(p.name + '\t' + content_hash(p) for p, _ in rows)
            g['full_content_files'] = len(rows)
        out['groups'].append(g)
    out['files'] = sum(g['files'] for g in out['groups'])
    out['bytes'] = sum(g['bytes'] for g in out['groups'])
    out['file_size_digest'] = digest(g['directory'] + '\t' + g['file_size_digest'] for g in out['groups'])
    # Cap individual link examples while preserving counts. They are aliases, not new datasets.
    out['symlink_count'] = len(out['symlinks'])
    out['symlinks'] = out['symlinks'][:12]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('rabbit02', 'RB05'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    catalog = RABBIT if args.host == 'rabbit02' else RB05
    result = dict(host=args.host, hostname=socket.gethostname(), root=str(args.root.resolve()),
                  captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  method='stat + all readable audio headers + bounded deterministic samples; no whole-audio content hash',
                  collector_sha256=content_hash(Path(__file__)), assets=[])
    for name, relative in catalog.items():
        item = profile(name, args.root / relative)
        result['assets'].append(item)
        print(name, 'files=' + str(item.get('files', 0)), 'bytes=' + str(item.get('bytes', 0)), file=sys.stderr, flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
