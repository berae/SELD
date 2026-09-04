"""Audit the historical EINV2 implementation using actual server artifacts.

No optimization, checkpoint changes, or dataset writes. --output is the only write.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load_model(variant, cfg):
    import torch
    path = ROOT / 'models/einv2/variants' / variant / 'seld/methods/ein_seld/models/seld_causal.py'
    spec = importlib.util.spec_from_file_location('audit_model_' + variant, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(2026)
    model = getattr(module, cfg['training']['model'])(cfg, SimpleNamespace(label_set=list(range(14))))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import h5py
    import numpy as np
    import torch
    from ruamel.yaml import YAML
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / 'models/einv2/variants/C0/seld'))
    from methods.feature_causal import CausalLogmelIntensity_Extractor
    from methods.ein_seld.training import Trainer
    from methods.ein_seld.inference import Inferer
    yaml = YAML(typ='safe')
    cfg = yaml.load((ROOT / 'configs/einv2/C0.yaml').read_text())
    hdf5 = args.project_root / 'EINV2/C0_CausalEINV2_seed2026/_hdf5'
    scalar = hdf5 / 'dcase2020task3/scalar/foa_causal_logmel&intensity_sr24000_nfft1024_hop600_mel256.h5'
    cfg.update(hdf5_dir=str(hdf5), causal_scalar_path=str(scalar))
    dataset = SimpleNamespace(clip_length=60, label_resolution=.1, label_set=list(range(14)))
    valid = SimpleNamespace(num_segments=15, valid_gt_sed_metrics2019=None,
                            valid_gt_doa_metrics2019=None, gt_metrics2020_dict=None)
    trainer = Trainer(SimpleNamespace(cuda=False), cfg, dataset, None, valid, None, None, None, None)
    inferer = Inferer(cfg, dataset, None, None, False)
    out = {'scope': 'historical code audit, no training', 'normalization': {
        'scalar': str(scalar), 'scalar_sha256': hashlib.sha256(scalar.read_bytes()).hexdigest(),
        'mean_max_abs_diff': float(np.max(np.abs(trainer.mean - inferer.mean))),
        'std_max_abs_diff': float(np.max(np.abs(trainer.std - inferer.std)))}, 'variants': {}}
    torch.manual_seed(999)
    x = torch.randn(1, 7, 24, 32)
    audio = torch.randn(1, 4, 19200) * .05
    feature = CausalLogmelIntensity_Extractor(cfg).eval()
    base = None
    with torch.no_grad():
        for variant in ('C0', 'C1', 'C2', 'C3'):
            recipe = yaml.load((ROOT / 'configs/einv2' / (variant + '.yaml')).read_text())
            model = load_model(variant, recipe)
            rng_after_init = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            if base is None:
                base = {key: value.clone() for key, value in model.state_dict().items()}
            common_diff = max((model.state_dict()[key] - value).abs().max().item() for key, value in base.items())
            model.eval()
            eval_pred = model(x)
            model.train()
            torch.manual_seed(777)
            train_pred = model(x)
            if variant == 'C0':
                reference_eval, reference_train = eval_pred, train_pred
            result = {'shared_init_max_abs_diff': common_diff, 'rng_after_init': rng_after_init,
                      'eval_max_abs_diff': max((eval_pred[k] - reference_eval[k]).abs().max().item() for k in ('sed', 'doa')),
                      'train_same_rng_max_abs_diff': max((train_pred[k] - reference_train[k]).abs().max().item() for k in ('sed', 'doa')),
                      'audio_prefix_checks': {}}
            model.eval()
            pred = model(feature(audio))
            for cutoff in (9600, 11400, 12000):
                changed = audio.clone()
                changed[..., cutoff:] = .2 * torch.randn_like(changed[..., cutoff:])
                after = model(feature(changed))
                result['audio_prefix_checks'][str(cutoff)] = {
                    'first_four_frames_max_diff': max((pred[k][:, :4] - after[k][:, :4]).abs().max().item() for k in ('sed', 'doa')),
                    'frame_index_4_max_diff': max((pred[k][:, 4] - after[k][:, 4]).abs().max().item() for k in ('sed', 'doa'))}
            out['variants'][variant] = result
            del model
    provenance = json.loads((ROOT / 'provenance/source_snapshot.json').read_text())
    checked = []
    for row in provenance['files']:
        if not row['path'].startswith('models/einv2/variants/') or '/C0_legacy/' in row['path']:
            continue
        if not row['path'].endswith('.py'):
            continue
        source = Path(row['source'])
        actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        checked.append({'path': row['source'], 'snapshot_sha256': row['sha256'], 'current_sha256': actual})
    out['server_sources'] = checked
    out['source_changed_since_snapshot'] = [r['path'] for r in checked if r['current_sha256'] != r['snapshot_sha256']]
    with h5py.File(scalar) as hf:
        out['scalar_metadata'] = {str(k): str(v) for k, v in hf.attrs.items()}
    out['scalar_generator'] = {'path': str(args.project_root / 'EINV2/C0_CausalEINV2_seed2026/seld/methods/data.py'),
                              'dev_file_count': len(list((hdf5 / 'dcase2020task3/data/24000fs/dev/foa').glob('*.h5'))),
                              'filter': 'BaseDataset includes all dev files, no train_fold filter'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'server_sources'}, indent=2))


if __name__ == '__main__':
    main()
