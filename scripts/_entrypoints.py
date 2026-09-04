"""Portable, explicit train/infer/preprocess entry points. No automatic GPU selection."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
EINV2 = ('C0', 'C1', 'C2', 'C3', 'E0', 'E1', 'E2', 'E3')
MULTI = ('A0', 'A1', 'A2', 'A3', 'C0', 'C1', 'C2', 'C3')


def bootstrap(path):
    sys.path.insert(0, str(path))


def common(parser, variants):
    parser.add_argument('--variant', choices=variants, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--dataset-root', default=os.environ.get('SELD_DATA_ROOT'))
    parser.add_argument('--output-root', default=str(ROOT / 'outputs'))
    parser.add_argument('--run-id')
    parser.add_argument('--dry-run', action='store_true')


def einv2(mode):
    parser = argparse.ArgumentParser(description='EINV2 ' + mode + ' (explicit model/config/checkpoint).')
    common(parser, EINV2)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--hdf5-root', default=os.environ.get('SELD_EINV2_HDF5_ROOT'))
    parser.add_argument('--scalar-path', default=os.environ.get('SELD_EINV2_SCALAR_PATH'))
    parser.add_argument('--velocity-root', default=os.environ.get('SELD_EINV2_VELOCITY_ROOT'))
    parser.add_argument('--jepa-root', default=os.environ.get('SELD_EINV2_JEPA_ROOT'))
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--cpu', action='store_true')
    if mode == 'infer':
        parser.add_argument('--checkpoint', type=Path, required=True)
        parser.add_argument('--split', choices=('validation', 'evaluation'), default='evaluation')
    if mode == 'preprocess':
        parser.add_argument('--operation', choices=('extract_data', 'extract_scalar', 'extract_meta'), required=True)
        parser.add_argument('--dataset-type', choices=('dev', 'eval'), default='dev')
    args = parser.parse_args()
    config = args.config or ROOT / 'configs/einv2' / (args.variant + '.yaml')
    source_variant = 'C0' if args.variant == 'E0' else args.variant
    source = ROOT / 'models/einv2/variants' / source_variant / 'seld'
    run_id = args.run_id or '{}_seed{}_v010{}'.format(
        args.variant, args.seed, '_' + args.split if mode == 'infer' else '')
    if Path(run_id).name != run_id or '/' in run_id or '\\' in run_id or run_id in ('.', '..'):
        parser.error('--run-id must be a single directory name')
    print(json.dumps(dict(model='EINV2', variant=args.variant, mode=mode, seed=args.seed,
                         config=str(config), source=str(source), run_id=run_id,
                         metric_protocol='dcase2023_micro_full_duration', options=vars(args)),
                     default=str, indent=2))
    if args.dry_run:
        return 0
    if not args.dataset_root or not args.hdf5_root:
        parser.error('--dataset-root and --hdf5-root (or their SELD_* env vars) are required')
    from ruamel.yaml import YAML
    cfg = YAML(typ='safe').load(config.read_text(encoding='utf-8'))
    cfg['dataset_dir'] = str(Path(args.dataset_root).resolve())
    cfg['hdf5_dir'] = str(Path(args.hdf5_root).resolve())
    cfg['workspace_dir'] = str(Path(args.output_root).resolve())
    for arg, key in ((args.scalar_path, 'causal_scalar_path'),
                     (args.velocity_root, 'velocity_hdf5_dir'), (args.jepa_root, 'jepa_hdf5_dir')):
        if arg:
            cfg[key] = str(Path(arg).resolve())
    # Inference constructs these Path objects but does not read auxiliary targets.
    for key in ('velocity_hdf5_dir', 'jepa_hdf5_dir'):
        if key in cfg and cfg[key] is None:
            cfg[key] = str(Path(args.hdf5_root).resolve() / key)
    # Cache/scaler conventions differ between causal and offline historical runs.
    # Require the matching explicit scalar instead of silently changing IV preprocessing.
    if mode != 'preprocess' and 'causal_scalar_path' in cfg and not args.scalar_path:
        parser.error('this config requires --scalar-path; choose the matching causal/offline scaler')
    for key, required in (('velocity_hdf5_dir', 'velocity_hdf5_dir' in cfg),
                          ('jepa_hdf5_dir', 'jepa_hdf5_dir' in cfg)):
        if mode == 'train' and required and not (args.velocity_root if key.startswith('velocity') else args.jepa_root):
            parser.error('training requires explicit --' + ('velocity-root' if key.startswith('velocity') else 'jepa-root'))
    cfg['training']['train_id'] = run_id
    cfg['training']['remark'] = run_id
    cfg['inference']['infer_id'] = run_id
    cfg['inference']['models'] = cfg['training']['model']
    cfg['metric_protocol'] = 'dcase2023_micro_full_duration'
    if mode == 'infer':
        if not args.checkpoint.is_file():
            parser.error('--checkpoint does not exist')
        cfg['inference']['checkpoint_paths'] = [str(args.checkpoint.resolve())]
        cfg['inference']['testset_type'] = 'eval' if args.split == 'evaluation' else 'dev'
        cfg['inference']['test_fold'] = 'None' if args.split == 'evaluation' else '1'
    ns = SimpleNamespace(mode=mode, seed=args.seed, num_workers=args.num_workers,
                         no_cuda=args.cpu, read_into_mem=False,
                         preproc_mode=getattr(args, 'operation', None),
                         dataset_type=getattr(args, 'dataset_type', None))
    bootstrap(source)
    from main import main
    return main(ns, cfg)


def multi_train():
    parser = argparse.ArgumentParser(description='Train Multi-ACCDOA; never access the held-out evaluation split.')
    common(parser, MULTI)
    parser.add_argument('--feature-root', default=os.environ.get('SELD_FEATURE_DIR'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--stage', choices=('full', 'smoke'), default='full')
    args = parser.parse_args()
    config = args.config or ROOT / 'configs/multi_accdoa' / (args.variant + '.json')
    recipe = json.loads(config.read_text(encoding='utf-8'))
    if recipe['variant'] != args.variant:
        parser.error('--variant differs from config')
    run_id = args.run_id or '{}_seed{}_v010'.format(args.variant, args.seed)
    if Path(run_id).name != run_id or '/' in run_id or '\\' in run_id or run_id in ('.', '..'):
        parser.error('--run-id must be a single directory name')
    run_root = Path(args.output_root).resolve() / 'multi_accdoa' / run_id
    print(json.dumps(dict(mode='train', recipe=recipe, run_root=str(run_root), options=vars(args)), default=str, indent=2))
    if args.dry_run:
        return 0
    if not args.dataset_root or not args.feature_root:
        parser.error('--dataset-root and --feature-root are required')
    if '2020' not in str(args.dataset_root):
        parser.error('upstream split detection requires "2020" in dataset-root; e.g. TAU2020_SELD_dataset')
    if run_root.exists():
        raise FileExistsError('Run output exists: ' + str(run_root))
    os.environ.update(SELD_SEED=str(args.seed), SELD_EVALUATE_TEST='0',
                      SELD_TAU2020_ROOT=str(Path(args.dataset_root).resolve()),
                      SELD_FEATURE_DIR=str(Path(args.feature_root).resolve()),
                      SELD_RUN_ROOT=str(run_root),
                      SELD_LAMBDA_VELOCITY=str(recipe['lambda_velocity']),
                      SELD_LAMBDA_JEPA=str(recipe['lambda_jepa']))
    bootstrap(ROOT / 'models/multi_accdoa')
    from experiment_matrix import get_task_id
    from train_seldnet import main
    return main(['train_seldnet.py', get_task_id(args.variant, args.stage), run_id])


def multi_infer():
    parser = argparse.ArgumentParser(description='Evaluate a saved Multi-ACCDOA checkpoint; no optimizer or training.')
    parser.add_argument('--manifest', type=Path, required=True, help='completed training run manifest')
    parser.add_argument('--checkpoint', type=Path, help='override the manifest checkpoint after migration')
    parser.add_argument('--dataset-root')
    parser.add_argument('--feature-root')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'evaluation'), default='evaluation')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest.get('status') not in ('completed', 'validation_completed'):
        parser.error('manifest is not a completed training run')
    params = dict(manifest['params'])
    # Use the declared common comparison protocol, independent of legacy manifests.
    params.update(average='micro', lad_doa_thresh=20, max_audio_len_s=60,
                  label_hop_len_s=0.1, metric_protocol='dcase2023_micro_full_duration')
    for arg, key in ((args.dataset_root, 'dataset_dir'), (args.feature_root, 'feat_label_dir')):
        if arg:
            params[key] = str(Path(arg).resolve())
    manifest['params'] = params
    if args.checkpoint:
        manifest['checkpoint'] = str(args.checkpoint.resolve())
    print(json.dumps(dict(mode='infer', checkpoint=manifest['checkpoint'], options=vars(args)), default=str, indent=2))
    if args.dry_run:
        return 0
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    bootstrap(ROOT / 'models/multi_accdoa')
    import torch
    from analysis.evaluate_threshold_sensitivity import evaluate_seed
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    rows = evaluate_seed(manifest['variant'], manifest['seed'], args.split, [args.threshold], device,
                         manifest=manifest, run_label=manifest['job_id'], output_root=args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(metric_protocol='dcase2023_micro_full_duration', manifest=str(args.manifest.resolve()),
                  checkpoint=manifest['checkpoint'], split=args.split, results=rows)
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0


def multi_preprocess():
    parser = argparse.ArgumentParser(description='Extract Multi-ACCDOA features (separate from training/testing).')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--split', choices=('dev', 'eval'), required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))
    if args.dry_run:
        return 0
    os.environ.update(SELD_TAU2020_ROOT=str(Path(args.dataset_root).resolve()),
                      SELD_FEATURE_DIR=str(Path(args.feature_root).resolve()))
    bootstrap(ROOT / 'models/multi_accdoa')
    from parameters import get_params
    from cls_feature_class import FeatureClass
    params = get_params('35')
    feature = FeatureClass(params, is_eval=args.split == 'eval')
    feature.extract_all_feature()
    feature.preprocess_features()
    if args.split == 'dev':
        feature.extract_all_labels()
    return 0
