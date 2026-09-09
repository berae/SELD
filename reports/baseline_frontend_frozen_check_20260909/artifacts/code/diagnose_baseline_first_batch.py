"""Authorized single-batch diagnosis only; never repairs or releases caches."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def difference(a, b):
    import numpy as np
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(array_equal=bool(np.array_equal(a, b)), shape=list(a.shape),
                unequal_elements=int(np.count_nonzero(a != b)), elements=int(a.size),
                max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                p99_abs=float(np.quantile(delta, .99)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--baseline-seed', type=int, choices=[2027, 2028], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frontend-frozen-only', action='store_true',
                   help='One separately authorized forward with exact default model flags and frozen frontend')
    p.add_argument('--previous-root', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    root = a.source_root.resolve()
    prepared = json.loads((root / 'PREPARED.json').read_text())
    source = next(x for x in prepared['sources'] if x['C0_seed'] == a.baseline_seed)
    for rel, expected in prepared['code_hashes'].items():
        assert digest(root / 'code' / rel) == expected, rel
    for name, expected in prepared['config_hashes'].items():
        assert digest(root / 'configs' / name) == expected, name
    run = root / 'source' / ('C0_%d' % a.baseline_seed)
    assert digest(run / 'best.pth') == source['C0_sha256']
    assert digest(run / 'config.json') == source['config_sha256']
    relocation = json.loads((root / 'configs' / ('relocation_C0_%d.json' % a.baseline_seed)).read_text())
    assert Path(relocation['run']).resolve() == run.resolve()
    runtime = Path(relocation['runtime'])
    sys.path[:0] = [str(runtime / 'models/einv2/audited'),
                   str(root / 'code/refinement_v2'), str(root / 'code/motion_decomposition')]
    import numpy as np
    import torch
    import runner
    from candidate import fixed_records
    from experiment_core import csv_bytes
    from probe_frozen_c0 import state_digest
    torch.set_num_threads(4)
    cfg_disk = json.loads((run / 'config.json').read_text())
    assert cfg_disk['audit']['seed'] == a.baseline_seed
    assert runner.source_manifest() == cfg_disk['audit']['source_hashes']
    assert digest(relocation['causal_scalar_path']) == cfg_disk['audit']['scalar_sha256']
    ck = torch.load(run / 'best.pth', map_location='cpu', weights_only=False)
    assert ck['config'] == cfg_disk
    for key in ('optimizer', 'scheduler', 'rng', 'cuda_rng', 'np_rng', 'random_rng'):
        ck.pop(key, None)
    runner.seed_all(a.baseline_seed)
    cfg = copy.deepcopy(cfg_disk)
    for key in ('causal_scalar_path', 'dataset_dir', 'hdf5_dir'):
        cfg[key] = relocation[key]
    assert cfg['inference']['batch_size'] == 32
    cfg['inference'].update(testset_type='dev', test_fold='1')
    dataset = runner.get_dataset(cfg['dataset'], cfg['dataset_dir'])
    ds, generator, _ = runner.get_generator(runner.data_args(a.baseline_seed, 4), cfg, dataset, 'test')
    assert len(ds) == 1500
    model = runner.AuditedEINV2(cfg, dataset).cuda().eval()
    default_model_flags = {name: param.requires_grad for name, param in model.named_parameters()}
    model.requires_grad_(False)
    frontend = runner.Frontend(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model'])
    frontend.load_state_dict(ck['frontend'])
    del ck
    before = state_digest(model), state_digest(frontend)
    iterator = iter(generator)
    batch = next(iterator)  # Exactly one batch; no loop over the dataset.
    del iterator
    wave = batch['waveform'].cuda(non_blocking=True)
    names = [str(x) for x in batch['filename']]
    segments = [int(x) for x in batch['n_segment']]
    assert len(names) == 32 and names[0] == 'fold1_room1_mix001_ov1'
    assert all(0 <= s < 15 for s in segments)
    history = Path(relocation['original_validation'])
    historical = {}
    reference_hashes = {}
    for name in sorted(set(names)):
        rel = 'validation/float/' + name + '.npz'
        path = history / 'float' / (name + '.npz')
        assert digest(path) == source['files_sha256'][rel]
        reference_hashes[rel] = digest(path)
        with np.load(path) as data:
            historical[name] = {k: data[k] for k in ('sed', 'probability', 'doa')}
    reference = {k: np.stack([historical[n][k][s*40:(s+1)*40]
                             for n, s in zip(names, segments)]) for k in ('sed', 'probability', 'doa')}
    np.savez_compressed(a.output / 'historical_first_batch.npz', **reference,
                        filename=np.asarray(names), segment=np.asarray(segments))
    manifest = dict(baseline_seed=a.baseline_seed, checkpoint_sha256=source['C0_sha256'],
                    prepared_sha256=digest(root / 'PREPARED.json'),
                    source_hashes=runner.source_manifest(), config=cfg_disk, relocation=relocation,
                    config_hashes=prepared['config_hashes'], code_hashes=prepared['code_hashes'],
                    script_sha256=digest(__file__), reference_hashes=reference_hashes,
                    actual_python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                    visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), command=sys.argv,
                    optimizer_updates=0, batches=1, batch_chunks=len(names),
                    batch_mapping=list(zip(names, segments)),
                    input_tensor_sha256=hashlib.sha256(batch['waveform'].contiguous().numpy().tobytes()).hexdigest(),
                    inference_scope='Same first batch only, controlled export contexts; not new method candidates',
                    cache_release_authorized=False, full_validation_recovery_authorized=False)
    # Only hash waveform files represented in this batch; do not read new batches.
    paths = sorted({Path(str(x).split('%')[0]) for x in ds.paths_list
                    if Path(str(x).split('%')[0]).stem in set(names)})
    manifest['waveform_files_sha256'] = {str(x): digest(x) for x in paths}
    if a.frontend_frozen_only:
        assert a.previous_root is not None
        previous = a.previous_root / ('baseline_%d' % a.baseline_seed)
        old_hashes = json.loads((previous / 'OUTPUT_HASHES.json').read_text())
        for rel in ('INPUT_MANIFEST.json', 'frozen_hook_no_grad/first_batch.npz'):
            assert digest(previous / rel) == old_hashes[rel], rel
        prior_manifest = json.loads((previous / 'INPUT_MANIFEST.json').read_text())
        assert manifest['input_tensor_sha256'] == prior_manifest['input_tensor_sha256']
        assert manifest['waveform_files_sha256'] == prior_manifest['waveform_files_sha256']
        assert [list(x) for x in manifest['batch_mapping']] == prior_manifest['batch_mapping']
        assert manifest['checkpoint_sha256'] == prior_manifest['checkpoint_sha256']
        manifest.update(previous_manifest_sha256=digest(previous / 'INPUT_MANIFEST.json'),
                        same_input_as_previous_diagnostic=True,
                        default_model_requires_grad=default_model_flags,
                        frontend_requires_grad={n: p.requires_grad for n,p in frontend.named_parameters()},
                        forward_count=1)
        assert not any(p.requires_grad for p in frontend.parameters())
    save(a.output / 'INPUT_MANIFEST.json', manifest)
    cases = [('frozen_hook_no_grad', False, True, False),
             ('frozen_no_hook_no_grad', False, False, False),
             ('historical_flags_no_hook_no_grad', True, False, False),
             ('historical_flags_hook_no_grad', True, True, False),
             ('frozen_hook_inference_mode', False, True, True)]
    if a.frontend_frozen_only:
        cases = [('default_model_flags_frontend_frozen', True, True, False)]
    outputs = {}
    results = {}
    for label, requires_grad, use_hook, inference in cases:
        if a.frontend_frozen_only:
            for name, param in model.named_parameters():
                param.requires_grad_(default_model_flags[name])
            assert not any(p.requires_grad for p in frontend.parameters())
        else:
            model.requires_grad_(requires_grad)
            frontend.requires_grad_(requires_grad)
        hidden = {}
        def capture(index):
            def hook(module, inputs):
                hidden[index] = inputs[0].detach().clone()
            return hook
        handles = [head.register_forward_pre_hook(capture(i)) for i, head in enumerate(model.doa_heads)] if use_hook else []
        with torch.inference_mode() if inference else torch.no_grad():
            pred = model(frontend(wave))
            values = {k: pred[k].detach().cpu().numpy().copy() for k in ('sed', 'doa')}
            values['probability'] = pred['sed'].sigmoid().cpu().numpy().copy()
            if use_hook:
                values['doa_features'] = torch.stack([hidden[i] for i in range(2)], dim=2).cpu().numpy().copy()
        for handle in handles:
            handle.remove()
        assert all(np.isfinite(v).all() for v in values.values())
        outputs[label] = values
        case_dir = a.output / label
        case_dir.mkdir()
        np.savez_compressed(case_dir / 'first_batch.npz', **values)
        comparisons = {k: difference(values[k], reference[k]) for k in reference}
        active = values['probability'].max(-1) > .5
        old_active = reference['probability'].max(-1) > .5
        classes = values['probability'].argmax(-1)
        old_classes = reference['probability'].argmax(-1)
        entry = dict(requires_grad_flag=requires_grad, grad_enabled_during_forward=False,
                     hook=use_hook, inference_mode=inference, vs_historical=comparisons,
                     activity_changes=int(np.count_nonzero(active != old_active)),
                     active_class_changes=int(np.count_nonzero(active & old_active & (classes != old_classes))),
                     complete_recordings=[])
        if a.frontend_frozen_only:
            entry.update(frontend_frozen=True, model_default_flags_restored=True)
        metric_new = runner.AlignedMetrics(classes=14, threshold=20, frames=600, frames_per_second=10)
        metric_old = runner.AlignedMetrics(classes=14, threshold=20, frames=600, frames_per_second=10)
        for name in sorted(set(names)):
            positions = sorted((s, i) for i, (n, s) in enumerate(zip(names, segments)) if n == name)
            if [s for s, _ in positions] != list(range(15)):
                continue
            out = {k: np.concatenate([values[k][i] for _, i in positions]) for k in reference}
            records = fixed_records(out['probability'], out['doa'])
            raw = csv_bytes(records)
            path = case_dir / (name + '.csv')
            with path.open('xb') as f:
                f.write(raw)
            old_path = history / 'raw' / (name + '.csv')
            rel = 'validation/raw/' + name + '.csv'
            assert digest(old_path) == source['files_sha256'][rel]
            old_records = fixed_records(historical[name]['probability'], historical[name]['doa'])
            assert csv_bytes(old_records) == old_path.read_bytes()
            gt_path = Path(cfg['dataset_dir']) / 'metadata_dev' / (name + '.csv')
            hist_manifest = json.loads((history / 'input_manifest.json').read_text())
            assert digest(gt_path) == hist_manifest['gt_files'][name + '.csv']
            gt = runner.load_csv(gt_path, 'polar5')
            metric_new.update(runner.load_csv(path, 'polar4'), gt)
            metric_old.update(runner.load_csv(old_path, 'polar4'), gt)
            entry['complete_recordings'].append(dict(filename=name, records=len(records),
                historical_records=len(old_records), csv_bytes_equal=raw == old_path.read_bytes(),
                ordered_detection_entries_equal=[x[:3] for x in records] == [x[:3] for x in old_records],
                changed_rows=sum(x != y for x, y in zip(records, old_records)) if len(records) == len(old_records) else None,
                historical_csv_sha256=digest(old_path), diagnostic_csv_sha256=digest(path), gt_sha256=digest(gt_path)))
        if entry['complete_recordings']:
            old_scores, new_scores = metric_old.scores(), metric_new.scores()
            entry['subset_scores'] = dict(historical=old_scores, diagnostic=new_scores,
                delta={p: {k: new_scores[p][k] - old_scores[p][k] for k in old_scores[p]} for p in old_scores},
                scope='Only complete recordings contained in first batch; NOT full validation score')
        results[label] = entry
        save(case_dir / 'COMPARISON.json', entry)
        print(json.dumps(dict(case=label, baseline=a.baseline_seed, comparisons=comparisons,
                              elapsed_seconds=time.time()-start)), flush=True)
    pairwise = {}
    if a.frontend_frozen_only:
        with np.load(previous / 'frozen_hook_no_grad/first_batch.npz') as old:
            pairwise['vs_previous_frozen_export'] = {
                k: difference(outputs[cases[0][0]][k], old[k]) for k in outputs[cases[0][0]]}
    for label in outputs:
        if label == cases[0][0]:
            continue
        pairwise[label + '_vs_current_export'] = {
            k: difference(outputs[label][k], outputs[cases[0][0]][k])
            for k in outputs[label] if k in outputs[cases[0][0]]}
    model.requires_grad_(False)
    frontend.requires_grad_(False)
    assert before == (state_digest(model), state_digest(frontend))
    assert all(p.grad is None for m in (model, frontend) for p in m.parameters())
    save(a.output / 'RESULTS.json', dict(baseline_seed=a.baseline_seed,
        status='DIAGNOSTIC_COMPLETED_NOT_CACHE_RELEASE', elapsed_seconds=time.time()-start,
        cases=results, pairwise=pairwise, parameters_buffers_unchanged=True, optimizer_updates=0,
        full_validation_run=False, training_started=False, strict_gate_unchanged=True,
        historical_script_bytes_at_original_execution_not_proven=True,
        frontend_frozen_control=a.frontend_frozen_only,
        cache_release=False, peak_memory_allocated_bytes=torch.cuda.max_memory_allocated()))
    save(a.output / 'OUTPUT_HASHES.json', {str(p.relative_to(a.output)): digest(p)
         for p in sorted(a.output.rglob('*')) if p.is_file()})


if __name__ == '__main__':
    main()
