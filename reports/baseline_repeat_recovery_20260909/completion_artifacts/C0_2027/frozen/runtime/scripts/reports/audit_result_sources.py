"""Read-only file-evidence audit; no training, inference or checkpoint loading.

Run on rabbit02 with --repository-root pointing to the exported repository.
JSON goes to stdout. Checkpoint existence/size does not prove loadability.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_info(value, hash_content=True):
    path = Path(value.split('#', 1)[0])
    info = {'path': value, 'exists': path.is_file()}
    if info['exists']:
        stat = path.stat()
        info.update(bytes=stat.st_size, modified_at_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
        if hash_content:
            info['sha256'] = sha256(path)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    repo, root = args.repository_root, args.project_root
    evidence = list(csv.DictReader((repo / 'reports/aligned_20260904/evidence_index.csv').open(encoding='utf-8-sig')))
    out = {'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
           'scope': 'File existence, sizes, config/result/code hashes and prediction filename coverage. No training/inference rerun or checkpoint deserialization.',
           'rows': [], 'source_tables': [], 'architecture_sources': []}
    for row in evidence:
        item = {key: row[key] for key in ('family', 'context', 'variant', 'weight', 'seed', 'split', 'run_id')}
        for key in ('config', 'checkpoint', 'source'):
            item[key] = file_info(row[key], hash_content=key != 'checkpoint')
        pred = Path(row['prediction_dir'])
        names = {p.name for p in pred.glob('*.csv') if not p.name.startswith('.')}
        meta = root / 'EINV2/dataset_root' / ('metadata_eval' if row['split'] == 'evaluation' else 'metadata_dev')
        refs = {p.name for p in meta.glob('*.csv' if row['split'] == 'evaluation' else 'fold1*.csv') if not p.name.startswith('.')}
        item['predictions'] = {'path': str(pred), 'exists': pred.is_dir(), 'files': len(names),
                               'expected': len(refs), 'missing': sorted(refs - names), 'extra': sorted(names - refs)}
        if row['family'] == 'EINV2':
            item['training_metrics'] = file_info(str(Path(row['checkpoint']).parent / 'metrics_statistics.csv'))
        out['rows'].append(item)
    aligned = root / 'experiment_completion_20260904/results/metric_alignment_20260904'
    for name in ('aligned_run_metrics.csv', 'aligned_aggregate_metrics.csv', 'aligned_paired_seed_deltas.csv', 'status.json'):
        out['source_tables'].append(file_info(str(aligned / name)))
    final = root / 'experiment_completion_20260904/final_statistics_20260904'
    for name in ('completion_motion_per_run.csv', 'completion_motion_support.csv', 'completion_new_training.csv', 'completion_qa_checks.json'):
        out['source_tables'].append(file_info(str(final / name)))
    provenance = json.loads((repo / 'provenance/source_snapshot.json').read_text(encoding='utf-8'))
    for row in provenance['files']:
        p = row['path']
        if ('/ein_seld/models/' in p or p.endswith('/ein_seld/losses.py') or p.endswith('/ein_seld/training.py')
                or p.endswith('/feature_causal.py') or p.endswith('/model_utilities.py') or '/tools/build_' in p
                or p in ('models/multi_accdoa/seldnet_model.py', 'models/multi_accdoa/parameters.py', 'models/multi_accdoa/train_seldnet.py')):
            if '/C0_legacy/' not in p:
                item = file_info(row['source'])
                item['repository_path'] = p
                item['repository_sha256'] = sha256(repo / p)
                item['matches_repository'] = item.get('sha256') == item['repository_sha256']
                out['architecture_sources'].append(item)
    out['summary'] = {
        'run_split_rows': len(out['rows']),
        'missing_config_checkpoint_or_result': sum(not r[k]['exists'] for r in out['rows'] for k in ('config', 'checkpoint', 'source')),
        'prediction_coverage_failures': sum(bool(r['predictions']['missing'] or r['predictions']['extra']) or not r['predictions']['exists'] for r in out['rows']),
        'zero_byte_checkpoints': sum(r['checkpoint'].get('bytes') == 0 for r in out['rows']),
        'code_matches': sum(r['matches_repository'] for r in out['architecture_sources']),
        'code_checked': len(out['architecture_sources']),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
