"""Read already-produced files only; no model loading, forward pass, or retry."""
import argparse
import json
from pathlib import Path
from experiment_core import digest,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--old-base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;old=json.loads((a.old_base/'cache/validation/input_manifest.json').read_text());rows=[]
    for seed in (2027,2028):
        cohort=root/('C0_%d'%seed);source=root/'source'/('C0_%d'%seed)
        prior=json.loads((source/'validation/input_manifest.json').read_text())
        current=json.loads((cohort/'cache/validation/input_manifest.json').read_text())
        assert current['checkpoint_sha256']==prior['checkpoint_sha256']==digest(source/'best.pth')
        assert current['config']==prior['config']
        assert current['gt_files']==prior['gt_files']==old['gt_files']
        assert current['source_hashes']==prior['source_hashes']==old['source_hashes']
        assert current['waveform_inputs_sha256']==old['waveform_inputs_sha256']
        expected=json.loads((source/'validation/float_manifest.json').read_text())
        for filename,h in expected.items():assert digest(source/'validation/float'/filename)==h
        failed=root/'guards/cache_both_C0'/('%d_validation'%seed)
        exit_record=json.loads((failed/'EXIT.json').read_text());assert exit_record['exit_code']==1
        log=(failed/'process.log').read_text()
        assert "('fold1_room1_mix001_ov1', 'sed', 'original raw mismatch')" in log
        train=json.loads((cohort/'cache/train/COMPLETED.json').read_text());assert train['status']=='PASS' and train['files']==500
        assert not (cohort/'heads').exists()
        rows.append(dict(C0_seed=seed,checkpoint_identity_exact=True,checkpoint_config_exact=True,
            runtime_and_GT_hashes_exact=True,current_waveform_hashes_equal_prior_C0_2026_validation=True,
            historical_float_hashes_verified=len(expected),historical_export_command=prior['command'],
            historical_gpu=prior['gpu'],current_gpu=current['gpu'],historical_torch=prior['torch'],current_torch=current['torch'],
            failed_recording='fold1_room1_mix001_ov1',failed_field='sed',failure='bitwise_float_mismatch',
            numerical_delta=None,CSV_comparison=None,official_metric_comparison=None,
            unavailable_reason='assertion precedes writing first-record float/CSV; no diagnostic re-forward authorized or performed',
            validation_saved_float_files=len(list((cohort/'cache/validation/float').glob('*.npz'))),
            original_exit=exit_record,train_cache_completion=train,formal_head_runs_started=0,
            source_hashes=dict(historical_manifest=digest(source/'validation/input_manifest.json'),
                current_manifest=digest(cohort/'cache/validation/input_manifest.json'),failure_log=digest(failed/'process.log'))))
    save_json(a.output,dict(status='BLOCKED_AT_VALIDATION_FLOAT_REGRESSION',rows=rows,
        new_training_runs=0,maximum_authorized_training_runs=8,automatic_retry=False,
        suspected_cause='Unresolved; differing historical exporter contexts and GPU hardware are investigation leads, not established causes',
        script_sha256=digest(__file__),scope='read-only provenance/file checks; zero new inference or training'))
    print(json.dumps(dict(status='AUDIT_COMPLETE',new_training_runs=0,reference_float_files_verified=200)))


if __name__=='__main__':main()
