"""Read-only comparison of rebuilt training caches against preserved originals."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment_core import digest, save_json


def main():
    p = argparse.ArgumentParser()
    for key in ('old', 'new', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    reports = []
    for seed in (2027, 2028):
        old = a.old / f'C0_{seed}/cache/train'
        new = a.new / f'C0_{seed}/cache/train'
        manifests = [json.loads((r/'file_manifest.json').read_text()) for r in (old, new)]
        inputs = [json.loads((r/'input_manifest.json').read_text()) for r in (old, new)]
        assert inputs[0]['checkpoint_sha256'] == inputs[1]['checkpoint_sha256']
        assert inputs[0]['gt_files'] == inputs[1]['gt_files']
        assert inputs[0]['waveform_inputs_sha256'] == inputs[1]['waveform_inputs_sha256']
        for root in (old, new):
            done = json.loads((root/'COMPLETED.json').read_text())
            assert done['status'] == 'PASS' and done['files'] == 500
        assert set(manifests[0]) == set(manifests[1])
        totals = {}; changed_files = []
        for name in sorted(manifests[0]):
            for root, manifest in zip((old, new), manifests):
                assert digest(root/name) == manifest[name], (root, name)
            if not name.endswith('.npz'):
                key = name.split('/')[0] + '_bytes'
                equal = (old/name).read_bytes() == (new/name).read_bytes()
                totals.setdefault(key, dict(files=0, different_files=0))
                totals[key]['files'] += 1
                totals[key]['different_files'] += int(not equal)
                if not equal:
                    changed_files.append(dict(file=name, field='bytes'))
                continue
            with np.load(old/name, allow_pickle=False) as x, np.load(new/name, allow_pickle=False) as y:
                assert set(x.files) == set(y.files)
                for field in x.files:
                    u, v = x[field], y[field]
                    assert u.shape == v.shape and u.dtype == v.dtype
                    key = name.split('/')[0] + '/' + field
                    item = totals.setdefault(key, dict(files=0, elements=0, different_files=0, different_elements=0, max_abs_delta=None))
                    different = int(np.count_nonzero(u != v))
                    item['files'] += 1; item['elements'] += int(u.size)
                    item['different_files'] += int(different > 0); item['different_elements'] += different
                    if np.issubdtype(u.dtype, np.number) or u.dtype == np.bool_:
                        delta = float(np.max(np.abs(u.astype(np.float64) - v.astype(np.float64)))) if u.size else 0.0
                        item['max_abs_delta'] = max(item['max_abs_delta'] or 0.0, delta)
                    if different:
                        changed_files.append(dict(file=name, field=field, different_elements=different))
        reports.append(dict(baseline_seed=seed, totals=totals, changed_files=changed_files,
                            old_manifest_sha256=digest(old/'file_manifest.json'),
                            new_manifest_sha256=digest(new/'file_manifest.json'),
                            actual_waveforms_and_GT_identical=True, old_cache_reused=False))
        print(json.dumps(dict(baseline_seed=seed, compared_files=len(manifests[0]), status='COMPARED')), flush=True)
    save_json(a.output, dict(status='COMPARED', reports=reports, script_sha256=digest(__file__),
                            interpretation='Differences are retained, not treated as tolerance-based PASS; all fitting uses the rebuilt cache exclusively.'))


if __name__ == '__main__':
    main()
