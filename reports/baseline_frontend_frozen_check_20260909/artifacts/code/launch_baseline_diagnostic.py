"""Single authorized diagnostic dispatch, two GPUs, no retry or follow-on work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)


def free_gpus():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                    '--format=csv,noheader'], text=True)
    occupied = set(apps.splitlines())
    gpus = []
    for line in rows.splitlines():
        index, uuid, name, memory = [x.strip() for x in line.split(',')]
        if '3090' in name and uuid not in occupied and int(memory) < 256:
            gpus.append(dict(index=index, uuid=uuid, name=name, memory_mib=int(memory)))
    return gpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--frontend-frozen-only', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    assert (root / 'AUTHORIZATION.md').exists()
    assert not (root / 'PLAN.json').exists(), 'No repeat dispatch'
    gpus = free_gpus()
    assert len(gpus) >= 2, 'No sharing, preemption, or GPU computation started'
    code = Path(__file__).resolve().parent
    old_code = a.source_root / 'code/refinement_v2'
    jobs = []
    for seed, gpu in zip((2027, 2028), gpus):
        guard = root / 'guards' / ('baseline_%d' % seed)
        command = [sys.executable, str(old_code / 'run_guarded.py'), '--directory', str(guard),
                   '--seconds', '3600', '--', sys.executable, str(code / 'diagnose_baseline_first_batch.py'),
                   '--source-root', str(a.source_root), '--baseline-seed', str(seed),
                   '--output', str(root / ('baseline_%d' % seed))]
        if a.frontend_frozen_only:
            command += ['--frontend-frozen-only', '--previous-root',
                        str(root.parent / 'baseline_repeat_diagnostic_20260909')]
        jobs.append(dict(baseline_seed=seed, gpu=gpu, command=command, guard=str(guard)))
    save(root / 'PLAN.json', dict(jobs=jobs, hard_timeout_seconds_per_baseline=3600,
        authorization_sha256=hashlib.sha256((root/'AUTHORIZATION.md').read_bytes()).hexdigest(),
        diagnostic_sha256=hashlib.sha256((code/'diagnose_baseline_first_batch.py').read_bytes()).hexdigest(),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started_unix=time.time(), automatic_retry=False, new_training_runs=0))
    processes = []
    for job in jobs:
        assert job['gpu']['uuid'] in {g['uuid'] for g in free_gpus()}, 'GPU became occupied'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=job['gpu']['uuid'], PYTHONHASHSEED='2026',
                   PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(old_code)+':'+str(a.source_root/'code/motion_decomposition'))
        with (root / ('launcher_%d.log' % job['baseline_seed'])).open('x') as log:
            proc = subprocess.Popen(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        processes.append(proc)
        save(root / ('DISPATCHED_%d.json' % job['baseline_seed']), dict(**job, guardian_pid=proc.pid))
    while any(p.poll() is None for p in processes):
        print(json.dumps(dict(alive=[p.poll() is None for p in processes], time=time.time())), flush=True)
        time.sleep(10)
    codes = [p.returncode for p in processes]
    save(root / 'EXIT.json', dict(exit_codes=codes, automatic_retry=False, no_follow_on=True))
    if any(codes):
        raise RuntimeError('Diagnostic failed; preserve evidence and request direction')


if __name__ == '__main__':
    main()
