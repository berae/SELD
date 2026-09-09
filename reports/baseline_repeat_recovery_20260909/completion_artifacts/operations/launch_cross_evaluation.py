"""Two authorized evaluation jobs, gated on all eight validation-selected seals."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('cache', 'score'), required=True)
    a = p.parse_args(); root = a.root.resolve()
    sys.path.insert(0, str(root/'code/refinement_v2'))
    sys.path.insert(0, str(root/'code/motion_decomposition'))
    from experiment_core import digest, save_json
    from launch_stability import free_gpus
    all_sealed = json.loads((root/'ALL_EIGHT_FROZEN.json').read_text())
    assert all_sealed['status'] == 'PASS' and len(all_sealed['runs']) == 8
    assert all_sealed['new_formal_runs'] == 8 and all_sealed['head_seed'] == 2026
    for s in all_sealed['seals']:
        frozen = root/f"C0_{s['C0_seed']}/frozen"
        assert digest(frozen/'FROZEN.json') == s['manifest_sha256']
        seal = json.loads((frozen/'FROZEN.json').read_text())
        for rel, h in seal['files_sha256'].items():
            assert digest(frozen/rel) == h, rel
    if a.stage == 'score':
        assert json.loads((root/'dispatch/evaluation_cache/COMPLETED.json').read_text())['status'] == 'PASS'
    dispatch = root/'dispatch'/('evaluation_'+a.stage)
    assert not dispatch.exists()
    devices = free_gpus() if a.stage == 'cache' else [None, None]
    assert len(devices) >= 2, 'Need two unused RTX3090s; no sharing, no launch'
    dispatch.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed, gpu in zip((2027, 2028), devices):
        cohort = root/f'C0_{seed}'; frozen = cohort/'frozen'; code = frozen/'code/refinement_v2'
        if a.stage == 'cache':
            assert json.loads((root/'PREPARED.json').read_text())['preserve_source_model_flags']
            command = [sys.executable, str(code/'cache_features.py'), '--config', str(frozen/'configs/pilot.json'),
                '--relocation', str(frozen/'configs/evaluation_relocation.json'), '--split', 'evaluation',
                '--output', str(cohort/'cache/evaluation'), '--preserve-source-model-flags']
        else:
            command = [sys.executable, str(code/'evaluate_frozen.py'), '--frozen', str(frozen),
                '--cache', str(cohort/'cache/evaluation'), '--output', str(cohort/'evaluation')]
        guard = root/'guards'/f'evaluation_{a.stage}_{seed}'
        wrapper = [sys.executable, str(code/'run_guarded.py'), '--directory', str(guard), '--seconds', '1800', '--', *command]
        jobs.append(dict(C0_seed=seed, gpu=gpu, command=wrapper, guard=str(guard), code=str(code)))
    save_json(dispatch/'PLAN.json', dict(jobs=jobs, script_sha256=digest(__file__),
        all_eight_frozen_sha256=digest(root/'ALL_EIGHT_FROZEN.json'), created_unix=time.time(),
        automatic_retry=False, new_blind_test=False, selection_on_evaluation=False))
    processes = []
    for job in jobs:
        if job['gpu']:
            assert job['gpu']['uuid'] in {g['uuid'] for g in free_gpus()}
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=job['gpu']['uuid'] if job['gpu'] else '',
            PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='2026',
            PYTHONPATH=job['code']+':'+str(Path(job['code']).parent/'motion_decomposition'))
        with (dispatch/f"{job['C0_seed']}.log").open('x') as log:
            proc = subprocess.Popen(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        job['guardian_pid'] = proc.pid; processes.append(proc)
    save_json(dispatch/'DISPATCHED.json', jobs)
    while any(p.poll() is None for p in processes):
        print(json.dumps(dict(stage=a.stage, alive=[p.poll() is None for p in processes])), flush=True)
        time.sleep(30)
    codes = [p.returncode for p in processes]
    save_json(dispatch/'EXIT.json', dict(exit_codes=codes, automatic_retry=False))
    assert not any(codes), 'Failed evaluation stage: no retry or next stage'
    save_json(dispatch/'COMPLETED.json', dict(status='PASS', jobs=2, stage=a.stage))


if __name__ == '__main__':
    main()
