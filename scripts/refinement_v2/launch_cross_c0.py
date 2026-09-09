"""One explicit wave only; no resume, retry, expansion, or evaluation selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiment_core import digest,save_json
from launch_stability import free_gpus


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['cache','train'],required=True)
    p.add_argument('--C0-seed',type=int,choices=[2027,2028])
    a=p.parse_args();root=a.root.resolve();code=Path(__file__).parent.resolve()
    prepared=json.loads((root/'PREPARED.json').read_text());assert prepared['status']=='PASS'
    assert code == (root/'code/refinement_v2').resolve(), 'Execute the verified snapshot itself'
    for relative,h in prepared['code_hashes'].items():assert digest(root/'code'/relative)==h,relative
    for name,h in prepared['config_hashes'].items():assert digest(root/'configs'/name)==h,name
    wave='cache_both_C0' if a.stage=='cache' else 'train_C0_%d'%a.C0_seed
    dispatch=root/'dispatch'/wave;dispatch.mkdir(parents=True,exist_ok=False)
    jobs=[]
    if a.stage=='cache':
        assert a.C0_seed is None
        items=[(seed,split) for seed in (2027,2028) for split in ('validation','train')]
    else:
        assert a.C0_seed in (2027,2028)
        cohort=root/('C0_%d'%a.C0_seed)
        assert json.loads((cohort/'PREFLIGHT.json').read_text())['status']=='PASS'
        items=[(a.C0_seed,c) for c in ('F-Deriv','R0','R1','R2')]
        assert not (cohort/'heads').exists(),'Never retrain an existing cohort head wave'
    gpus=free_gpus()
    if len(gpus)<len(items):
        save_json(dispatch/'BLOCKED.json',dict(reason='fewer than four unused RTX3090s',launched=0))
        raise RuntimeError('No sharing or preemption; no jobs launched')
    for (seed,item),gpu in zip(items,gpus):
        cohort=root/('C0_%d'%seed);config=root/'configs'/('pilot_C0_%d.json'%seed)
        if a.stage=='cache':
            command=[sys.executable,str(code/'cache_features.py'),'--config',str(config),
                '--relocation',str(root/'configs'/('relocation_C0_%d.json'%seed)),
                '--split',item,'--output',str(cohort/'cache'/item)]
            if prepared.get('preserve_source_model_flags',False):
                command.append('--preserve-source-model-flags')
            seconds=1800
        else:
            command=[sys.executable,str(code/'train_heads.py'),'--config',str(config),
                '--execution',str(root/'configs/execution_cross_c0.json'),'--cache',str(cohort/'cache'),
                '--scorer',str(root/'scorer'),'--condition',item,'--output',str(cohort/'heads'/item)]
            seconds=14400
        guard=root/'guards'/wave/('%d_%s'%(seed,item))
        wrapper=[sys.executable,str(code/'run_guarded.py'),'--directory',str(guard),'--seconds',str(seconds),'--',*command]
        jobs.append(dict(C0_seed=seed,head_seed=2026 if a.stage=='train' else None,item=item,gpu=gpu,
                         command=wrapper,guard=str(guard)))
    save_json(dispatch/'PLAN.json',dict(jobs=jobs,authorization_sha256=prepared['authorization_sha256'],
        preregistration_sha256=prepared['preregistration_sha256'],created_unix=time.time(),automatic_retry=False))
    processes=[]
    for job in jobs:
        assert job['gpu']['uuid'] in {g['uuid'] for g in free_gpus()},'GPU became occupied before launch'
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=job['gpu']['uuid'],PYTHONHASHSEED='2026')
        with (dispatch/('%d_%s.log'%(job['C0_seed'],job['item']))).open('x') as log:
            proc=subprocess.Popen(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        job['guardian_pid']=proc.pid;processes.append(proc)
    save_json(dispatch/'DISPATCHED.json',jobs)
    while any(proc.poll() is None for proc in processes):
        print(json.dumps(dict(wave=wave,alive=[p.poll() is None for p in processes],time=time.time())),flush=True)
        time.sleep(30)
    codes=[proc.returncode for proc in processes]
    save_json(dispatch/'EXIT.json',dict(exit_codes=codes,automatic_retry=False))
    if any(codes):raise RuntimeError('A job failed; no automatic retry or next phase')
    save_json(dispatch/'COMPLETED.json',dict(status='PASS',jobs=len(jobs),stage=a.stage))


if __name__=='__main__':main()
