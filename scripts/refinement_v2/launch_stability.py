"""Bounded, predeclared two-head-seed repeat; same frozen C0, no sweep/retry."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiment_core import save_json,digest


def free_gpus():
    used=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
    rows=csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader,nounits'],text=True)))
    return [dict(index=r[0].strip(),uuid=r[1].strip()) for r in rows if '3090' in r[2] and r[1].strip() not in used and int(r[3])<256]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False)
    conditions=['F-Deriv','R0','R1','R2'];seeds=[2027,2028]
    for c in conditions:
        assert json.loads((a.base/'pilot'/c/'COMPLETED.json').read_text())['status']=='PASS'
    for s in ('train','validation'):
        assert json.loads((a.base/'cache'/s/'COMPLETED.json').read_text())['status']=='PASS'
    execution=json.loads((a.base/'configs/execution_r1.json').read_text())
    save_json(a.root/'PREREGISTERED.json',dict(created_unix=time.time(),head_seeds=seeds,conditions=conditions,
        primary_comparisons=['R0-F0','R0-F-Deriv','R1-R0','R2-R1'],primary_metrics=['SELD_LR','LE_CD'],
        C0_seed=2026,cache_manifest_sha256={s:digest(a.base/'cache'/s/'file_manifest.json') for s in ('train','validation')},
        analysis='Report all head seeds including original 2026, mean/sample SD and paired differences; no p-values or independent-C0 claims',
        scope='initialization and shuffle sensitivity only; no parameter search, no new data/backbone/C0 weights',
        authorization='2026-09-08 user: 查看实验结果，并分析实验结果，如需进一步实验可以在rabbit02上进行',
        code_hashes={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}))
    for seed in seeds:
        spec=dict(execution,seed=seed,authorized_scope='same-C0 head-seed stability under 2026-09-08 authorization',
                  no_automatic_three_seed_expansion=False,no_additional_C0_seed=True)
        config=a.root/('execution_seed%d.json'%seed);save_json(config,spec)
        gpus=free_gpus()
        if len(gpus)<4:
            save_json(a.root/('BLOCKED_seed%d.json'%seed),dict(reason='fewer than four unused 3090s; no jobs launched for this seed'))
            raise RuntimeError('Insufficient free GPUs; no preemption or automatic retry')
        jobs=[]
        for c,gpu in zip(conditions,gpus):
            run=a.root/('seed%d'%seed)/c;guard=a.root/'guards'/('seed%d'%seed)/c
            command=[sys.executable,str(Path(__file__).with_name('train_heads.py')),'--cache',str(a.base/'cache'),
                '--config',str(a.base/'configs/pilot.json'),'--execution',str(config),'--scorer',str(a.base/'scorer'),
                '--output',str(run),'--condition',c]
            wrapper=[sys.executable,str(Path(__file__).with_name('run_guarded.py')),'--directory',str(guard),
                     '--seconds','14400','--',*command]
            jobs.append(dict(condition=c,gpu=gpu,command=wrapper))
        save_json(a.root/('plan_seed%d.json'%seed),jobs)
        processes=[]
        for job in jobs:
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=job['gpu']['uuid'],PYTHONHASHSEED=str(seed))
            with (a.root/('launcher_%d_%s.log'%(seed,job['condition']))).open('x') as log:
                proc=subprocess.Popen(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            processes.append(proc);job['guardian_pid']=proc.pid
        save_json(a.root/('dispatched_seed%d.json'%seed),jobs)
        while any(proc.poll() is None for proc in processes):
            print(json.dumps(dict(seed=seed,alive=[proc.poll() is None for proc in processes],time=time.time())),flush=True)
            time.sleep(30)
        codes=[proc.returncode for proc in processes]
        save_json(a.root/('exit_seed%d.json'%seed),codes)
        if any(codes):raise RuntimeError('Experiment failure; no automatic retry or next seed')
    save_json(a.root/'COMPLETED.json',dict(status='PASS',head_seeds=seeds,conditions=conditions,C0_seeds=[2026]))


if __name__=='__main__':main()
