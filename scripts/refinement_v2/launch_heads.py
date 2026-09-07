"""Launch only the registered four heads on currently unused 3090 GPUs."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiment_core import save_json, digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','pilot'],required=True)
    a=p.parse_args();root=a.root
    for split in ('train','validation'):
        assert json.loads((root/'cache'/split/'COMPLETED.json').read_text())['status']=='PASS'
    assert json.loads((root/'execution_tests.json').read_text())['status']=='PASS'
    spec=json.loads((root/'configs/execution_r1.json').read_text())
    conditions=['F-Deriv','R0','R1','R2']
    if a.mode=='pilot':
        for cond in conditions:
            assert json.loads((root/'smoke'/cond/'COMPLETED.json').read_text())['status']=='SMOKE_PASS'
            assert json.loads((root/'guards_smoke'/cond/'EXIT.json').read_text())['exit_code']==0
    app_text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    used={x.strip() for x in app_text.splitlines()}
    gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    free=[row for row in csv.reader(io.StringIO(gpu_text)) if row[1].strip() not in used and '3090' in row[2]]
    if len(free)<len(conditions):
        raise RuntimeError('Fewer than four unused 3090s; no jobs launched, do not preempt other jobs')
    for cond in conditions:
        assert not (root/a.mode/cond).exists() and not (root/('guards_'+a.mode)/cond).exists()
    plan=[]
    for cond,gpu in zip(conditions,free):
        command=[sys.executable,str(Path(__file__).with_name('train_heads.py')),
                 '--cache',str(root/'cache'),'--config',str(root/'configs/pilot.json'),
                 '--execution',str(root/'configs/execution_r1.json'),'--scorer',str(root/'scorer'),
                 '--output',str(root/a.mode/cond),'--condition',cond]
        if a.mode=='smoke':command+=['--smoke-batches',str(spec['smoke_batches_per_condition'])]
        seconds=spec['smoke_hard_timeout_seconds'] if a.mode=='smoke' else spec['pilot_hard_timeout_seconds']
        wrapper=[sys.executable,str(Path(__file__).with_name('run_guarded.py')),
                 '--directory',str(root/('guards_'+a.mode)/cond),'--seconds',str(seconds),'--',*command]
        plan.append(dict(condition=cond,gpu_index=gpu[0].strip(),gpu_uuid=gpu[1].strip(),command=wrapper))
    save_json(root/(a.mode+'_launch_plan.json'),dict(mode=a.mode,created_unix=time.time(),jobs=plan,
              config_sha256=digest(root/'configs/pilot.json'),execution_sha256=digest(root/'configs/execution_r1.json'),
              future_expansion=False,automatic_retry=False))
    for job in plan:
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=job['gpu_uuid'];env['PYTHONHASHSEED']='2026'
        with (root/(a.mode+'_'+job['condition']+'_launcher.log')).open('x') as log:
            process=subprocess.Popen(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,start_new_session=True)
        job['guardian_pid']=process.pid
    save_json(root/(a.mode+'_dispatched.json'),dict(mode=a.mode,jobs=plan,
              status='DISPATCHED_NOT_COMPLETED',created_unix=time.time()))
    print(json.dumps(plan))


if __name__=='__main__':main()
