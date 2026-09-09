"""Detach only a named already-authorized stage from the SSH transport."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('train-2028','evaluation-cache','evaluation-score'),required=True)
    a=p.parse_args(); root=a.root.resolve(); code=root/'code/refinement_v2'
    sys.path.insert(0,str(code)); sys.path.insert(0,str(root/'code/motion_decomposition'))
    from experiment_core import digest,save_json
    if a.stage=='train-2028':
        assert json.loads((root/'dispatch/train_C0_2027/COMPLETED.json').read_text())['status']=='PASS'
        command=[sys.executable,str(code/'launch_cross_c0.py'),'--root',str(root),'--stage','train','--C0-seed','2028']
    else:
        assert json.loads((root/'ALL_EIGHT_FROZEN.json').read_text())['status']=='PASS'
        command=[sys.executable,str(root/'operations/launch_cross_evaluation.py'),'--root',str(root),
                 '--stage',a.stage.split('-')[1]]
    out=root/'transport_launches'/a.stage; out.mkdir(parents=True,exist_ok=False)
    save_json(out/'PLAN.json',dict(command=command,script_sha256=digest(__file__),
        child_entry_sha256=digest(command[1]),reason='Server-side log avoids SSH stdout lifetime dependency',
        training_recipe_changed=False,automatic_retry=False))
    env=os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(code)+':'+str(root/'code/motion_decomposition'))
    with (out/'launcher.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=root,env=env,start_new_session=True,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    save_json(out/'STARTED.json',dict(pid=proc.pid,command=command,log=str(out/'launcher.log')))
    print(json.dumps(dict(dispatched=True,pid=proc.pid,stage=a.stage)))


if __name__=='__main__':main()
