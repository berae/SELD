"""Read existing rabbit02 C2 checkpoints, single GPU, validation diagnostics."""
import json
from pathlib import Path
import subprocess
import sys
import os
from export import save

root=Path('/work/zhanghc/Myllm/SELD');out=root/'reports/motion_decomposition_20260907';code=out/'code'
runs=json.loads((out/'live_inputs/input_manifest.json').read_text())['runs']
for r in runs:
    cfg=r['config']
    if cfg['audit']['variant']!='C2':continue
    seed=cfg['audit']['seed'];dest=out/'exports_C2'/f'C2_{seed}'/'validation'
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i','0'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert uuid not in processes
    exporter=code/'export_v2.py' if (code/'export_v2.py').exists() else code/'export.py'
    cmds=[[sys.executable,str(exporter),'--run',r['run'],'--runtime',r['exact_runtime_candidates'][0],'--output',str(dest),'--split','validation','--latent'],
          [sys.executable,str(code/'diagnostics.py'),'--export',str(dest),'--output',str(out/'diagnostics'/f'C2_{seed}'/'validation'),'--variant','C2']]
    for i,cmd in enumerate(cmds):
        with (out/f'C2_{seed}_{i}.log').open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},timeout=1800)
        if result.returncode:raise RuntimeError(str(cmd))
    print(seed,'COMPLETED',flush=True)
save(out/'C2_COMPLETED.json',dict(seeds=[2026,2027,2028],split='validation',host='rabbit02',untrained_velocity_not_interpretable=True))
