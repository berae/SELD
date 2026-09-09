import json
from pathlib import Path
import subprocess
import sys
import time
from export import save

root=Path('/home/zhanghc/SELD/reports/motion_decomposition_20260907');code=root/'code';started=time.time()
while not (root/'MATRIX_COMPLETED.json').exists() or len(list((root/'diagnostics').glob('*/*/COMPLETED.json')))<17:
    if time.time()-started>3600:raise TimeoutError('Matrix/diagnostics not complete; no report fabricated')
    time.sleep(10)
cmds=[['aggregate.py','--root',str(root),'--output',str(root/'summary')],
      ['cases.py','--root',str(root),'--output',str(root/'cases')]]
for cmd in cmds:
    subprocess.run([sys.executable,str(code/cmd[0]),*cmd[1:]],check=True)
import csv
with (root/'summary/metrics_per_run.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==106,len(rows)
assert sum(r['variant'] in ('C0','C1','C3_j005') for r in rows)==96
for r in rows:assert int(r['files'])==(100 if r['split']=='validation' else 200)
with (root/'summary/common_match_breakdown.csv').open() as f:
    for r in csv.DictReader(f):assert sum(int(r[k]) for k in ('both_matched','baseline_only','variant_only','neither'))==int(r['gt_denominator'])
save(root/'final_qa.json',dict(status='PASS',primary_decoder_run_splits=48,appendix_decoder_run_splits=5,profiles=2,metrics_rows=106,common_partitions_exhaustive=True))
subprocess.run([sys.executable,str(code/'package_results.py'),'--root',str(root),'--output',str(root/'share')],check=True)
print('FINALIZED',flush=True)
