"""Run CPU mechanism diagnostics only for completed exports; no training."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(sys.argv[1]);code=root/'code';done=set()
while True:
    for path in sorted(list((root/'exports').glob('*/*/COMPLETED.json'))+list((root/'exports_v2').glob('*/*/COMPLETED.json'))):
        name=path.parent.parent.name;split=path.parent.name;variant=name.rsplit('_',1)[0]
        if variant=='C0' or (name,split) in done:continue
        output=root/'diagnostics'/name/split
        if (output/'COMPLETED.json').exists():done.add((name,split));continue
        if output.exists():raise RuntimeError('Incomplete diagnostic '+str(output))
        cmd=[sys.executable,str(code/'diagnostics.py'),'--export',str(path.parent),'--output',str(output),'--variant',variant]
        with (root/'commands.sh').open('a') as f:
            import shlex
            f.write(shlex.join(cmd)+'\n')
        with (root/(name+'_'+split+'_diag.log')).open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode:raise RuntimeError('Diagnostic failed '+str(output))
        print(name,split,'COMPLETED',flush=True);done.add((name,split))
    if (root/'MATRIX_COMPLETED.json').exists():break
    time.sleep(10)
