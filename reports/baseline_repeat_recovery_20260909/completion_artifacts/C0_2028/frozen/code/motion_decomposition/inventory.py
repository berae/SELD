"""Freeze live inputs without using chat/reported summary numbers."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone
from export import digest, save

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
entries=[]
for config in sorted(a.root.glob('experiment_einv2*/**/config.json')):
    run=config.parent
    if not (run/'best.pth').exists() or 'smoke' in str(run): continue
    cfg=json.loads(config.read_text())
    sources=cfg['audit']['source_hashes']
    matches=[]
    for runtime in sorted((a.root/'code_releases').iterdir()):
        if runtime.is_dir() and all((runtime/k).exists() and digest(runtime/k)==v for k,v in sources.items()): matches.append(str(runtime))
    record=dict(run=str(run),config=cfg,config_sha256=digest(config),checkpoint_sha256=digest(run/'best.pth'),
                checkpoint_bytes=(run/'best.pth').stat().st_size,exact_runtime_candidates=matches,
                scalar_sha256=digest(cfg['causal_scalar_path']),saved_results={})
    for split in ('validation','evaluation'):
        directory=run/split
        record['saved_results'][split]=dict(csv_hashes={f.name:digest(f) for f in sorted(directory.glob('*.csv'))},
            metrics=json.loads((directory/'metrics.json').read_text()) if (directory/'metrics.json').exists() else None)
    save(a.out/(run.name+'.json'),record); entries.append(record)
save(a.out/'input_manifest.json',dict(host=socket.gethostname(),root=str(a.root),collected_utc=datetime.now(timezone.utc).isoformat(),
    nvidia_smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,runs=entries))
print(json.dumps(dict(runs=len(entries),runtime_verified=sum(bool(r['exact_runtime_candidates']) for r in entries))))
