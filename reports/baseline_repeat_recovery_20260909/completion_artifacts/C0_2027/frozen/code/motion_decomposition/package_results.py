"""Small share package: never include weights, audio or full predictions."""
import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime,timezone
from export import save,digest

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--summary',default='summary');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
def copy(path,relative=None):
    dest=a.output/(relative or path.relative_to(a.root));dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
for path in (a.root/a.summary).glob('*'):
    if path.is_file():copy(path,Path(path.name))
for path in a.root.glob('*'):
    if path.is_file() and path.suffix in ('.json','.jsonl','.log','.sh'):copy(path,Path('execution')/path.name)
for folder in ('code','data_audit','cases'):
    for path in (a.root/folder).rglob('*'):
        if path.is_file() and '__pycache__' not in str(path):copy(path)
for path in (a.root/'live_inputs').glob('input_manifest.json'):copy(path)
for folder in ('exports','exports_v2','exports_C2'):
    for path in (a.root/folder).glob('*/*/*.json'):copy(path)
for path in (a.root/'diagnostics').glob('*/*/*'):
    if path.is_file():copy(path)
for path in (a.root/'analysis').glob('*/*/*.json'):copy(path)
for path in (a.root/'analysis').glob('*/*/recording_metrics.csv'):copy(path)
files={str(p.relative_to(a.output)):digest(p) for p in a.output.rglob('*') if p.is_file()}
save(a.output/'package_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_server_root=str(a.root),files=files,
    omitted='Weights/audio/full floating predictions/frame GT matches remain on source server'))
print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file()))))
