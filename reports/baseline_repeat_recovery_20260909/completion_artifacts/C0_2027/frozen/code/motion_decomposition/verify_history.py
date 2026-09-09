import json
from pathlib import Path
import sys
from export import digest,save

root=Path(sys.argv[1]);data=json.loads((root/'live_inputs/input_manifest.json').read_text());checked=0;fail=[]
for r in data['runs']:
    run=Path(r['run'])
    files={run/'config.json':r['config_sha256'],run/'best.pth':r['checkpoint_sha256'],Path(r['config']['causal_scalar_path']):r['scalar_sha256']}
    for split,entry in r['saved_results'].items():
        files.update({run/split/n:h for n,h in entry['csv_hashes'].items()})
    for path,expected in files.items():
        actual=digest(path);checked+=1
        if actual!=expected:fail.append(str(path))
    for runtime in r['exact_runtime_candidates']:
        for name,h in r['config']['audit']['source_hashes'].items():
            checked+=1
            if digest(Path(runtime)/name)!=h:fail.append(str(Path(runtime)/name))
save(root/'history_integrity.json',dict(checked_files=checked,changed_files=fail,status='PASS' if not fail else 'FAIL'))
assert not fail
