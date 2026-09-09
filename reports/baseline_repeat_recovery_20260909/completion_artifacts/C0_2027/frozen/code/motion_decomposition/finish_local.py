"""Assemble direct-entry deliverables from independently verified host packages."""
import csv
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

root=Path(sys.argv[1]);repo=root.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('metrics_per_run.csv','paired_deltas.csv','motion_strata.csv','common_match_breakdown.csv','velocity_diagnostics.csv','latent_diagnostics.csv','seed_summary.csv'):
    assert not (root/name).exists();shutil.copy2(root/'RB05'/name,root/name)
for name in ('protocol.json','tests.json','commands.sh'):
    assert not (root/name).exists();shutil.copy2(root/'RB05/execution'/name,root/name)
with (root/'commands.sh').open('a',encoding='utf-8') as f:
    f.write('\n# Additional executed entry points; commands are an execution record, not an overwrite-safe replay script.\n')
    f.write('# rabbit02: existing_raw.py /work/zhanghc/Myllm/SELD\n# rabbit02: c2_diagnostics.py\n# Both hosts: inventory.py, verify_history.py, environment.py\n# RB05 first validation: export.py C0_2026 validation; C3 failed export.py retained, corrected export_v2.py in exports_v2.\n')
with (root/'common_match_breakdown.csv').open() as f:rs=list(csv.DictReader(f))
groups=defaultdict(list)
for r in rs:groups[tuple(r[k] for k in ('baseline','variant','split','stratum'))].append(r)
out=[]
for k,rs in groups.items():
    for field in ('common_delta_LE','baseline_recall','variant_recall','baseline_only','variant_only'):
        vals=[float(r[field]) for r in rs if r[field]!='']
        out.append(dict(zip(('baseline','variant','split','stratum'),k),metric=field,n_seeds=len(vals),mean=statistics.mean(vals) if vals else '',sample_SD=statistics.stdev(vals) if len(vals)>1 else ''))
with (root/'common_seed_summary.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),repository='berae/SELD',
    taskbook_upstream_commit='51096fff3c4cb8270c64a9d1843bafe1a53c236e',upstream_commit_existence_verified_by_web=True,
    upstream_main_current_head_verified=False,local_snapshot_tree_equality_to_upstream_verified=False,
    local_import_commit='d76dc1f24fdeb978008da9ebad68cd6551d7557c',analysis_code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip(),
    taskbook_sha256=sha(Path('E:/SELDR/01_Codex_实验执行任务书.md')),
    hosts={'RB05':{'live_manifest':'RB05/live_inputs/input_manifest.json','root':'/home/zhanghc/SELD/reports/motion_decomposition_20260907'},
           'rabbit02':{'live_manifest':'rabbit02/live_inputs/input_manifest.json','root':'/work/zhanghc/Myllm/SELD/reports/motion_decomposition_20260907'}},
    root_csv_scope='RB05 primary and specified exploratory appendix; rabbit02 remains in its own subdirectory',
    execution=dict(new_training=0,lambda_search=0,rb05_checkpoint_run_splits=23,rb05_decoder_run_splits=53,rabbit02_existing_raw_run_splits=12,rabbit02_C2_validation_exports=3),
    code_source_hashes={str(p.relative_to(repo)):sha(p) for p in (repo/'scripts/motion_decomposition').glob('*.py')},
    runtime_and_checkpoint_hashes='Nested live manifests and per-export input_manifest.json contain exact config/checkpoint/scaler/runtime source hashes; data_audit contains waveform/label hashes',
    host_packages={h:sha(root/h/'package_manifest.json') for h in ('RB05','rabbit02')})
with (root/'input_manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
with (root/'VISUAL_QA.json').open('x') as f:json.dump(dict(plots_inspected=7,status='PASS',axes_and_titles_readable=True,missing_matches_are_gaps=True,
    duplicate_dynamic_turn_case='Expected from the predeclared lexicographic rule; not independent evidence',vertical_red_lines='GT-defined turn markers, offline diagnosis only'),f,indent=2)
