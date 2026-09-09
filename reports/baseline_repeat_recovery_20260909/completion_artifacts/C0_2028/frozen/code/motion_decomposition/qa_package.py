import csv
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
def read(name):
    with (root/name).open() as f:return list(csv.DictReader(f))
rows=read('metrics_per_run.csv');common=read('common_match_breakdown.csv');strata=read('motion_strata.csv')
assert len(rows)==106
groups={}
for r in rows:
    assert r['decoder']!='fusion' or r['variant']!='C0'
    k=r['variant'],r['seed'],r['split'],r['profile']
    groups.setdefault(k,[]).append(r)
for rs in groups.values():
    for field in ('sed_micro_F1','sed_macro_F1','sed_TP','sed_FP','sed_FN','gt_denominator','matched'):
        assert len({r[field] for r in rs})==1,(field,rs)
for r in common:
    assert sum(int(r[k]) for k in ('both_matched','baseline_only','variant_only','neither'))==int(r['gt_denominator'])
    assert int(r['both_matched'])+int(r['baseline_only'])==int(r['baseline_matched'])
    assert int(r['both_matched'])+int(r['variant_only'])==int(r['variant_matched'])
keys={}
for r in strata:keys.setdefault((r['variant'],r['seed'],r['split'],r['decoder']),{})[r['stratum']]=r
for k,s in keys.items():
    assert sum(int(s[g]['gt_denominator']) for g in ('static','dynamic'))==int(s['all']['gt_denominator'])
    assert sum(int(s[g]['gt_denominator']) for g in ('same_class','different_class','single'))==int(s['all']['gt_denominator'])
    for r in s.values():assert int(r['within20'])<=int(r['matched'])<=int(r['gt_denominator'])
with (root/'additional_qa.json').open('x') as f:json.dump(dict(status='PASS',checks=['106 metric rows','no C0 fusion','SED and frame counts identical across decoders','common partitions exhaustive','motion and overlap partitions exhaustive','within20 <= matched <= GT']),f,indent=2)
