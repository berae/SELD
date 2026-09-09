"""Check diagnostic I/O optimization without rescoring or resampling a dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
from candidate import fixed_records
from diagnose_fixed_targets import errors
from experiment_core import digest,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cache=a.previous/'cache/evaluation'
    name=sorted(json.loads((cache/'input_manifest.json').read_text())['gt_files'])[0]
    stem=Path(name).stem
    with np.load(cache/'float'/(stem+'.npz')) as z,np.load(cache/'targets'/(stem+'.npz')) as t:
        records=fixed_records(z['probability'],z['doa']);materialized={k:t[k] for k in t.files}
        old=errors(a.previous/'evaluation/F0'/name,records,t)
        new=errors(a.previous/'evaluation/F0'/name,records,materialized)
        assert np.array_equal(old,new,equal_nan=True)
        assert all(np.array_equal(materialized[k],t[k]) for k in t.files)
    save_json(a.output,dict(status='PASS',filename=name,all_arrays_exact=True,fixed_errors_exact=True,
        scope='One-file NPZ materialization equivalence; no model training, bootstrap, or metric selection',
        source_target_sha256=digest(cache/'targets'/(stem+'.npz')),script_sha256=digest(__file__)))
    print(json.dumps(dict(status='PASS',NPZ_materialization_exact=True)))


if __name__=='__main__':main()
