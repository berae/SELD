import json
from pathlib import Path
import sys
import h5py
import numpy as np
from diagnostics import identities
from export import save,digest

cfg=json.loads(Path(sys.argv[1]).read_text());dest=Path(sys.argv[2]);files=0;pairs=np.zeros(3,int);hashes={}
for p in sorted((Path(cfg['jepa_hdf5_dir'])/'dev').glob('fold1*.h5')):
    meta=Path(cfg['hdf5_dir'])/cfg['dataset']/'meta/dev'/p.name
    with h5py.File(meta) as f:sed=f['sed_label'][:];doa=f['doa_label'][:]
    ids=identities(Path(cfg['dataset_dir'])/'metadata_dev'/(p.stem+'.csv'),sed,doa)
    with h5py.File(p) as f:
        np.testing.assert_array_equal(ids,f['identity_label'][:]);actual=f['jepa_valid_mask'][:]
    expected=np.zeros_like(actual)
    for i,h in enumerate(cfg['training']['jepa_horizons_frames']):
        m=ids[:-h,:,0]>=0
        for j in range(1,h+1):m&=(ids[:-h]==ids[j:600-h+j]).all(-1)
        expected[:-h,:,i]=m
    np.testing.assert_array_equal(expected,actual);pairs+=actual.sum((0,1)).astype(int);files+=1;hashes[str(p)]=digest(p)
save(dest,dict(files=files,all_mask_entries_equal=True,valid_pairs_before_chunk_filter=pairs.tolist(),files_sha256=hashes))
