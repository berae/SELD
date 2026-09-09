"""Hash exact waveform/label inputs and count existing cross-chunk train targets."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from export import digest,save

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
root=Path(cfg['hdf5_dir'])/cfg['dataset'];hashes={}; boundary=valid=moving=active=files=0
for subset in ('dev','eval'):
    for path in sorted((root/'data'/f"{cfg['data']['sample_rate']}fs"/subset/cfg['data']['type']).glob('*.h5')):
        if path.name.startswith('.'):continue
        hashes[str(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
    for path in sorted((root/'meta'/subset).glob('*.h5')):
        if not path.name.startswith('.'):hashes[str(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
for path in sorted((Path(cfg['velocity_hdf5_dir'])/'dev').glob('*.h5')):
    if path.name.startswith('.') or path.stem.startswith('fold1'):continue
    hashes[str(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
    with h5py.File(path,'r') as f:
        m=f['velocity_mask'][:]> .5;v=f['velocity_label'][:]
    with h5py.File(root/'meta/dev'/path.name,'r') as f:s=f['sed_label'][:].max(-1)>.5
    files+=1; valid+=int(m.sum());active+=int(s[40::40].sum());boundary+=int(m[40::40].sum());moving+=int((m[40::40]&(np.linalg.norm(v[40::40],axis=-1)>1e-6)).sum())
save(a.output/'data_hashes.json',hashes)
save(a.output/'train_chunk_boundary.json',dict(train_files=files,all_valid_targets=valid,boundary_active_targets=active,boundary_valid_targets=boundary,boundary_moving_targets=moving,
    boundary_fraction=boundary/valid,supervision_changed=False,source='Existing velocity HDF5; frozen loader slices without resetting the mask'))
