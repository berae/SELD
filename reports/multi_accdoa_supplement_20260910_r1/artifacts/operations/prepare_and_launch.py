"""One-shot, scope-bound source sealing and supervised preflight launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

P = Path('/work/zhanghc/Myllm/SELD/DynamicCausalMultiACCDOA_TAU2020')
R = Path('/work/zhanghc/Myllm/SELD/reports/multi_accdoa_supplement_20260910_r1')
REV = 'ca05f674b3c719c7b1912417a766f3219197e1a1'
PYTHON = P/'.venvs/dcase2023-official-py38/bin/python'
WEIGHT = 'runs/models/35_paper_v1_C0_seed2026_eval_split0_multiaccdoa_foa_model.h5'
MANIFEST = 'runs/manifests/35_paper_v1_C0_seed2026_eval_split0_multiaccdoa_foa.json'
REF = P/'runs/analysis/threshold_predictions/Completion_20260904_paper_v1_C0_seed2026_validation_seed2026_validation_t050'
COMPLETION = Path('/work/zhanghc/Myllm/SELD/experiment_completion_20260904/runtime/manifests/Completion_20260904_paper_v1_C0_seed2026_validation.json')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def prepare():
    assert not (R/'SOURCE.json').exists()
    source = R/'source'
    source.mkdir(exist_ok=False)
    runtime = source/'runtime'
    runtime.mkdir()
    names = subprocess.check_output(['git', 'ls-tree', '--name-only', REV], cwd=P, text=True).splitlines()
    for name in names:
        if name.endswith('.py'):
            (runtime/name).write_bytes(subprocess.check_output(['git', 'show', REV+':'+name], cwd=P))
    shutil.copy2(P/'scripts/evaluate_threshold_sensitivity.py', source/'historical_threshold_decoder.py')
    shutil.copy2(P/WEIGHT, source/'C0.h5')
    shutil.copy2(P/MANIFEST, source/'manifest.json')
    shutil.copy2(COMPLETION, source/'completion_manifest.json')
    m = json.loads((source/'manifest.json').read_text())
    cm = json.loads((source/'completion_manifest.json').read_text())
    assert m['checkpoint'] == cm['checkpoint'] == str(P/WEIGHT)
    assert m['params'] == cm['params'] and m['seed'] == cm['seed'] == 2026
    assert sha(source/'C0.h5') == '8c0fc23d13f35468d69b9e35e96a3868011b06384f9de9196ac06f392f40bcc1'
    feat = Path(m['params']['feat_label_dir'])
    shutil.copy2(feat/'foa_wts', source/'scaler')
    shutil.copy2(feat/'foa_wts.json', source/'scaler.json')
    assert sha(source/'scaler') == '35f60fdb0d7467e005a8f530a0786a29057e8d160e88b409c4cd26e9cd2a1b27'
    inputs = {}
    records = {}
    for split, stem in [('train', 'fold2_room1_mix001_ov1'), ('validation', 'fold1_room1_mix001_ov1')]:
        paths = dict(features=feat/'foa_dev_norm'/(stem+'.npy'),
                     labels=feat/'foa_dev_adpit_label'/(stem+'.npy'),
                     gt=Path(m['params']['dataset_dir'])/'metadata_dev/source'/(stem+'.csv'),
                     audio=Path(m['params']['dataset_dir'])/'foa_dev/source'/(stem+'.wav'))
        if split == 'validation':
            paths['historical_csv'] = REF/(stem+'.csv')
        records[split] = {k:str(p) for k,p in paths.items()}
        inputs.update({str(p):sha(p) for p in paths.values()})
    for split in ['foa_dev_norm', 'foa_eval_norm']:
        # Names only for whole splits; no evaluation contents or model inference.
        write(source/(split+'_filenames.json'), sorted(p.name for p in (feat/split).glob('*.npy')))
    seal = dict(created=time.time(), historical_commit=REV, records=records,
                inputs=inputs, files={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file()},
                scope='two fixed complete recordings, synthetic boundaries, no optimizer, no evaluation',
                timeout_seconds=1800, automatic_retry=False)
    write(R/'SOURCE.json', seal)
    print(json.dumps(dict(prepared=True, source_sha256=sha(R/'SOURCE.json'))), flush=True)


def launch():
    seal = json.loads((R/'SOURCE.json').read_text())
    for rel, digest in seal['files'].items():
        assert sha(R/rel) == digest, rel
    for path, digest in seal['inputs'].items():
        assert sha(path) == digest, path
    gpu = subprocess.check_output(['nvidia-smi', '--id=4', '--query-gpu=uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, name, mem, util = [s.strip() for s in gpu.split(',')]
    assert '3090' in name and int(mem) == 0 and int(util) == 0, gpu
    lock = R.parent/('multi_preflight_gpu_'+uuid+'.lock')
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        command = [str(PYTHON), '-B', '-u', str(R/'operations/preflight.py')]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONHASHSEED='2026', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MPLCONFIGDIR=str(R/'matplotlib_cache'))
        write(R/'LAUNCH.json', dict(command=command, gpu=gpu, started=time.time(),
              source_sha256=sha(R/'SOURCE.json'), timeout_seconds=1800, retries=0))
        with (R/'preflight.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(R/'PROCESS.json', dict(pid=proc.pid, guard_pid=os.getpid(), started=time.time()))
            started = time.monotonic()
            timed_out = False
            try:
                code = proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                timed_out = True
                print('Hard timeout reached; terminating only preflight process group.', flush=True)
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait()
            write(R/'EXIT.json', dict(exit_code=code, timed_out=timed_out, seconds=time.monotonic()-started))
            print(json.dumps(dict(exit_code=code, timed_out=timed_out)), flush=True)
    finally:
        lock.unlink()


def dispatch():
    # Persist the one authorized guard independently of the SSH connection.
    with (R/'guard.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'launch'],
                                cwd=R, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(R/'DISPATCH.json', dict(pid=proc.pid, started=time.time(), stage='single_preflight'))
    print(json.dumps(dict(guard_pid=proc.pid)), flush=True)


if __name__ == '__main__':
    mode = argparse.ArgumentParser()
    mode.add_argument('mode', choices=['prepare', 'launch', 'dispatch'])
    {'prepare':prepare, 'launch':launch, 'dispatch':dispatch}[mode.parse_args().mode]()
