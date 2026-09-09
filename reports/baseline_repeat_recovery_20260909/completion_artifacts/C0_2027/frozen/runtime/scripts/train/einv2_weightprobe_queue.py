"""Finite same-host dynamic-pair pilot; validation only, no automatic retry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--scalar', type=Path, required=True)
    parser.add_argument('--hdf5-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--timeout-hours', type=float, default=24)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    if args.background:
        command = [sys.executable, str(Path(__file__).resolve()), *[x for x in sys.argv[1:] if x != '--background']]
        with (args.experiment_root / 'queue_supervisor.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'supervisor_pid': process.pid, 'command': command, 'experiment_root': str(args.experiment_root)}))
        return
    gpus = [int(g) for g in args.gpus.split(',')]
    assert len(set(gpus)) == len(gpus) == 2
    logs = args.experiment_root / 'queue_logs'
    logs.mkdir(exist_ok=False)
    jobs = queue.Queue()
    rows = []
    recipes = (
        ('D1_motionpairs_v020_seed2026_rb05_v1', 'C1', None),
        ('D3_motionpairs_v020_jepa005_seed2026_rb05_v1', 'C3', .05),
    )
    for base_id, variant, lambda_jepa in recipes:
        row = dict(id=('smoke_' if args.smoke else '') + base_id, base_id=base_id,
                   seed=2026, variant=variant, lambda_jepa=lambda_jepa,
                   velocity_min_norm=1e-6, status='pending')
        rows.append(row)
        jobs.put(row)
    lock = threading.Lock()
    stop = threading.Event()

    def record(row=None, **changes):
        with lock:
            if row is not None:
                row.update(changes)
            state = dict(supervisor_pid=os.getpid(), updated_at=now(), stopped_after_failure=stop.is_set(),
                         source_root=str(ROOT), timeout_hours=args.timeout_hours, jobs=rows)
            path = args.experiment_root / 'queue_status.json'
            temp = path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
            temp.replace(path)

    def execute(row, gpu, phase, command):
        path = logs / (row['id'] + '.' + phase + '.log')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(row['seed']),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        with path.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            start, last_change, last_size = time.monotonic(), time.monotonic(), 0
            record(row, status=phase, gpu=gpu, pid=process.pid, command=command, log=str(path), phase_started_at=now())
            while process.poll() is None:
                elapsed = time.monotonic() - start
                size = path.stat().st_size
                if size != last_size:
                    last_change, last_size = time.monotonic(), size
                record(row, elapsed_seconds=round(elapsed), log_bytes=size,
                       output_stall_advisory=time.monotonic() - last_change > 600)
                if elapsed > args.timeout_hours * 3600:
                    record(row, status='timeout_warning', timeout_notification_at=now())
                    print(json.dumps({'timeout_warning': row['id'], 'pid': process.pid, 'grace_seconds': 30}), flush=True)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    record(row, status='timeout', exit_code=process.returncode)
                    return False
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        record(row, exit_code=process.returncode)
        return process.returncode == 0

    def worker(gpu):
        while not stop.is_set():
            try:
                row = jobs.get_nowait()
            except queue.Empty:
                return
            # Recheck before every new training job; do not evict other GPU users.
            while not stop.is_set():
                used = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                if int(used.strip()) < 700:
                    break
                record(row, status='waiting_for_gpu', gpu=gpu)
                time.sleep(30)
            if stop.is_set():
                return
            train = [sys.executable, str(ROOT / 'scripts/train/einv2_audited.py'), '--project-root', str(args.project_root),
                     '--output-root', str(args.experiment_root / 'runs'), '--scalar', str(args.scalar),
                     '--hdf5-dir', str(args.hdf5_dir), '--velocity-min-norm', str(row['velocity_min_norm']),
                     '--run-id', row['base_id'], '--variant', row['variant'], '--seed', str(row['seed'])]
            if row['lambda_jepa'] is not None:
                train += ['--lambda-jepa', str(row['lambda_jepa'])]
            if args.smoke:
                train += ['--smoke-batches', '2']
            run = args.experiment_root / 'runs' / row['id']
            commands = [('training', train)] + [(split, [sys.executable, str(ROOT / 'scripts/eval/einv2_audited.py'),
                         '--run', str(run), '--split', split]) for split in ('validation',)]
            for phase, command in commands:
                if not execute(row, gpu, phase, command):
                    stop.set()
                    record(row, status='failed_' + phase, finished_at=now())
                    return
            record(row, status='completed', finished_at=now(), run=str(run))

    record()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    record()
    print(json.dumps({'status': 'failed' if stop.is_set() else 'completed', 'jobs': len(rows)}), flush=True)
    if stop.is_set():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
