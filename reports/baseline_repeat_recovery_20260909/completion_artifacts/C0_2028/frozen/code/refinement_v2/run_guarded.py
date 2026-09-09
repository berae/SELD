"""Bounded process runner: preserves failures, never retries, reports alive/stall."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from experiment_core import save_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--seconds',type=int,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command
    if cmd and cmd[0]=='--':cmd=cmd[1:]
    assert cmd and a.seconds>0
    a.directory.mkdir(parents=True,exist_ok=False)
    start=time.time();timed_out=False;last_size=0;stalled=0
    with (a.directory/'process.log').open('x') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(a.directory/'STARTED.json',dict(command=cmd,pid=process.pid,guardian_pid=os.getpid(),start_unix=start,
                  hard_timeout_seconds=a.seconds,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        with (a.directory/'heartbeat.jsonl').open('x') as hb:
            while process.poll() is None:
                now=time.time();size=(a.directory/'process.log').stat().st_size
                stalled=stalled+1 if size==last_size else 0;last_size=size
                record=dict(elapsed_seconds=now-start,pid=process.pid,alive=True,log_bytes=size,
                            advisory='OUTPUT_STALL_SUSPECTED' if stalled>=3 else None)
                hb.write(json.dumps(record)+'\n');hb.flush()
                if now-start>=a.seconds:
                    hb.write(json.dumps(dict(event='HARD_TIMEOUT_NOTIFY_BEFORE_TERMINATION',pid=process.pid))+'\n');hb.flush()
                    os.killpg(process.pid,signal.SIGTERM);timed_out=True
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                time.sleep(min(30,max(1,a.seconds-(now-start))))
        code=process.wait()
    save_json(a.directory/'EXIT.json',dict(exit_code=code,timed_out=timed_out,elapsed_seconds=time.time()-start,
              status='TIMED_OUT' if timed_out else 'PROCESS_COMPLETED' if code==0 else 'FAILED',automatic_retry=False))
    raise SystemExit(124 if timed_out else code)


if __name__=='__main__':main()
