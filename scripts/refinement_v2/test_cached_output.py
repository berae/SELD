"""Full validation zero-correction regression on existing C0 predictions.

Uses zero synthetic h ONLY to exercise zero-initialized new heads. No claim that
the missing real h cache, new extraction path, or waveform causality is tested.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from candidate import ResidualHead, prediction_history, correct_direction, fixed_records, replace_record_directions


def lab(records):
    result={}
    for frame,cls,rid,az,el in records:result.setdefault(frame,[]).append([cls,rid,az,el])
    return result


def activity(records):
    x=np.zeros((600,14),bool)
    for frame,cls,*_ in records:x[frame,cls]=True
    return x


def sed_metrics(counts):
    tp,fp,fn=counts
    f=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(14),where=(2*tp+fp+fn)>0)
    return {"TP":tp.tolist(),"FP":fp.tolist(),"FN":fn.tolist(),
            "micro_F1":float(2*tp.sum()/max(1,2*tp.sum()+fp.sum()+fn.sum())),
            "macro_F1":float(f.mean())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--export",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    start=time.perf_counter();torch.set_num_threads(2)
    manifest=json.loads((args.export/"input_manifest.json").read_text())
    assert manifest["split"]=="validation","This test must not select on evaluation"
    sys.path.insert(0,str(Path(manifest["runtime"])/"evaluation"))
    from aligned_metrics import AlignedMetrics,load_csv
    raw_metric=AlignedMetrics();zero_metric=AlignedMetrics()
    counts={k:np.zeros((3,14),np.int64) for k in ("raw","zero","nonzero")}
    head0=ResidualHead();head1=ResidualHead(True)
    hashes={};num_records=0
    names=sorted(manifest["gt_files"]);assert len(names)==100
    for filename in names:
        path=args.export/"float"/(Path(filename).stem+".npz")
        with np.load(path,allow_pickle=False) as data:
            p=data["probability"];raw=data["doa"]
            frames=data["frame_index"];chunks=data["chunk_index"]
        assert raw.shape==(600,2,3) and p.shape==(600,2,14)
        probability_hash=hashlib.sha256(p.tobytes()).hexdigest()
        records=fixed_records(p,raw);num_records+=len(records)
        m=prediction_history(p,raw,frames,chunks)
        h=torch.zeros(600,2,512);r=torch.from_numpy(raw);active=torch.from_numpy(p.max(-1)>.5)
        with torch.no_grad():
            q=correct_direction(r,head0(h,r,m),active)
            q1=correct_direction(r,head1(h,r,m),active)
            nonzero=correct_direction(r,torch.ones_like(r)*.05,active).numpy()
        assert torch.equal(q,r) and torch.equal(q1,r)
        zero=replace_record_directions(records,q.numpy())
        changed=replace_record_directions(records,nonzero)
        assert zero==records
        out=io.StringIO(newline="");w=csv.writer(out)
        for f,c,s,a,e in zero:w.writerow([f,c,a,e])
        assert out.getvalue().encode()==(args.export/"raw"/filename).read_bytes()
        gt=load_csv(Path(manifest["config"]["dataset_dir"])/"metadata_dev"/filename,"polar5")
        raw_metric.update(lab(records),gt);zero_metric.update(lab(zero),gt)
        gt_active=np.zeros((600,14),bool)
        for f,rows in gt.items():
            for row in rows:gt_active[f,row[0]]=True
        for mode,rs in (("raw",records),("zero",zero),("nonzero",changed)):
            pred=activity(rs)
            counts[mode]+=np.stack([(pred&gt_active).sum(0),(pred&~gt_active).sum(0),(~pred&gt_active).sum(0)])
        assert probability_hash==hashlib.sha256(p.tobytes()).hexdigest()
        hashes[filename]={"probability_sha256":probability_hash,
                         "record_identity_sha256":hashlib.sha256(json.dumps([x[:3] for x in records]).encode()).hexdigest()}
    raw_scores=raw_metric.scores();zero_scores=zero_metric.scores()
    assert raw_scores==zero_scores
    previous=json.loads((args.export/"raw_regression.json").read_text())["scores"]
    max_delta=max(abs(raw_scores[profile][key]-previous[profile][key])
                  for profile in raw_scores for key in raw_scores[profile])
    assert max_delta<=1e-6
    assert np.array_equal(counts["raw"],counts["zero"]) and np.array_equal(counts["raw"],counts["nonzero"])
    result={"scope":"C0 seed2026 full validation cached-output regression; no real h export or waveform forward",
            "status":"PASS","files":100,"frames_per_file":600,"records":num_records,
            "zero_float_exact":True,"zero_csv_bytes_exact":True,
            "zero_official_score_exact":True,"historical_max_score_delta":max_delta,
            "nonzero_record_identity_and_sed_equal":True,"sed":{k:sed_metrics(v) for k,v in counts.items()},
            "raw_scores":raw_scores,"zero_scores":zero_scores,"hashes":hashes,
            "elapsed_seconds":time.perf_counter()-start}
    with args.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("hashes","raw_scores","zero_scores","sed")}))


if __name__=="__main__":main()
