"""Read existing RB05 C0 caches; write only a new requested JSON report."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--exports",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result={"collected_utc":datetime.now(timezone.utc).isoformat(),"scope":str(args.exports),
            "raw_inputs_modified":False,"entries":[]}
    for seed in (2026,2027,2028):
        for split,expected in (("train",500),("validation",100),("evaluation",200)):
            root=args.exports/f"C0_{seed}"/split
            row={"seed":seed,"split":split,"root":str(root),"expected_files":expected}
            if not root.exists():
                row["status"]="NOT_PRESENT_IN_AUDITED_EXPORT_ROOT"
                result["entries"].append(row);continue
            paths=sorted((root/"float").glob("*.npz"))
            hashes=json.loads((root/"float_manifest.json").read_text())
            manifest=json.loads((root/"input_manifest.json").read_text())
            bad=[];schemas=set();h_count=0
            for p in paths:
                if hashes.get(p.name)!=digest(p):bad.append(p.name)
                with np.load(p,allow_pickle=False) as z:
                    schemas.add(tuple(z.files))
                    if "doa_features" in z and z["doa_features"].shape==(600,2,512):h_count+=1
            run=Path(manifest["run"]);cfg=json.loads((run/"config.json").read_text())
            source_report={}
            for rel,expected_hash in manifest["source_hashes"].items():
                src=Path(manifest["runtime"])/rel
                source_report[rel]=src.exists() and digest(src)==expected_hash
            row.update(status="EXISTING_PREDICTIONS_ONLY" if not h_count else "HAS_HIDDEN_FEATURES",
                       files=len(paths),hash_mismatches=bad,
                       file_coverage_ok=len(paths)==expected and set(hashes)=={p.name for p in paths},
                       schemas=[list(x) for x in schemas],doa_features_files=h_count,
                       run=str(run),checkpoint=str(run/"best.pth"),
                       checkpoint_sha256=digest(run/"best.pth"),
                       checkpoint_hash_matches=digest(run/"best.pth")==manifest["checkpoint_sha256"],
                       scalar=cfg["causal_scalar_path"],
                       scalar_hash_matches=digest(Path(cfg["causal_scalar_path"]))==manifest["scalar_sha256"],
                       runtime=manifest["runtime"],source_checks=source_report,
                       dataset_dir=cfg["dataset_dir"],audit_version=cfg["audit"]["version"],
                       raw_regression=json.loads((root/"raw_regression.json").read_text()),
                       completed=(root/"COMPLETED.json").exists())
            result["entries"].append(row)
    result["new_candidate_training_cache_ready"]=all(
        r.get("doa_features_files")==r["expected_files"] and r.get("file_coverage_ok",False)
        for r in result["entries"] if r["split"] in ("train","validation"))
    with args.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({"cache_ready":result["new_candidate_training_cache_ready"],
                      "entries":[{k:r.get(k) for k in ("seed","split","status","files","doa_features_files")}
                                 for r in result["entries"]]}))


if __name__=="__main__":main()
