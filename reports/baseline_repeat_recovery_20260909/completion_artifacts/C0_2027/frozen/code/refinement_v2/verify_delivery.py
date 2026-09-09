"""Validate the first-delivery evidence and create an exclusive SHA256 manifest."""
import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = repo / "reports/refinement_v2_20260907"
    config = read_json(repo / "configs/refinement_v2/pilot.json")
    registry = read_json(report / "run_registry.json")
    module = read_json(report / "module_tests.json")
    cached = read_json(report / "cached_output_tests.json")
    probe = read_json(report / "frozen_c0_probe.json")
    audit = read_json(report / "cache_audit.json")
    assert config["version"] == registry["version"] == "unified_v2_20260907"
    assert not config["execution"]["training_started"]
    assert not registry["pilot"]["training_started"]
    assert not registry["pilot"]["old_fusion_oracle_dependency"]
    assert not config["promotion"]["factorial_2x2"]
    assert len(module["tests"]) == 8 and all(t["status"] == "PASS" for t in module["tests"])
    for cond in config["conditions"]:
        if cond["id"] in module["parameter_counts"]:
            assert cond["parameters"] == module["parameter_counts"][cond["id"]]
    assert cached["status"] == probe["status"] == "PASS"
    assert cached["files"] == 100 and cached["records"] == 48134
    for key in ("zero_float_exact", "zero_csv_bytes_exact", "zero_official_score_exact",
                "nonzero_record_identity_and_sed_equal"):
        assert cached[key] is True
    assert cached["historical_max_score_delta"] == 0
    assert probe["training_updates"] == 0 and not probe["optimizer_created"]
    assert probe["parameters_and_buffers_unchanged"]
    assert probe["checkpoint_sha256"] == config["baseline_checkpoint_sha256"]
    assert probe["future_prefix_max_delta"] == probe["future_hidden_prefix_max_delta"] == 0
    existing = [e for e in audit["entries"] if e["status"] == "EXISTING_PREDICTIONS_ONLY"]
    assert len(existing) == 6 and sum(e["files"] for e in existing) == 900
    assert all(e["file_coverage_ok"] and not e["hash_mismatches"] and
               e["doa_features_files"] == 0 and e["checkpoint_hash_matches"] and
               e["scalar_hash_matches"] and all(e["source_checks"].values()) for e in existing)
    missing = [e for e in audit["entries"] if e["split"] == "train"]
    assert len(missing) == 3 and all(e["status"] == "NOT_PRESENT_IN_AUDITED_EXPORT_ROOT" for e in missing)
    for filename, expected in module["code_sha256"].items():
        local = repo / "scripts/refinement_v2" / filename
        assert hashlib.sha256(local.read_bytes()).hexdigest() == expected, filename
    files = [args.decision.resolve()]
    for folder in ("scripts/refinement_v2", "configs/refinement_v2", "reports/refinement_v2_20260907"):
        files.extend(p for p in (repo / folder).iterdir()
                     if p.is_file() and p.resolve() != args.output.resolve())
    files.append(repo / "scripts/motion_decomposition/decoder.py")
    syntax_checked = []
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            syntax_checked.append(str(path))
    result = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local delivery consistency, report assertions, code syntax and hashes; not rerunning GPU tests or training",
        "version": config["version"],
        "base_commit": registry["base_commit"],
        "syntax_checked": syntax_checked,
        "files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                   "bytes": p.stat().st_size} for p in sorted(set(files))],
        "not_ready": ["full feature cache", "fixed targets/masks", "F-KF implementation", "trainer", "new-head experiment results"]
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "files": len(result["files"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
