"""Dependency-light tests for entrypoint separation and import-time safety."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_python_syntax(self):
        for p in ROOT.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))

    def test_entrypoint_help(self):
        for path in ('train/einv2.py', 'train/multi_accdoa.py', 'eval/einv2.py',
                     'eval/multi_accdoa.py', 'preprocess/einv2.py', 'preprocess/multi_accdoa.py'):
            result = subprocess.run([sys.executable, str(ROOT / 'scripts' / path), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_dry_runs_without_training(self):
        for family, variants in [('einv2', ('C0', 'C1', 'C2', 'C3', 'E0', 'E1', 'E2', 'E3')),
                                 ('multi_accdoa', ('A0', 'A1', 'A2', 'A3', 'C0', 'C1', 'C2', 'C3'))]:
            for variant in variants:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/train' / (family + '.py')),
                                         '--variant', variant, '--dry-run'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_provenance(self):
        imported = json.loads((ROOT / 'provenance/source_snapshot.json').read_text(encoding='utf-8'))['files']
        changes = json.loads((ROOT / 'provenance/relocation_changes.json').read_text(encoding='utf-8'))
        latest = {r['path']: r['after_sha256'] for r in changes}
        for row in imported:
            actual = hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest()
            self.assertEqual(actual, latest.get(row['path'], row['sha256']), row['path'])

    def test_no_training_from_evaluation_entrypoint(self):
        text = (ROOT / 'scripts/_entrypoints.py').read_text(encoding='utf-8')
        tree = ast.parse(text)
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'multi_infer')
        calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
        self.assertFalse(any('optimizer' in c or 'train_epoch' in c for c in calls))
        self.assertIn("SELD_EVALUATE_TEST='0'", text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
