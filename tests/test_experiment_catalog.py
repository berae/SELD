"""Lightweight reproducibility checks for result and diagram artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'reports/catalog_20260904'


class CatalogTests(unittest.TestCase):
    def test_run_index_coverage(self):
        with (CATALOG / 'run_index.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 84)
        keys = {(r['family'], r['context'], r['variant'], r['seed'], r['split']) for r in rows}
        self.assertEqual(len(keys), 84)
        self.assertEqual({r['profile'] for r in rows}, {'dcase2023_micro'})
        self.assertEqual({r['seed'] for r in rows}, {'2026', '2027', '2028'})
        for row in rows:
            self.assertTrue((ROOT / row['portable_recipe_not_historical_config']).is_file())
            self.assertGreater(int(row['checkpoint_bytes']), 0)
            self.assertEqual(int(row['prediction_files']), 200 if row['split'] == 'evaluation' else 100)

    def test_recorded_source_differences_are_accounted_for(self):
        audit = json.loads((CATALOG / 'source_audit.json').read_text(encoding='utf-8'))
        changes = json.loads((ROOT / 'provenance/relocation_changes.json').read_text(encoding='utf-8'))
        for source in audit['architecture_sources']:
            actual = hashlib.sha256((ROOT / source['repository_path']).read_bytes()).hexdigest()
            self.assertEqual(actual, source['repository_sha256'])
            if not source['matches_repository']:
                self.assertTrue(any(r['path'] == source['repository_path'] and r['before_sha256'] == source['sha256']
                                    and r['after_sha256'] == actual for r in changes), source['repository_path'])

    def test_six_editable_svg_figures(self):
        paths = sorted((ROOT / 'docs/figures').glob('*.svg'))
        self.assertEqual(len(paths), 6)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        for path in paths:
            svg = ET.fromstring(path.read_text(encoding='utf-8'))
            self.assertIsNotNone(svg.find('svg:title', ns))
            self.assertGreater(len(svg.findall('.//svg:text', ns)), 20)
            self.assertFalse(svg.findall('.//svg:script', ns))

    def test_document_links_and_table_widths(self):
        for rel in ('reports/EXPERIMENT_CATALOG.md', 'reports/catalog_20260904/RUN_INDEX.md', 'docs/METHOD_AND_ARCHITECTURE.md'):
            path = ROOT / rel
            text = path.read_text(encoding='utf-8')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' not in target:
                    self.assertTrue((path.parent / target.split('#')[0]).is_file(), target)
            width = None
            for line in text.splitlines():
                if line.startswith('|'):
                    count = line.count('|')
                    if width is None:
                        width = count
                    self.assertEqual(count, width, line)
                else:
                    width = None

    def test_generators_are_reproducible(self):
        paths = [ROOT / 'reports/EXPERIMENT_CATALOG.md', CATALOG / 'run_index.csv', CATALOG / 'RUN_INDEX.md',
                 CATALOG / 'catalog_checks.json', *sorted((ROOT / 'docs/figures').glob('*.svg'))]
        before = {p: p.read_bytes() for p in paths}
        for script in ('scripts/reports/build_experiment_catalog.py', 'scripts/figures/build_architecture_figures.py'):
            result = subprocess.run([sys.executable, str(ROOT / script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        for path in paths:
            self.assertEqual(before[path], path.read_bytes(), str(path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
