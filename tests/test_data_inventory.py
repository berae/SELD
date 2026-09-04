import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('coverage_check', ROOT / 'scripts/data/check_label_coverage.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class InventoryTests(unittest.TestCase):
    def test_sidecar_is_not_a_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'fold1.csv').write_text('0,0,0,0,0\n')
            (root / '._fold1.csv').write_bytes(b'not-a-label')
            clips, sidecars = coverage.names(root, '.csv')
            self.assertEqual(clips, {'fold1'})
            self.assertEqual(len(sidecars), 1)

    def test_subdirectory_identity_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('train', 'test'):
                (root / folder).mkdir()
                (root / folder / 'same.csv').write_text('')
            clips, _ = coverage.names(root, '.csv')
            self.assertEqual(len(clips), 2)

    def test_recorded_coverage_is_consistent(self):
        path = ROOT / 'docs/data_inventory/2026-09-04/label_coverage.json'
        rows = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 11)
        for row in rows:
            self.assertTrue(row['matched'])
            if row['matched']:
                self.assertEqual(row['audio'], row['labels'])
                self.assertFalse(row['audio_without_labels'])
                self.assertFalse(row['labels_without_audio'])

    def test_snapshot_headers_and_collector(self):
        collector_hash = hashlib.sha256((ROOT / 'scripts/data/collect_inventory.py').read_bytes()).hexdigest()
        for host in ('rabbit02', 'RB05'):
            path = ROOT / 'docs/data_inventory/2026-09-04' / (host + '.json')
            snapshot = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(snapshot['collector_sha256'], collector_hash)
            for asset in snapshot['assets']:
                self.assertFalse(asset['errors'])
                self.assertFalse(asset['broken_symlinks'])
                for group in asset['groups']:
                    self.assertEqual(group['audio']['files'], group['audio']['headers_read'])
                    self.assertFalse(group['audio_header_errors'])

    def test_notebook_recomputes_migration(self):
        notebook = json.loads((ROOT / 'notebooks/data_inventory_20260904.ipynb').read_text(encoding='utf-8'))
        namespace = {'snapshots': {
            host: json.loads((ROOT / 'docs/data_inventory/2026-09-04' / (host + '.json')).read_text(encoding='utf-8'))
            for host in ('rabbit02', 'RB05')
        }, 'base': ROOT / 'docs/data_inventory/2026-09-04', 'json': json}
        cell = next(c for c in notebook['cells'] if c['id'] == 'migration')
        exec(compile(''.join(cell['source']), 'notebook:migration', 'exec'), namespace)


if __name__ == '__main__':
    unittest.main(verbosity=2)
