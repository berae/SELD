"""Exercise actual argument/config resolution while replacing the training call."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import _entrypoints as entry


class ConfigTests(unittest.TestCase):
    def test_einv2_resolves_all_variants_without_training(self):
        for variant in entry.EINV2:
            captured = []
            fake = SimpleNamespace(main=lambda ns, cfg: captured.append((ns, cfg)))
            argv = ['einv2.py', '--variant', variant, '--seed', '2028', '--dataset-root', '/data/TAU2020',
                    '--hdf5-root', '/cache/hdf5', '--scalar-path', '/cache/scaler.h5',
                    '--velocity-root', '/cache/velocity', '--jepa-root', '/cache/jepa']
            previous = sys.modules.get('main')
            sys.modules['main'] = fake
            try:
                with patch.object(sys, 'argv', argv):
                    entry.einv2('train')
            finally:
                if previous is None:
                    sys.modules.pop('main', None)
                else:
                    sys.modules['main'] = previous
            ns, cfg = captured[0]
            self.assertEqual(ns.mode, 'train')
            self.assertEqual(ns.seed, 2028)
            self.assertIn('2028', cfg['training']['train_id'])
            self.assertTrue(cfg['hdf5_dir'].endswith('/cache/hdf5'))
            self.assertEqual(cfg['training']['max_epoch'], 90)
            self.assertEqual(cfg['training']['valid_fold'], 1)

    def test_multi_train_disables_test_access_and_uses_promoted_weights(self):
        for variant in entry.MULTI:
            captured = []
            fake = SimpleNamespace(main=lambda argv: captured.append((argv, dict(os.environ))))
            with tempfile.TemporaryDirectory() as folder:
                argv = ['multi.py', '--variant', variant, '--dataset-root', '/data/TAU2020',
                        '--feature-root', '/cache/features', '--output-root', folder, '--seed', '2027']
                with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'train_seldnet': fake}), patch.dict(os.environ):
                    entry.multi_train()
            actual, env = captured[0]
            self.assertEqual(env['SELD_EVALUATE_TEST'], '0')
            self.assertEqual(env['SELD_SEED'], '2027')
            self.assertEqual(float(env['SELD_LAMBDA_VELOCITY']), .05 if variant[-1] in '13' else 0.)
            self.assertEqual(float(env['SELD_LAMBDA_JEPA']), .05 if variant[-1] in '23' else 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
