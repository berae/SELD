"""Tests of the one-variable probe; the original seven regressions also apply."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models/einv2/audited'))
from runtime import configuration, loss_values, seed_all, AuditedEINV2, make_teacher
import torch
from types import SimpleNamespace


class WeightProbeTests(unittest.TestCase):
    def test_only_jepa_coefficient_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            scalar = Path(directory) / 'scalar'
            scalar.touch()
            base = configuration(Path(directory), scalar, 'C3', 2026, hdf5_dir=directory, lambda_jepa=.2)
            low = configuration(Path(directory), scalar, 'C3', 2026, hdf5_dir=directory, lambda_jepa=.05)
            low['training']['lambda_jepa'] = .2
            self.assertEqual(base, low)
            self.assertEqual(base['training']['lambda_velocity'], .2)
            self.assertEqual(base['training']['max_epoch'], 90)

    def test_weight_change_preserves_outputs_and_scales_gradient(self):
        torch.set_num_threads(2)
        from ruamel.yaml import YAML
        base = YAML(typ='safe').load((ROOT / 'configs/einv2/C3.yaml').read_text())
        base['audit'] = {'alignment': 'current_frame_75ms'}
        low = copy.deepcopy(base)
        low['training']['lambda_jepa'] = .05
        seed_all(2026)
        model = AuditedEINV2(base, SimpleNamespace(label_set=list(range(14)))).train()
        teacher = make_teacher(model)
        x = torch.randn(1, 7, 32, 32)
        target = dict(sed=torch.zeros(1, 8, 2, 14), doa=torch.randn(1, 8, 2, 3) * .1,
                      velocity=torch.randn(1, 8, 2, 3), velocity_mask=torch.ones(1, 8, 2),
                      jepa_valid_mask=torch.ones(1, 8, 2, 3))
        target['sed'][:, :, 0, 0] = 1
        with torch.no_grad():
            t = teacher(x)
        pred = model(x)
        a, b = loss_values(pred, target, t, base), loss_values(pred, target, t, low)
        for name in ('sed', 'doa', 'velocity', 'jepa'):
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        torch.testing.assert_close(a['all'] - b['all'], .15 * a['jepa'])
        params = tuple(model.latent_predictor.parameters())
        ga = torch.autograd.grad(a['all'], params, retain_graph=True)
        gb = torch.autograd.grad(b['all'], params)
        self.assertGreater(sum(g.square().sum().item() for g in ga), 0)
        for first, second in zip(ga, gb):
            torch.testing.assert_close(first * .25, second, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
