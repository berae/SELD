"""Regression tests for the dynamic-pair velocity-loss filter."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models/einv2/audited'))
from runtime import loss_values


class DynamicMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = YAML(typ='safe').load((ROOT / 'configs/einv2/C3.yaml').read_text())
        cls.cfg['training'].update(lambda_velocity=.2, lambda_jepa=0.)

    def inputs(self):
        pred = {
            'sed': torch.zeros(1, 2, 2, 14),
            'doa': torch.zeros(1, 2, 2, 3),
            'velocity': torch.ones(1, 2, 2, 3),
        }
        target = {
            'sed': torch.zeros_like(pred['sed']),
            'doa': torch.zeros_like(pred['doa']),
            'velocity': torch.zeros_like(pred['velocity']),
            'velocity_mask': torch.ones(1, 2, 2),
            'jepa_valid_mask': torch.ones(1, 2, 2, 3),
        }
        target['velocity'][0, 1, 0, 0] = 1.
        return pred, target

    def test_zero_threshold_preserves_all_valid_pairs(self):
        pred, target = self.inputs()
        cfg = copy.deepcopy(self.cfg)
        cfg['training']['velocity_min_norm'] = 0.
        value = loss_values(pred, target, None, cfg)['velocity']
        expected = torch.nn.functional.smooth_l1_loss(
            pred['velocity'], target['velocity'], reduction='none').mean(-1).mean()
        torch.testing.assert_close(value, expected)

    def test_positive_threshold_keeps_only_moving_pairs(self):
        pred, target = self.inputs()
        cfg = copy.deepcopy(self.cfg)
        cfg['training']['velocity_min_norm'] = 1e-6
        value = loss_values(pred, target, None, cfg)['velocity']
        expected = torch.nn.functional.smooth_l1_loss(
            pred['velocity'][0, 1, 0], target['velocity'][0, 1, 0], reduction='none').mean()
        torch.testing.assert_close(value, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
