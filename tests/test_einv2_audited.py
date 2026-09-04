"""CPU regression tests; no dataset access, optimizer runs, or training jobs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models/einv2/audited'))
from runtime import (AuditedEINV2, CausalLogmelIntensity_Extractor, Frontend, WEIGHTS,
                     assignment, canonical, loss_values, make_teacher, update_teacher)
import h5py
import numpy as np
import torch
from ruamel.yaml import YAML
from types import SimpleNamespace
from methods.ein_seld.losses import Losses as HistoricalLoss


class AuditedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base = YAML(typ='safe').load((ROOT / 'configs/einv2/C3.yaml').read_text())
        cls.base['audit'] = {'alignment': 'current_frame_75ms'}
        cls.dataset = SimpleNamespace(label_set=list(range(14)))

    def model(self, cfg):
        torch.manual_seed(2026)
        return AuditedEINV2(cfg, self.dataset)

    def labels(self, frames=8):
        torch.manual_seed(999)
        sed = torch.zeros(1, frames, 2, 14)
        sed[:, :, 0, 1] = 1
        sed[:, :, 1, 4] = 1
        return dict(sed=sed, doa=torch.randn(1, frames, 2, 3) * .3,
                    velocity=torch.randn(1, frames, 2, 3), velocity_mask=torch.ones(1, frames, 2),
                    jepa_valid_mask=torch.ones(1, frames, 2, 3))

    def test_paired_initialization_and_dropout(self):
        torch.manual_seed(19)
        x = torch.randn(1, 7, 32, 32)
        first = None
        for variant, weights in WEIGHTS.items():
            cfg = copy.deepcopy(self.base)
            cfg['training']['lambda_velocity'], cfg['training']['lambda_jepa'] = weights
            model = self.model(cfg).train()
            rng = torch.get_rng_state().clone()
            with torch.no_grad():
                out = model(x)
            if first is None:
                first = ({k: v.clone() for k, v in model.state_dict().items()}, rng, out)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, first[0][key], rtol=0, atol=0)
            self.assertTrue(torch.equal(rng, first[1]), variant)
            for key in out:
                torch.testing.assert_close(out[key], first[2][key], rtol=0, atol=0)

    def test_audio_frame_causality_train_and_eval(self):
        model = self.model(self.base)
        feature = CausalLogmelIntensity_Extractor(self.base)
        torch.manual_seed(19)
        audio = torch.randn(1, 4, 19200) * .05
        changed = audio.clone()
        changed[..., 12000:] = torch.randn_like(changed[..., 12000:])
        with torch.no_grad():
            for training in (False, True):
                model.train(training)
                torch.manual_seed(88)
                out = model(feature(audio))
                torch.manual_seed(88)
                after = model(feature(changed))
                self.assertEqual(out['sed'].shape, (1, 8, 2, 14))
                for key in out:
                    torch.testing.assert_close(out[key][:, :5], after[key][:, :5], rtol=0, atol=0)
        # Check the exact 75 ms bound for frame index 4 in eval mode.
        model.eval()
        with torch.no_grad():
            out = model(feature(audio))
            changed = audio.clone()
            changed[..., 11400:] *= -3
            after = model(feature(changed))
            torch.testing.assert_close(out['doa'][:, :5], after['doa'][:, :5], rtol=0, atol=0)

    def test_frontend_train_eval_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scalar.h5'
            with h5py.File(path, 'w') as hf:
                hf.create_dataset('mean', data=np.arange(7, dtype=np.float32).reshape(1, 7, 1, 1) * np.ones((1, 7, 1, 256)))
                hf.create_dataset('std', data=np.ones((1, 7, 1, 256), dtype=np.float32))
                hf.attrs.update(channel_order='logmel_WYZX_IV_XYZ', folds='2,3,4,5,6')
            cfg = copy.deepcopy(self.base)
            cfg['causal_scalar_path'] = str(path)
            train, infer = Frontend(cfg).train(), Frontend(cfg).eval()
            audio = torch.randn(1, 4, 4800)
            torch.testing.assert_close(train(audio), infer(audio), rtol=0, atol=0)

    def test_teacher_slot_permutation_invariance(self):
        model = self.model(self.base).eval()
        with torch.no_grad():
            pred = model(torch.randn(1, 7, 32, 32))
        target = self.labels()
        teacher = {key: val.clone() for key, val in pred.items()}
        flipped = {key: val.clone() for key, val in teacher.items()}
        for key in flipped:
            flipped[key][:, 1::2] = flipped[key][:, 1::2].flip(2)
        before = loss_values(pred, target, teacher, self.base)
        after = loss_values(pred, target, flipped, self.base)
        torch.testing.assert_close(before['jepa'], after['jepa'], rtol=0, atol=0)

    def test_historical_loss_equivalence_when_assignments_agree(self):
        model = self.model(self.base).eval()
        with torch.no_grad():
            pred = model(torch.randn(1, 7, 32, 32))
        target = self.labels()
        new = loss_values(pred, target, pred, self.base)
        old = HistoricalLoss(self.base).calculate(pred, target, pred['latent'])
        for key in ('all', 'sed', 'doa', 'velocity', 'jepa'):
            torch.testing.assert_close(new[key], old[key], rtol=1e-6, atol=1e-7)

    def test_auxiliary_gradient_routes(self):
        for variant, (lv, lj) in WEIGHTS.items():
            cfg = copy.deepcopy(self.base)
            cfg['training'].update(lambda_velocity=lv, lambda_jepa=lj)
            model = self.model(cfg).train()
            x = torch.randn(1, 7, 32, 32)
            teacher = make_teacher(model)
            with torch.no_grad():
                target_pred = teacher(x)
            values = loss_values(model(x), self.labels(), target_pred, cfg)
            values['all'].backward()
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            for name, module, enabled in [('velocity', model.velocity_heads, lv), ('jepa', model.latent_predictor, lj)]:
                grads = [p.grad for p in module.parameters()]
                self.assertEqual(any(g is not None for g in grads), bool(enabled), (variant, name))
                self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))
            self.assertTrue(torch.isfinite(model.doa_heads[0].weight.grad).all())

    def test_teacher_ema(self):
        model = self.model(self.base)
        teacher = make_teacher(model)
        before = next(teacher.parameters()).clone()
        with torch.no_grad():
            next(model.parameters()).add_(1)
        update_teacher(teacher, model, .996)
        torch.testing.assert_close(next(teacher.parameters()), before + .004, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
