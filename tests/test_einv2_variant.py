"""One isolated CPU forward/import check; invoke once per variant (no training)."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('C0', 'C1', 'C2', 'C3', 'E0', 'E1', 'E2', 'E3'), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'models/einv2/variants' / ('C0' if args.variant == 'E0' else args.variant) / 'seld'))
    import torch
    from ruamel.yaml import YAML
    import main as entry
    from methods.ein_seld import models
    from methods.utils.SELD_evaluation_metrics_aligned import METRIC_PROTOCOL
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    cfg = YAML(typ='safe').load((ROOT / 'configs/einv2' / (args.variant + '.yaml')).read_text())
    model = getattr(models, cfg['training']['model'])(cfg, SimpleNamespace(label_set=list(range(14)))).eval()
    x = torch.randn(1, 7, 16, 32)
    with torch.no_grad():
        pred = model(x)
        assert pred['sed'].shape == (1, 4, 2, 14), pred['sed'].shape
        assert pred['doa'].shape == (1, 4, 2, 3), pred['doa'].shape
        assert all(torch.isfinite(v).all() for v in pred.values())
        if args.variant.startswith('C'):
            changed = x.clone()
            changed[:, :, 8:] = torch.randn_like(changed[:, :, 8:])
            after = model(changed)
            for key in ('sed', 'doa'):
                torch.testing.assert_close(pred[key][:, :2], after[key][:, :2], rtol=0, atol=0)
    print(json.dumps(dict(variant=args.variant, model=cfg['training']['model'],
                          metric_protocol=METRIC_PROTOCOL, forward='PASS',
                          feature_level_causal_prefix='PASS' if args.variant.startswith('C') else 'N/A',
                          note='Synthetic feature-level forward; not a full retraining or end-to-end audio causality claim.')))


if __name__ == '__main__':
    main()
