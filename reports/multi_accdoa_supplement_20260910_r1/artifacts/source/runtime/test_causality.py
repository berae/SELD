import copy

import torch

import parameters
from seldnet_model import SeldModel


def main():
    torch.manual_seed(2026)
    params = parameters.get_params('34')
    params['dropout_rate'] = 0.0
    model = SeldModel((2, 7, 250, 64), (2, 50, 126), params).eval()

    original = torch.randn(2, 7, 250, 64)
    changed = copy.deepcopy(original)
    output_cutoff = 19
    feature_cutoff = (output_cutoff + 1) * 5
    changed[:, :, feature_cutoff:, :] = torch.randn_like(changed[:, :, feature_cutoff:, :])

    with torch.no_grad():
        y_original = model(original)
        y_changed = model(changed)

    prefix_delta = (y_original[:, : output_cutoff + 1] - y_changed[:, : output_cutoff + 1]).abs().max().item()
    suffix_delta = (y_original[:, output_cutoff + 1 :] - y_changed[:, output_cutoff + 1 :]).abs().max().item()
    print('prefix_max_abs_delta={:.9g}'.format(prefix_delta))
    print('suffix_max_abs_delta={:.9g}'.format(suffix_delta))
    assert prefix_delta < 1e-6, 'Future input changed a causal output prefix'
    assert suffix_delta > 1e-6, 'Test perturbation did not affect future outputs'
    print('CAUSALITY_TEST=PASS')


if __name__ == '__main__':
    main()
