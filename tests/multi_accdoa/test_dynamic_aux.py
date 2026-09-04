import torch

from parameters import get_params
from seldnet_model import DynamicMSELoss_ADPIT, MSELoss_ADPIT, SeldModel


def build_two_source_target(frames=2, classes=2):
    target = torch.zeros(1, frames, 6, 4, classes)
    # Two same-class sources B0/B1.
    target[:, :, 1, 0, 0] = 1.0
    target[:, :, 1, 1:, 0] = torch.tensor([1.0, 0.0, 0.0])
    target[:, :, 2, 0, 0] = 1.0
    target[:, :, 2, 1:, 0] = torch.tensor([0.0, 1.0, 0.0])
    return target


def candidate_output(order, frames=2, classes=2):
    b0 = torch.zeros(3, classes)
    b1 = torch.zeros(3, classes)
    b0[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    b1[:, 0] = torch.tensor([0.0, 1.0, 0.0])
    tracks = {'0': b0, '1': b1}
    packed = torch.cat([tracks[item] for item in order], dim=0)
    return packed.unsqueeze(0).unsqueeze(0).repeat(1, frames, 1, 1).reshape(1, frames, -1)


def test_adpit_alignment():
    target = build_two_source_target()
    output = candidate_output('101')
    loss, aligned, assignment = MSELoss_ADPIT().calculate_with_alignment(output, target)
    assert loss.item() == 0.0
    assert torch.all(assignment[:, :, 0] == 5), assignment  # loss_5 = B1/B0/B1
    assert torch.all(assignment[:, :, 1] == 0), assignment  # inactive class tie -> first candidate
    expected = output.reshape(1, 2, 3, 3, 2).permute(0, 1, 2, 4, 3)
    assert torch.equal(aligned, expected)


def test_track_switch_mask():
    params = get_params('40')
    target = build_two_source_target()
    doa = torch.cat((candidate_output('001', frames=1), candidate_output('010', frames=1)), dim=1)
    output = {'doa': doa, 'velocity': torch.zeros(1, 2, 3, 2, 3)}
    values = DynamicMSELoss_ADPIT(params)(output, target)
    assert values['velocity_count'].item() == 0.0
    assert torch.isfinite(values['total'])


def test_causality_and_shapes():
    for task in ('40', '41', '42'):
        params = get_params(task)
        params['unique_classes'] = 14
        model = SeldModel((2, 7, 25, 64), (2, 5, 126), params).eval()
        x = torch.randn(2, 7, 25, 64)
        x_changed = x.clone()
        x_changed[:, :, 15:] = torch.randn_like(x_changed[:, :, 15:])
        with torch.no_grad():
            left = model(x)
            right = model(x_changed)
        for key in left:
            assert left[key].shape == right[key].shape
            assert torch.equal(left[key][:, :3], right[key][:, :3]), (task, key)
        assert left['doa'].shape == (2, 5, 126)
        if params['use_velocity']:
            assert left['velocity'].shape == (2, 5, 3, 14, 3)
        if params['use_jepa']:
            assert left['future_latent'].shape == (2, 5, 3, 14, 3, 128)


if __name__ == '__main__':
    test_adpit_alignment()
    test_track_switch_mask()
    test_causality_and_shapes()
    print('DYNAMIC_AUX_TESTS_PASS')
