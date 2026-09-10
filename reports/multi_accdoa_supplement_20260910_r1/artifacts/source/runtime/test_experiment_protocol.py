import os

import numpy as np

from cls_feature_class import FeatureClass
from experiment_matrix import DEFAULT_SEEDS, TASK_CONFIGS, TASK_IDS, VARIANTS
from parameters import get_params


def test_matrix_is_complete():
    assert set(VARIANTS) == {'A0', 'A1', 'A2', 'A3', 'C0', 'C1', 'C2', 'C3'}
    assert DEFAULT_SEEDS == (2026, 2027, 2028)
    assert len(TASK_CONFIGS) == 16
    assert all(set(stages) == {'full', 'smoke'} for stages in TASK_IDS.values())


def test_paper_tasks_share_frontend_protocol():
    feature_dirs = set()
    for task_id, task in TASK_CONFIGS.items():
        params = get_params(task_id)
        assert params['causal_frontend'] is True
        assert params['normalization_fit_splits'] == [2, 3, 4, 5, 6]
        assert params['frontend_lookahead_s'] == 0.0
        assert params['experiment_variant'] == task['variant']
        feature_dirs.add(params['feat_label_dir'])
    assert len(feature_dirs) == 1


def test_seed_override():
    previous = os.environ.get('SELD_SEED')
    os.environ['SELD_SEED'] = '2028'
    try:
        assert get_params('32')['seed'] == 2028
    finally:
        if previous is None:
            del os.environ['SELD_SEED']
        else:
            os.environ['SELD_SEED'] = previous


def test_train_only_scaler_file_selection():
    files = [
        'fold1_room1_mix001.npy',
        'fold2_room1_mix001.npy',
        'fold6_room1_mix001.npy',
        'notes.txt',
    ]
    assert FeatureClass._select_normalization_fit_files(files, [2, 3, 4, 5, 6]) == [
        'fold2_room1_mix001.npy', 'fold6_room1_mix001.npy'
    ]


def test_causal_stft_prefix_invariance():
    feature = FeatureClass.__new__(FeatureClass)
    feature._nfft = 16
    feature._hop_len = 4
    feature._win_len = 8
    feature._causal_frontend = True

    rng = np.random.RandomState(2026)
    original = rng.randn(64, 2)
    changed = original.copy()
    cutoff_frame = 5
    changed[(cutoff_frame + 1) * feature._hop_len:] = rng.randn(
        len(changed) - (cutoff_frame + 1) * feature._hop_len, 2
    )

    left = feature._spectrogram(original, 16)
    right = feature._spectrogram(changed, 16)
    np.testing.assert_allclose(left[:cutoff_frame + 1], right[:cutoff_frame + 1])
    assert np.max(np.abs(left[cutoff_frame + 1:] - right[cutoff_frame + 1:])) > 0


if __name__ == '__main__':
    test_matrix_is_complete()
    test_paper_tasks_share_frontend_protocol()
    test_seed_override()
    test_train_only_scaler_file_selection()
    test_causal_stft_prefix_invariance()
    print('EXPERIMENT_PROTOCOL_TESTS_PASS')
