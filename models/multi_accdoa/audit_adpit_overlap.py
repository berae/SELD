from pathlib import Path

import numpy as np


LABEL_DIR = Path(
    '/work/zhanghc/Myllm/SELD/CausalMultiACCDOA_TAU2020/'
    'features/tau2020_foa_multiaccdoa/foa_dev_adpit_label'
)


def main():
    files = sorted(LABEL_DIR.glob('*.npy'))
    counts = np.zeros(6, dtype=np.int64)
    examples = [[] for _ in range(6)]
    shapes = set()
    for path in files:
        label = np.load(str(path), mmap_mode='r')
        shapes.add(tuple(label.shape[1:]))
        active = label[:, :, 0, :] > 0.5
        per_slot = active.sum(axis=(0, 2)).astype(np.int64)
        counts += per_slot
        for slot, value in enumerate(per_slot):
            if value and len(examples[slot]) < 3:
                examples[slot].append((path.name, int(value)))

    print('files={}'.format(len(files)))
    print('shapes={}'.format(sorted(shapes)))
    for slot, count in enumerate(counts):
        print('slot{}={}; examples={}'.format(slot, int(count), examples[slot]))


if __name__ == '__main__':
    main()
