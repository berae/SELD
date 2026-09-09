"""Score one saved prediction directory using the same official metric code."""
import argparse
import json
from pathlib import Path

from aligned_metrics import AlignedMetrics, SCHEMAS, UPSTREAM, load_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', required=True)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--reference-glob', default='*.csv')
    parser.add_argument('--prediction-schema', choices=SCHEMAS, required=True)
    parser.add_argument('--reference-schema', choices=SCHEMAS, default='polar5')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reference = {p.name: p for p in sorted(Path(args.reference_dir).glob(args.reference_glob))}
    prediction = {p.name: p for p in Path(args.prediction_dir).glob('*.csv')}
    if not reference or set(reference) != set(prediction):
        raise ValueError('Reference/prediction filename sets must match exactly; check --reference-glob.')
    metric = AlignedMetrics(classes=14, frames=600, frames_per_second=10, threshold=20)
    for name, path in reference.items():
        metric.update(load_csv(prediction[name], args.prediction_schema), load_csv(path, args.reference_schema))
    result = dict(dataset='TAU2020', classes=14, frames_per_clip=600, frame_seconds=.1,
                  segment_seconds=1, threshold_degrees=20, files=len(reference), upstream=UPSTREAM,
                  scores=metric.scores(), counters=metric.counters(), inputs=vars(args))
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['scores'], indent=2))


if __name__ == '__main__':
    main()
