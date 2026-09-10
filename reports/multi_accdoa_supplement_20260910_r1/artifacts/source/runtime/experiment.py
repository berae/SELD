"""Inspect or launch one frozen experiment-matrix cell."""

import argparse
import os
import subprocess

from experiment_matrix import DEFAULT_SEEDS, TASK_IDS, VARIANTS, get_task_id


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    parser.add_argument('--stage', choices=('smoke', 'full'), default='full')
    parser.add_argument('--seed', type=int, choices=DEFAULT_SEEDS, default=DEFAULT_SEEDS[0])
    parser.add_argument('--job-id')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timeout-hours', type=int, default=24)
    parser.add_argument('--launch', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    task_id = get_task_id(args.variant, args.stage)
    job_id = args.job_id or '{}_{}_seed{}_strictcausal_v1'.format(
        args.variant, args.stage, args.seed
    )
    project_dir = os.path.dirname(os.path.abspath(__file__))
    command = [
        os.path.join(project_dir, 'launch_dynamic_detached.sh'),
        str(args.gpu), task_id, job_id, str(args.timeout_hours), str(args.seed),
    ]
    print('variant={} stage={} task_id={} seed={}'.format(
        args.variant, args.stage, task_id, args.seed
    ))
    print('command={}'.format(' '.join(command)))
    if args.launch:
        subprocess.check_call(command)


if __name__ == '__main__':
    main()
