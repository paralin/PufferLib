"""Sequential Protein optimization of SAC, followed by finalist continuation."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from pufferlib.sweep import Protein


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-factory', required=True)
    parser.add_argument('--score-function', required=True, help='module:function(checkpoint, seed, episodes) -> rows')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-at', type=datetime.fromisoformat, required=True)
    parser.add_argument('--no-continuation', action='store_true', help='finish after bounded tuning trials')
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--trial-steps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=88000000)
    parser.add_argument('--eval-seed', type=int, default=90000000)
    parser.add_argument('--eval-episodes', type=int, default=32)
    args = parser.parse_args(argv)
    if args.stop_at.tzinfo is None:
        parser.error('--stop-at requires timezone')
    if min(args.trials, args.trial_steps, args.eval_episodes) <= 0:
        parser.error('budgets must be positive')
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    module, name = args.score_function.split(':')
    score = getattr(importlib.import_module(module), name)
    config = dict(metric='combined_return', metric_distribution='linear', goal='maximize',
                  downsample=1, early_stop_quantile=.3, use_gpu=False,
                  learning_rate=dict(distribution='log_normal', min=3e-5, max=1e-3, scale=1),
                  batch_size=dict(distribution='uniform_pow2', min=128, max=512, scale=1),
                  entropy_fraction=dict(distribution='uniform', min=.2, max=.8, scale=1),
                  spr_weight=dict(distribution='uniform', min=0., max=.5, scale=1))
    protein = Protein(config, num_random_samples=3, gp_training_iter=30,
                      suggestions_per_pareto=64, infer_batch_size=256, cost_param='unused')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps(dict(search=config, options={
        k: str(v) if isinstance(v, (Path, datetime)) else v for k, v in vars(args).items()}), indent=2))
    rows = []
    defaults = dict(learning_rate=3e-4, batch_size=256, entropy_fraction=.5, spr_weight=.1)

    def train(path, hypers, steps, stop_at, resume=None):
        command = [sys.executable, '-u', '-m', 'pufferlib.sac_training',
                   '--env-factory', args.env_factory, '--output', str(path),
                   '--stop-at', stop_at.isoformat(), '--steps', str(steps), '--seed', str(args.seed)]
        for key, value in hypers.items():
            command.extend(['--' + key.replace('_', '-'), str(value)])
        if resume:
            command.extend(['--resume', str(resume)])
        with path.with_suffix('.log').open('w') as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode

    for index in range(args.trials):
        if time.time() >= args.stop_at.timestamp() - 1800:
            break
        hypers = deepcopy(defaults)
        if index:
            hypers, _ = protein.suggest(hypers)
        path = args.output / f'trial-{index:03d}'
        started = time.monotonic()
        # Each trial has both a sample cap and a bounded wall-clock cap.
        deadline = min(args.stop_at - timedelta(minutes=30), datetime.now(args.stop_at.tzinfo) + timedelta(minutes=15))
        status = train(path, hypers, args.trial_steps, deadline)
        if status:
            protein.observe(hypers, 0., time.monotonic() - started, is_failure=True)
            rows.append(dict(trial=index, hypers=hypers, status=status))
        else:
            evaluation = score(path / 'actor.pt', args.eval_seed, args.eval_episodes)
            value = float(np.mean([row['combined_return'] for row in evaluation]))
            cost = time.monotonic() - started
            protein.observe(hypers, value, cost)
            result = json.loads((path / 'results.json').read_text())
            rows.append(dict(trial=index, hypers=hypers, status=0, score=value, cost=cost,
                             decisions=result['decisions'], evaluation=evaluation))
        (args.output / 'trials.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps({k: v for k, v in rows[-1].items() if k != 'evaluation'}), flush=True)
    successes = [row for row in rows if row['status'] == 0]
    if not successes:
        raise RuntimeError('no successful SAC trial; inspect trial logs')
    best = max(successes, key=lambda row: row['score'])
    (args.output / 'selected.json').write_text(json.dumps(best, indent=2) + '\n')
    if args.no_continuation:
        return
    status = train(args.output / 'continuation', best['hypers'], 100_000_000, args.stop_at,
                   args.output / f"trial-{best['trial']:03d}" / 'checkpoint.pt')
    if status:
        raise RuntimeError(f'SAC continuation exited {status}')
    print(json.dumps(dict(event='complete', selected_trial=best['trial'])), flush=True)


if __name__ == '__main__':
    main()
