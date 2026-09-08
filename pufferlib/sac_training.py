"""Train optional SAC through a Gymnasium factory with bounded GPU replay."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import importlib
import json
import signal
from pathlib import Path
import time

import numpy as np
import torch

from pufferlib.sac import Learner, Replay


def emit(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-factory', required=True, help='module:callable accepting seed=')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='trusted checkpoint.pt; restores learner/replay, resets matches')
    parser.add_argument('--init-learner', type=Path, help='transfer learner and optimizer state with fresh replay for a new opponent')
    parser.add_argument('--init-actor', type=Path, help='transfer actor weights only; fresh critics, optimizers and replay')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stop-at', type=datetime.fromisoformat, required=True)
    parser.add_argument('--steps', type=int, default=100_000_000)
    parser.add_argument('--seed', type=int, default=88000000)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--capacity', type=int, default=131072)
    parser.add_argument('--warmup', type=int, default=4096)
    parser.add_argument('--history', type=int, default=4)
    parser.add_argument('--checkpoint-seconds', type=float, default=300)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--reward-scale', type=float, default=.1)
    parser.add_argument('--entropy-fraction', type=float, default=.5)
    parser.add_argument('--spr-weight', type=float, default=.1)
    args = parser.parse_args(argv)
    if args.stop_at.tzinfo is None:
        parser.error('--stop-at requires a timezone')
    if min(args.steps, args.num_envs, args.batch_size, args.capacity, args.warmup, args.history, args.checkpoint_seconds) <= 0:
        parser.error('budgets and dimensions must be positive')
    if args.capacity < max(args.num_envs, args.batch_size):
        parser.error('--capacity must fit a collection and training batch')
    module, name = args.env_factory.split(':')
    factory = getattr(importlib.import_module(module), name)
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = {k: str(v) if isinstance(v, (Path, datetime)) else v for k, v in vars(args).items()}
    configuration.update(algorithm='factorized-discrete-sac-spr', torch=torch.__version__, hip=torch.version.hip)
    (args.output / 'config.json').write_text(json.dumps(configuration, indent=2) + '\n')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.monotonic()
    with ExitStack() as stack:
        envs = [stack.enter_context(factory(seed=args.seed + i)) for i in range(args.num_envs)]
        agents_per_env = getattr(envs[0], "agents_per_env", 1)
        rows = args.num_envs * agents_per_env
        width = envs[0].observation_space.shape[0]
        limits = envs[0].action_space.nvec.tolist()
        observations = np.stack([env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)]).reshape(rows, width)
        history = np.repeat(observations[:, None, :], args.history, axis=1)
        action_schedule = bool(getattr(envs[0], "action_schedule", False))
        configuration["action_schedule"] = action_schedule
        (args.output / 'config.json').write_text(json.dumps(configuration, indent=2) + '\n')
        learner = Learner(width * args.history, limits, args.device, args.learning_rate, spr_weight=args.spr_weight, entropy_fraction=args.entropy_fraction, action_schedule=action_schedule)
        collector = deepcopy(learner.actor).cpu().eval()
        replay = Replay(args.capacity, width * args.history, len(limits), args.device)
        returns = np.zeros(rows)
        lengths = np.zeros(rows, dtype=int)
        resets = np.zeros(args.num_envs, dtype=int)
        episode_file = stack.enter_context((args.output / 'episodes.jsonl').open('w'))
        decisions, completed = 0, 0
        if args.resume:
            state = torch.load(args.resume, map_location=args.device, weights_only=False)
            for key in ('env_factory', 'capacity', 'history', 'learning_rate', 'reward_scale', 'spr_weight', 'seed', 'num_envs', 'entropy_fraction', 'action_schedule'):
                if state['configuration'].get(key, .5 if key == 'entropy_fraction' else False if key == 'action_schedule' else None) != configuration[key]:
                    raise ValueError(f'resume must preserve {key}')
            learner.load_state_dict(state['learner'])
            replay.load_state_dict(state['replay'])
            decisions = state['decisions']
            # Begin fresh matches on unused seeds, not the original opening states.
            resets[:] = state.get('resets', np.zeros(args.num_envs, dtype=int)) + 1
            for i, env in enumerate(envs):
                initial_obs = np.asarray(env.reset(seed=args.seed + i + args.num_envs * int(resets[i]))[0]).reshape(agents_per_env, width)
                history[i*agents_per_env:(i+1)*agents_per_env] = initial_obs[:, None]
            torch.set_rng_state(state['torch_rng'].cpu())
            np.random.set_state(state['numpy_rng'])
            if args.device.startswith('cuda'):
                torch.cuda.set_rng_state(state['device_rng'].cpu())
            collector.load_state_dict(learner.actor.state_dict())
        if args.init_learner:
            if args.resume or args.init_actor:
                parser.error('--init-learner cannot be combined with another initialization')
            initial = torch.load(args.init_learner, map_location=args.device, weights_only=False)
            if initial['width'] != width or initial['limits'] != limits or initial['configuration']['history'] != args.history:
                raise ValueError('initial learner observation or action contract differs')
            for key in ('learning_rate', 'reward_scale', 'spr_weight', 'entropy_fraction'):
                if initial['configuration'][key] != configuration[key]:
                    raise ValueError(f'learner transfer must preserve {key}')
            learner.load_state_dict(initial['learner'])
            if initial['configuration'].get('action_schedule', False) != action_schedule:
                # The old temperature was optimized for a different entropy space.
                with torch.no_grad():
                    learner.log_alpha.fill_(-3.)
                learner.alpha_opt.state.clear()
            collector.load_state_dict(learner.actor.state_dict())
        if args.init_actor:
            if args.resume:
                parser.error('--init-actor cannot be combined with --resume')
            initial = torch.load(args.init_actor, map_location=args.device, weights_only=True)
            if initial['history'] != args.history or initial['width'] != width or initial['limits'] != limits:
                raise ValueError('initial actor observation or action contract differs')
            learner.actor.load_state_dict(initial['actor'])
            collector.load_state_dict(learner.actor.state_dict())
        starting_decisions = decisions
        next_save, next_report = started + args.checkpoint_seconds, started + 30
        metrics = None
        torch.save(dict(actor=collector.state_dict(), limits=limits, width=width, history=args.history, action_schedule=action_schedule), args.output / 'initial.pt')

        def save(final=False):
            payload = dict(learner=learner.state_dict(), replay=replay.state_dict(),
                           decisions=decisions, resets=resets.copy(), configuration=configuration, limits=limits, width=width,
                           torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                           device_rng=torch.cuda.get_rng_state() if args.device.startswith('cuda') else None)
            temporary = args.output / 'checkpoint.tmp'
            torch.save(payload, temporary)
            temporary.replace(args.output / 'checkpoint.pt')
            policy = dict(actor={k: v.cpu() for k, v in learner.actor.state_dict().items()},
                          limits=limits, width=width, history=args.history, decisions=decisions, action_schedule=action_schedule)
            torch.save(policy, args.output / f'actor-{decisions:012d}.pt')
            torch.save(policy, args.output / 'actor.pt')
            emit(event='checkpoint', decisions=decisions, updates=learner.updates, final=final)

        stopping = False
        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not stopping and decisions < args.steps and time.time() < args.stop_at.timestamp():
            flat = history.reshape(rows, -1)
            with torch.no_grad():
                if decisions < args.warmup:
                    action = np.stack([np.random.randint(n, size=rows) for n in limits], -1)
                else:
                    action = collector.sample(torch.from_numpy(flat))[0][0].numpy()
            match_actions = action.reshape(args.num_envs, agents_per_env, -1)
            results = [env.step(a if agents_per_env > 1 else a[0]) for env, a in zip(envs, match_actions)]
            following = np.stack([row[0] for row in results]).reshape(rows, width)
            rewards = np.asarray([row[1] for row in results], dtype=np.float32).reshape(rows)
            terminal = np.repeat(np.asarray([row[2] for row in results], dtype=np.float32), agents_per_env)
            next_history = np.concatenate((history[:, 1:], following[:, None]), axis=1)
            replay.add(torch.as_tensor(flat, device=args.device), torch.as_tensor(action, device=args.device),
                       torch.as_tensor(rewards * args.reward_scale, device=args.device),
                       torch.as_tensor(next_history.reshape(rows, -1), device=args.device),
                       torch.as_tensor(terminal, device=args.device))
            decisions += rows
            returns += rewards
            lengths += 1
            for i, row in enumerate(results):
                if row[2] or row[3]:
                    for seat in range(agents_per_env):
                        agent = i * agents_per_env + seat
                        episode_file.write(json.dumps(dict(seed=args.seed + i + args.num_envs * int(resets[i]),
                            seat=seat, reward=float(returns[agent]), decisions=int(lengths[agent]), terminated=bool(row[2]))) + '\n')
                        returns[agent], lengths[agent] = 0, 0
                    episode_file.flush()
                    completed += 1
                    resets[i] += 1
                    obs, _ = envs[i].reset(seed=args.seed + i + args.num_envs * int(resets[i]))
                    next_history[i*agents_per_env:(i+1)*agents_per_env] = np.asarray(obs).reshape(agents_per_env, width)[:, None]
            history = next_history
            if decisions >= args.warmup and replay.size >= args.batch_size:
                indices, weights, batch = replay.sample(args.batch_size, min(1., .4 + .6 * decisions / 1_000_000))
                errors, metrics = learner.update(batch, weights)
                replay.update(indices, errors)
                if learner.updates % 8 == 0:
                    collector.load_state_dict(learner.actor.state_dict())
            now = time.monotonic()
            if now >= next_report:
                emit(event='progress', decisions=decisions, updates=learner.updates, episodes=completed,
                     agents_per_env=agents_per_env, environment_steps=(decisions-starting_decisions)//agents_per_env,
                     decisions_per_second=(decisions - starting_decisions) / (now - started), replay=replay.size,
                     metrics=metrics.tolist() if metrics is not None else None)
                next_report = now + 30
            if now >= next_save:
                save()
                next_save = time.monotonic() + args.checkpoint_seconds
        save(final=True)
        result = dict(decisions=decisions, fresh_decisions=decisions - starting_decisions, updates=learner.updates, episodes=completed,
                      agents_per_env=agents_per_env, environment_steps=(decisions-starting_decisions)//agents_per_env,
                      seconds=time.monotonic() - started, stopped_at=datetime.now().astimezone().isoformat())
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        emit(event='complete', **result)


if __name__ == '__main__':
    main()
