"""Frozen-proposal Q learning with bounded prioritized replay.

The adapter supplies environment rollouts and actor/action contracts. This
module owns collection retention, critic updates, continuation and evaluation.
The optional qplan library supplies immutable replay and categorical Q types.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from pufferlib.qtraining import CriticTrainer, initialization_identity_matches
from qplan import CandidateBatch, ContextBatch, critic_diagnostics
from qplan.critic import FactorCritic
from qplan.replay import ReplayStore, ShardWriter


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def resolve_resume_metadata(manifest: dict[str, object]) -> tuple[str | None, str | None, str, list[tuple[int, int]]]:
    """Resume metadata from a prior manifest: init provenance, collection policy, seed lineage."""
    init_from = manifest.get("init_from")
    init_checkpoint_hash = manifest.get("init_checkpoint_sha256")
    collection_policy = manifest.get("collection_policy", "fixed-actor")
    used_ranges = lineage_seed_ranges(manifest)
    return init_from, init_checkpoint_hash, collection_policy, used_ranges


def lineage_seed_ranges(manifest: dict[str, object]) -> list[tuple[int, int]]:
    """Half-open [start, end) seed ranges one experiment manifest commits to.

    used_seed_ranges covers the experiment and every critic ancestor it was
    initialized from. Historical manifests without the field held one
    contiguous block from seed through training and held-out seeds.
    """
    recorded = manifest.get("used_seed_ranges")
    if recorded is None:
        start = manifest["seed"]
        recorded = [[start, start + manifest["episodes"] + len(manifest["held_out_seeds"])]]
    return sorted({(int(lo), int(hi)) for lo, hi in recorded})


def overlapping_range(used: list[tuple[int, int]], candidate: tuple[int, int]) -> tuple[int, int] | None:
    """Return the first used half-open range intersecting candidate, else None."""
    for lo, hi in used:
        if lo < candidate[1] and candidate[0] < hi:
            return (lo, hi)
    return None


def tensors(batch):
    context = np.concatenate((batch.observations, batch.actor_contexts.reshape(len(batch.actions), -1)), 1)
    next_context = np.concatenate((batch.next_observations, batch.next_actor_contexts.reshape(len(batch.actions), -1)), 1)
    return tuple(torch.from_numpy(value) for value in (
        context, batch.actions, batch.reward_targets, next_context, batch.next_actions,
        batch.terminals | batch.truncated))


def evaluate_rows(model, records, discount=None):
    rows = [row for episode in records for row in episode]
    contexts = torch.from_numpy(np.stack([np.concatenate((row.observation, row.actor_context.ravel())) for row in rows]))
    actions = torch.from_numpy(np.stack([row.action for row in rows]))
    realized = []
    for episode in records:
        if discount is None:
            realized.extend([episode[-1].reward_target] * len(episode))
        else:
            returns_to_go = []
            value = 0.
            for row in reversed(episode):
                value = row.reward_target + discount * value
                returns_to_go.append(value)
            realized.extend(reversed(returns_to_go))
    returns = torch.tensor(realized)
    with torch.no_grad():
        prediction = model.score(ContextBatch(contexts), CandidateBatch(actions[:, None]))[:, 0]
        shuffled = model.score(ContextBatch(contexts), CandidateBatch(actions.roll(1, 0)[:, None]))[:, 0]
        facts = critic_diagnostics(prediction, returns, 0)
        probabilities = model(contexts, actions).softmax(-1)
    return {**{key: value if not isinstance(value, float) or np.isfinite(value) else None
               for key, value in asdict(facts).items()},
            "return_target_min": float(returns.min()), "return_target_max": float(returns.max()),
            "shuffled_action_mae": float((shuffled - returns).abs().mean()),
            "edge_probability": float(probabilities[:, [0, -1]].sum(-1).mean()),
            "has_return_variation": bool(returns.max() > returns.min()),
            "is_degenerate": facts.is_degenerate or bool(returns.max() == returns.min())}


def main(adapter, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path, help="continue a prior experiment in a fresh output directory")
    group.add_argument("--init-from", type=Path,
                       help="start a new experiment whose collector and initial critic weights come from a prior experiment")
    parser.add_argument("--episodes", type=int, help="training episodes (default 64; inherited on resume)")
    parser.add_argument("--held-out", type=int, help="matched evaluation seeds (default 16; inherited on resume)")
    parser.add_argument("--updates", type=int, default=5000, help="total completed critic updates, including resumed updates")
    parser.add_argument("--replay-capacity", type=int, help="maximum retained transitions (default 131072; inherited on resume)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, help="initial collection seed (default 7000; inherited on resume)")
    parser.add_argument("--allow-environment-change", action="store_true",
                        help="allow --init-from weights on a different kernel or reset distribution; actor and learned formats must match")
    parser.add_argument("--reset-value-head", action="store_true",
                        help="with --init-from, retain critic features but reset the return head for a changed reward objective")
    args = parser.parse_args(argv)
    if args.reset_value_head and args.init_from is None:
        parser.error("--reset-value-head requires --init-from")
    if args.allow_environment_change and args.init_from is None:
        parser.error("--allow-environment-change requires --init-from")
    previous_manifest = None
    if args.resume is not None:
        args.resume = args.resume.resolve()
        previous_manifest = json.loads((args.resume / "experiment.json").read_text())
    for name, default in (("episodes", 64), ("held_out", 16), ("seed", 7000), ("replay_capacity", 131072)):
        previous_value = previous_manifest.get(name, default) if previous_manifest else default
        supplied = getattr(args, name)
        if previous_manifest and supplied is not None and supplied != previous_value:
            parser.error(f"--{name.replace('_', '-')} must match the resumed experiment")
        setattr(args, name, previous_value if supplied is None else supplied)
    if min(args.episodes, args.held_out, args.updates, args.replay_capacity) < 1 or args.seed < 0:
        parser.error("episode counts, updates and replay capacity must be positive; seed must be nonnegative")
    if args.seed + args.episodes + args.held_out > 2**32:
        parser.error("training and held-out seeds must fit the kernel's unsigned 32-bit seed range")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Refuse accidental replacement of an earlier experiment's replay or critic.
    manifest_path = args.output / "experiment.json"
    if manifest_path.exists():
        parser.error("output already contains an experiment; choose a fresh output directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    actor = adapter.load_actor(args.actor)
    identity = adapter.identity(args.actor)
    digest = identity.actor_hash
    held_out = list(range(args.seed + args.episodes, args.seed + args.episodes + args.held_out))
    used_range = (args.seed, args.seed + args.episodes + args.held_out)
    replay_path = args.output / "replay"
    trainer = CriticTrainer(args.device, args.seed, context_size=adapter.context_size,
                            limits=adapter.action_limits, replay_capacity=args.replay_capacity,
                            reward_discount=getattr(adapter, "reward_discount", None))
    resume_checkpoint_hash = None
    batch = None
    if previous_manifest:
        if previous_manifest["identity"] != asdict(identity):
            parser.error("resumed actor, kernel, formats or objective differ")
        if previous_manifest["held_out_seeds"] != held_out:
            parser.error("resumed held-out seeds differ")
        replay_path = Path(previous_manifest.get("replay_path", args.resume / "replay")).resolve()
        checkpoint = args.resume / "critic.pt"
        batch = ReplayStore.open(replay_path, identity, held_out_seeds=held_out).read_all(max_rows=args.replay_capacity)
        replay_digest = hashlib.sha256((replay_path / "manifest.jsonl").read_bytes()).hexdigest()
        trainer.restore(checkpoint, identity, replay_digest)
        if args.updates <= trainer.updates:
            parser.error("--updates must exceed the checkpoint's completed update count")
        resume_checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    init_from = None
    init_checkpoint_hash = None
    init_updates = previous_manifest.get("init_updates", 0) if previous_manifest else 0
    collector = None
    collection_policy = "fixed-actor"
    used_ranges = [used_range]
    if previous_manifest:
        # Resume preserves init provenance as metadata; never recollects, never infers --init-from.
        init_from, init_checkpoint_hash, collection_policy, used_ranges = resolve_resume_metadata(previous_manifest)
    if args.init_from is not None:
        init_from = args.init_from.resolve()
        if not (init_from / "experiment.json").is_file():
            parser.error("--init-from points to a directory without an experiment manifest")
        prior_manifest = json.loads((init_from / "experiment.json").read_text())
        prior_ranges = lineage_seed_ranges(prior_manifest)
        clash = overlapping_range(prior_ranges, used_range)
        if clash is not None:
            parser.error(f"seeds overlap the prior experiment's range {list(clash)}; choose a disjoint --seed")
        checkpoint = init_from / "critic.pt"
        if not checkpoint.is_file():
            parser.error("--init-from points to a directory without a critic checkpoint")
        if not initialization_identity_matches(prior_manifest.get("identity", {}), asdict(identity), args.allow_environment_change, args.reset_value_head):
            parser.error("--init-from manifest actor, kernel, formats or objective differ")
        if not isinstance(prior_manifest.get("updates"), int):
            parser.error("--init-from manifest must record completed updates")
        try:
            init_updates = trainer.load_weights(checkpoint, identity, prior_manifest["updates"],
                                                allow_environment_change=args.allow_environment_change,
                                                reset_value_head=args.reset_value_head)
        except ValueError as error:
            parser.error(str(error))
        collector = FactorCritic(adapter.context_size, adapter.action_limits)
        collector.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
        collector.eval().requires_grad_(False)
        init_checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        collection_policy = "q-guided"
        used_ranges = sorted(set(prior_ranges) | {used_range})
    manifest = {**vars(args), "actor": str(args.actor), "output": str(args.output),
                "resume": str(args.resume) if args.resume else None,
                "resume_checkpoint_sha256": resume_checkpoint_hash,
                "init_from": str(init_from) if init_from else None,
                "init_checkpoint_sha256": init_checkpoint_hash,
                "collection_policy": collection_policy,
                "collection_objective": prior_manifest["identity"]["reward_target"] if init_from else identity.reward_target,
                "used_seed_ranges": [list(r) for r in used_ranges],
                "starting_updates": trainer.updates, "replay_path": str(replay_path),
                "init_updates": init_updates,
                "identity": asdict(identity), "held_out_seeds": held_out, "candidates": 16,
                **adapter.runtime_metadata,
                "actor_device": "cpu", "critic_device": args.device,
                "replay_sampling": "proportional-td-error", "replay_eviction": "fifo",
                "reward_discount": trainer.reward_discount, "target_ema": .01, "learning_rate": .0003, "batch_size": 256,
                "runtime_note": adapter.runtime_note}
    with manifest_path.open("x") as output:
        json.dump(manifest, output, indent=2)
    started = time.perf_counter()
    collected = 0
    if previous_manifest is None:
        emit("collecting", episodes=args.episodes, actor_hash=digest,
             collection_policy=collection_policy)
        retained = deque(maxlen=args.replay_capacity)
        for first in range(0, args.episodes, 8):
            seeds = list(range(args.seed + first, args.seed + min(first + 8, args.episodes)))
            records, latency = (adapter.episodes(actor, seeds, 16, collector)
                                if collector is not None else
                                adapter.episodes(actor, seeds))
            for episode in records:
                retained.extend(episode)
                collected += len(episode)
            emit("collected", episodes=first + len(seeds), combined_returns=[sum(float(r.reward_diagnostics[0]) for r in e) for e in records],
                 actor_p95_ms=latency, retained_transitions=len(retained),
                 evicted_transitions=max(0, collected - len(retained)))
        with ShardWriter(replay_path, identity, held_out_seeds=held_out, shard_capacity=4096) as writer:
            for row in retained:
                writer.add(row)
        del retained, records, episode, row
    if batch is None:
        batch = ReplayStore.open(replay_path, identity, held_out_seeds=held_out).read_all(max_rows=args.replay_capacity)
    replay_digest = hashlib.sha256((replay_path / "manifest.jsonl").read_bytes()).hexdigest()
    data = tensors(batch)
    del batch
    collection_seconds = time.perf_counter() - started
    update_started = time.perf_counter()
    emit("training", device=args.device, transitions=len(data[0]), updates=args.updates,
         starting_updates=trainer.updates)
    while trainer.updates < args.updates:
        loss = trainer.step(data)
        if trainer.updates % 500 == 0 or trainer.updates == args.updates:
            trainer.save(args.output / "critic.pt", identity, replay_digest)
            emit("trained", update=trainer.updates, loss=loss)
    update_seconds = time.perf_counter() - update_started
    evaluation_started = time.perf_counter()
    model = trainer.model
    model.cpu().eval()
    loaded = FactorCritic(adapter.context_size, adapter.action_limits)
    loaded.load_state_dict(torch.load(args.output / "critic.pt", map_location="cpu", weights_only=True)["model"])
    with torch.no_grad():
        assert torch.equal(model(data[0][:32], data[1][:32]), loaded(data[0][:32], data[1][:32]))
    baseline_records, guided_records = [], []
    latencies = []
    for first in range(0, len(held_out), 8):
        seeds = held_out[first:first + 8]
        baseline, _ = adapter.episodes(actor, seeds, evaluation=True)
        guided, latency = adapter.episodes(actor, seeds, 16, model, evaluation=True)
        baseline_records.extend(baseline)
        guided_records.extend(guided)
        latencies.append(latency)
        emit("compared", seeds=len(baseline_records), baseline_combined_return=[sum(float(r.reward_diagnostics[0]) for r in e) for e in baseline],
             guided_combined_return=[sum(float(r.reward_diagnostics[0]) for r in e) for e in guided],
             planner_p95_ms=latency)
    baseline_returns = [sum(float(row.reward_diagnostics[0]) for row in episode)
                        for episode in baseline_records]
    guided_returns = [sum(float(row.reward_diagnostics[0]) for row in episode)
                      for episode in guided_records]
    facts = evaluate_rows(model, baseline_records, trainer.reward_discount)
    result = {"learner": "q", "training_decisions": collected,
              "retained_transitions": len(data[0]), "updates": trainer.updates,
              "collection_seconds": collection_seconds, "update_seconds": update_seconds,
              "evaluation_seconds": time.perf_counter() - evaluation_started,
              "evaluation": {
                  **adapter.evaluation_metadata,
                  "seeds": held_out, "baseline": baseline_returns, "guided": guided_returns,
                  "mean_paired_gain": float(np.mean(np.subtract(guided_returns, baseline_returns))),
                  "decisions": sum(map(len, baseline_records)) + sum(map(len, guided_records))},
              "diagnostics": facts, "baseline": baseline_returns,
              "guided": guided_returns,
              "planner_max_p95_ms": max(latencies), "elapsed_seconds": time.perf_counter() - started,
              "accepted": False, "acceptance_note": adapter.acceptance_note}
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    emit("complete", **result)
    return result

