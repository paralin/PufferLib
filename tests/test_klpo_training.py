"""Exercise the built LLB native trainer (set PUFFER_KLPO_BINARY)."""
import configparser
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope="module")
def trainer():
    binary = os.environ.get("PUFFER_KLPO_BINARY")
    if not binary:
        pytest.skip("set PUFFER_KLPO_BINARY to the built LLB trainer")
    return Path(binary).resolve()


def run(trainer, root, name, *, horizon=4096, steps=65536, **overrides):
    options = {
        "train.learner": "klpo", "train.klpo_returns": "discounted",
        "train.horizon": horizon, "train.minibatch_size": horizon,
        "train.replay_ratio": 1, "train.total_timesteps": steps,
        "train.learning_rate": 0.00015, "vec.total_agents": 8,
        "vec.num_buffers": 1, "vec.num_threads": 4, "base.async": 0,
        "base.cudagraphs": -1, "base.run_id": name,
        "base.checkpoint_dir": str(root), "base.log_dir": str(root / "logs"),
        "base.eval_episodes": 0, "base.save_training_state": 1,
        "selfplay.enabled": 0,
    }
    options.update(overrides)
    with (root / f"{name}.log").open("w") as log:
        subprocess.run([str(trainer), "train", *(f"--{k}={v}" for k, v in options.items()
                                                if v is not None)],
                       cwd=trainer.parents[2], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=90)
    output = root / "llb-rust" / name
    progress = configparser.ConfigParser()
    progress.read(output / "progress.ini")
    return output, dict(progress["metrics"])


def weights(path):
    result = np.fromfile(path, np.float32)
    assert len(result) == 563456
    assert np.isfinite(result).all()
    return result


def test_training_and_resume(trainer, tmp_path):
    output, metrics = run(trainer, tmp_path, "eager")
    initial = weights(output / "0000000000000000.bin")
    final = weights(output / "actor.bin")
    # Encoder and each recurrent layer must learn; the spare value row stays inert.
    for start, end in ((0, 256 * 88), (170240, 366848), (366848, 563456)):
        assert np.max(np.abs(final[start:end] - initial[start:end])) > 1e-7
    value = slice(256 * 88 + 576 * 256, 256 * 88 + 577 * 256)
    np.testing.assert_array_equal(final[value], initial[value])
    assert float(metrics["klpo/complete_episodes"]) > 0
    assert 0 < float(metrics["klpo/used_fraction"]) < 1
    assert float(metrics["loss/value"]) == 0
    state = output / "0000000000065536.state"
    np.testing.assert_array_equal(final, weights(state / "weights.f32"))
    resumed, after = run(trainer, tmp_path, "resume", steps=98304,
                         **{"base.resume_training_state": str(state)})
    assert float(after["agent_steps"]) == 98304
    assert float(after["klpo/used_decisions"]) > float(metrics["klpo/used_decisions"])
    assert not np.array_equal(final, weights(resumed / "actor.bin"))
    graph, _ = run(trainer, tmp_path, "graph", **{"base.cudagraphs": 1})
    np.testing.assert_allclose(weights(graph / "actor.bin"), final, atol=1e-6, rtol=1e-5)


def test_no_complete_episode_does_not_step(trainer, tmp_path):
    output, metrics = run(trainer, tmp_path, "empty", horizon=8, steps=128)
    assert float(metrics["klpo/used_decisions"]) == 0
    np.testing.assert_array_equal(weights(output / "0000000000000000.bin"),
                                  weights(output / "actor.bin"))
    momentum = np.fromfile(output / "0000000000000128.state/momentum.f32", np.float32)
    assert not np.any(momentum)


@pytest.mark.parametrize("mode", ["whole_match", "ppo", "async"])
def test_other_paths(trainer, tmp_path, mode):
    options = {"train.klpo_returns": "whole_match"} if mode == "whole_match" else (
        {"train.learner": "ppo"} if mode == "ppo" else {"base.async": 1, "base.cudagraphs": 1})
    output, metrics = run(trainer, tmp_path, mode, **options)
    assert np.isfinite(float(metrics["loss/policy"]))
    assert not np.array_equal(weights(output / "actor.bin"),
                              weights(output / "0000000000000000.bin"))
    if mode == "ppo":
        state = output / "0000000000065536.state"
        for name in ("config.ini", "state.ini"):
            path = state / name
            lines = path.read_text().splitlines(keepends=True)
            path.write_text("".join(line for line in lines
                                    if not line.startswith(("learner =", "klpo"))))
        _, resumed = run(trainer, tmp_path, "old-ppo", steps=98304, **{
            "base.resume_training_state": str(state), "train.learner": None,
            "train.klpo_returns": None,
        })
        assert float(resumed["agent_steps"]) == 98304
