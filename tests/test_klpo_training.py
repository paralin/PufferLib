"""Exercise the built LLB native trainer (set PUFFER_KLPO_BINARY)."""
import configparser
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pytest


@pytest.fixture(scope="module")
def trainer():
    binary = os.environ.get("PUFFER_KLPO_BINARY")
    if not binary:
        pytest.skip("set PUFFER_KLPO_BINARY to the built LLB trainer")
    return Path(binary).resolve()


def command(trainer, root, name, *, horizon=4096, steps=32768, **overrides):
    options = {
        "train.learner": "klpo", "train.klpo_returns": "discounted",
        "train.klpo_collect_steps": 256,
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
    return [str(trainer), "train", *(f"--{k}={v}" for k, v in options.items()
                                    if v is not None)]


def read_ini(path, section):
    ini = configparser.ConfigParser()
    ini.read(path)
    return dict(ini[section])


def run(trainer, root, name, **options):
    with (root / f"{name}.log").open("w") as log:
        subprocess.run(command(trainer, root, name, **options),
                       cwd=trainer.parents[2], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=90)
    output = root / "llb-rust" / name
    return output, read_ini(output / "progress.ini", "metrics")


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
    assert float(metrics["klpo/used_fraction"]) == 1
    assert float(metrics["fresh_decisions"]) == float(metrics["agent_steps"])
    assert float(metrics["klpo/pending_decisions"]) == 0
    assert float(metrics["agent_steps"]) >= 32768
    assert float(metrics["loss/value"]) == 0
    state = max(output.glob("*.state"))
    np.testing.assert_array_equal(final, weights(state / "weights.f32"))
    resumed, after = run(trainer, tmp_path, "resume", steps=98304,
                         **{"base.resume_training_state": str(state)})
    assert float(after["agent_steps"]) >= 98304
    assert float(after["klpo/used_decisions"]) > float(metrics["klpo/used_decisions"])
    assert not np.array_equal(final, weights(resumed / "actor.bin"))
    graph, _ = run(trainer, tmp_path, "graph", **{"base.cudagraphs": 1})
    np.testing.assert_allclose(weights(graph / "actor.bin"), final, atol=1e-6, rtol=1e-5)


def test_capacity_cannot_silently_drop_a_match(trainer, tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run(trainer, tmp_path, "capacity", horizon=8, steps=128,
            **{"train.klpo_collect_steps": 4})
    assert "exceeded train.horizon capacity" in (tmp_path / "capacity.log").read_text()


@pytest.mark.parametrize("mode", ["whole_match", "ppo", "async"])
def test_other_paths(trainer, tmp_path, mode):
    options = {"train.klpo_returns": "whole_match"} if mode == "whole_match" else (
        {"train.learner": "ppo"} if mode == "ppo" else {"base.async": 1, "base.cudagraphs": 1})
    output, metrics = run(trainer, tmp_path, mode, **options)
    assert np.isfinite(float(metrics["loss/policy"]))
    assert not np.array_equal(weights(output / "actor.bin"),
                              weights(output / "0000000000000000.bin"))
    if mode != "ppo":
        assert float(metrics["klpo/used_fraction"]) == 1
        assert float(metrics["klpo/pending_decisions"]) == 0
        assert float(metrics["fresh_decisions"]) == float(metrics["agent_steps"])
    if mode == "ppo":
        state = max(output.glob("*.state"))
        for name in ("config.ini", "state.ini"):
            path = state / name
            lines = path.read_text().splitlines(keepends=True)
            path.write_text("".join(line for line in lines
                                    if not line.startswith(("learner =", "klpo"))))
        _, resumed = run(trainer, tmp_path, "old-ppo", steps=98304, **{
            "base.resume_training_state": str(state), "train.learner": None,
            "train.klpo_returns": None, "train.klpo_collect_steps": None,
        })
        assert float(resumed["agent_steps"]) == 98304


def test_checkpoint_and_stop_drain_prefetch(trainer, tmp_path):
    output = tmp_path / "llb-rust" / "drain"
    cmd = command(trainer, tmp_path, "drain", steps=1000000, **{
        "base.async": 1, "base.cudagraphs": 1, "base.checkpoint_interval": 2,
        "base.checkpoint_seconds": 0, "vec.num_buffers": 2,
    })
    with (tmp_path / "drain.log").open("w") as log:
        process = subprocess.Popen(cmd, cwd=trainer.parents[2], stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                assert process.poll() is None, "trainer exited before pending collection was observed"
                progress = output / "progress.ini"
                if progress.exists():
                    metrics = read_ini(progress, "metrics")
                    if (float(metrics["epoch"]) >= 4
                            and float(metrics["klpo/pending_decisions"]) > 0):
                        break
                time.sleep(0.05)
            else:
                pytest.fail("never observed a queued batch after a periodic checkpoint")
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=25) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    final = read_ini(output / "progress.ini", "metrics")
    assert float(final["stopped"]) == 1
    assert float(final["klpo/pending_decisions"]) == 0
    assert float(final["fresh_decisions"]) == float(final["agent_steps"])
    assert float(final["klpo/used_fraction"]) == 1
    assert float(final["agent_steps"]) >= float(metrics["fresh_decisions"])
    states = sorted(output.glob("*.state"))
    assert len(states) >= 3  # Initial, periodic, and drained final state.
    for state in states:
        counters = read_ini(state / "state.ini", "state")
        assert counters["collected_steps"] == counters["global_step"]
        assert counters["klpo_decisions"] == counters["global_step"]
    np.testing.assert_array_equal(weights(output / "actor.bin"),
                                  weights(states[-1] / "weights.f32"))
