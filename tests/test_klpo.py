"""Native KLPO gradients against independent float64 schedule enumeration.

KLPO_TEST_HIP=1 compiles/runs the same production arithmetic on the GPU.
"""
import ctypes
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    library = tmp_path_factory.mktemp("klpo") / "probe.so"
    hip = os.environ.get("KLPO_TEST_HIP") == "1"
    compiler = ["hipcc", "-DKLPO_HIP"] if hip else ["c++"]
    subprocess.run(
        [*compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I", str(root / "src"),
         str(root / "tests/klpo_probe.cpp"), "-o", str(library)],
        check=True, timeout=90,
    )
    result = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    result.score.argtypes = [pointer] * 5
    if not hip:
        result.targets.argtypes = [pointer] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_bool, np.ctypeslib.ndpointer(dtype=np.int32)]
    return result


def schedules(logits, size):
    initial = logits[:size].log_softmax(0)
    final = logits[size:size + size * size].reshape(size, size).log_softmax(1)
    time = logits[size + size * size:].reshape(size, size, 10).log_softmax(2)
    atoms = (initial[:, None, None] + final[:, :, None] + time).exp()
    classes = torch.empty((size, size, 10), dtype=torch.long)
    next_class = size
    for i in range(size):
        for f in range(size):
            for k in range(10):
                if i == f or k == 0:
                    classes[i, f, k] = f
                else:
                    classes[i, f, k] = next_class
                    next_class += 1
    mass = torch.zeros(next_class, dtype=torch.float64).scatter_add(
        0, classes.flatten(), atoms.flatten())
    return mass, classes


@pytest.mark.parametrize("scale", [0.0, 1.0, 8.0])
@pytest.mark.parametrize("constant", [False, True])
def test_native_score_and_gradient(native, scale, constant):
    rng = np.random.default_rng(121)
    p = (rng.normal(size=576) * scale).astype(np.float32)
    q = (rng.normal(size=576) * scale).astype(np.float32)
    actions = np.array([0, 2, 0 if constant else 7, 1, 1, 3,
                        4, 1, 8, 0, 1, 2, 1, 0, 6], dtype=np.float32)
    gradient, stats = np.zeros(576, np.float32), np.zeros(4, np.float32)
    assert native.score(p, q, actions, gradient, stats) == 0
    learner = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    sampler = torch.tensor(q, dtype=torch.float64)
    center = ratio = entropy = kl = 0
    offset = 0
    for group, size in enumerate((3, 3, 5, 2, 2)):
        width = size + 11 * size * size
        pm, classes = schedules(learner[offset:offset + width], size)
        qm, _ = schedules(sampler[offset:offset + width], size)
        selected = classes[tuple(int(x) for x in actions[group * 3:group * 3 + 3])]
        center = center + pm[selected].log() - (qm * pm.log()).sum()
        ratio = ratio + pm[selected].log() - qm[selected].log()
        entropy = entropy - (pm * pm.log()).sum()
        kl = kl + (qm * (qm.log() - pm.log())).sum()
        offset += width
    expected = torch.stack((center, ratio, entropy, kl)).detach().numpy()
    np.testing.assert_allclose(stats, expected, atol=4e-5, rtol=2e-5)
    # A nontrivial detached feedback verifies the loss, not just its score.
    feedback = (2.5 - 0.3 * ratio).detach()
    (-feedback * center).backward()
    np.testing.assert_allclose(-float(feedback) * gradient,
                               learner.grad.numpy(), atol=6e-5, rtol=3e-5)


def test_complete_episode_targets(native):
    if os.environ.get("KLPO_TEST_HIP") == "1":
        pytest.skip("boundary scan covered by the host probe and native train smoke")
    # Two complete matches cross a notional target at action 2. Outcome index
    # six belongs to the final executed action, not to a seventh action.
    rewards = np.array([99, 1, 2, 3, 4, 5, 6, 88, 77], np.float32)
    dones = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1], np.float32)
    output = np.empty(9, np.float32)
    counts = np.zeros(2, np.int32)
    native.targets(rewards, dones, output, 9, 6, 0.5, False, counts)
    np.testing.assert_allclose(output, [3.25, 4.5, 5, 4, 8, 6,
                                       np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(counts, [2, 6])
    native.targets(rewards, dones, output, 9, 6, 0.5, True, counts)
    np.testing.assert_allclose(output, [10, 10, 10, 10, 11, 11, np.nan, np.nan, np.nan])
    # Padding can look terminal and contain arbitrary rewards without creating
    # another episode or changing the last match's target.
    np.testing.assert_array_equal(counts, [2, 6])


def test_prioritized_episode_average():
    # Rows contain unequal episode counts; average over all episodes, never
    # per sampled row/length. Full inverse sampling weights recover this mean.
    row_gradients = np.array([3.0, 0, -2, 8])
    episodes = np.array([1, 0, 2, 3])
    probabilities = np.array([0.6, 0, 0.1, 0.3])
    used = probabilities > 0
    weights = 1 / (len(episodes) * probabilities[used])
    expectation = np.sum(probabilities[used] * weights * len(episodes)
                         / episodes.sum() * row_gradients[used])
    assert expectation == pytest.approx(row_gradients.sum() / episodes.sum())
