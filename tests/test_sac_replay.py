"""SAC replay preserves transitions and corrects nonuniform sampling."""

import pytest
import torch

from pufferlib.sac import Replay


@pytest.mark.parametrize("count", [1, 2, 512])
@pytest.mark.parametrize("beta", [0.0, 0.4, 1.0])
def test_importance_weights_use_the_entire_replay(count: int, beta: float) -> None:
    """A transition's correction does not depend on its sampled companions."""
    replay = Replay(2, 1, 1, "cpu")
    observations = torch.arange(2).float()[:, None]
    replay.add(observations, observations.long(), torch.zeros(2), observations, torch.zeros(2))
    replay.priority[:] = torch.tensor([1.0, 9.0]).pow(1 / 0.6)

    torch.manual_seed(19)
    indices, weights, _ = replay.sample(count, beta)
    expected = torch.tensor([1.0, 9.0 ** -beta])[indices]
    torch.testing.assert_close(weights, expected)


def test_full_correction_preserves_a_zero_expected_gradient() -> None:
    """Unequal sampling must not move a critic off a balanced replay optimum."""
    replay = Replay(2, 1, 1, "cpu")
    observations = torch.arange(2).float()[:, None]
    replay.add(observations, observations.long(), torch.zeros(2), observations, torch.zeros(2))
    replay.priority[:] = torch.tensor([1.0, 9.0]).pow(1 / 0.6)

    # Recover each transition's singleton correction through the production sampler.
    torch.manual_seed(19)
    correction = torch.zeros(2)
    seen = torch.zeros(2, dtype=torch.bool)
    for _ in range(100):
        indices, weights, _ = replay.sample(1, 1.0)
        correction[indices] = weights
        seen[indices] = True
    assert seen.all()

    # The two replay transitions have opposite gradients at the uniform optimum.
    expected_gradient = (torch.tensor([0.1, 0.9]) * correction * torch.tensor([-1.0, 1.0])).sum()
    torch.testing.assert_close(expected_gradient, torch.tensor(0.0), atol=1e-7, rtol=0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_oversized_collection_retains_latest_transitions(device: str) -> None:
    """A collection larger than the ring retains its newest capacity rows."""
    replay = Replay(4, 1, 1, device)
    observations = torch.arange(3, device=device).float()[:, None]
    replay.add(observations, observations.long(), observations[:, 0], observations + 1, torch.zeros(3, device=device))
    end = 100006
    observations = torch.arange(3, end, device=device).float()[:, None]
    replay.add(observations, observations.long(), observations[:, 0], observations + 1, torch.zeros(end - 3, device=device))
    assert replay.size == 4
    assert replay.position == 2
    indices = (torch.arange(4, device=device) + replay.position) % replay.capacity
    expected = torch.arange(end - 4, end, device=device)
    torch.testing.assert_close(replay.obs[indices, 0], expected.float())
    torch.testing.assert_close(replay.actions[indices, 0], expected)
    torch.testing.assert_close(replay.reward[indices], expected.float())
    torch.testing.assert_close(replay.next_obs[indices, 0], expected.float() + 1)
