"""Observable SAC distribution, replay and learning contracts."""
import torch
from pufferlib.sac import Learner, Replay, project_distribution


def test_projection_preserves_exact_and_clipped_mass():
    support = torch.linspace(-1, 1, 3)
    probability = torch.tensor([[.2, .3, .5]])
    assert torch.allclose(project_distribution(probability, support[None], support), probability)
    clipped = project_distribution(probability, torch.tensor([[-9., 0., 9.]]), support)
    assert torch.allclose(clipped, probability)
    midpoint = project_distribution(probability, torch.full((1, 3), .5), support)
    assert torch.allclose(midpoint, torch.tensor([[0., .5, .5]]))


def test_ring_retains_latest_and_duplicate_priorities_use_max():
    replay = Replay(4, 2, 1, 'cpu')
    for start in (0, 3):
        obs = torch.arange(start, start + 3).float()[:, None].repeat(1, 2)
        replay.add(obs, torch.zeros(3, 1, dtype=torch.long), torch.zeros(3), obs, torch.zeros(3))
    assert replay.size == 4
    assert sorted(replay.obs[:, 0].tolist()) == [2, 3, 4, 5]
    replay.update(torch.tensor([1, 1]), torch.tensor([2., 8.]))
    assert torch.isclose(replay.priority[1], torch.tensor(8.0001))
    indices, weights, _ = replay.sample(8, .4)
    assert indices.max() < 4 and torch.isfinite(weights).all()


def test_terminal_reward_changes_critics_and_actor_stays_legal():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    learner = Learner(4, [2, 3], 'cpu')
    observations = torch.randn(16, 4)
    actions = torch.stack((torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)), -1)
    before = learner.critics[0].value(observations, actions).detach().mean()
    for _ in range(12):
        errors, metrics = learner.update((observations, actions, torch.ones(16) * 8,
                                          observations, torch.ones(16)), torch.ones(16))
        assert torch.isfinite(errors).all() and torch.isfinite(metrics).all()
    after = learner.critics[0].value(observations, actions).detach().mean()
    assert after > before
    samples, _, _ = learner.actor.sample(observations, 4)
    assert samples.shape == (4, 16, 2)
    assert (samples[..., 0] < 2).all() and (samples[..., 1] < 3).all()
