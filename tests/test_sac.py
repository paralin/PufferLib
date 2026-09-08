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


def test_schedule_entropy_counts_controls_and_merges_aliases():
    from pufferlib.sac import schedule_statistics, canonical_schedule
    import itertools
    logits = [torch.tensor([[.3, -.4]], requires_grad=True),
              torch.tensor([[-.2, .7]], requires_grad=True),
              torch.tensor([[.1, -.5, .4]], requires_grad=True)]
    actions = torch.tensor(list(itertools.product(range(2), range(2), range(3))))[:, None]
    canonical = canonical_schedule(actions)
    masses = {}
    for action, key in zip(actions[:, 0], canonical[:, 0]):
        probability = torch.stack([logits[i].softmax(-1)[0, action[i]] for i in range(3)]).prod()
        key = tuple(key.tolist())
        masses[key] = masses.get(key, 0) + probability
    logp, entropy = schedule_statistics(logits, actions)
    expected = torch.stack([masses[tuple(key.tolist())].log() for key in canonical[:, 0]])
    torch.testing.assert_close(logp[:, 0], expected)
    expected_entropy = -sum(p * p.log() for p in masses.values())
    torch.testing.assert_close(entropy[0], expected_entropy)
    actual_grads = torch.autograd.grad(entropy.sum(), logits, retain_graph=True)
    expected_grads = torch.autograd.grad(expected_entropy, logits)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)
    deterministic = [torch.tensor([[0., -1000.]]), torch.tensor([[0., -1000.]]), torch.zeros(1, 10)]
    _, entropy = schedule_statistics(deterministic, torch.zeros(1, 1, 3, dtype=torch.long))
    assert abs(entropy.item()) < 1e-6


def test_schedule_critic_aliases_and_learning():
    from pufferlib.sac import Learner
    learner = Learner(4, (2, 2, 3), device='cpu', action_schedule=True)
    obs = torch.randn(8, 4)
    a = torch.tensor([[0, 0, 0]]).expand(8, -1)
    b = torch.tensor([[1, 0, 0]]).expand(8, -1)
    c = torch.tensor([[0, 0, 2]]).expand(8, -1)
    torch.testing.assert_close(learner.critics[0](obs, a), learner.critics[0](obs, b))
    torch.testing.assert_close(learner.critics[0](obs, a), learner.critics[0](obs, c))
    _, metrics = learner.update((obs, a, torch.ones(8), obs, torch.zeros(8)), torch.ones(8))
    assert torch.isfinite(metrics).all()
