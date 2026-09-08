"""Factorized discrete SAC with distributional critics and predictive features.

Adapts SAC-BBF's sampled policy gradient and self-prediction to structured
observations and MultiDiscrete actions. This is not the Atari SAC-BBF recipe:
there are no image augmentations, multi-step SPR rollouts or periodic resets.
Torch's CUDA API dispatches to HIP on ROCm; replay and learner stay on device.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 256), nn.LayerNorm(256), nn.SiLU(),
                         nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, outputs))


class Actor(nn.Module):
    """Independent categorical factors; samples are legal whole actions."""
    def __init__(self, observations, limits):
        super().__init__()
        self.limits = tuple(limits)
        self.network = mlp(observations, sum(limits))

    def forward(self, observations):
        return self.network(observations).split(self.limits, dim=-1)

    def sample(self, observations, count=1):
        logits = self(observations)
        distributions = [torch.distributions.Categorical(logits=x) for x in logits]
        actions = [d.sample((count,)) for d in distributions]
        logp = sum(d.log_prob(a) for d, a in zip(distributions, actions))
        entropy = sum(d.entropy() for d in distributions)
        return torch.stack(actions, -1), logp, entropy


class Critic(nn.Module):
    def __init__(self, observations, limits, bins=101):
        super().__init__()
        self.limits = tuple(limits)
        self.encoder = mlp(observations, 128)
        self.head = mlp(128 + sum(limits), bins)
        self.predictor = mlp(128 + sum(limits), 128)
        # Symmetric support is in scaled reward units.
        self.register_buffer('support', torch.linspace(-100, 100, bins))

    def features(self, observations, actions):
        encoded = self.encoder(observations)
        onehot = torch.cat([F.one_hot(actions[..., i], n) for i, n in enumerate(self.limits)], -1)
        return torch.cat((encoded, onehot.to(encoded.dtype)), -1)

    def forward(self, observations, actions):
        return self.head(self.features(observations, actions))

    def value(self, observations, actions):
        return (self(observations, actions).softmax(-1) * self.support).sum(-1)


def project_distribution(probabilities, values, support):
    """C51 projection preserves mass, including exact-bin and clipped targets."""
    position = (values.clamp(support[0], support[-1]) - support[0]) / (support[1] - support[0])
    low = position.floor().long().clamp(0, len(support) - 1)
    high = (low + 1).clamp(max=len(support) - 1)
    fraction = position - low
    result = torch.zeros_like(probabilities)
    result.scatter_add_(-1, low, probabilities * (1 - fraction))
    result.scatter_add_(-1, high, probabilities * fraction)
    return result


class Replay:
    """Bounded GPU ring with proportional priorities and importance weights."""
    def __init__(self, capacity, observations, factors, device):
        self.capacity, self.size, self.position = capacity, 0, 0
        self.obs = torch.empty(capacity, observations, device=device)
        self.next_obs = torch.empty_like(self.obs)
        self.actions = torch.empty(capacity, factors, dtype=torch.long, device=device)
        self.reward = torch.empty(capacity, device=device)
        self.terminal = torch.empty(capacity, device=device)
        self.priority = torch.ones(capacity, device=device)

    def add(self, obs, actions, reward, next_obs, terminal):
        """Retain the newest capacity transitions, including oversized collections."""
        n = len(obs)
        start = max(0, n - self.capacity)
        indices = (torch.arange(start, n, device=self.obs.device) + self.position) % self.capacity
        self.obs[indices], self.actions[indices] = obs[start:], actions[start:]
        self.reward[indices] = reward[start:]
        self.next_obs[indices], self.terminal[indices] = next_obs[start:], terminal[start:]
        self.priority[indices] = self.priority[:max(self.size, 1)].max()
        self.position = (self.position + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, count, beta):
        """Normalize importance weights against the least likely replay transition."""
        mass = self.priority[:self.size].pow(.6)
        indices = torch.multinomial(mass, count, replacement=True)
        # The replay-wide maximum IS weight cancels the common N and total mass.
        weights = (mass.min() / mass[indices]).pow(beta)
        return indices, weights, (self.obs[indices], self.actions[indices], self.reward[indices],
                                 self.next_obs[indices], self.terminal[indices])

    def update(self, indices, errors):
        # Duplicate draws must be order-independent.
        self.priority[indices.unique()] = 1e-4
        self.priority.scatter_reduce_(0, indices, errors.detach().abs() + 1e-4, reduce='amax', include_self=True)

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity:
            raise ValueError('replay capacity changed')
        self.size, self.position = state['size'], state['position']
        for name in ('obs', 'next_obs', 'actions', 'reward', 'terminal', 'priority'):
            getattr(self, name)[:self.size].copy_(state[name])

    def state_dict(self):
        return {name: value[:self.size].cpu() if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()}


class Learner:
    """SAC owns actor, twin critics, entropy temperature and target updates."""
    def __init__(self, observations, limits, device='cuda', learning_rate=3e-4,
                 gamma=.99, tau=.005, spr_weight=.1, entropy_fraction=.5):
        self.actor = Actor(observations, limits).to(device)
        self.critics = nn.ModuleList([Critic(observations, limits) for _ in range(2)]).to(device)
        self.targets = deepcopy(self.critics).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(-3., device=device))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=learning_rate)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.gamma, self.tau, self.spr_weight = gamma, tau, spr_weight
        self.target_entropy = entropy_fraction * sum(math.log(n) for n in limits)
        self.updates = 0

    def update(self, batch, weights):
        obs, actions, rewards, next_obs, terminal = batch
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_actions, next_logp, _ = self.actor.sample(next_obs)
            distributions = torch.stack([q(next_obs, next_actions[0]).softmax(-1) for q in self.targets])
            support = self.targets[0].support
            values = (distributions * support).sum(-1)
            which = values.argmin(0)
            probabilities = distributions[which, torch.arange(len(obs), device=obs.device)]
            bellman = rewards[:, None] + self.gamma * (1 - terminal[:, None]) * (support - alpha * next_logp[0, :, None])
            target = project_distribution(probabilities, bellman, support)
            clipped = ((bellman < support[0]) | (bellman > support[-1])).float().mean()
        critic_loss, errors = 0., 0.
        for critic, slow in zip(self.critics, self.targets):
            features = critic.features(obs, actions)
            logits = critic.head(features)
            ce = -(target * logits.log_softmax(-1)).sum(-1)
            prediction = critic.predictor(features)
            with torch.no_grad():
                latent_target = slow.encoder(next_obs)
            spr = (F.normalize(prediction, dim=-1) - F.normalize(latent_target, dim=-1)).square().sum(-1)
            critic_loss = critic_loss + (weights * (ce + self.spr_weight * spr * (1 - terminal))).mean()
            errors = errors + ((logits.softmax(-1) - target) * support).sum(-1).abs()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), 10., error_if_nonfinite=True)
        self.critic_opt.step()

        # Leave-one-out baseline gives an unbiased sampled discrete policy gradient.
        sampled, logp, entropy = self.actor.sample(obs, count=4)
        with torch.no_grad():
            repeated = obs.unsqueeze(0).expand(4, -1, -1).reshape(-1, obs.shape[-1])
            sampled_q = torch.stack([q.value(repeated, sampled.reshape(-1, sampled.shape[-1]))
                                     for q in self.critics]).amin(0).reshape(4, -1)
            baseline = (sampled_q.sum(0, keepdim=True) - sampled_q) / 3
        actor_loss = -(logp * (sampled_q - baseline)).mean() - alpha * entropy.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10., error_if_nonfinite=True)
        self.actor_opt.step()
        alpha_loss = self.log_alpha * (entropy.detach().mean() - self.target_entropy)
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10, 2)
            for target_parameter, parameter in zip(self.targets.parameters(), self.critics.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        self.updates += 1
        metrics = torch.stack((critic_loss.detach(), actor_loss.detach(), entropy.detach().mean(), alpha, clipped))
        if not torch.isfinite(metrics).all():
            raise FloatingPointError('nonfinite SAC update')
        return errors / 2, metrics

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critics=self.critics.state_dict(),
                    targets=self.targets.state_dict(), log_alpha=self.log_alpha.detach(),
                    actor_opt=self.actor_opt.state_dict(), critic_opt=self.critic_opt.state_dict(),
                    alpha_opt=self.alpha_opt.state_dict(), updates=self.updates)

    def load_state_dict(self, state):
        for name in ('actor', 'critics', 'targets', 'actor_opt', 'critic_opt', 'alpha_opt'):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        self.updates = state['updates']
