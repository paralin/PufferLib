"""Own critic updates and the state needed to continue them exactly."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from qplan import CandidateBatch, ContextBatch
from qplan.critic import FactorCritic

ALPHA = .6
BETA_START = .4
BETA_END = 1.
BETA_ANNEAL_UPDATES = 1_000_000
EPSILON = 1e-6


def initialization_identity_matches(actual: dict, expected: dict, allow_environment_change: bool = False, reset_value_head: bool = False, transfer_value_head: bool = False) -> bool:
    """Explicit transfer can replace the environment or value objective, never the actor."""
    if allow_environment_change:
        actual = {key: value for key, value in actual.items() if key != "opponent"}
        expected = {key: value for key, value in expected.items() if key != "opponent"}
    if reset_value_head or transfer_value_head:
        actual = {key: value for key, value in actual.items() if key != "reward_target"}
        expected = {key: value for key, value in expected.items() if key != "reward_target"}
    return actual == expected


class PrioritizedSampler:
    """Sample proportional TD priorities in O(batch log capacity) on the CPU."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.generator = torch.Generator().manual_seed(seed)
        self._base = 1 << (capacity - 1).bit_length()
        self._tree = np.zeros(2 * self._base, dtype=np.float64)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def append(self, count: int) -> None:
        """Initialize one immutable snapshot with equal priorities."""
        if self._size or count < 1 or count > self.capacity:
            raise ValueError("replay snapshot must be nonempty, bounded and initialized once")
        self._size = count
        self._tree[self._base:self._base + count] = 1.
        for level in range(self._base - 1, 0, -1):
            self._tree[level] = self._tree[2 * level] + self._tree[2 * level + 1]

    def update(self, indices: np.ndarray, residuals: np.ndarray) -> None:
        """Duplicate draws use the greatest absolute TD error, independent of order."""
        if not np.isfinite(residuals).all() or np.any(indices < 0) or np.any(indices >= self._size):
            raise ValueError("invalid replay priority update")
        unique, inverse = np.unique(indices, return_inverse=True)
        errors = np.zeros(len(unique), dtype=np.float64)
        np.maximum.at(errors, inverse, np.abs(residuals))
        nodes = unique + self._base
        self._tree[nodes] = (errors + EPSILON) ** ALPHA
        while nodes.size and nodes[0] > 1:
            nodes = np.unique(nodes // 2)
            self._tree[nodes] = self._tree[2 * nodes] + self._tree[2 * nodes + 1]

    def sample(self, count: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
        """Sample with replacement and normalize importance weights within the batch."""
        if not self._size or count < 1:
            raise ValueError("cannot sample an empty replay or batch")
        total = self._tree[1]
        targets = torch.rand(count, dtype=torch.float64, generator=self.generator).numpy() * total
        nodes = np.ones(count, dtype=np.int64)
        while nodes[0] < self._base:
            left = 2 * nodes
            mass = self._tree[left]
            right = targets >= mass
            targets -= np.where(right, mass, 0.)
            nodes = left + right
        probabilities = self._tree[nodes] / total
        weights = (self._size * probabilities) ** (-beta)
        return nodes - self._base, weights / weights.max()

    def state(self) -> dict:
        # Preserve the tree exactly; rebuilding floating-point sums can alter draws.
        return {"tree": torch.from_numpy(self._tree.copy()), "size": self._size,
                "capacity": self.capacity, "alpha": ALPHA, "epsilon": EPSILON,
                "beta_start": BETA_START, "beta_end": BETA_END,
                "beta_updates": BETA_ANNEAL_UPDATES, "rng": self.generator.get_state()}

    def load(self, state: dict) -> None:
        expected = {"capacity": self.capacity, "alpha": ALPHA, "epsilon": EPSILON,
                    "beta_start": BETA_START, "beta_end": BETA_END,
                    "beta_updates": BETA_ANNEAL_UPDATES}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint replay capacity or priority configuration differs")
        size = state["size"]
        tree = state["tree"].numpy()
        if not 0 < size <= self.capacity or tree.shape != self._tree.shape or not np.isfinite(tree).all():
            raise ValueError("invalid checkpoint priority tree")
        if np.any(tree[self._base:self._base + size] <= 0) or np.any(tree[self._base + size:] != 0):
            raise ValueError("invalid checkpoint priority leaves")
        if not np.array_equal(tree[1:self._base], tree[2::2] + tree[3::2]):
            raise ValueError("invalid checkpoint priority sums")
        self._size = size
        self._tree[:] = tree
        self.generator.set_state(state["rng"])


class CriticTrainer:
    def __init__(self, device: str, seed: int, *, context_size: int,
                 limits: tuple[int, ...], replay_capacity: int = 131_072, reward_discount: float | None = None, gaussian_sigma: float | None = None, learning_rate: float = .0003,
                 target_ema: float = .01, canonical_schedules: bool = False):
        if reward_discount is not None and not 0 <= reward_discount <= 1:
            raise ValueError("reward discount must be between zero and one")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not np.isfinite(target_ema) or not 0 <= target_ema <= 1:
            raise ValueError("target EMA must be finite and between zero and one")
        self.target_ema = float(target_ema)
        self.device = device
        self.reward_discount = reward_discount
        self.model = FactorCritic(context_size, limits, reward_mode=reward_discount is not None, gaussian_sigma=gaussian_sigma, canonical_schedules=canonical_schedules).to(device)
        self.target = deepcopy(self.model).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))
        self.updates = 0
        self.replay_capacity = replay_capacity
        self.sampler = PrioritizedSampler(replay_capacity, seed)
        self.generator = self.sampler.generator
        self.last_sampled_indices = np.zeros(0, dtype=np.int64)

    @property
    def runtime(self) -> dict[str, str]:
        return {"device": self.device, "torch": str(torch.__version__),
                "accelerator": torch.cuda.get_device_name() if self.device == "cuda" else "cpu"}

    @property
    def beta(self) -> float:
        fraction = min(self.updates / BETA_ANNEAL_UPDATES, 1.)
        return BETA_START + (BETA_END - BETA_START) * fraction

    def step(self, data) -> float:
        if len(data[0]) > self.replay_capacity:
            raise ValueError(f"replay holds {len(data[0])} transitions above the "
                             f"{self.replay_capacity} capacity")
        if not len(self.sampler):
            self.sampler.append(len(data[0]))
        elif len(self.sampler) != len(data[0]):
            raise ValueError("replay row count differs from checkpoint priorities")
        indices, weights = self.sampler.sample(256, self.beta)
        self.last_sampled_indices = indices
        context, action, reward, next_context, next_action, done = [
            torch.as_tensor(x)[torch.from_numpy(indices)].to(self.device) for x in data]
        with torch.no_grad():
            bootstrap = self.target.score(
                ContextBatch(next_context), CandidateBatch(next_action[:, None]))[:, 0]
            returns = (torch.where(done, reward, bootstrap) if self.reward_discount is None else
                       reward + self.reward_discount * torch.where(done, 0., bootstrap))
            # Categorical projection saturates only outside the declared return support.
            returns = returns.clamp(self.model.support[0], self.model.support[-1])
        rows = self.model.loss(context, action, returns, reduction="none")
        loss = (rows * torch.as_tensor(weights, device=self.device, dtype=rows.dtype)).sum() / 256
        if not torch.isfinite(loss):
            raise RuntimeError("critic loss became nonfinite")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10., error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            residual = (self.model.score(ContextBatch(context), CandidateBatch(action[:, None]))[:, 0]
                        - returns).abs()
            self.sampler.update(indices, residual.cpu().numpy())
            for slow, fast in zip(self.target.parameters(), self.model.parameters()):
                slow.lerp_(fast, self.target_ema)
        self.updates += 1
        return float(loss.detach())

    def save(self, path: Path, identity, replay_sha256: str) -> None:
        temporary = path.with_suffix(".tmp")
        torch.save({"model": self.model.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                    "identity": asdict(identity), "rng": self.generator.get_state(),
                    "replay_sha256": replay_sha256, "training_runtime": self.runtime,
                    "gaussian_sigma": self.model.gaussian_sigma, "target_ema": self.target_ema,
                    "priorities": self.sampler.state()}, temporary)
        temporary.replace(path)

    def restore(self, path: Path, identity, replay_sha256: str) -> None:
        # Keep the sampling RNG on CPU; optimizer.load_state_dict moves its
        # tensors to their parameter devices.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        encoding = checkpoint["model"].get("_canonical_schedules", torch.tensor(False))
        if bool(encoding.item()) != self.model.canonical_schedules:
            raise ValueError("checkpoint action encoding differs; use explicit weight initialization")
        # Historical checkpoints used the fixed .01 update coefficient.
        if checkpoint.get("target_ema", .01) != self.target_ema:
            raise ValueError("checkpoint target EMA differs; use explicit weight initialization")
        if checkpoint.get("gaussian_sigma") != self.model.gaussian_sigma:
            raise ValueError("checkpoint categorical loss differs")
        if checkpoint.get("replay_sha256") != replay_sha256:
            raise ValueError("checkpoint replay digest differs or is absent")
        if checkpoint.get("training_runtime") != self.runtime:
            raise ValueError("checkpoint training runtime differs or is absent")
        if checkpoint["identity"] != asdict(identity):
            raise ValueError("checkpoint actor, runtime, formats or objective differ")
        if not isinstance(checkpoint["updates"], int) or checkpoint["updates"] < 1:
            raise ValueError("checkpoint must contain completed critic updates")
        if "priorities" not in checkpoint:
            raise ValueError("checkpoint lacks prioritized replay state; exact "
                             "continuation from a legacy uniform-sampling checkpoint "
                             "is not possible; retrain or start a new experiment")
        self.model.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.generator.set_state(checkpoint["rng"])
        self.updates = checkpoint["updates"]
        self.sampler.load(checkpoint["priorities"])

    def load_weights(self, path: Path, identity, expected_updates: int | None = None, *,
                     allow_environment_change: bool = False, reset_value_head: bool = False,
                     transfer_value_head: bool = False) -> int:
        """Load model weights into model and target from one checkpoint.

        Requires a matching actor and learned contract plus completed updates.
        Environment changes require explicit opt-in. A new objective requires
        explicitly resetting or transferring the value head. Transfer retains
        its initial estimates; subsequent targets use the new objective. Optimizer, sampler, RNG
        and update counter stay fresh. Returns the checkpoint's completed update count.
        """
        if reset_value_head and transfer_value_head:
            raise ValueError("cannot both reset and transfer the value head")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not initialization_identity_matches(checkpoint.get("identity", {}), asdict(identity), allow_environment_change, reset_value_head, transfer_value_head):
            raise ValueError("init checkpoint actor, runtime, formats or objective differ")
        updates = checkpoint.get("updates")
        if not isinstance(updates, int) or updates < 1:
            raise ValueError("init checkpoint must contain completed critic updates")
        if expected_updates is not None and updates != expected_updates:
            raise ValueError("init manifest does not match its critic checkpoint updates")
        if not reset_value_head and checkpoint.get("gaussian_sigma") != self.model.gaussian_sigma:
            raise ValueError("init categorical loss differs; reset the value head explicitly")
        # Explicit initialization transfers parameters into the requested encoding.
        # Replay stores raw actions, so their transition meaning is unchanged.
        weights = {**checkpoint["model"],
                   "_canonical_schedules": torch.tensor(self.model.canonical_schedules)}
        if transfer_value_head and not torch.equal(weights["support"], self.model.support.cpu()):
            raise ValueError("value support differs; reset the value head instead")
        if reset_value_head:
            weights = {key: value for key, value in weights.items()
                       if key not in ("support", "network.4.weight", "network.4.bias")}
        loaded = self.model.load_state_dict(weights, strict=not reset_value_head)
        if reset_value_head and (set(loaded.missing_keys) != {"support", "network.4.weight", "network.4.bias"}
                                 or loaded.unexpected_keys):
            raise ValueError("init checkpoint critic feature weights differ")
        self.target.load_state_dict(self.model.state_dict())
        return updates
