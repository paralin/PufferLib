"""Exact categorical chain for LLB's fifteen scheduled controller factors."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

GROUP_SIZES = (3, 3, 5, 2, 2)
SWITCH_SIZE = 10
NUM_FACTORS = 15
NUM_LOGITS = sum(size + size * size + size * size * SWITCH_SIZE
                 for size in GROUP_SIZES)
_NUM_GROUPS = len(GROUP_SIZES)
_MAX_GROUP_SIZE = max(GROUP_SIZES)


@dataclass(frozen=True)
class AutoregressiveLogits:
    """Raw tables for the initial, final, and switch choices in each group."""

    raw: torch.Tensor


class AutoregressiveCategorical:
    """The LLB action law p(initial) p(final|initial) p(switch|both)."""

    def __init__(self, raw: torch.Tensor) -> None:
        if raw.ndim != 2 or raw.shape[1] != NUM_LOGITS:
            raise ValueError(f"expected [batch, {NUM_LOGITS}] logits")

        invalid_logit = torch.finfo(raw.dtype).min
        initial_tables = []
        final_tables = []
        switch_tables = []
        offset = 0
        for size in GROUP_SIZES:
            initial_end = offset + size
            final_end = initial_end + size * size
            switch_end = final_end + size * size * SWITCH_SIZE
            initial_tables.append(F.pad(raw[:, offset:initial_end],
                                        (0, _MAX_GROUP_SIZE - size), value=invalid_logit))
            final_tables.append(F.pad(
                raw[:, initial_end:final_end].reshape(-1, size, size),
                (0, _MAX_GROUP_SIZE - size, 0, _MAX_GROUP_SIZE - size),
                value=invalid_logit))
            switch_tables.append(F.pad(
                raw[:, final_end:switch_end].reshape(-1, size, size, SWITCH_SIZE),
                (0, 0, 0, _MAX_GROUP_SIZE - size, 0, _MAX_GROUP_SIZE - size),
                value=invalid_logit))
            offset = switch_end

        self._initial = torch.stack(initial_tables, dim=1)
        self._final = torch.stack(final_tables, dim=1)
        self._switch = torch.stack(switch_tables, dim=1)
        sizes = raw.new_tensor(GROUP_SIZES, dtype=torch.long)
        choices = torch.arange(_MAX_GROUP_SIZE, device=raw.device)
        valid = choices[None, :] < sizes[:, None]
        self._final = torch.where(valid[None, :, :, None], self._final, 0.0)
        valid_pairs = valid[:, :, None] & valid[:, None, :]
        self._switch = torch.where(
            valid_pairs[None, :, :, :, None], self._switch, 0.0)

    def sample(self) -> torch.Tensor:
        batch = torch.arange(self._initial.shape[0], device=self._initial.device)[:, None]
        group = torch.arange(_NUM_GROUPS, device=self._initial.device)[None, :]
        initial = Categorical(logits=self._initial).sample()
        final = Categorical(logits=self._final[batch, group, initial]).sample()
        switch = Categorical(logits=self._switch[batch, group, initial, final]).sample()
        return torch.stack((initial, final, switch), dim=2).reshape(-1, NUM_FACTORS)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape != (self._initial.shape[0], NUM_FACTORS):
            raise ValueError(f"expected [{self._initial.shape[0]}, {NUM_FACTORS}] actions")
        choices = actions.long().reshape(-1, _NUM_GROUPS, 3)
        initial, final, switch = choices.unbind(dim=2)
        batch = torch.arange(actions.shape[0], device=actions.device)[:, None]
        group = torch.arange(_NUM_GROUPS, device=actions.device)[None, :]

        initial_log_probability = self._initial.log_softmax(dim=-1).gather(
            -1, initial.unsqueeze(-1)).squeeze(-1)
        final_logits = self._final[batch, group, initial]
        final_log_probability = final_logits.log_softmax(dim=-1).gather(
            -1, final.unsqueeze(-1)).squeeze(-1)
        switch_logits = self._switch[batch, group, initial, final]
        switch_log_probability = switch_logits.log_softmax(dim=-1).gather(
            -1, switch.unsqueeze(-1)).squeeze(-1)
        return (initial_log_probability + final_log_probability
                + switch_log_probability).sum(dim=1)

    @staticmethod
    def _conditional_entropy(logits: torch.Tensor) -> torch.Tensor:
        log_probabilities = logits.log_softmax(dim=-1)
        return -(log_probabilities.exp() * log_probabilities).sum(dim=-1)

    @staticmethod
    def _conditional_kl(logits: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        log_probabilities = logits.log_softmax(dim=-1)
        other_log_probabilities = other.log_softmax(dim=-1)
        terms = log_probabilities.exp() * (log_probabilities - other_log_probabilities)
        return terms.sum(dim=-1)

    def kl_divergence(self, other: "AutoregressiveCategorical") -> torch.Tensor:
        initial_log_probabilities = self._initial.log_softmax(dim=-1)
        initial_probabilities = initial_log_probabilities.exp()
        final_log_probabilities = self._final.log_softmax(dim=-1)
        final_probabilities = final_log_probabilities.exp()

        initial_kl = self._conditional_kl(self._initial, other._initial)
        final_kl = self._conditional_kl(self._final, other._final)
        switch_kl = self._conditional_kl(self._switch, other._switch)
        return (initial_kl
                + (initial_probabilities * final_kl).sum(dim=2)
                + (initial_probabilities.unsqueeze(-1) * final_probabilities
                   * switch_kl).sum(dim=(2, 3))).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        initial_probabilities = self._initial.softmax(dim=-1)
        final_probabilities = self._final.softmax(dim=-1)
        initial_entropy = self._conditional_entropy(self._initial)
        final_entropy = self._conditional_entropy(self._final)
        switch_entropy = self._conditional_entropy(self._switch)
        return (initial_entropy
                + (initial_probabilities * final_entropy).sum(dim=2)
                + (initial_probabilities.unsqueeze(-1) * final_probabilities
                   * switch_entropy).sum(dim=(2, 3))).sum(dim=1)
