from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ReplaySample:
    data: dict[str, np.ndarray]
    indices: np.ndarray
    weights: np.ndarray


class PrioritizedReplay:
    """Checkpoint-friendly proportional prioritized replay with stratified sampling."""

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6, seed: int = 1) -> None:
        if capacity <= 0 or not 0 <= alpha <= 1:
            raise ValueError("invalid replay configuration")
        self.capacity, self.alpha, self.epsilon = capacity, alpha, epsilon
        self.rng = np.random.default_rng(seed)
        self.storage: dict[str, np.ndarray] = {}
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.size = 0

    def add(self, **transition: Any) -> None:
        if not transition:
            raise ValueError("transition cannot be empty")
        if not self.storage:
            for key, value in transition.items():
                arr = np.asarray(value)
                self.storage[key] = np.empty((self.capacity, *arr.shape), dtype=arr.dtype)
        if set(transition) != set(self.storage):
            raise ValueError("transition fields changed")
        for key, value in transition.items():
            self.storage[key][self.position] = value
        self.priorities[self.position] = self.priorities[: self.size].max(initial=1.0)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> ReplaySample:
        if self.size == 0 or batch_size <= 0 or not 0 <= beta <= 1:
            raise ValueError("invalid sample request")
        scaled = np.power(self.priorities[: self.size] + self.epsilon, self.alpha)
        probs = scaled / scaled.sum()
        cdf = np.cumsum(probs)
        points = np.arange(batch_size) / batch_size + self.rng.random(batch_size) / batch_size
        indices = np.searchsorted(cdf, points, side="right").clip(max=self.size - 1)
        weights = np.power(self.size * probs[indices], -beta)
        weights /= weights.max(initial=1.0)
        return ReplaySample({k: v[indices] for k, v in self.storage.items()}, indices, weights.astype(np.float32))

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        if indices.shape != priorities.shape or np.any(priorities < 0):
            raise ValueError("invalid priorities")
        self.priorities[indices] = priorities + self.epsilon

    def __len__(self) -> int:
        return self.size
