from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


class Space(Protocol):
    @property
    def flat_dim(self) -> int: ...
    def sample(self, rng: np.random.Generator, batch: int | None = None) -> np.ndarray: ...
    def contains(self, value: np.ndarray) -> bool: ...


@dataclass(frozen=True, slots=True)
class Box:
    low: float | np.ndarray
    high: float | np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype = np.dtype(np.float32)

    def __post_init__(self) -> None:
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError("shape must contain positive dimensions")
        low = np.broadcast_to(np.asarray(self.low, dtype=self.dtype), self.shape)
        high = np.broadcast_to(np.asarray(self.high, dtype=self.dtype), self.shape)
        if np.any(low > high):
            raise ValueError("low must be <= high")

    @property
    def flat_dim(self) -> int:
        return int(np.prod(self.shape))

    def sample(self, rng: np.random.Generator, batch: int | None = None) -> np.ndarray:
        shape = self.shape if batch is None else (batch, *self.shape)
        low = np.broadcast_to(np.asarray(self.low, dtype=np.float64), self.shape)
        high = np.broadcast_to(np.asarray(self.high, dtype=np.float64), self.shape)
        return rng.uniform(low, high, size=shape).astype(self.dtype)

    def contains(self, value: np.ndarray) -> bool:
        value = np.asarray(value)
        low = np.broadcast_to(np.asarray(self.low), self.shape)
        high = np.broadcast_to(np.asarray(self.high), self.shape)
        return value.shape == self.shape and bool(np.all(value >= low) and np.all(value <= high))


@dataclass(frozen=True, slots=True)
class Discrete:
    n: int

    def __post_init__(self) -> None:
        if self.n <= 1:
            raise ValueError("n must be greater than one")

    @property
    def flat_dim(self) -> int:
        return 1

    def sample(self, rng: np.random.Generator, batch: int | None = None) -> np.ndarray:
        size = None if batch is None else batch
        return rng.integers(0, self.n, size=size, dtype=np.int64)

    def contains(self, value: np.ndarray) -> bool:
        arr = np.asarray(value)
        return bool(np.issubdtype(arr.dtype, np.integer) and np.all((arr >= 0) & (arr < self.n)))


@dataclass(frozen=True, slots=True)
class MultiDiscrete:
    nvec: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.nvec or any(n <= 1 for n in self.nvec):
            raise ValueError("all action cardinalities must be greater than one")

    @property
    def flat_dim(self) -> int:
        return len(self.nvec)

    @property
    def logits_dim(self) -> int:
        return sum(self.nvec)

    def sample(self, rng: np.random.Generator, batch: int | None = None) -> np.ndarray:
        shape = (len(self.nvec),) if batch is None else (batch, len(self.nvec))
        out = np.empty(shape, dtype=np.int64)
        for i, n in enumerate(self.nvec):
            out[..., i] = rng.integers(0, n, size=out[..., i].shape)
        return out

    def contains(self, value: np.ndarray) -> bool:
        arr = np.asarray(value)
        if arr.shape != (len(self.nvec),) or not np.issubdtype(arr.dtype, np.integer):
            return False
        return all(0 <= int(v) < n for v, n in zip(arr, self.nvec, strict=True))
