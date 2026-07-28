from __future__ import annotations

from collections import deque
import numpy as np
from .envs import StepBatch, VectorEnv


class VectorWrapper:
    def __init__(self, env: VectorEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.obs_size = env.obs_size
        self.num_actions = env.num_actions

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self.env.reset(seed)

    def step(self, actions: np.ndarray) -> StepBatch:
        return self.env.step(actions)

    def drain_stats(self) -> dict[str, float | int]:
        return self.env.drain_stats()

    def close(self) -> None:
        self.env.close()


class ClipReward(VectorWrapper):
    def __init__(self, env: VectorEnv, low: float = -1.0, high: float = 1.0) -> None:
        super().__init__(env)
        self.low, self.high = float(low), float(high)

    def step(self, actions: np.ndarray) -> StepBatch:
        batch = self.env.step(actions)
        return StepBatch(batch.observations, np.clip(batch.rewards, self.low, self.high), batch.terminated, batch.truncated, batch.info)


class NormalizeObservation(VectorWrapper):
    def __init__(self, env: VectorEnv, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        super().__init__(env)
        self.epsilon, self.clip = epsilon, clip
        self.count = 0
        self.mean = np.zeros(env.obs_size, dtype=np.float64)
        self.m2 = np.zeros(env.obs_size, dtype=np.float64)

    def _update(self, obs: np.ndarray) -> np.ndarray:
        for row in np.asarray(obs, dtype=np.float64):
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)
        variance = self.m2 / max(self.count - 1, 1)
        return np.clip((obs - self.mean) / np.sqrt(variance + self.epsilon), -self.clip, self.clip).astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self._update(self.env.reset(seed))

    def step(self, actions: np.ndarray) -> StepBatch:
        batch = self.env.step(actions)
        return StepBatch(self._update(batch.observations), batch.rewards, batch.terminated, batch.truncated, batch.info)


class FrameStack(VectorWrapper):
    def __init__(self, env: VectorEnv, frames: int = 4) -> None:
        if frames <= 0:
            raise ValueError("frames must be positive")
        super().__init__(env)
        self.frames = frames
        self.obs_size = env.obs_size * frames
        self._history: deque[np.ndarray] = deque(maxlen=frames)

    def _stack(self) -> np.ndarray:
        return np.concatenate(tuple(self._history), axis=-1)

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = self.env.reset(seed)
        self._history.clear()
        self._history.extend(np.array(obs, copy=True) for _ in range(self.frames))
        return self._stack()

    def step(self, actions: np.ndarray) -> StepBatch:
        batch = self.env.step(actions)
        self._history.append(np.array(batch.observations, copy=True))
        return StepBatch(self._stack(), batch.rewards, batch.terminated, batch.truncated, batch.info)
