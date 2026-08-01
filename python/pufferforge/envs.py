from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True)
class StepBatch:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, float | int]

    @property
    def done(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)


@runtime_checkable
class VectorEnv(Protocol):
    num_envs: int
    obs_size: int
    num_actions: int

    def reset(self, seed: int | None = None) -> np.ndarray: ...
    def step(self, actions: np.ndarray) -> StepBatch: ...
    def drain_stats(self) -> dict[str, float | int]: ...
    def close(self) -> None: ...


class NativeLineWorld:
    """C++ vector environment with autoreset and zero-copy NumPy views."""

    def __init__(
        self,
        num_envs: int,
        world_size: int = 15,
        max_steps: int = 64,
        seed: int = 1,
    ) -> None:
        try:
            from . import _core
        except ImportError as exc:
            raise RuntimeError(
                "PufferForge native extension is unavailable. Build with "
                "`python -m pip install -e .` or run `./scripts/build_local.sh`."
            ) from exc
        self._env = _core.LineWorldVec(num_envs, world_size, max_steps, seed)
        self.num_envs = int(self._env.num_envs)
        self.obs_size = int(self._env.obs_size)
        self.num_actions = int(self._env.num_actions)
        self._seed = seed

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed
        return np.asarray(self._env.reset(self._seed), dtype=np.float32)

    def step(self, actions: np.ndarray) -> StepBatch:
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        observations, rewards, terminated, truncated = self._env.step(actions)
        return StepBatch(
            observations=np.asarray(observations, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=np.uint8).astype(bool, copy=False),
            truncated=np.asarray(truncated, dtype=np.uint8).astype(bool, copy=False),
            info={},
        )

    def drain_stats(self) -> dict[str, float | int]:
        return dict(self._env.drain_stats())

    def close(self) -> None:
        self._env = None


class ScalarEnv(Protocol):
    obs_size: int
    num_actions: int

    def reset(self, seed: int | None = None) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    def close(self) -> None: ...


class PythonVectorEnv:
    """Reference vectorizer for custom Python environments.

    It is intentionally simple and deterministic. Production custom environments
    can later move their hot step/reset kernels behind the same VectorEnv protocol.
    """

    def __init__(self, factories: list[Callable[[], ScalarEnv]], seed: int = 1) -> None:
        if not factories:
            raise ValueError("at least one environment factory is required")
        self._envs = [factory() for factory in factories]
        self.num_envs = len(self._envs)
        self.obs_size = int(self._envs[0].obs_size)
        self.num_actions = int(self._envs[0].num_actions)
        if self.obs_size <= 0 or self.num_actions <= 1:
            self.close()
            raise ValueError("environments must expose obs_size > 0 and num_actions > 1")
        incompatible = [
            index
            for index, env in enumerate(self._envs[1:], start=1)
            if int(env.obs_size) != self.obs_size
            or int(env.num_actions) != self.num_actions
        ]
        if incompatible:
            self.close()
            raise ValueError(
                "all environments must share one space; "
                f"incompatible indices: {incompatible}"
            )
        self._seed = seed
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._stats = {"episodes": 0, "return_sum": 0.0, "length_sum": 0}

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed
        observations = [
            np.asarray(env.reset(self._seed + i), dtype=np.float32).reshape(self.obs_size)
            for i, env in enumerate(self._envs)
        ]
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        return np.stack(observations)

    def step(self, actions: np.ndarray) -> StepBatch:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.size != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} actions, received shape {actions.shape}"
            )
        actions = actions.reshape(self.num_envs)
        if np.any((actions < 0) | (actions >= self.num_actions)):
            raise ValueError(f"actions must be in [0, {self.num_actions})")
        observations = np.empty((self.num_envs, self.obs_size), dtype=np.float32)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        for i, (env, action) in enumerate(zip(self._envs, actions, strict=True)):
            transition = env.step(int(action))
            if not isinstance(transition, tuple) or len(transition) != 5:
                raise TypeError(
                    "scalar env step() must return "
                    "(obs, reward, terminated, truncated, info)"
                )
            obs, reward, term, trunc, _ = transition
            if not np.isfinite(reward):
                raise ValueError(f"environment {i} returned a non-finite reward")
            rewards[i] = reward
            terminated[i] = term
            truncated[i] = trunc
            self._episode_returns[i] += reward
            self._episode_lengths[i] += 1
            if term or trunc:
                self._stats["episodes"] += 1
                self._stats["return_sum"] += float(self._episode_returns[i])
                self._stats["length_sum"] += int(self._episode_lengths[i])
                obs = env.reset(self._seed + i + int(self._stats["episodes"]))
                self._episode_returns[i] = 0.0
                self._episode_lengths[i] = 0
            observations[i] = np.asarray(obs, dtype=np.float32).reshape(self.obs_size)

        return StepBatch(observations, rewards, terminated, truncated, {})

    def drain_stats(self) -> dict[str, float | int]:
        episodes = int(self._stats["episodes"])
        output = dict(self._stats)
        output["mean_return"] = float(self._stats["return_sum"]) / episodes if episodes else 0.0
        output["mean_length"] = float(self._stats["length_sum"]) / episodes if episodes else 0.0
        self._stats = {"episodes": 0, "return_sum": 0.0, "length_sum": 0}
        return output

    def close(self) -> None:
        for env in self._envs:
            env.close()
