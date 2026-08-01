from __future__ import annotations

import numpy as np
import pytest
from pufferforge.envs import PythonVectorEnv


class CounterEnv:
    obs_size = 1
    num_actions = 2

    def __init__(self) -> None:
        self.value = 0

    def reset(self, seed=None):
        self.value = 0
        return np.array([0.0], dtype=np.float32)

    def step(self, action: int):
        self.value += 1 if action else -1
        done = abs(self.value) >= 2
        return np.array([self.value], dtype=np.float32), 1.0, done, False, {}

    def close(self):
        pass


def test_python_vector_autoreset() -> None:
    env = PythonVectorEnv([CounterEnv, CounterEnv])
    obs = env.reset(1)
    assert obs.shape == (2, 1)
    env.step(np.array([1, 1]))
    result = env.step(np.array([1, 1]))
    assert result.done.all()
    assert (result.observations == 0).all()
    assert env.drain_stats()["episodes"] == 2


def test_python_vector_validates_actions() -> None:
    env = PythonVectorEnv([CounterEnv, CounterEnv])
    with pytest.raises(ValueError, match="expected 2 actions"):
        env.step(np.array([1]))
    with pytest.raises(ValueError, match="actions must be"):
        env.step(np.array([1, 2]))
    env.close()


def test_python_vector_rejects_mixed_spaces() -> None:
    class OtherCounter(CounterEnv):
        obs_size = 2

    with pytest.raises(ValueError, match="incompatible indices"):
        PythonVectorEnv([CounterEnv, OtherCounter])
