from __future__ import annotations

import numpy as np

from pufferforge import NativeLineWorld
from pufferforge.native import compute_gae


def test_native_env_shapes_and_autoreset() -> None:
    env = NativeLineWorld(64, world_size=7, max_steps=4, seed=3)
    obs = env.reset(3)
    assert obs.shape == (64, 4)
    assert obs.dtype == np.float32

    saw_done = False
    for _ in range(10):
        result = env.step(np.full(64, 2, dtype=np.int64))
        assert result.observations.shape == (64, 4)
        assert result.rewards.shape == (64,)
        saw_done = saw_done or bool(result.done.any())
    assert saw_done
    stats = env.drain_stats()
    assert stats["episodes"] > 0
    env.close()


def test_native_gae_matches_reference() -> None:
    rng = np.random.default_rng(7)
    rewards = rng.normal(size=(8, 5)).astype(np.float32)
    dones = (rng.random((8, 5)) < 0.15).astype(np.uint8)
    values = rng.normal(size=(8, 5)).astype(np.float32)
    next_values = rng.normal(size=5).astype(np.float32)

    advantages, returns = compute_gae(rewards, dones, values, next_values, 0.99, 0.95)

    reference = np.zeros_like(rewards)
    last = np.zeros(5, dtype=np.float32)
    for t in range(7, -1, -1):
        next_value = next_values if t == 7 else values[t + 1]
        nonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + 0.99 * next_value * nonterminal - values[t]
        last = delta + 0.99 * 0.95 * nonterminal * last
        reference[t] = last

    np.testing.assert_allclose(advantages, reference, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(returns, reference + values, rtol=1e-5, atol=1e-5)
