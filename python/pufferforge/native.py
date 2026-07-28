from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE with the C++ kernel, falling back to NumPy."""
    rewards = np.ascontiguousarray(rewards, dtype=np.float32)
    dones = np.ascontiguousarray(dones, dtype=np.uint8)
    values = np.ascontiguousarray(values, dtype=np.float32)
    next_values = np.ascontiguousarray(next_values, dtype=np.float32)
    try:
        from . import _core
        advantages, returns = _core.compute_gae(
            rewards, dones, values, next_values, float(gamma), float(gae_lambda)
        )
        return np.asarray(advantages), np.asarray(returns)
    except ImportError:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
        for t in range(rewards.shape[0] - 1, -1, -1):
            next_value = next_values if t == rewards.shape[0] - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t].astype(np.float32)
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values
