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
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("rewards must have shape (horizon, num_envs) with non-zero dimensions")
    if dones.shape != rewards.shape or values.shape != rewards.shape:
        raise ValueError("dones and values must have the same shape as rewards")
    if next_values.shape != (rewards.shape[1],):
        raise ValueError(f"next_values must have shape ({rewards.shape[1]},)")
    finite = (
        np.isfinite(rewards).all()
        and np.isfinite(values).all()
        and np.isfinite(next_values).all()
    )
    if not finite:
        raise ValueError("rewards and values must contain only finite values")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
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
