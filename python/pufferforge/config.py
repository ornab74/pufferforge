from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    seed: int = 1
    total_timesteps: int = 250_000
    num_envs: int = 256
    horizon: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 4096
    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    adam_epsilon: float = 1e-5
    gae_ensemble: tuple[tuple[float, float], ...] = ()
    consensus_power: float = 1.0
    value_heads: int = 1
    critic_bootstrap_probability: float = 1.0
    uncertainty_coef: float = 0.0
    anneal_lr: bool = True
    normalize_advantage: bool = True
    device: str = "auto"
    hidden_size: int = 128
    hidden_layers: int = 2
    checkpoint_interval: int = 25
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 1

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.num_envs <= 0 or self.horizon <= 0:
            raise ValueError("num_envs and horizon must be positive")
        batch_size = self.num_envs * self.horizon
        if self.minibatch_size <= 0 or self.minibatch_size > batch_size:
            raise ValueError(f"minibatch_size must be in [1, {batch_size}]")
        if batch_size % self.minibatch_size != 0:
            raise ValueError("num_envs * horizon must be divisible by minibatch_size")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive or None")
        if self.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be positive")
        for index, pair in enumerate(self.gae_ensemble):
            if len(pair) != 2 or not all(0.0 <= float(value) <= 1.0 for value in pair):
                raise ValueError(
                    f"gae_ensemble[{index}] must contain gamma and lambda in [0, 1]"
                )
        if self.consensus_power < 0 or self.uncertainty_coef < 0:
            raise ValueError("consensus_power and uncertainty_coef must be non-negative")
        if self.value_heads <= 0:
            raise ValueError("value_heads must be positive")
        if not 0.0 < self.critic_bootstrap_probability <= 1.0:
            raise ValueError("critic_bootstrap_probability must be in (0, 1]")
        non_negative = (
            "clip_coef",
            "value_clip_coef",
            "entropy_coef",
            "value_coef",
            "max_grad_norm",
        )
        for name in non_negative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.hidden_size <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden_size and hidden_layers must be positive")
        if self.checkpoint_interval < 0 or self.log_interval <= 0:
            raise ValueError(
                "checkpoint_interval must be non-negative and log_interval positive"
            )
        if self.device != "auto" and not self.device.strip():
            raise ValueError("device cannot be empty")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.horizon

    @property
    def updates(self) -> int:
        return max(1, ceil(self.total_timesteps / self.batch_size))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("training configuration must be a JSON object")
        config = cls(**payload)
        config.validate()
        return config

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
