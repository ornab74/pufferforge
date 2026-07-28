from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


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

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.horizon

    @property
    def updates(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
