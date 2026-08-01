from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import save_checkpoint
from .config import TrainConfig
from .envs import VectorEnv
from .models import ActorCritic
from .native import compute_gae


@dataclass(slots=True)
class TrainMetrics:
    update: int
    global_step: int
    steps_per_second: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    episodes: int
    mean_return: float
    mean_length: float
    learning_rate: float
    gradient_norm: float
    optimizer_epochs: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "update": self.update,
            "global_step": self.global_step,
            "steps_per_second": self.steps_per_second,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "explained_variance": self.explained_variance,
            "episodes": self.episodes,
            "mean_return": self.mean_return,
            "mean_length": self.mean_length,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "optimizer_epochs": self.optimizer_epochs,
        }


class PPOTrainer:
    def __init__(
        self,
        env: VectorEnv,
        config: TrainConfig,
        model: ActorCritic | None = None,
    ) -> None:
        config.validate()
        if env.num_envs != config.num_envs:
            raise ValueError(
                f"env.num_envs ({env.num_envs}) must equal config.num_envs ({config.num_envs})"
            )

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.env = env
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = model or ActorCritic(
            env.obs_size,
            env.num_actions,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
        )

        shape = (config.horizon, config.num_envs)
        self.obs = np.empty((*shape, env.obs_size), dtype=np.float32)
        self.actions = np.empty(shape, dtype=np.int64)
        self.log_probs = np.empty(shape, dtype=np.float32)
        self.rewards = np.empty(shape, dtype=np.float32)
        self.dones = np.empty(shape, dtype=np.uint8)
        self.values = np.empty(shape, dtype=np.float32)

        self.current_obs = np.asarray(env.reset(config.seed), dtype=np.float32)
        if self.current_obs.shape != (config.num_envs, env.obs_size):
            raise ValueError(
                f"reset returned {self.current_obs.shape}; expected "
                f"({config.num_envs}, {env.obs_size})"
            )
        self.global_step = 0
        self.start_time = time.perf_counter()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return device

    def collect_rollout(self) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        for t in range(self.config.horizon):
            obs_tensor = torch.as_tensor(self.current_obs, device=self.device)
            with torch.inference_mode():
                actions, log_probs, _, values = self.model.act(obs_tensor)

            actions_np = actions.cpu().numpy().astype(np.int64, copy=False)
            result = self.env.step(actions_np)

            self.obs[t] = self.current_obs
            self.actions[t] = actions_np
            self.log_probs[t] = log_probs.cpu().numpy()
            self.values[t] = values.cpu().numpy()
            self.rewards[t] = result.rewards
            self.dones[t] = result.done.astype(np.uint8, copy=False)
            self.current_obs = np.asarray(result.observations, dtype=np.float32)
            self.global_step += self.config.num_envs

        with torch.inference_mode():
            _, next_values = self.model(
                torch.as_tensor(self.current_obs, device=self.device)
            )
        advantages, returns = compute_gae(
            self.rewards,
            self.dones,
            self.values,
            next_values.cpu().numpy(),
            self.config.gamma,
            self.config.gae_lambda,
        )
        return advantages, returns

    def update_policy(
        self, advantages: np.ndarray, returns: np.ndarray
    ) -> dict[str, float]:
        self.model.train()
        cfg = self.config
        batch_size = cfg.batch_size

        observations = torch.as_tensor(
            self.obs.reshape(batch_size, self.env.obs_size), device=self.device
        )
        actions = torch.as_tensor(
            self.actions.reshape(batch_size), device=self.device
        )
        old_log_probs = torch.as_tensor(
            self.log_probs.reshape(batch_size), device=self.device
        )
        old_values = torch.as_tensor(
            self.values.reshape(batch_size), device=self.device
        )
        advantages_t = torch.as_tensor(
            advantages.reshape(batch_size), device=self.device
        )
        returns_t = torch.as_tensor(returns.reshape(batch_size), device=self.device)

        if cfg.normalize_advantage:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
            "minibatches": 0.0,
            "optimizer_epochs": 0.0,
        }

        indices = np.arange(batch_size)
        for epoch in range(cfg.update_epochs):
            epoch_kl = 0.0
            epoch_minibatches = 0
            np.random.shuffle(indices)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb_idx = torch.as_tensor(
                    indices[start : start + cfg.minibatch_size], device=self.device
                )
                _, new_log_prob, entropy, new_value = self.model.act(
                    observations[mb_idx], actions[mb_idx]
                )
                log_ratio = new_log_prob - old_log_probs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > cfg.clip_coef
                    ).float().mean()

                mb_advantages = advantages_t[mb_idx]
                policy_loss_unclipped = -mb_advantages * ratio
                policy_loss_clipped = -mb_advantages * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                policy_loss = torch.maximum(
                    policy_loss_unclipped, policy_loss_clipped
                ).mean()

                value_delta = new_value - old_values[mb_idx]
                value_clipped = old_values[mb_idx] + torch.clamp(
                    value_delta, -cfg.value_clip_coef, cfg.value_clip_coef
                )
                value_loss_unclipped = (new_value - returns_t[mb_idx]).pow(2)
                value_loss_clipped = (value_clipped - returns_t[mb_idx]).pow(2)
                value_loss = 0.5 * torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                ).mean()

                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                totals["policy_loss"] += float(policy_loss.detach())
                totals["value_loss"] += float(value_loss.detach())
                totals["entropy"] += float(entropy_mean.detach())
                totals["approx_kl"] += float(approx_kl.detach())
                totals["clip_fraction"] += float(clip_fraction.detach())
                totals["gradient_norm"] += float(gradient_norm.detach())
                totals["minibatches"] += 1.0
                epoch_kl += float(approx_kl.detach())
                epoch_minibatches += 1

            totals["optimizer_epochs"] = float(epoch + 1)
            mean_epoch_kl = epoch_kl / max(1, epoch_minibatches)
            if cfg.target_kl is not None and mean_epoch_kl > cfg.target_kl:
                break

        count = max(1.0, totals.pop("minibatches"))
        epochs = totals.pop("optimizer_epochs")
        averaged = {key: value / count for key, value in totals.items()}
        averaged["optimizer_epochs"] = epochs
        return averaged

    def train(
        self,
        callback: Callable[[TrainMetrics], None] | None = None,
    ) -> list[TrainMetrics]:
        history: list[TrainMetrics] = []
        for update in range(1, self.config.updates + 1):
            if self.config.anneal_lr:
                fraction = 1.0 - (update - 1.0) / self.config.updates
                lr = fraction * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lr
            else:
                lr = self.config.learning_rate

            advantages, returns = self.collect_rollout()
            losses = self.update_policy(advantages, returns)
            stats = self.env.drain_stats()

            value_flat = self.values.reshape(-1)
            returns_flat = returns.reshape(-1)
            variance = np.var(returns_flat)
            explained_variance = (
                float(1.0 - np.var(returns_flat - value_flat) / variance)
                if variance > 1e-12
                else 0.0
            )
            elapsed = max(time.perf_counter() - self.start_time, 1e-9)
            metrics = TrainMetrics(
                update=update,
                global_step=self.global_step,
                steps_per_second=self.global_step / elapsed,
                policy_loss=losses["policy_loss"],
                value_loss=losses["value_loss"],
                entropy=losses["entropy"],
                approx_kl=losses["approx_kl"],
                clip_fraction=losses["clip_fraction"],
                explained_variance=explained_variance,
                episodes=int(stats.get("episodes", 0)),
                mean_return=float(stats.get("mean_return", 0.0)),
                mean_length=float(stats.get("mean_length", 0.0)),
                learning_rate=lr,
                gradient_norm=losses["gradient_norm"],
                optimizer_epochs=int(losses["optimizer_epochs"]),
            )
            history.append(metrics)
            if callback is not None:
                callback(metrics)

            if self.config.checkpoint_interval > 0 and (
                update % self.config.checkpoint_interval == 0
                or update == self.config.updates
            ):
                path = (
                    Path(self.config.checkpoint_dir)
                    / f"step_{self.global_step:012d}.pt"
                )
                save_checkpoint(
                    path,
                    model=self.model,
                    optimizer=self.optimizer,
                    global_step=self.global_step,
                    update=update,
                    config=self.config.to_dict(),
                )
        return history

    def close(self) -> None:
        self.env.close()
