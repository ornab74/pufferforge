from __future__ import annotations

import random
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig
from .device import select_device
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
    early_stopped: bool
    advantage_consensus: float
    value_uncertainty: float
    update_rejected: bool
    update_rolled_back: bool
    rollback_reason: str | None
    device: str

    def to_dict(self) -> dict[str, object]:
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
            "early_stopped": self.early_stopped,
            "advantage_consensus": self.advantage_consensus,
            "value_uncertainty": self.value_uncertainty,
            "update_rejected": self.update_rejected,
            "update_rolled_back": self.update_rolled_back,
            "rollback_reason": self.rollback_reason,
            "device": self.device,
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
        self.device_info = select_device(
            config.device,
            deterministic=config.cuda_deterministic,
            allow_tf32=config.cuda_allow_tf32,
        )
        self.device = self.device_info.device
        self.model = model or ActorCritic(
            env.obs_size,
            env.num_actions,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            value_heads=config.value_heads,
        )
        model_heads = int(getattr(self.model, "value_heads", 1))
        if model_heads != config.value_heads:
            raise ValueError(
                f"model has {model_heads} value heads; config requests "
                f"{config.value_heads}"
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
        self.value_uncertainty = np.zeros(shape, dtype=np.float32)
        self.advantage_confidence = np.ones(shape, dtype=np.float32)

        self.current_obs = np.asarray(env.reset(config.seed), dtype=np.float32)
        if self.current_obs.shape != (config.num_envs, env.obs_size):
            raise ValueError(
                f"reset returned {self.current_obs.shape}; expected "
                f"({config.num_envs}, {env.obs_size})"
            )
        self.global_step = 0
        self.completed_updates = 0
        self.start_time = time.perf_counter()

    def resume(self, path: str | Path, *, load_optimizer: bool = True) -> dict:
        """Restore training progress at a rollout boundary.

        Environments are reset by construction, so a resumed run continues with
        fresh episodes while retaining policy, optimizer, and schedule progress.
        """
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer if load_optimizer else None,
            map_location=self.device,
        )
        saved_config = payload["config"]
        if not isinstance(saved_config, dict):
            raise TypeError("checkpoint config must be a dictionary")
        shape_fields = (
            "num_envs",
            "horizon",
            "hidden_size",
            "hidden_layers",
            "value_heads",
        )
        mismatches = {
            name: (saved_config.get(name), getattr(self.config, name))
            for name in shape_fields
            if saved_config.get(name) != getattr(self.config, name)
        }
        if mismatches:
            details = ", ".join(
                f"{name}={saved!r} (checkpoint) != {current!r} (current)"
                for name, (saved, current) in mismatches.items()
            )
            raise ValueError(f"checkpoint is incompatible with this trainer: {details}")
        self.global_step = int(payload["global_step"])
        self.completed_updates = int(payload["update"])
        if self.global_step < 0 or self.completed_updates < 0:
            raise ValueError("checkpoint progress counters cannot be negative")
        self.start_time = time.perf_counter()
        return payload

    def collect_rollout(self) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        for t in range(self.config.horizon):
            obs_tensor = torch.as_tensor(self.current_obs, device=self.device)
            with torch.inference_mode():
                actions, log_probs, _, values, value_distribution = (
                    self.model.act_ensemble(obs_tensor)
                )

            actions_np = actions.cpu().numpy().astype(np.int64, copy=False)
            result = self.env.step(actions_np)

            self.obs[t] = self.current_obs
            self.actions[t] = actions_np
            self.log_probs[t] = log_probs.cpu().numpy()
            self.values[t] = values.cpu().numpy()
            self.value_uncertainty[t] = (
                value_distribution.std(dim=-1, correction=0).cpu().numpy()
            )
            self.rewards[t] = result.rewards
            self.dones[t] = result.done.astype(np.uint8, copy=False)
            self.current_obs = np.asarray(result.observations, dtype=np.float32)
            self.global_step += self.config.num_envs

        with torch.inference_mode():
            _, next_values = self.model(
                torch.as_tensor(self.current_obs, device=self.device)
            )
        estimators = [(self.config.gamma, self.config.gae_lambda)]
        estimators.extend(
            (float(gamma), float(gae_lambda))
            for gamma, gae_lambda in self.config.gae_ensemble
        )
        advantage_estimates = [
            compute_gae(
                self.rewards,
                self.dones,
                self.values,
                next_values.cpu().numpy(),
                gamma,
                gae_lambda,
            )[0]
            for gamma, gae_lambda in estimators
        ]
        stacked = np.stack(advantage_estimates)
        advantages = np.median(stacked, axis=0).astype(np.float32)
        if len(advantage_estimates) == 1:
            self.advantage_confidence.fill(1.0)
        else:
            sign_agreement = np.abs(np.sign(stacked).mean(axis=0))
            scale = np.mean(np.abs(stacked), axis=0) + 1e-6
            relative_dispersion = np.std(stacked, axis=0) / scale
            self.advantage_confidence = np.clip(
                sign_agreement / (1.0 + relative_dispersion), 0.0, 1.0
            ).astype(np.float32)
        returns = advantages + self.values
        return advantages, returns

    def update_policy(
        self, advantages: np.ndarray, returns: np.ndarray
    ) -> dict[str, object]:
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
        confidence_t = torch.as_tensor(
            self.advantage_confidence.reshape(batch_size), device=self.device
        )
        uncertainty_t = torch.as_tensor(
            self.value_uncertainty.reshape(batch_size), device=self.device
        )

        if cfg.normalize_advantage:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )
        confidence_weight = confidence_t.pow(cfg.consensus_power)
        relative_uncertainty = uncertainty_t / (old_values.abs() + 1.0)
        uncertainty_weight = torch.exp(-cfg.uncertainty_coef * relative_uncertainty)
        advantages_t = advantages_t * confidence_weight * uncertainty_weight

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
        early_stopped = False
        invalid_reason = None
        for epoch in range(cfg.update_epochs):
            epoch_kl = 0.0
            epoch_minibatches = 0
            np.random.shuffle(indices)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb_idx = torch.as_tensor(
                    indices[start : start + cfg.minibatch_size], device=self.device
                )
                _, new_log_prob, entropy, _new_value, new_value_distribution = (
                    self.model.act_ensemble(observations[mb_idx], actions[mb_idx])
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

                old_value = old_values[mb_idx].unsqueeze(-1)
                value_delta = new_value_distribution - old_value
                value_clipped = old_value + torch.clamp(
                    value_delta, -cfg.value_clip_coef, cfg.value_clip_coef
                )
                target = returns_t[mb_idx].unsqueeze(-1)
                value_loss_unclipped = (new_value_distribution - target).pow(2)
                value_loss_clipped = (value_clipped - target).pow(2)
                value_errors = torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                )
                if cfg.value_heads > 1 and cfg.critic_bootstrap_probability < 1.0:
                    mask = (
                        torch.rand_like(value_errors)
                        < cfg.critic_bootstrap_probability
                    ).float()
                    empty_rows = mask.sum(dim=0) == 0
                    mask[0, empty_rows] = 1.0
                    value_loss = 0.5 * (value_errors * mask).sum() / mask.sum()
                else:
                    value_loss = 0.5 * value_errors.mean()

                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )

                finite_tensors = (
                    loss,
                    policy_loss,
                    value_loss,
                    entropy_mean,
                    approx_kl,
                )
                if cfg.rollback_on_nonfinite and not all(
                    bool(torch.isfinite(value)) for value in finite_tensors
                ):
                    invalid_reason = "non_finite_loss"
                    break

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                if cfg.rollback_on_nonfinite and not bool(
                    torch.isfinite(gradient_norm)
                ):
                    invalid_reason = "non_finite_gradient"
                    self.optimizer.zero_grad(set_to_none=True)
                    break
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

            if invalid_reason is not None:
                break
            totals["optimizer_epochs"] = float(epoch + 1)
            mean_epoch_kl = epoch_kl / max(1, epoch_minibatches)
            if cfg.target_kl is not None and mean_epoch_kl > cfg.target_kl:
                early_stopped = True
                break

        count = max(1.0, totals.pop("minibatches"))
        epochs = totals.pop("optimizer_epochs")
        averaged = {key: value / count for key, value in totals.items()}
        averaged["optimizer_epochs"] = epochs
        averaged["early_stopped"] = early_stopped
        averaged["invalid_reason"] = invalid_reason
        return averaged

    def _snapshot_update_state(self) -> tuple[dict, dict]:
        model_state = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        return model_state, deepcopy(self.optimizer.state_dict())

    def _restore_update_state(self, snapshot: tuple[dict, dict]) -> None:
        model_state, optimizer_state = snapshot
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def train(
        self,
        callback: Callable[[TrainMetrics], None] | None = None,
    ) -> list[TrainMetrics]:
        history: list[TrainMetrics] = []
        for update in range(self.completed_updates + 1, self.config.updates + 1):
            if self.config.anneal_lr:
                fraction = 1.0 - (update - 1.0) / self.config.updates
                lr = fraction * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lr
            else:
                lr = self.config.learning_rate

            advantages, returns = self.collect_rollout()
            snapshot = (
                self._snapshot_update_state()
                if self.config.transactional_updates
                else None
            )
            losses = self.update_policy(advantages, returns)
            rollback_reason = losses["invalid_reason"]
            if (
                rollback_reason is None
                and self.config.transactional_updates
                and self.config.rollback_kl is not None
                and float(losses["approx_kl"]) > self.config.rollback_kl
            ):
                rollback_reason = "kl_budget_exceeded"
            update_rolled_back = rollback_reason is not None and snapshot is not None
            if update_rolled_back:
                self._restore_update_state(snapshot)
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
                early_stopped=bool(losses["early_stopped"]),
                advantage_consensus=float(self.advantage_confidence.mean()),
                value_uncertainty=float(self.value_uncertainty.mean()),
                update_rejected=rollback_reason is not None,
                update_rolled_back=update_rolled_back,
                rollback_reason=(
                    str(rollback_reason) if update_rolled_back else None
                ),
                device=str(self.device),
            )
            self.completed_updates = update
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
