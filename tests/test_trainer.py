from __future__ import annotations

import pytest
import torch
from pufferforge import NativeLineWorld, PPOTrainer, TrainConfig


def test_trainer_smoke(tmp_path) -> None:
    config = TrainConfig(
        num_envs=16,
        horizon=16,
        total_timesteps=512,
        minibatch_size=128,
        update_epochs=1,
        hidden_size=32,
        hidden_layers=1,
        checkpoint_interval=1,
        checkpoint_dir=str(tmp_path),
        device="cpu",
    )
    env = NativeLineWorld(config.num_envs, world_size=7, max_steps=16, seed=1)
    trainer = PPOTrainer(env, config)
    history = trainer.train()
    trainer.close()

    assert len(history) == 2
    assert history[-1].global_step == 512
    assert history[-1].gradient_norm >= 0
    assert history[-1].optimizer_epochs == 1
    assert isinstance(history[-1].early_stopped, bool)
    assert history[-1].device == "cpu"
    assert list(tmp_path.glob("*.pt"))


def test_training_budget_rounds_up_to_complete_rollouts() -> None:
    config = TrainConfig(
        total_timesteps=257,
        num_envs=16,
        horizon=16,
        minibatch_size=128,
    )
    assert config.batch_size == 256
    assert config.updates == 2


def test_ppo_advanced_controls_are_validated() -> None:
    for changes in ({"target_kl": 0.0}, {"adam_epsilon": 0.0}):
        config = TrainConfig(**changes)
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid configuration was accepted: {changes}")


def test_trainer_resumes_model_optimizer_and_progress(tmp_path) -> None:
    common = {
        "num_envs": 8,
        "horizon": 8,
        "minibatch_size": 32,
        "update_epochs": 1,
        "hidden_size": 16,
        "hidden_layers": 1,
        "checkpoint_interval": 1,
        "checkpoint_dir": str(tmp_path),
        "device": "cpu",
    }
    first_config = TrainConfig(total_timesteps=64, **common)
    first = PPOTrainer(
        NativeLineWorld(8, world_size=7, max_steps=8, seed=1), first_config
    )
    first_history = first.train()
    first.close()
    checkpoint = next(tmp_path.glob("step_*.pt"))

    resumed_config = TrainConfig(total_timesteps=128, **common)
    resumed = PPOTrainer(
        NativeLineWorld(8, world_size=7, max_steps=8, seed=1), resumed_config
    )
    payload = resumed.resume(checkpoint)
    resumed_history = resumed.train()
    resumed.close()

    assert payload["format_version"] == 2
    assert first_history[-1].update == 1
    assert resumed_history[-1].update == 2
    assert resumed_history[-1].global_step == 128


def test_resume_rejects_incompatible_rollout_shape(tmp_path) -> None:
    source_config = TrainConfig(
        total_timesteps=64,
        num_envs=8,
        horizon=8,
        minibatch_size=32,
        update_epochs=1,
        hidden_size=16,
        hidden_layers=1,
        checkpoint_interval=1,
        checkpoint_dir=str(tmp_path),
        device="cpu",
    )
    source = PPOTrainer(
        NativeLineWorld(8, world_size=7, max_steps=8, seed=1), source_config
    )
    source.train()
    source.close()

    incompatible_config = TrainConfig(
        total_timesteps=128,
        num_envs=8,
        horizon=16,
        minibatch_size=32,
        hidden_size=16,
        hidden_layers=1,
        checkpoint_interval=0,
        device="cpu",
    )
    incompatible = PPOTrainer(
        NativeLineWorld(8, world_size=7, max_steps=8, seed=1), incompatible_config
    )
    with pytest.raises(ValueError, match="checkpoint is incompatible"):
        incompatible.resume(next(tmp_path.glob("step_*.pt")))
    incompatible.close()


def test_consensus_gae_and_bootstrapped_critic_train_together() -> None:
    config = TrainConfig(
        total_timesteps=64,
        num_envs=8,
        horizon=8,
        minibatch_size=32,
        update_epochs=2,
        hidden_size=16,
        hidden_layers=1,
        checkpoint_interval=0,
        device="cpu",
        gae_ensemble=((0.95, 0.85), (0.995, 0.97)),
        consensus_power=1.5,
        value_heads=3,
        critic_bootstrap_probability=0.7,
        uncertainty_coef=1.0,
    )
    trainer = PPOTrainer(
        NativeLineWorld(8, world_size=7, max_steps=8, seed=3), config
    )
    history = trainer.train()
    trainer.close()

    metrics = history[-1]
    assert 0.0 <= metrics.advantage_consensus <= 1.0
    assert metrics.advantage_consensus < 1.0
    assert metrics.value_uncertainty > 0.0
    assert metrics.value_loss >= 0.0


@pytest.mark.parametrize(
    "changes",
    [
        {"gae_ensemble": ((1.1, 0.9),)},
        {"value_heads": 0},
        {"critic_bootstrap_probability": 0.0},
        {"uncertainty_coef": -1.0},
    ],
)
def test_consensus_and_ensemble_configuration_is_validated(changes) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**changes).validate()


def test_transactional_update_restores_model_and_optimizer() -> None:
    config = TrainConfig(
        total_timesteps=64,
        num_envs=8,
        horizon=8,
        minibatch_size=32,
        update_epochs=1,
        hidden_size=16,
        hidden_layers=1,
        checkpoint_interval=0,
        device="cpu",
        transactional_updates=True,
        rollback_kl=0.01,
    )
    trainer = PPOTrainer(
        NativeLineWorld(8, world_size=7, max_steps=8, seed=5), config
    )
    before = {
        name: value.detach().clone() for name, value in trainer.model.state_dict().items()
    }

    def hazardous_update(advantages, returns):
        del advantages, returns
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.add_(10.0)
        trainer.optimizer.param_groups[0]["lr"] = 0.9
        return {
            "policy_loss": 1.0,
            "value_loss": 1.0,
            "entropy": 0.0,
            "approx_kl": 0.5,
            "clip_fraction": 1.0,
            "gradient_norm": 1.0,
            "optimizer_epochs": 1.0,
            "early_stopped": True,
            "invalid_reason": None,
        }

    trainer.update_policy = hazardous_update
    metrics = trainer.train()[-1]

    assert metrics.update_rejected
    assert metrics.update_rolled_back
    assert metrics.rollback_reason == "kl_budget_exceeded"
    assert trainer.optimizer.param_groups[0]["lr"] == config.learning_rate
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[name])
    trainer.close()


def test_transaction_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="rollback_kl"):
        TrainConfig(rollback_kl=0.0).validate()
