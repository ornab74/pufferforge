from __future__ import annotations

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
    assert list(tmp_path.glob("*.pt"))
