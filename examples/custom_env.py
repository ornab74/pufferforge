from __future__ import annotations

import numpy as np

from pufferforge import PPOTrainer, PythonVectorEnv, TrainConfig


class TinyBalance:
    obs_size = 2
    num_actions = 2

    def __init__(self) -> None:
        self.rng = np.random.default_rng()
        self.position = 0.0
        self.velocity = 0.0
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.position = float(self.rng.uniform(-0.25, 0.25))
        self.velocity = 0.0
        self.steps = 0
        return np.array([self.position, self.velocity], dtype=np.float32)

    def step(self, action: int):
        force = -0.04 if action == 0 else 0.04
        self.velocity = 0.95 * self.velocity + force - 0.01 * self.position
        self.position += self.velocity
        self.steps += 1
        terminated = abs(self.position) > 1.0
        truncated = self.steps >= 128
        reward = 1.0 - abs(self.position)
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        return np.array([self.position, self.velocity], dtype=np.float32)

    def close(self) -> None:
        pass


def main() -> None:
    num_envs = 32
    env = PythonVectorEnv([TinyBalance for _ in range(num_envs)])
    config = TrainConfig(
        num_envs=num_envs,
        horizon=64,
        minibatch_size=512,
        total_timesteps=65_536,
        checkpoint_interval=0,
    )
    trainer = PPOTrainer(env, config)
    trainer.train(lambda m: print(m.to_dict()))
    trainer.close()


if __name__ == "__main__":
    main()
