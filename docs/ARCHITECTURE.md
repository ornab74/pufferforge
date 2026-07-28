# Architecture

## Design boundary

PufferForge uses Python as the control plane and C++ as the data plane.

Python owns:

- PPO orchestration and optimization
- Torch policies and checkpoint formats
- configuration, CLI, evaluation, and logging
- generic Python environment compatibility
- future sweep and distributed-process coordination

C++ owns:

- tightly packed vector-environment state
- batched reset/step loops
- autoreset and episode-stat aggregation
- zero-copy NumPy views over native memory
- CPU-parallel Generalized Advantage Estimation

This boundary keeps research code editable while removing Python from the hottest
per-transition loops.

## Runtime flow

1. `PPOTrainer` requests policy actions from Torch.
2. Actions are converted to one contiguous `int64` NumPy array.
3. `LineWorldVec::step` releases the GIL and advances all environments.
4. C++ returns views of observations, rewards, and done flags without copying.
5. Python records the rollout and asks the native GAE kernel for advantages.
6. Torch performs PPO minibatch updates.
7. Checkpoints are atomically written as Torch state dictionaries.

## Extension contract

A new environment backend should expose:

- `num_envs`, `obs_size`, and `num_actions`
- `reset(seed) -> float32[N, O]`
- `step(int64[N]) -> observations, rewards, terminated, truncated`
- `drain_stats() -> mapping`

The Python `VectorEnv` protocol mirrors this contract. A custom simulation can be
prototyped with `PythonVectorEnv` and later moved to C++ without changing the trainer.

## Planned next layers

- native continuous and multidiscrete action distributions
- shared-memory multiprocess environment workers
- CUDA rollout storage and fused categorical sampling
- recurrent policies with sequence-safe minibatching
- V-trace / prioritized segment replay
- multi-agent slots, policy banks, and self-play matchmaking
- NCCL multi-GPU synchronization
- pluggable environment ABI for separately compiled simulations
