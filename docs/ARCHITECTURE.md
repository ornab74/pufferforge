# Architecture

## Design boundary

PufferForge uses Python as the control plane and C++ as the data plane.

Python owns:

- PPO orchestration and optimization
- Torch policies and checkpoint formats
- configuration, CLI, evaluation, and logging
- generic Python environment compatibility
- future sweep and distributed-process coordination
- CUDA/CPU selection, CUDA backend policy, and accelerator diagnostics

C++ owns:

- tightly packed vector-environment state
- batched reset/step loops
- autoreset and episode-stat aggregation
- zero-copy NumPy views over native memory
- CPU-parallel Generalized Advantage Estimation

This boundary keeps research code editable while removing Python from the hottest
per-transition loops.

Native environments remain CPU-resident and feed contiguous NumPy batches to the
selected Torch device. Policy inference and PPO optimization run on CUDA whenever
`select_device("auto")` detects a usable GPU; otherwise the identical code path
runs on CPU. Explicit CUDA indices are validated before model allocation.

## Runtime flow

1. `PPOTrainer` requests policy actions from Torch.
2. Actions are converted to one contiguous `int64` NumPy array.
3. `LineWorldVec::step` releases the GIL and advances all environments.
4. C++ returns views of observations, rewards, and done flags without copying.
5. Python records the rollout and asks the native GAE kernel for each configured
   temporal estimator.
6. Multi-timescale advantages are combined by median and weighted by estimator
   consensus.
7. Torch performs PPO minibatch updates; bootstrapped critic heads expose
   epistemic uncertainty that tempers policy updates in uncertain states.
8. KL guards can stop optimizer epochs early, and checkpoints are atomically
   written as versioned Torch state dictionaries.
9. When transactional updates are enabled, the pre-update model and optimizer
   snapshot is restored if numerical checks or the final KL budget fail.

## Confidence-aware PPO

Conventional PPO is recovered with one value head and no additional GAE
estimators. The advanced path composes two independent confidence signals:

- temporal confidence from sign agreement and dispersion among GAE timescales;
- model confidence from disagreement among bootstrapped value heads.

Only the policy advantage is tempered. Every critic still learns from its
bootstrap sample, allowing uncertainty to shrink when the shared representation
and value heads acquire sufficient evidence.

## Transaction boundary

The rollout is the transaction boundary. PufferForge snapshots model parameters,
buffers, and optimizer state after collection but before minibatch optimization.
Non-finite loss or gradient checks can abort inside an epoch; a final approximate
KL check protects against finite but excessively large policy movement. Rollback
does not rewind environments because the rejected rollout remains valid on-policy
data for the pre-update policy and environment stepping is intentionally outside
the optimizer transaction.

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
