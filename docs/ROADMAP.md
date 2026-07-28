# Roadmap to a full high-throughput RL stack

## Stage 1 — completed in this bundle

- Python/C++ control-plane/data-plane split
- native vector stepping and autoreset
- zero-copy NumPy views
- native GAE kernel
- discrete PPO trainer, checkpoints, evaluation, tests, and benchmark
- Python reference vectorizer for fast environment prototyping

## Stage 2 — native environment ABI

- stable C ABI for separately compiled environment plugins
- metadata descriptors for observation and action layouts
- structure-of-arrays state helpers and aligned arenas
- deterministic per-environment random streams
- optional render hooks that are isolated from training workers

## Stage 3 — richer algorithms and memory

- multidiscrete and continuous policies
- recurrent sequence rollouts and burn-in
- V-trace and prioritized trajectory-segment replay
- fused reward normalization, GAE, return, and minibatch-index kernels
- memory-mapped replay for long-running experiments

## Stage 4 — GPU execution

- CUDA rollout storage exposed through DLPack
- fused categorical sampling and log-probability kernels
- CUDA Graph capture for stable-shape update loops
- mixed precision with FP32 master statistics
- stream-aware environment/policy overlap

## Stage 5 — distributed and self-play

- NCCL all-reduce with deterministic rank seeding
- policy banks and immutable frozen snapshots
- agent-slot permutation for symmetric multi-agent games
- Elo/TrueSkill matchmaking and curriculum sampling
- asynchronous checkpoint evaluation workers

## Stage 6 — experiment system

- typed hierarchical configuration
- local and multi-node sweep scheduler
- early stopping from learning-curve models
- JSONL/SQLite metrics with optional W&B adapters
- reproducibility manifest containing source revision, compiler, device, seed,
  environment ABI, and checkpoint schema versions
