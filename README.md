# PufferForge

PufferForge is a clean-room Python/C++ reconstruction of the essential architecture
behind high-throughput reinforcement-learning systems such as PufferLib. It is not
a line-for-line fork. The first release is a compact, functional foundation designed
to be understandable, portable, and easy to extend.

## What is implemented

- C++20 batched environment with OpenMP stepping
- zero-copy C++ memory exposed as NumPy arrays through pybind11
- native CPU Generalized Advantage Estimation kernel
- Python vector-environment protocol and pure-Python reference vectorizer
- PyTorch discrete actor-critic model
- PPO rollout collection, clipped policy/value losses, entropy regularization,
  gradient clipping, learning-rate annealing, and explained variance
- atomic checkpoints, evaluation CLI, JSONL metrics, tests, and benchmarks
- native autoreset and aggregated episode statistics

## Architecture

```text
Python control plane
  CLI -> config -> PPOTrainer -> Torch policy/optimizer/checkpoints
                         | actions
                         v
C++ data plane      LineWorldVec / future simulation plugins
                    packed state + OpenMP + zero-copy arrays
                         |
                         +---- native GAE kernel
```

Python remains the research layer. C++ owns transition-heavy loops and contiguous
memory. See `docs/ARCHITECTURE.md` for the extension contract.

## Build

Standard editable install:

```bash
python -m pip install -e .
```

For an offline environment that already has PyTorch but not the standalone
`pybind11` package, use PyTorch's bundled pybind11 headers:

```bash
./scripts/build_local.sh
export PYTHONPATH="$PWD/python"
```

Disable OpenMP when necessary:

```bash
PUFFERFORGE_OPENMP=0 ./scripts/build_local.sh
```

## Train

```bash
pufferforge train \
  --num-envs 256 \
  --horizon 128 \
  --minibatch-size 4096 \
  --total-timesteps 250000 \
  --json-log runs/lineworld.jsonl
```

Equivalent module invocation:

```bash
PYTHONPATH=python python -m pufferforge train --total-timesteps 65536
```

## Benchmark native stepping

```bash
pufferforge bench --num-envs 4096 --steps 5000
```

## Evaluate

```bash
pufferforge eval checkpoints/step_000000250000.pt --episodes 2048
```

The model dimensions used at evaluation must match training. The defaults match the
default training configuration.

## Custom Python environment

`examples/custom_env.py` demonstrates the scalar environment contract and wraps many
instances with `PythonVectorEnv`. Once a custom environment is stable, its `step` and
`reset` methods can be moved into a C++ class implementing the native contract.

## Repository layout

```text
cpp/include/pufferforge/core.hpp  Native runtime interfaces
cpp/src/bindings.cpp              C++ implementation + pybind11 module
python/pufferforge/envs.py        Vector environment protocols/adapters
python/pufferforge/models.py      Torch actor-critic models
python/pufferforge/trainer.py     PPO runtime
python/pufferforge/native.py      Native GAE dispatch with NumPy fallback
benchmarks/                       Throughput tools
tests/                            Native, Python-vector, and trainer tests
```

## Relationship to PufferLib

PufferLib 4.0 combines a Python CLI/control layer with a compiled backend, Torch and
native training paths, CUDA/NCCL integration, vector environments, self-play,
hyperparameter sweeps, and a large catalog of native simulations. PufferForge starts
with the stable architectural seam shared by those systems: a Python research plane
over a native vectorized execution plane. CUDA kernels, distributed training,
self-play policy banks, and simulation plugins are explicit next milestones rather
than hidden behind a monolithic build script.

PufferLib is MIT licensed. This project uses the same permissive license but contains
an independently written implementation and its own API.
