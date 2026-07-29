# PufferForge

PufferForge Python/C++ reconstruction of the essential architecture
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
- AtlasLab predictive mapping simulations with hidden controls and changing worlds
- counterfactual world/atlas beam planning
- temporal semantic forecasting and causal command learning
- bandwidth-limited multi-agent map fusion and frontier auctions

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

```bash
python -m pip install -e .
```

For an offline environment that already has PyTorch but not standalone `pybind11`,
the build now discovers PyTorch's bundled compatible headers automatically.

Build the pure-Python/NumPy fallback without compiling C++:

```bash
PUFFERFORGE_BUILD_NATIVE=0 python -m pip install -e . --no-build-isolation
```

The fallback keeps `PythonVectorEnv`, PPO, Torch policies, and NumPy GAE available.
Native LineWorld stepping requires the compiled extension.

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

## Benchmark native stepping

```bash
pufferforge bench --num-envs 4096 --steps 5000
```

## Evaluate

```bash
pufferforge eval checkpoints/step_000000250000.pt --episodes 2048
```

## AtlasLab predictive mapping

Run paired strategy comparisons across identical procedural worlds:

```bash
pufferforge atlas-suite \
  --seeds 1,2,3,4,5 \
  --strategies random,frontier,atlas_dreamer \
  --steps 100 \
  --output atlaslab/suite.json
```

Run bandwidth-limited coordinated cartography:

```bash
pufferforge atlas-swarm \
  --agents 4 \
  --steps 120 \
  --sync-interval 8 \
  --bandwidth-bytes 32768 \
  --output atlaslab/swarm.json
```

AtlasLab combines semantic Dirichlet beliefs, per-cell temporal transition models,
causal command learning, hidden rotated controls, world state snapshots,
counterfactual beam search, source/sequence map packets, and frontier auctions.
See `docs/ATLASLAB.md` for details.

## SurfGuard-USA Colab

`SurfGuard_USA_PufferForge_Colab.ipynb` is a research pipeline for contiguous-U.S.
coastal hazard reconstruction, calibrated risk modeling, and a PufferForge PPO alert
policy. The repaired workflow uses NOAA-safe 31-day CO-OPS request windows, optional
GRIB dependencies, corrected NDBC missing-value parsing, public top-level PufferForge
imports, and configurable OpenAI API model identifiers.

Run installation, configuration, and the runtime self-check before starting network
collection. See `docs/SURFGUARD_COLAB.md` for recovery instructions and scale controls.

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
python/pufferforge/atlaslab.py    Predictive mapping research vertical
benchmarks/                       Throughput tools
tests/                            Native, Python-vector, trainer, and mapping tests
```

## Relationship to PufferLib

PufferLib 4.0 combines a Python CLI/control layer with a compiled backend, Torch and
native training paths, CUDA/NCCL integration, vector environments, self-play,
hyperparameter sweeps, and a large catalog of native simulations. PufferForge starts
with the stable architectural seam shared by those systems: a Python research plane
over a native vectorized execution plane, then extends it with independently written
predictive mapping and multi-agent cartography systems.

PufferLib is MIT licensed. This project uses the same permissive license but contains
an independently written implementation and its own API.
