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
  gradient clipping, learning-rate annealing, KL early stopping, and explained variance
- gradient-norm and optimizer-epoch telemetry for diagnosing unstable PPO updates
- reproducible CPU/CUDA seeding and complete-rollout timestep budgeting
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

For an offline environment that already has PyTorch but not standalone
`pybind11`, the local build helper can use PyTorch's bundled headers:

```bash
./scripts/build_local.sh
export PYTHONPATH="$PWD/python"
```

Build the pure-Python/NumPy fallback without compiling C++:

```bash
PUFFERFORGE_BUILD_NATIVE=0 python -m pip install --no-build-isolation --editable .
```

Native discovery defaults to `auto`: when pybind11 headers are unavailable,
packaging completes with the pure-Python runtime. Set
`PUFFERFORGE_BUILD_NATIVE=1` when a missing native extension should be fatal.

Disable OpenMP when necessary:

```bash
PUFFERFORGE_OPENMP=0 ./scripts/build_local.sh
```

## Colab-safe editable install

Upgrade the packaging toolchain before an editable install and use the already-installed runtime build toolchain:

```bash
python -m pip install --upgrade "pip>=24" "setuptools>=61" wheel "pybind11>=2.13"
python -m pip install --no-build-isolation --editable .
```

Set `PUFFERFORGE_BUILD_NATIVE=0` to skip the optional C++ extension. The Python
package still provides `PPOTrainer`, `PythonVectorEnv`, and the NumPy GAE path.

## Train

```bash
pufferforge train \
  --num-envs 256 \
  --horizon 128 \
  --minibatch-size 4096 \
  --total-timesteps 250000 \
  --target-kl 0.02 \
  --json-log runs/lineworld.jsonl
```

### Advanced PPO controls

PufferForge uses clipped PPO with GAE, value clipping, entropy regularization,
global gradient clipping, and optional linear learning-rate annealing. A positive
`target_kl` stops the remaining optimizer epochs when an epoch's mean approximate
KL exceeds the trust-region budget. Set it to `null` in JSON to disable this
guard. Metrics include `gradient_norm` and `optimizer_epochs`, making it easy to
see whether updates are saturating the gradient clip or stopping early.

`total_timesteps` is a lower bound: training completes whole vector rollouts and
therefore rounds up to `num_envs * horizon`. This avoids silently training for
fewer transitions than requested.

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

[![Open SurfGuard-USA in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ornab74/pufferforge/blob/main/examples/SurfGuard_USA_PufferForge_Colab.ipynb)

[`examples/SurfGuard_USA_PufferForge_Colab.ipynb`](examples/SurfGuard_USA_PufferForge_Colab.ipynb)
builds a research pipeline for contiguous-U.S. coastal hazard reconstruction,
calibrated risk modeling, and a PufferForge PPO alert policy. The notebook now:

- splits NOAA CO-OPS requests into service-safe 31-day windows;
- installs PufferForge with a native-first, pure-Python fallback;
- treats GRIB dependencies as optional;
- validates NDBC month/minute headers and documented missing-value sentinels;
- imports PPO APIs from the supported top-level `pufferforge` package; and
- uses configurable OpenAI Responses and speech model IDs.

Open the notebook in Colab and run the installation, configuration, and runtime self-check cells first. See `docs/SURFGUARD_COLAB.md` for failure recovery and scale controls.

The GRIB stack is skipped by default because the historical workflow does not
need it. Set `SURFGUARD_INSTALL_GRIB=1` before the installation cell only when
enabling operational GFS Wave collection. The repository keeps one clean,
output-free notebook under `examples/` so fixes cannot drift across aliases.

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
python/pufferforge/trainer.py     PPO runtime with KL guards and diagnostics
python/pufferforge/atlaslab.py    Predictive mapping research vertical
examples/                         Runnable Python and Colab workflows
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
