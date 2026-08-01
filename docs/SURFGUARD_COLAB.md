# SurfGuard-USA Colab validation and recovery

The notebook is a research workflow, not an operational NOAA/NWS warning product.

## Recommended run order

1. Run **Runtime installation**.
2. Run **Configuration and storage**.
3. Run **Runtime self-check** and confirm `pufferforge_available=True`.
4. Run the collection and training cells in order.
5. Leave `RUN_OPERATIONAL_GFSWAVE`, `RUN_GPT_WARNINGS`, `RUN_AUDIO`, and `RUN_GPT_INCIDENT_CURATION` disabled until the base pipeline succeeds.

## CUDA selector

The notebook uses `select_device("auto")`. In a Colab GPU runtime it selects the
current CUDA device and runs policy inference and PPO optimization there; in a
standard runtime it falls back to CPU automatically. The runtime self-check
prints the selected device, GPU name, compute capability, memory, cuDNN version,
determinism, and TF32 status.

Choose **Runtime > Change runtime type > GPU** before installation when CUDA is
desired. No notebook edit is required. The PPO configuration keeps
`device="auto"`; explicit selectors such as `cuda:0` are also supported.

## Scale controls

The default demonstration uses the last complete historical calendar year and a limited station-year workload. Override without editing the notebook:

```python
import os
os.environ["SURFGUARD_DEMO_MAX_STATION_YEARS"] = "24"
os.environ["SURFGUARD_HISTORICAL_END_YEAR"] = "2025"
```

Set `FULL_SCALE=True` only after validating storage and NOAA request volume.

## PufferForge installation

The installer first upgrades `pip`, `setuptools`, `wheel`, and `pybind11`, then attempts the native C++ extension using the runtime build toolchain:

```bash
python -m pip install --upgrade "pip>=24" "setuptools>=61" wheel "pybind11>=2.13"
python -m pip install --no-build-isolation --editable .
```

If compilation fails, it retries with:

```bash
PUFFERFORGE_BUILD_NATIVE=0 python -m pip install --no-build-isolation --editable .
```

Outside the notebook, packaging defaults to automatic native discovery and
falls back cleanly when pybind11 headers are absent. The notebook uses explicit
`1` then `0` modes so its status report can say which runtime was installed.

The notebook no longer marks the second packaging attempt as fatal. It inserts the repository's `python/` directory into `sys.path` as a final deterministic source-tree fallback, so PPO with `PythonVectorEnv` remains available even if editable-package metadata fails. Native LineWorld benchmarks still require the compiled extension. Failed commands print their complete trailing pip output rather than only a generic `CalledProcessError`.

## NOAA CO-OPS

CO-OPS retrievals are split into inclusive windows of at most 31 days. Do not replace the chunking helper with one annual request.

## NDBC

Historical headers distinguish uppercase `MM` (month) from lowercase `mm` (minute). The parser normalizes the minute field to `MN` before uppercasing other fields. Historical missing values encoded with 9s are converted to `NaN`.

## Optional GFS Wave support

`cfgrib` and `eccodes` are optional and are no longer installed by default. To
use operational GFS Wave collection, set the following before running the
installation cell, then set `RUN_OPERATIONAL_GFSWAVE=True` later:

```python
import os
os.environ["SURFGUARD_INSTALL_GRIB"] = "1"
```

If either dependency fails to install, keep `RUN_OPERATIONAL_GFSWAVE=False`;
the historical tide and supervised pipeline still runs.

## OpenAI calls

The notebook defaults to `gpt-5` for Responses API text and `gpt-4o-mini-tts` for speech. Override them with:

```python
os.environ["SURFGUARD_OPENAI_MODEL"] = "your-accessible-model-id"
os.environ["SURFGUARD_OPENAI_TTS_MODEL"] = "your-accessible-tts-model-id"
```

OpenAI sections are opt-in and require `OPENAI_API_KEY`.

## Repeated SimpleImputer warnings or NaN ROC/PR metrics

If tide observations or NDBC history are unavailable for the selected station-years,
whole feature columns can contain only missing values. The repaired notebook now:

- records those columns as `dropped_all_missing_features`;
- trains only on features with at least one observed training value;
- preserves the requested and active feature lists in the model bundle/model card;
- uses incident-group temporal splitting so one dangerous event cannot leak across
  train, validation, and test;
- guarantees both classes in validation/test when at least three independent
  dangerous episodes are available;
- emits JSON-safe `null` metrics rather than serialized `NaN` if a ranking metric is
  genuinely undefined.

The PPO import must use the public package surface:

```python
from pufferforge import PPOTrainer, PythonVectorEnv, TrainConfig
```

Do not use `from pufferforge.pufferforge ...`; that submodule does not exist.

## Compact example notebook

The single committed notebook lives at
`examples/SurfGuard_USA_PufferForge_Colab.ipynb`. Its execution counts, outputs,
widget state, and stale Colab metadata are removed, and every code cell is
compiled by the test suite. Keeping one authoritative example prevents repaired
and output variants from drifting apart.
