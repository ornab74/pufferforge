# SurfGuard-USA Colab validation and recovery

The notebook is a research workflow, not an operational NOAA/NWS warning product.

## Recommended run order

1. Run **Runtime installation**.
2. Run **Configuration and storage**.
3. Run **Runtime self-check** and confirm `pufferforge_available=True`.
4. Run the collection and training cells in order.
5. Leave `RUN_OPERATIONAL_GFSWAVE`, `RUN_GPT_WARNINGS`, `RUN_AUDIO`, and `RUN_GPT_INCIDENT_CURATION` disabled until the base pipeline succeeds.

## Scale controls

Use the last complete historical calendar year by default. Large national runs can generate many NOAA requests and substantial local or Drive storage.

## PufferForge installation

The repaired installer attempts the native C++ extension first. If compilation fails, retry with:

```bash
PUFFERFORGE_BUILD_NATIVE=0 python -m pip install -e . --no-build-isolation
```

PPO with `PythonVectorEnv` remains available in fallback mode. Native LineWorld benchmarks require the compiled extension.

## NOAA CO-OPS

CO-OPS retrievals must be split into inclusive windows of at most 31 days. Do not replace the notebook's chunking helper with one annual request.

## NDBC

Historical headers distinguish uppercase `MM` (month) from lowercase `mm` (minute). The repaired parser preserves this distinction and converts documented all-9 missing sentinels to `NaN`.

## Optional GFS Wave support

`cfgrib` and `eccodes` are optional. If either fails to install, keep `RUN_OPERATIONAL_GFSWAVE=False`; the historical tide and supervised pipeline still runs.

## OpenAI calls

Use configurable API model identifiers rather than hard-coding an account-specific alias:

```python
import os
os.environ["SURFGUARD_OPENAI_MODEL"] = "gpt-5"
os.environ["SURFGUARD_OPENAI_TTS_MODEL"] = "gpt-4o-mini-tts"
```

OpenAI sections are opt-in and require `OPENAI_API_KEY`.

## Repaired notebook validation

The repaired notebook was checked for valid notebook JSON, Python syntax in every code cell, NOAA request chunking, NDBC missing-value parsing, and compatibility with the public PufferForge PPO imports.
