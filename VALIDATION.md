# Validation record

Validated in the build container with Python 3.13, NumPy 2.3.5, PyTorch 2.10.0
CPU, GCC 14.2, CMake 3.31.6, and OpenMP enabled.

## Tests

```text
4 passed in 1.51s
```

Coverage includes:

- native vector environment shapes and autoreset
- native C++ GAE equivalence to a NumPy reference
- Python vector-environment autoreset
- end-to-end PPO rollout, update, and checkpoint smoke test

## Native stepping benchmark

Command:

```bash
PYTHONPATH=python python -m pufferforge bench --num-envs 1024 --steps 300
```

Observed result:

```json
{
  "transitions": 307200,
  "seconds": 0.005628579000017453,
  "transitions_per_second": 54578606.78495362,
  "num_envs": 1024
}
```

This is a microbenchmark of the included LineWorld step loop, not a claim about
performance on complex simulations or other hardware.

## PPO smoke run

A 1,024-transition CPU run completed two updates and wrote a checkpoint. Final
reported throughput was approximately 34,706 environment transitions per second.
