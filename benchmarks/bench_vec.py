from __future__ import annotations

import argparse
import time
import numpy as np

from pufferforge import NativeLineWorld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    env = NativeLineWorld(args.num_envs)
    rng = np.random.default_rng(1)
    env.reset(1)
    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(rng.integers(0, 3, size=args.num_envs, dtype=np.int64))
    elapsed = time.perf_counter() - start
    transitions = args.num_envs * args.steps
    print(f"{transitions / elapsed:,.0f} transitions/s ({transitions:,} in {elapsed:.3f}s)")


if __name__ == "__main__":
    main()
