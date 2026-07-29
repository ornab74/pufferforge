from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .atlaslab import AtlasWorldConfig, run_atlas_suite, run_atlas_swarm, write_atlas_report
from .checkpoint import load_checkpoint
from .config import TrainConfig
from .envs import NativeLineWorld
from .models import ActorCritic
from .trainer import PPOTrainer, TrainMetrics


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--world-size", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--json-log", type=Path)


def build_config(args: argparse.Namespace) -> TrainConfig:
    if args.config:
        config = TrainConfig.from_json(args.config)
    else:
        config = TrainConfig(
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            horizon=args.horizon,
            minibatch_size=args.minibatch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            hidden_layers=args.hidden_layers,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
        )
    config.validate()
    return config


def format_metrics(metrics: TrainMetrics) -> str:
    return (
        f"update={metrics.update:4d} step={metrics.global_step:10d} "
        f"sps={metrics.steps_per_second:9.0f} return={metrics.mean_return:7.3f} "
        f"pi={metrics.policy_loss:8.4f} vf={metrics.value_loss:8.4f} "
        f"ent={metrics.entropy:7.4f} kl={metrics.approx_kl:8.5f}"
    )


def train_command(args: argparse.Namespace) -> int:
    config = build_config(args)
    env = NativeLineWorld(config.num_envs, world_size=args.world_size, max_steps=args.max_steps, seed=config.seed)
    trainer = PPOTrainer(env, config)
    log_file = None
    if args.json_log:
        args.json_log.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.json_log.open("w", encoding="utf-8")

    def on_metrics(metrics: TrainMetrics) -> None:
        print(format_metrics(metrics), flush=True)
        if log_file is not None:
            log_file.write(json.dumps(metrics.to_dict()) + "\n")
            log_file.flush()

    try:
        trainer.train(on_metrics)
    finally:
        trainer.close()
        if log_file is not None:
            log_file.close()
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    env = NativeLineWorld(args.num_envs, world_size=args.world_size, max_steps=args.max_steps, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    env.reset(args.seed)
    started = time.perf_counter()
    for _ in range(args.steps):
        env.step(rng.integers(0, env.num_actions, size=env.num_envs, dtype=np.int64))
    elapsed = time.perf_counter() - started
    transitions = args.steps * args.num_envs
    print(json.dumps({
        "transitions": transitions,
        "seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
        "num_envs": args.num_envs,
    }, indent=2))
    env.close()
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    env = NativeLineWorld(args.num_envs, world_size=args.world_size, max_steps=args.max_steps, seed=args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model = ActorCritic(env.obs_size, env.num_actions, args.hidden_size, args.hidden_layers).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()
    obs = env.reset(args.seed)
    completed = 0
    return_sum = 0.0
    while completed < args.episodes:
        with torch.inference_mode():
            logits, _ = model(torch.as_tensor(obs, device=device))
            actions = logits.argmax(dim=-1).cpu().numpy()
        result = env.step(actions)
        obs = result.observations
        stats = env.drain_stats()
        completed += int(stats.get("episodes", 0))
        return_sum += float(stats.get("return_sum", 0.0))
    print(json.dumps({"episodes": completed, "mean_return": return_sum / completed}, indent=2))
    env.close()
    return 0


def atlas_config(args: argparse.Namespace) -> AtlasWorldConfig:
    return AtlasWorldConfig(
        height=args.height,
        width=args.width,
        obstacle_density=args.obstacle_density,
        artifacts=args.artifacts,
        hazards=args.hazards,
        shift_period=args.shift_period,
        max_steps=max(args.steps, 1),
        seed=args.seed,
    )


def atlas_suite_command(args: argparse.Namespace) -> int:
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    strategies = tuple(value for value in args.strategies.split(",") if value)
    report = run_atlas_suite(atlas_config(args), seeds=seeds, strategies=strategies, max_steps=args.steps)
    write_atlas_report(report, args.output)
    print(json.dumps(report, indent=2))
    return 0


def atlas_swarm_command(args: argparse.Namespace) -> int:
    report = run_atlas_swarm(
        atlas_config(args),
        seed=args.seed,
        agents=args.agents,
        steps=args.steps,
        sync_interval=args.sync_interval,
        budget_bytes=args.bandwidth_bytes,
    )
    write_atlas_report(report, args.output)
    print(json.dumps(report, indent=2))
    return 0


def add_atlas_world_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--height", type=int, default=19)
    parser.add_argument("--width", type=int, default=19)
    parser.add_argument("--obstacle-density", type=float, default=0.24)
    parser.add_argument("--artifacts", type=int, default=5)
    parser.add_argument("--hazards", type=int, default=8)
    parser.add_argument("--shift-period", type=int, default=31)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pufferforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train PPO on the native LineWorld environment")
    add_train_args(train_parser)
    train_parser.set_defaults(func=train_command)

    bench_parser = subparsers.add_parser("bench", help="benchmark native vector stepping")
    bench_parser.add_argument("--num-envs", type=int, default=4096)
    bench_parser.add_argument("--steps", type=int, default=2000)
    bench_parser.add_argument("--world-size", type=int, default=15)
    bench_parser.add_argument("--max-steps", type=int, default=64)
    bench_parser.add_argument("--seed", type=int, default=1)
    bench_parser.set_defaults(func=benchmark_command)

    eval_parser = subparsers.add_parser("eval", help="evaluate a checkpoint")
    eval_parser.add_argument("checkpoint", type=Path)
    eval_parser.add_argument("--episodes", type=int, default=1024)
    eval_parser.add_argument("--num-envs", type=int, default=256)
    eval_parser.add_argument("--world-size", type=int, default=15)
    eval_parser.add_argument("--max-steps", type=int, default=64)
    eval_parser.add_argument("--seed", type=int, default=2)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--hidden-size", type=int, default=128)
    eval_parser.add_argument("--hidden-layers", type=int, default=2)
    eval_parser.set_defaults(func=evaluate_command)

    suite_parser = subparsers.add_parser("atlas-suite", help="run paired predictive mapping simulations")
    add_atlas_world_args(suite_parser)
    suite_parser.add_argument("--seeds", default="1,2,3")
    suite_parser.add_argument("--strategies", default="random,frontier,atlas_dreamer")
    suite_parser.add_argument("--output", type=Path, default=Path("atlaslab/suite.json"))
    suite_parser.set_defaults(func=atlas_suite_command)

    swarm_parser = subparsers.add_parser("atlas-swarm", help="run bandwidth-limited multi-agent cartography")
    add_atlas_world_args(swarm_parser)
    swarm_parser.add_argument("--agents", type=int, default=3)
    swarm_parser.add_argument("--sync-interval", type=int, default=8)
    swarm_parser.add_argument("--bandwidth-bytes", type=int, default=32768)
    swarm_parser.add_argument("--output", type=Path, default=Path("atlaslab/swarm.json"))
    swarm_parser.set_defaults(func=atlas_swarm_command)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
