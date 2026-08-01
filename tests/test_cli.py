from __future__ import annotations

from pufferforge.cli import build_config, make_parser
from pufferforge.config import TrainConfig


def test_runtime_device_flags_override_json_config(tmp_path) -> None:
    path = tmp_path / "train.json"
    TrainConfig(
        num_envs=8,
        horizon=8,
        minibatch_size=32,
        device="cuda:1",
        cuda_deterministic=False,
        cuda_allow_tf32=True,
    ).save_json(path)
    args = make_parser().parse_args(
        [
            "train",
            "--config",
            str(path),
            "--device",
            "cpu",
            "--cuda-deterministic",
            "--no-cuda-tf32",
        ]
    )

    config = build_config(args)
    assert config.device == "cpu"
    assert config.cuda_deterministic
    assert not config.cuda_allow_tf32


def test_json_device_settings_survive_without_cli_override(tmp_path) -> None:
    path = tmp_path / "train.json"
    TrainConfig(
        num_envs=8,
        horizon=8,
        minibatch_size=32,
        device="cpu",
        cuda_deterministic=True,
        cuda_allow_tf32=False,
    ).save_json(path)
    args = make_parser().parse_args(["train", "--config", str(path)])

    config = build_config(args)
    assert config.device == "cpu"
    assert config.cuda_deterministic
    assert not config.cuda_allow_tf32
