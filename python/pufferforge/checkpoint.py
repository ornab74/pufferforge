from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

CHECKPOINT_VERSION = 2


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    update: int,
    config: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "update": update,
            "config": config,
        },
        temp,
    )
    temp.replace(output)
    return output


def read_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    required = {"model", "global_step", "update", "config"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {sorted(missing)}")
    version = int(payload.get("format_version", 1))
    if version > CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint format {version} is newer than supported format "
            f"{CHECKPOINT_VERSION}"
        )
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = read_checkpoint(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
