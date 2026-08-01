from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvSpec:
    id: str
    entry_point: str | Callable[..., Any]
    default_kwargs: dict[str, Any]


_REGISTRY: dict[str, EnvSpec] = {}


def register(id: str, entry_point: str | Callable[..., Any], **default_kwargs: Any) -> None:
    if not id or ":" in id:
        raise ValueError("environment id must be a non-empty name without ':'")
    if id in _REGISTRY:
        raise ValueError(f"environment already registered: {id}")
    _REGISTRY[id] = EnvSpec(id, entry_point, dict(default_kwargs))


def spec(id: str) -> EnvSpec:
    try:
        return _REGISTRY[id]
    except KeyError as exc:
        raise KeyError(f"unknown environment {id!r}; available={sorted(_REGISTRY)}") from exc


def make(id: str, **kwargs: Any) -> Any:
    item = spec(id)
    factory = item.entry_point
    if isinstance(factory, str):
        module_name, attr = factory.split(":", 1)
        factory = getattr(import_module(module_name), attr)
    return factory(**{**item.default_kwargs, **kwargs})


def registered() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
