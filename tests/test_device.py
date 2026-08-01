from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pufferforge.device import select_device


def test_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = select_device("auto")

    assert info.device == torch.device("cpu")
    assert info.accelerator == "cpu"
    assert info.automatic
    assert info.to_dict()["device"] == "cpu"


def test_explicit_cuda_fails_early_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="is_available"):
        select_device("cuda:0")


def test_auto_cuda_selects_current_gpu_and_reports_hardware(monkeypatch) -> None:
    properties = SimpleNamespace(
        name="Test Accelerator",
        major=9,
        minor=0,
        total_memory=24 * 1024**3,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)

    info = select_device("auto", deterministic=True, allow_tf32=True)

    assert info.device == torch.device("cuda:1")
    assert info.cuda_devices == 2
    assert info.cuda_index == 1
    assert info.capability == (9, 0)
    assert info.total_memory_bytes == 24 * 1024**3
    assert info.deterministic
    assert not info.tf32


def test_cuda_selector_validates_device_index(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="detected 1 device"):
        select_device("cuda:3")
