from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device: torch.device
    requested: str
    automatic: bool
    accelerator: str
    name: str
    cuda_devices: int
    cuda_index: int | None = None
    capability: tuple[int, int] | None = None
    total_memory_bytes: int | None = None
    cudnn_version: int | None = None
    deterministic: bool = False
    tf32: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["device"] = str(self.device)
        return payload


def select_device(
    requested: str = "auto",
    *,
    deterministic: bool = False,
    allow_tf32: bool = True,
) -> DeviceInfo:
    """Select and configure a CUDA or CPU Torch device.

    ``auto`` chooses the current CUDA device when CUDA is usable and otherwise
    returns CPU. Explicit CUDA requests fail early with a useful error.
    """
    normalized = requested.strip().lower()
    if not normalized:
        raise ValueError("device selector cannot be empty")
    automatic = normalized == "auto"
    if automatic:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        try:
            device = torch.device(normalized)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"invalid Torch device selector: {requested!r}") from exc

    if device.type != "cuda":
        if device.type != "cpu":
            raise ValueError("PufferForge currently supports only CPU and CUDA devices")
        return DeviceInfo(
            device=device,
            requested=requested,
            automatic=automatic,
            accelerator="cpu",
            name="CPU",
            cuda_devices=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            deterministic=deterministic,
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but torch.cuda.is_available() is False"
        )
    count = torch.cuda.device_count()
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= count:
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; detected {count} device(s)"
        )
    device = torch.device("cuda", index)
    properties = torch.cuda.get_device_properties(index)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    tf32 = bool(allow_tf32 and not deterministic)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = tf32

    return DeviceInfo(
        device=device,
        requested=requested,
        automatic=automatic,
        accelerator="cuda",
        name=properties.name,
        cuda_devices=count,
        cuda_index=index,
        capability=(properties.major, properties.minor),
        total_memory_bytes=properties.total_memory,
        cudnn_version=torch.backends.cudnn.version(),
        deterministic=deterministic,
        tf32=tf32,
    )
