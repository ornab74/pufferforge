"""PufferForge: a clean-room Python/C++ reinforcement-learning runtime."""

from .config import TrainConfig
from .envs import NativeLineWorld, PythonVectorEnv, StepBatch
from .models import ActorCritic
from .trainer import PPOTrainer, TrainMetrics

try:
    from . import _core
    NATIVE_AVAILABLE = True
except ImportError:
    _core = None
    NATIVE_AVAILABLE = False

__all__ = [
    "ActorCritic",
    "NATIVE_AVAILABLE",
    "NativeLineWorld",
    "PPOTrainer",
    "PythonVectorEnv",
    "StepBatch",
    "TrainConfig",
    "TrainMetrics",
]

__version__ = "0.1.0"
