"""PufferForge: a clean-room Python/C++ reinforcement-learning runtime."""

from .config import TrainConfig
from .envs import NativeLineWorld, PythonVectorEnv, StepBatch
from .models import ActorCritic, RecurrentActorCritic
from .registry import make, register, registered, spec
from .replay import PrioritizedReplay, ReplaySample
from .selfplay import EloLeague, Player
from .spaces import Box, Discrete, MultiDiscrete
from .trainer import PPOTrainer, TrainMetrics
from .wrappers import ClipReward, FrameStack, NormalizeObservation

try:
    from . import _core
    NATIVE_AVAILABLE = True
except ImportError:
    _core = None
    NATIVE_AVAILABLE = False

__all__ = [
    "ActorCritic", "Box", "ClipReward", "Discrete", "EloLeague", "FrameStack",
    "MultiDiscrete", "NATIVE_AVAILABLE", "NativeLineWorld", "NormalizeObservation",
    "PPOTrainer", "Player", "PrioritizedReplay", "PythonVectorEnv",
    "RecurrentActorCritic", "ReplaySample", "StepBatch", "TrainConfig",
    "TrainMetrics", "make", "register", "registered", "spec",
]

__version__ = "0.2.0"
