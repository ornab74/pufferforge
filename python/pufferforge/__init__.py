"""PufferForge: a clean-room Python/C++ reinforcement-learning runtime."""

from .atlaslab import (
    AtlasAction,
    AtlasDreamer,
    AtlasTile,
    AtlasWorld,
    AtlasWorldConfig,
    MapChannel,
    PredictiveAtlas,
    run_atlas_episode,
    run_atlas_suite,
    run_atlas_swarm,
)
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
    "NATIVE_AVAILABLE",
    "ActorCritic",
    "AtlasAction",
    "AtlasDreamer",
    "AtlasTile",
    "AtlasWorld",
    "AtlasWorldConfig",
    "Box",
    "ClipReward",
    "Discrete",
    "EloLeague",
    "FrameStack",
    "MapChannel",
    "MultiDiscrete",
    "NativeLineWorld",
    "NormalizeObservation",
    "PPOTrainer",
    "Player",
    "PredictiveAtlas",
    "PrioritizedReplay",
    "PythonVectorEnv",
    "RecurrentActorCritic",
    "ReplaySample",
    "StepBatch",
    "TrainConfig",
    "TrainMetrics",
    "make",
    "register",
    "registered",
    "run_atlas_episode",
    "run_atlas_suite",
    "run_atlas_swarm",
    "spec",
]

__version__ = "0.7.3"
