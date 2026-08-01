from __future__ import annotations

import numpy as np
import torch
from pufferforge.distributions import multidiscrete, squashed_gaussian
from pufferforge.models import RecurrentActorCritic
from pufferforge.registry import make, register
from pufferforge.replay import PrioritizedReplay
from pufferforge.selfplay import EloLeague
from pufferforge.spaces import Box, Discrete, MultiDiscrete


def test_spaces_and_distributions() -> None:
    rng = np.random.default_rng(1)
    assert Discrete(3).contains(np.array(2))
    assert MultiDiscrete((2, 3)).sample(rng, 4).shape == (4, 2)
    assert Box(-1, 1, (3,)).sample(rng).shape == (3,)
    output = multidiscrete(torch.zeros(5, 5), (2, 3))
    assert output.action.shape == (5, 2)
    continuous = squashed_gaussian(torch.zeros(5, 2), torch.zeros(5, 2))
    assert continuous.action.abs().max() <= 1


def test_recurrent_policy_resets_state() -> None:
    model = RecurrentActorCritic(3, 2, hidden_size=8)
    obs = torch.randn(4, 2, 3)
    starts = torch.tensor([[0, 0], [0, 1], [0, 0], [1, 0]], dtype=torch.bool)
    logits, values, state = model.forward_sequence(obs, model.initial_state(2, "cpu"), starts)
    assert logits.shape == (4, 2, 2)
    assert values.shape == (4, 2)
    assert state.shape == (1, 2, 8)


def test_prioritized_replay_and_league() -> None:
    replay = PrioritizedReplay(16, seed=1)
    for i in range(12):
        replay.add(obs=np.array([i], dtype=np.float32), action=np.array(i % 2))
    sample = replay.sample(8)
    assert sample.data["obs"].shape == (8, 1)
    replay.update_priorities(sample.indices, np.ones(8) * 2)

    league = EloLeague(seed=1)
    league.add("a")
    league.add("b")
    league.record("a", "b", 1.0)
    assert league.leaderboard()[0].id == "a"
    assert league.opponent("a").id == "b"


def test_registry() -> None:
    try:
        register("unit-test-env", lambda value=1: {"value": value}, value=2)
    except ValueError:
        pass
    assert make("unit-test-env", value=3) == {"value": 3}
