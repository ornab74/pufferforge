from __future__ import annotations

import math

import torch
from torch import nn


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)
    return module


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden_size: int = 128, hidden_layers: int = 2) -> None:
        super().__init__()
        if obs_size <= 0 or num_actions <= 1:
            raise ValueError("invalid observation or action dimensions")
        if hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive")
        layers: list[nn.Module] = []
        in_features = obs_size
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_features, hidden_size), nn.Tanh()])
            in_features = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        self.encoder.apply(orthogonal_init)
        orthogonal_init(self.actor, gain=0.01)
        orthogonal_init(self.critic, gain=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations.float())
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(self, observations: torch.Tensor, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), values


class RecurrentActorCritic(nn.Module):
    """GRU actor-critic with explicit episode-boundary state resets."""

    def __init__(self, obs_size: int, num_actions: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.Tanh())
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        self.encoder.apply(orthogonal_init)
        orthogonal_init(self.actor, gain=0.01)
        orthogonal_init(self.critic, gain=1.0)

    def initial_state(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward_sequence(self, observations: torch.Tensor, state: torch.Tensor, episode_starts: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observations.float())
        outputs = []
        for t in range(encoded.shape[0]):
            if episode_starts is not None:
                state = state * (~episode_starts[t].bool()).view(1, -1, 1)
            out, state = self.rnn(encoded[t : t + 1], state)
            outputs.append(out)
        features = torch.cat(outputs, dim=0)
        return self.actor(features), self.critic(features).squeeze(-1), state
