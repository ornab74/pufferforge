from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class ActionOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


def categorical(logits: torch.Tensor, action: torch.Tensor | None = None) -> ActionOutput:
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample() if action is None else action.long()
    return ActionOutput(action, dist.log_prob(action), dist.entropy())


def multidiscrete(logits: torch.Tensor, nvec: tuple[int, ...], action: torch.Tensor | None = None) -> ActionOutput:
    if logits.shape[-1] != sum(nvec):
        raise ValueError("logit width does not match nvec")
    chunks = torch.split(logits, nvec, dim=-1)
    outputs = [categorical(chunk, None if action is None else action[..., i]) for i, chunk in enumerate(chunks)]
    return ActionOutput(
        torch.stack([o.action for o in outputs], dim=-1),
        torch.stack([o.log_prob for o in outputs], dim=-1).sum(-1),
        torch.stack([o.entropy for o in outputs], dim=-1).sum(-1),
    )


def squashed_gaussian(mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor | None = None, epsilon: float = 1e-6) -> ActionOutput:
    std = log_std.clamp(-20.0, 2.0).exp()
    dist = torch.distributions.Normal(mean, std)
    if action is None:
        raw = dist.rsample()
        action = torch.tanh(raw)
    else:
        action = action.clamp(-1 + epsilon, 1 - epsilon)
        raw = torch.atanh(action)
    correction = torch.log(1 - action.square() + epsilon)
    return ActionOutput(action, (dist.log_prob(raw) - correction).sum(-1), dist.entropy().sum(-1))
