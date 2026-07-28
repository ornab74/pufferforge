from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable


@dataclass(slots=True)
class Player:
    id: str
    rating: float = 1000.0
    games: int = 0
    metadata: dict[str, str | int | float] = field(default_factory=dict)


class EloLeague:
    def __init__(self, k_factor: float = 32.0, seed: int = 1) -> None:
        self.k_factor = k_factor
        self.players: dict[str, Player] = {}
        self.rng = random.Random(seed)

    def add(self, player_id: str, rating: float = 1000.0, **metadata: str | int | float) -> Player:
        if player_id in self.players:
            raise ValueError(f"duplicate player: {player_id}")
        player = Player(player_id, rating, metadata=metadata)
        self.players[player_id] = player
        return player

    @staticmethod
    def expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (b - a) / 400.0))

    def record(self, a_id: str, b_id: str, score_a: float) -> None:
        if score_a not in (0.0, 0.5, 1.0):
            raise ValueError("score must be 0, 0.5, or 1")
        a, b = self.players[a_id], self.players[b_id]
        delta = self.k_factor * (score_a - self.expected(a.rating, b.rating))
        a.rating += delta
        b.rating -= delta
        a.games += 1
        b.games += 1

    def opponent(self, player_id: str, temperature: float = 200.0, candidates: Iterable[str] | None = None) -> Player:
        player = self.players[player_id]
        ids = [i for i in (candidates or self.players) if i != player_id]
        if not ids:
            raise ValueError("no opponents available")
        weights = [math.exp(-abs(self.players[i].rating - player.rating) / max(temperature, 1e-6)) for i in ids]
        return self.players[self.rng.choices(ids, weights=weights, k=1)[0]]

    def leaderboard(self) -> list[Player]:
        return sorted(self.players.values(), key=lambda p: (-p.rating, p.id))
