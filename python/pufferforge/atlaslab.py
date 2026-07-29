from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


class AtlasTile(IntEnum):
    UNKNOWN = -1
    WALL = 0
    FLOOR = 1
    ARTIFACT = 2
    HAZARD = 3
    BEACON = 4
    SPRING = 5


class AtlasAction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    SCAN = 4
    INTERACT = 5


_MOVES = np.asarray(((-1, 0), (0, 1), (1, 0), (0, -1)), dtype=np.int16)
_CLASSES = tuple(int(tile) for tile in AtlasTile if tile is not AtlasTile.UNKNOWN)
_INDEX = {tile: index for index, tile in enumerate(_CLASSES)}


@dataclass(frozen=True, slots=True)
class AtlasWorldConfig:
    name: str = "causal_vault"
    height: int = 19
    width: int = 19
    obstacle_density: float = 0.24
    artifacts: int = 5
    hazards: int = 8
    beacons: int = 3
    springs: int = 2
    view_radius: int = 2
    scan_radius: int = 5
    max_steps: int = 240
    control_rotation: int = 1
    shift_period: int = 31
    seed: int = 1

    def validate(self) -> None:
        if self.height < 9 or self.width < 9 or self.height % 2 == 0 or self.width % 2 == 0:
            raise ValueError("height and width must be odd and at least 9")
        if not 0.0 <= self.obstacle_density < 0.6:
            raise ValueError("obstacle_density must be in [0, 0.6)")
        if self.control_rotation not in range(4):
            raise ValueError("control_rotation must be 0..3")
        if self.max_steps <= 0 or self.view_radius < 1 or self.scan_radius < self.view_radius:
            raise ValueError("invalid step or observation radius")

    def mutated(self, **changes: object) -> "AtlasWorldConfig":
        config = replace(self, **changes)
        config.validate()
        return config

    @property
    def stable_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()[:16]


@dataclass(slots=True)
class AtlasWorldSnapshot:
    terrain: np.ndarray
    known: np.ndarray
    visited: np.ndarray
    position: tuple[int, int]
    energy: float
    step: int
    score: float
    artifacts: int
    rotation: int
    rng_state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AtlasTransition:
    reward: float
    done: bool
    event: str


class AtlasWorld:
    """Partially observed world with hidden controls and non-stationary landmarks."""

    def __init__(self, config: AtlasWorldConfig | None = None) -> None:
        self.config = config or AtlasWorldConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.terrain = np.zeros((self.config.height, self.config.width), dtype=np.int8)
        self.known = np.full_like(self.terrain, int(AtlasTile.UNKNOWN))
        self.visited = np.zeros_like(self.terrain, dtype=np.uint8)
        self.position = (self.config.height // 2, self.config.width // 2)
        self.energy = 1.0
        self.step_count = 0
        self.score = 0.0
        self.artifacts_collected = 0
        self.rotation = self.config.control_rotation
        self._generate()
        self.reset(self.config.seed)

    def _generate(self) -> None:
        c = self.config
        rng = np.random.default_rng(c.seed)
        terrain = np.full((c.height, c.width), int(AtlasTile.FLOOR), dtype=np.int8)
        terrain[[0, -1], :] = int(AtlasTile.WALL)
        terrain[:, [0, -1]] = int(AtlasTile.WALL)
        noise = rng.random((c.height, c.width))
        for _ in range(3):
            noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                     + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5.0
        interior = noise[1:-1, 1:-1]
        terrain[1:-1, 1:-1][interior <= np.quantile(interior, c.obstacle_density)] = int(AtlasTile.WALL)
        center = (c.height // 2, c.width // 2)
        terrain[center[0], 1:-1] = int(AtlasTile.FLOOR)
        terrain[1:-1, center[1]] = int(AtlasTile.FLOOR)
        for row in range(1, c.height - 1, 4):
            terrain[row, 1:-1] = int(AtlasTile.FLOOR)
        for col in range(1, c.width - 1, 4):
            terrain[1:-1, col] = int(AtlasTile.FLOOR)
        cells = np.argwhere(terrain == int(AtlasTile.FLOOR))
        rng.shuffle(cells)
        cursor = 0
        for tile, count in ((AtlasTile.ARTIFACT, c.artifacts), (AtlasTile.HAZARD, c.hazards),
                            (AtlasTile.BEACON, c.beacons), (AtlasTile.SPRING, c.springs)):
            placed = 0
            while cursor < len(cells) and placed < count:
                cell = tuple(map(int, cells[cursor])); cursor += 1
                if abs(cell[0] - center[0]) + abs(cell[1] - center[1]) < 3:
                    continue
                terrain[cell] = int(tile); placed += 1
        terrain[center] = int(AtlasTile.FLOOR)
        self.terrain = terrain

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None and seed != self.config.seed:
            self.config = self.config.mutated(seed=int(seed)); self._generate()
        self.rng = np.random.default_rng(self.config.seed)
        self.known.fill(int(AtlasTile.UNKNOWN)); self.visited.fill(0)
        self.position = (self.config.height // 2, self.config.width // 2)
        self.energy = 1.0; self.step_count = 0; self.score = 0.0; self.artifacts_collected = 0
        self.rotation = self.config.control_rotation
        self._reveal(self.config.view_radius); self.visited[self.position] = 1
        return self.observation()

    def snapshot(self) -> AtlasWorldSnapshot:
        return AtlasWorldSnapshot(self.terrain.copy(), self.known.copy(), self.visited.copy(),
            self.position, self.energy, self.step_count, self.score, self.artifacts_collected,
            self.rotation, dict(self.rng.bit_generator.state))

    def restore(self, snapshot: AtlasWorldSnapshot) -> None:
        self.terrain = snapshot.terrain.copy(); self.known = snapshot.known.copy()
        self.visited = snapshot.visited.copy(); self.position = tuple(snapshot.position)
        self.energy = float(snapshot.energy); self.step_count = int(snapshot.step)
        self.score = float(snapshot.score); self.artifacts_collected = int(snapshot.artifacts)
        self.rotation = int(snapshot.rotation); self.rng = np.random.default_rng()
        self.rng.bit_generator.state = dict(snapshot.rng_state)

    def clone(self) -> "AtlasWorld":
        world = object.__new__(AtlasWorld); world.config = self.config; world.restore(self.snapshot()); return world

    @property
    def coverage(self) -> float:
        return float(self.visited.sum() / max(1, np.count_nonzero(self.terrain != int(AtlasTile.WALL))))

    @property
    def discovery(self) -> float:
        return float(np.count_nonzero(self.known != int(AtlasTile.UNKNOWN)) / self.known.size)

    def _reveal(self, radius: int) -> None:
        row, col = self.position
        for rr in range(max(0, row - radius), min(self.config.height, row + radius + 1)):
            for cc in range(max(0, col - radius), min(self.config.width, col + radius + 1)):
                if abs(rr - row) + abs(cc - col) <= radius:
                    self.known[rr, cc] = self.terrain[rr, cc]

    def _shift(self) -> None:
        movable = np.argwhere(np.isin(self.terrain, (int(AtlasTile.ARTIFACT), int(AtlasTile.HAZARD))))
        free = np.argwhere(self.terrain == int(AtlasTile.FLOOR))
        if not len(movable) or not len(free): return
        source = tuple(map(int, movable[int(self.rng.integers(len(movable)))]))
        target = tuple(map(int, free[int(self.rng.integers(len(free)))]))
        self.terrain[target], self.terrain[source] = self.terrain[source], int(AtlasTile.FLOOR)
        self.rotation = (self.rotation + 1) % 4

    def step(self, action: int | AtlasAction) -> AtlasTransition:
        action = AtlasAction(int(action)); reward = -0.002; event = "idle"
        if action.value < 4:
            delta = _MOVES[(action.value + self.rotation) % 4]
            target = self.position[0] + int(delta[0]), self.position[1] + int(delta[1])
            if self.terrain[target] != int(AtlasTile.WALL):
                self.position = target; self.visited[target] = 1; event = "move"; reward += 0.01
                tile = AtlasTile(int(self.terrain[target]))
                if tile is AtlasTile.HAZARD: self.energy -= 0.15; reward -= 0.2; event = "hazard"
                if tile is AtlasTile.SPRING: self.energy = min(1.0, self.energy + 0.35); event = "spring"
            else: reward -= 0.02; event = "blocked"
            self.energy -= 0.004
        elif action is AtlasAction.SCAN:
            self._reveal(self.config.scan_radius); self.energy -= 0.015; reward += 0.02; event = "scan"
        elif action is AtlasAction.INTERACT:
            if self.terrain[self.position] == int(AtlasTile.ARTIFACT):
                self.terrain[self.position] = int(AtlasTile.FLOOR); self.artifacts_collected += 1
                reward += 1.0; event = "artifact"
            elif self.terrain[self.position] == int(AtlasTile.BEACON):
                reward += 0.1; event = "beacon"
        self.step_count += 1; self._reveal(self.config.view_radius)
        if self.config.shift_period and self.step_count % self.config.shift_period == 0:
            self._shift(); event = "world_shift"
        self.score += reward
        return AtlasTransition(float(reward), self.energy <= 0 or self.step_count >= self.config.max_steps, event)

    def observation(self) -> np.ndarray:
        return np.concatenate((self.known.reshape(-1).astype(np.float32), np.asarray(
            (*self.position, self.energy, self.step_count / self.config.max_steps), dtype=np.float32)))


@dataclass(frozen=True, slots=True)
class AtlasMetrics:
    known_fraction: float
    confidence: float
    entropy: float
    volatility: float
    causal_confidence: float
    conflicts: int


class PredictiveAtlas:
    """Semantic, temporal, and causal map learned only from visible observations."""

    def __init__(self, height: int, width: int, *, prior: float = 0.2) -> None:
        self.height = int(height); self.width = int(width); self.prior = float(prior)
        classes = len(_CLASSES)
        self.alpha = np.full((height, width, classes), prior, dtype=np.float32)
        self.transitions = np.full((height, width, classes, classes), 0.1, dtype=np.float32)
        self.last_labels = np.full((height, width), int(AtlasTile.UNKNOWN), dtype=np.int8)
        self.last_seen = np.full((height, width), -1, dtype=np.int32)
        self.conflicts = np.zeros((height, width), dtype=np.int32)
        self.effects: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.step = 0

    def clone(self) -> "PredictiveAtlas":
        other = PredictiveAtlas(self.height, self.width, prior=self.prior)
        for name in ("alpha", "transitions", "last_labels", "last_seen", "conflicts"):
            setattr(other, name, getattr(self, name).copy())
        other.effects = {key: list(values) for key, values in self.effects.items()}; other.step = self.step
        return other

    def probabilities(self) -> np.ndarray:
        return self.alpha / np.maximum(self.alpha.sum(-1, keepdims=True), 1e-8)

    def labels(self) -> np.ndarray:
        labels = np.asarray(_CLASSES, dtype=np.int8)[self.alpha.argmax(-1)]
        labels[self.alpha.sum(-1) <= self.prior * len(_CLASSES) + 0.5] = int(AtlasTile.UNKNOWN)
        return labels

    def entropy(self) -> np.ndarray:
        p = self.probabilities(); return -np.sum(p * np.log(np.maximum(p, 1e-8)), axis=-1) / np.log(len(_CLASSES))

    def volatility(self) -> np.ndarray:
        p = self.transitions / np.maximum(self.transitions.sum(-1, keepdims=True), 1e-8)
        diag = np.diagonal(p, axis1=-2, axis2=-1)
        out = 1.0 - diag.mean(-1); out[self.last_seen < 0] = 0.0; return out.astype(np.float32)

    def observe(self, known: np.ndarray, *, step: int) -> int:
        previous = self.labels(); changes = 0
        for row, col in np.argwhere(known != int(AtlasTile.UNKNOWN)):
            row, col = int(row), int(col); tile = int(known[row, col]); old = int(self.last_labels[row, col])
            if old in _INDEX:
                self.transitions[row, col, _INDEX[old], _INDEX[tile]] += 1.0
            if previous[row, col] != int(AtlasTile.UNKNOWN) and previous[row, col] != tile:
                self.conflicts[row, col] += 1; changes += 1
            self.alpha[row, col, _INDEX[tile]] += 4.0
            self.last_labels[row, col] = tile; self.last_seen[row, col] = int(step)
        self.step = max(self.step, int(step)); return changes

    def observe_action(self, action: AtlasAction, before: tuple[int, int], after: tuple[int, int]) -> None:
        key = (0, int(action)); self.effects.setdefault(key, []).append((after[0] - before[0], after[1] - before[1]))

    def causal_confidence(self) -> float:
        values = []
        for action in range(4):
            samples = self.effects.get((0, action), [])
            if not samples: continue
            _, counts = np.unique(np.asarray(samples), axis=0, return_counts=True); values.append(counts.max() / counts.sum())
        return float(np.mean(values)) if values else 0.0

    def best_command(self, desired: tuple[int, int]) -> AtlasAction:
        choices = []
        for action in range(4):
            samples = self.effects.get((0, action), [])
            delta = np.mean(samples, axis=0) if samples else _MOVES[action]
            error = float(np.sum((delta - np.asarray(desired)) ** 2)); choices.append((error, AtlasAction(action)))
        return min(choices, key=lambda item: item[0])[1]

    def forecast(self, horizon: int = 1) -> np.ndarray:
        result = self.probabilities(); transition = self.transitions / np.maximum(self.transitions.sum(-1, keepdims=True), 1e-8)
        for _ in range(max(0, int(horizon))): result = np.einsum("hwc,hwcd->hwd", result, transition)
        return result

    def stale_priority(self) -> np.ndarray:
        age = np.maximum(0, self.step - self.last_seen).astype(np.float32)
        scale = max(1.0, float(np.percentile(age, 90)))
        score = self.entropy() + 1.5 * self.volatility() + 0.4 * age / scale
        score[self.last_seen < 0] = -np.inf
        return score

    def metrics(self) -> AtlasMetrics:
        evidence = self.alpha.sum(-1) - self.prior * len(_CLASSES); known = evidence > 0.5
        return AtlasMetrics(float(known.mean()), float(self.probabilities().max(-1)[known].mean()) if known.any() else 0.0,
            float(self.entropy()[known].mean()) if known.any() else 1.0,
            float(self.volatility()[known].mean()) if known.any() else 0.0,
            self.causal_confidence(), int(self.conflicts.sum()))


class AtlasDreamer:
    """Counterfactual beam planner carrying world and atlas state through branches."""

    name = "atlas_dreamer"

    def __init__(self, horizon: int = 3, beam_width: int = 8) -> None:
        self.horizon = int(horizon); self.beam_width = int(beam_width); self.atlas: PredictiveAtlas | None = None

    def reset(self, world: AtlasWorld) -> None:
        self.atlas = PredictiveAtlas(world.config.height, world.config.width); self.atlas.observe(world.known, step=0)

    @staticmethod
    def _value(world: AtlasWorld, atlas: PredictiveAtlas) -> float:
        metrics = atlas.metrics()
        return (world.score + 2.5 * world.coverage + 1.5 * world.discovery + 0.5 * world.artifacts_collected
            + 0.5 * metrics.causal_confidence + 0.3 * metrics.confidence - 0.1 * metrics.conflicts)

    def act(self, world: AtlasWorld) -> AtlasAction:
        if self.atlas is None: self.reset(world)
        assert self.atlas is not None
        beam = [(world.clone(), self.atlas.clone(), (), self._value(world, self.atlas))]
        for _ in range(self.horizon):
            candidates = []
            for branch_world, branch_atlas, sequence, _score in beam:
                for action in AtlasAction:
                    next_world = branch_world.clone(); next_atlas = branch_atlas.clone(); before = next_world.position
                    transition = next_world.step(action); next_atlas.observe_action(action, before, next_world.position)
                    next_atlas.observe(next_world.known, step=next_world.step_count)
                    candidates.append((next_world, next_atlas, sequence + (action,), self._value(next_world, next_atlas) + transition.reward))
            candidates.sort(key=lambda item: item[3], reverse=True); beam = candidates[:self.beam_width]
        return beam[0][2][0] if beam else AtlasAction.SCAN

    def update(self, world: AtlasWorld, action: AtlasAction, before: tuple[int, int]) -> None:
        assert self.atlas is not None
        self.atlas.observe_action(action, before, world.position); self.atlas.observe(world.known, step=world.step_count)


@dataclass(frozen=True, slots=True)
class MapPatch:
    source: str
    sequence: int
    rows: np.ndarray
    cols: np.ndarray
    evidence: np.ndarray

    @property
    def byte_size(self) -> int:
        return int(self.rows.nbytes + self.cols.nbytes + self.evidence.nbytes)


class MapChannel:
    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = int(budget_bytes); self.used = 0; self.sent = 0; self.dropped = 0; self.last_sequence: dict[str, int] = {}

    def make_patch(self, atlas: PredictiveAtlas, source: str, sequence: int, limit: int = 128) -> MapPatch | None:
        strength = np.maximum(0.0, atlas.alpha - atlas.prior).sum(-1) * (1.0 + atlas.entropy() + atlas.volatility())
        cells = np.argwhere(strength > 0); cells = cells[np.argsort(strength[cells[:, 0], cells[:, 1]])[::-1][:limit]]
        if not len(cells): return None
        patch = MapPatch(source, sequence, cells[:, 0].astype(np.int16), cells[:, 1].astype(np.int16),
            np.maximum(0.0, atlas.alpha[cells[:, 0], cells[:, 1]] - atlas.prior).astype(np.float16))
        if self.used + patch.byte_size > self.budget_bytes: self.dropped += 1; return None
        self.used += patch.byte_size; self.sent += 1; return patch

    def apply(self, atlas: PredictiveAtlas, patch: MapPatch) -> bool:
        if patch.sequence <= self.last_sequence.get(patch.source, -1): return False
        atlas.alpha[patch.rows, patch.cols] += patch.evidence.astype(np.float32)
        self.last_sequence[patch.source] = patch.sequence; return True


@dataclass(frozen=True, slots=True)
class FrontierTask:
    id: int
    cell: tuple[int, int]
    information: float
    risk: float


class FrontierAuction:
    def assign(self, tasks: Iterable[FrontierTask], agents: Mapping[str, tuple[tuple[int, int], float]]) -> dict[str, FrontierTask]:
        remaining = list(tasks); assignments = {}
        for agent, (position, energy) in sorted(agents.items()):
            if not remaining: break
            def bid(task: FrontierTask) -> float:
                travel = abs(position[0] - task.cell[0]) + abs(position[1] - task.cell[1])
                return task.information - 0.12 * travel - task.risk + 0.2 * energy
            chosen = max(remaining, key=bid); assignments[agent] = chosen; remaining.remove(chosen)
        return assignments


def _frontiers(atlas: PredictiveAtlas) -> list[FrontierTask]:
    labels = atlas.labels(); entropy = atlas.entropy(); tasks = []
    known = labels != int(AtlasTile.UNKNOWN)
    for row, col in np.argwhere(known & (labels != int(AtlasTile.WALL))):
        if any(0 <= row + dr < atlas.height and 0 <= col + dc < atlas.width
               and not known[row + dr, col + dc] for dr, dc in _MOVES):
            tasks.append(FrontierTask(len(tasks), (int(row), int(col)), float(entropy[row, col] + 1.0),
                float(atlas.probabilities()[row, col, _INDEX[int(AtlasTile.HAZARD)]])))
    return tasks


def run_atlas_episode(config: AtlasWorldConfig, *, seed: int, strategy: str = "atlas_dreamer",
                      max_steps: int | None = None, horizon: int = 3, beam_width: int = 8) -> dict[str, float | int | str]:
    world = AtlasWorld(config.mutated(seed=seed)); world.reset(seed)
    atlas = PredictiveAtlas(config.height, config.width); atlas.observe(world.known, step=0)
    dreamer = AtlasDreamer(horizon, beam_width); dreamer.atlas = atlas
    rng = np.random.default_rng(seed)
    for _ in range(min(max_steps or config.max_steps, config.max_steps)):
        if strategy == "random": action = AtlasAction(int(rng.integers(0, len(AtlasAction))))
        elif strategy == "frontier":
            tasks = _frontiers(atlas)
            if tasks:
                target = max(tasks, key=lambda task: task.information - 0.05 * (abs(world.position[0] - task.cell[0]) + abs(world.position[1] - task.cell[1])))
                delta = np.asarray(target.cell) - np.asarray(world.position)
                desired = (int(np.sign(delta[0])), 0) if abs(delta[0]) >= abs(delta[1]) else (0, int(np.sign(delta[1])))
                action = atlas.best_command(desired)
            else: action = AtlasAction.SCAN
        else: action = dreamer.act(world)
        before = world.position; transition = world.step(action)
        atlas.observe_action(action, before, world.position); atlas.observe(world.known, step=world.step_count)
        if transition.done: break
    metrics = atlas.metrics(); predicted = atlas.forecast(1)
    truth = np.eye(len(_CLASSES), dtype=np.float32)[np.clip(world.terrain, 0, len(_CLASSES) - 1)]
    brier = float(np.mean(np.sum((predicted - truth) ** 2, axis=-1)))
    return {"scenario": config.name, "strategy": strategy, "seed": seed, "score": world.score,
        "coverage": world.coverage, "discovery": world.discovery, "artifacts": world.artifacts_collected,
        "steps": world.step_count, "known_fraction": metrics.known_fraction, "confidence": metrics.confidence,
        "volatility": metrics.volatility, "causal_confidence": metrics.causal_confidence,
        "conflicts": metrics.conflicts, "forecast_brier": brier}


def run_atlas_suite(config: AtlasWorldConfig, *, seeds: Iterable[int] = (1, 2, 3),
                    strategies: Iterable[str] = ("random", "frontier", "atlas_dreamer"),
                    max_steps: int | None = None) -> dict[str, object]:
    episodes = [run_atlas_episode(config, seed=int(seed), strategy=strategy, max_steps=max_steps)
                for seed in seeds for strategy in strategies]
    summaries = {}
    for strategy in strategies:
        rows = [row for row in episodes if row["strategy"] == strategy]
        summaries[strategy] = {key: float(np.mean([float(row[key]) for row in rows]))
            for key in ("score", "coverage", "discovery", "confidence", "forecast_brier")}
    return {"config": asdict(config), "episodes": episodes, "summaries": summaries}


def run_atlas_swarm(config: AtlasWorldConfig, *, seed: int = 1, agents: int = 3,
                    steps: int = 80, sync_interval: int = 8, budget_bytes: int = 32768) -> dict[str, object]:
    worlds = [AtlasWorld(config.mutated(seed=seed)) for _ in range(agents)]
    atlases = [PredictiveAtlas(config.height, config.width) for _ in range(agents)]
    dreamers = [AtlasDreamer(2, 4) for _ in range(agents)]
    shared = PredictiveAtlas(config.height, config.width); channel = MapChannel(budget_bytes); auction = FrontierAuction()
    assignments = []
    for world, atlas, dreamer in zip(worlds, atlases, dreamers):
        world.reset(seed); atlas.observe(world.known, step=0); dreamer.atlas = atlas
    for step in range(steps):
        for world, atlas, dreamer in zip(worlds, atlases, dreamers):
            action = dreamer.act(world); before = world.position; transition = world.step(action)
            atlas.observe_action(action, before, world.position); atlas.observe(world.known, step=world.step_count)
            if transition.done: continue
        if (step + 1) % sync_interval == 0:
            channel.used = 0
            for index, atlas in enumerate(atlases):
                patch = channel.make_patch(atlas, str(index), step + 1)
                if patch is not None: channel.apply(shared, patch)
            tasks = _frontiers(shared)
            round_assignments = auction.assign(tasks, {str(i): (world.position, world.energy) for i, world in enumerate(worlds)})
            assignments.extend({"step": step + 1, "agent": agent, "task": task.id, "cell": task.cell}
                for agent, task in round_assignments.items())
    return {"agents": agents, "sent_patches": channel.sent, "dropped_patches": channel.dropped,
        "shared_known_fraction": shared.metrics().known_fraction,
        "mean_local_known_fraction": float(np.mean([atlas.metrics().known_fraction for atlas in atlases])),
        "assignments": assignments}


def write_atlas_report(report: Mapping[str, object], path: str | Path) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); return target


__all__ = [name for name in globals() if name.startswith("Atlas") or name.startswith("Predictive")
    or name.startswith("Map") or name.startswith("Frontier") or name.startswith("run_atlas")
    or name == "write_atlas_report"]
