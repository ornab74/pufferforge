from __future__ import annotations

import numpy as np
from pufferforge.atlaslab import (
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


def test_world_snapshot_and_hidden_rotation() -> None:
    world = AtlasWorld(AtlasWorldConfig(seed=3, control_rotation=1, shift_period=0))
    snapshot = world.snapshot()
    world.step(AtlasAction.NORTH)
    moved = world.position
    world.restore(snapshot)
    assert world.position != moved
    assert world.position == snapshot.position


def test_temporal_forecast_and_conflict_tracking() -> None:
    atlas = PredictiveAtlas(5, 5)
    grid = np.full((5, 5), int(AtlasTile.UNKNOWN), dtype=np.int8)
    grid[2, 2] = int(AtlasTile.ARTIFACT)
    atlas.observe(grid, step=1)
    grid[2, 2] = int(AtlasTile.FLOOR)
    atlas.observe(grid, step=2)
    assert atlas.conflicts[2, 2] == 1
    forecast = atlas.forecast(2)
    assert forecast.shape == (5, 5, 6)
    assert np.allclose(forecast.sum(-1), 1.0)


def test_dreamer_is_deterministic_from_snapshot() -> None:
    world = AtlasWorld(AtlasWorldConfig(seed=7, shift_period=0))
    dreamer = AtlasDreamer(horizon=2, beam_width=4)
    dreamer.reset(world)
    first = dreamer.act(world)
    snapshot = world.snapshot(); atlas = dreamer.atlas.clone()
    world.step(first)
    world.restore(snapshot); dreamer.atlas = atlas
    assert dreamer.act(world) == first


def test_map_channel_budget_and_duplicate_rejection() -> None:
    world = AtlasWorld(AtlasWorldConfig(seed=5))
    atlas = PredictiveAtlas(world.config.height, world.config.width)
    atlas.observe(world.known, step=0)
    channel = MapChannel(4096); patch = channel.make_patch(atlas, "a", 1)
    assert patch is not None and patch.byte_size <= 4096
    target = PredictiveAtlas(world.config.height, world.config.width)
    assert channel.apply(target, patch)
    evidence = target.alpha.sum()
    assert not channel.apply(target, patch)
    assert target.alpha.sum() == evidence


def test_suite_and_swarm_smoke() -> None:
    config = AtlasWorldConfig(seed=1, max_steps=30, shift_period=11)
    episode = run_atlas_episode(config, seed=1, strategy="atlas_dreamer", max_steps=8, horizon=2, beam_width=3)
    assert episode["steps"] == 8
    suite = run_atlas_suite(config, seeds=(1,), strategies=("random", "atlas_dreamer"), max_steps=6)
    assert set(suite["summaries"]) == {"random", "atlas_dreamer"}
    swarm = run_atlas_swarm(config, seed=1, agents=2, steps=6, sync_interval=3, budget_bytes=8192)
    assert swarm["sent_patches"] > 0
