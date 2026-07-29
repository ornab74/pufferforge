# AtlasLab: predictive mapping simulations

AtlasLab is a self-contained mapping research vertical inside PufferForge. It adds a
procedural partially observed world with hidden rotated controls and periodically
moving landmarks, plus a predictive atlas learned only from visible observations.

The atlas combines:

- Dirichlet semantic beliefs and calibrated uncertainty;
- per-cell semantic transition matrices for future-map forecasts;
- stale-region and volatility scores;
- empirical command-to-motion causal models;
- counterfactual beam planning over cloned world and atlas states;
- sparse source/sequence map packets under a hard byte budget;
- frontier auctions for decentralized multi-agent mapping.

## Paired strategy suite

```bash
pufferforge atlas-suite \
  --seeds 1,2,3,4,5 \
  --strategies random,frontier,atlas_dreamer \
  --steps 100 \
  --output atlaslab/suite.json
```

## Coordinated swarm

```bash
pufferforge atlas-swarm \
  --agents 4 \
  --steps 120 \
  --sync-interval 8 \
  --bandwidth-bytes 32768 \
  --output atlaslab/swarm.json
```

The report separates task score, coverage, discovery, confidence, causal mastery,
conflicts, temporal volatility, and forecast Brier score.
