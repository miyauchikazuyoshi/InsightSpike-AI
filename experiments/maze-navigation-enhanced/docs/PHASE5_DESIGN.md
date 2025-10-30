# Phase 5 Design: Episode Flush & Lazy Load / Index Scaling

## Objectives

- Prevent unbounded memory growth as episodes & vectors accumulate.
- Preserve recent/local working set in RAM while offloading cold episodes.
- Maintain query wiring behavior and parity with in-memory mode.

## Key Concepts

| Concept | Description |
|---------|-------------|
| flush_interval | Steps between eviction checks (e.g. every 200 steps) |
| max_in_memory | Hard cap on resident Episode objects (e.g. 10k) |
| cold criteria | Episodes not touched (visit_count unchanged) & outside sliding spatial window |
| vector store | Persisted weighted vectors (or raw vectors + recompute) |
| rehydrate | On-demand load of Episode metadata + vector when accessed |

## Proposed Architecture

1. Add `EpisodeStore` abstraction (backed by DataStore) for persisted records:
   - Schema: {episode_id, position, direction, base_vector(float[8]), visit_count, is_wall}
   - Optional separate collection for weighted vectors (versioning) else recompute.
2. Introduce `MemoryGuard` inside `MazeNavigator`:
   - Tracks insertion order & last access timestamps.
   - On interval, if len(episodes) > max_in_memory => select eviction batch.
3. Eviction policy (initial):
   - Score formula: `w1*recency_rank + w2*(1/(1+visit_count)) + w3*distance_from_current`
   - Keep top K by negative score (retain recent, frequent, spatially close).
4. Evicted Episodes:
   - Serialize & persist via EpisodeStore (idempotent: skip if already stored).
   - Remove from in-memory dict but retain lightweight tombstone mapping id -> persisted.
5. Lazy Load Path:
   - On observe/move when needing neighbor Episode that is missing but persisted, load & reinstate.
   - Reconstructed Episode excludes weighted cache; recomputed on first access.
6. Vector Index Interaction:
   - If using VectorIndex, keep vector even if metadata flushed OR remove and rely on index for similarity.
   - Option A (simpler initial): DO NOT remove from index (index grows; memory save from Episode structs).
   - Option B (future): maintain reference counts and remove from index when evicted if rarely contributing.

## Data Flows

```text
[Navigator] -> EpisodeManager (new episode) -> (MemoryGuard?) -> VectorIndex.add
                                              |-> if over cap -> persist + evict
                                              |-> later access -> EpisodeStore.load -> rehydrate
```

## Configuration Additions

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| flush_interval | int | 200 | Steps between eviction passes |
| max_in_memory | int | 10000 | Episode object soft cap |
| retain_recent_window | int | 250 | Steps considered "recent" for eviction weighting |
| spatial_keep_radius | int | 6 | Manhattan radius retained regardless of recency |

## Metrics

- `flush_events`: count
- `episodes_evicted_total`
- `episodes_rehydrated_total`
- `in_memory_current`
- `persisted_total`
- `rehydration_latency_ms` (avg / p95)

## Edge Cases

- Goal episode flushed: treat as pinned (never evict).
- Rapid re-access thrash: add min_residency_steps before eviction eligible.
- Weight version change: invalidate weighted cache only; persisted raw vector remains.

## Initial Implementation Steps

1. Add config params & stats fields (no effect) – feature flag `enable_flush`.
2. Implement EpisodeStore wrapper around DataStoreIndex or raw DataStore (linear JSON fallback).
3. Integrate MemoryGuard: track access & implement flush pass (skip if feature disabled).
4. Add instrumentation + stats.
5. Tests: eviction triggers after threshold; rehydration returns functional Episode; parity path length unaffected (<1% drift).

## Future Optimizations

- Batch persistence & async IO.
- Compressed vector blocks (float16 / delta encoding).
- Adaptive max_in_memory based on RSS monitor.
- Index pruning heuristics (remove very old low-degree nodes).

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Frequent load thrash | Add hysteresis (min_residency_steps) |
| Data corruption | Write temp + atomic rename |
| Performance regression | Benchmark wiring_ms before/after enabling flush |
| Index skew (stale episodes) | Periodic index consistency scan |

## Open Questions

- Should wall episodes be evicted earlier? (Likely yes: add wall_weight bonus to eviction score)
- Need separate persistence for event_log? (Phase 6 maybe)
- Multi-run reuse of persisted episodes? (Out of scope Phase 5)
