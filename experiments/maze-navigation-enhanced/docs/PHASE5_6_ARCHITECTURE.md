# Phase 5 & 6 Architecture Overview

## Scope

- Phase 5: Memory control (flush/evict + lazy rehydrate + persistence catalog + compaction)
- Phase 6: ANN acceleration (HNSW) & dynamic linear->ANN upgrade

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| MazeNavigator | navigation/maze_navigator.py | Orchestration, eviction scoring, ANN upgrade, metrics |
| Eviction Catalog | evicted_catalog.jsonl | Append-only JSONL of evicted episode metadata (LRU bounded) |
| Vector Index Abstraction | indexes/vector_index.py | Uniform API (add/search/remove/len) |
| HNSWLibIndex | indexes/hnsw_index.py | ANN backend wrapper (hnswlib) |
| Benchmark Script | experiments/ann_benchmark.py | Latency & recall comparison linear vs ANN |

## Eviction Flow

1. Periodic guard `_memory_guard_pass()` at `flush_interval` steps.
2. If `max_in_memory_positions` configured and exceeded -> position-group scoring eviction pass.
3. Then episode-level cap enforcement if still above `max_in_memory`.
4. Each evicted episode: metadata appended to in-memory LRU + JSONL file (if persistence_dir set).
5. Catalog size trimmed to `evicted_catalog_max` (LRU).

### Scoring

```text
score = 0.6 * recency_rank + 1.2 * inverse_visit + 0.2 * manhattan_distance
inverse_visit = 1/(1+visit_count)
```

Anchors (start/current/goal) get huge protective bias at position-level.

## Lazy Rehydrate

Triggered when current position is stepped into. For each catalog entry with matching position and direction not already live:

- Reconstruct episode vector using VectorProcessor to keep distribution consistent.
- Reinsert into EpisodeManager and vector index (if not wall, best-effort).
- Count & event log (`rehydration`).

## Catalog Compaction

Optional final step (`catalog_compaction_on_close`): rewrite JSONL with current LRU map to eliminate tombstoned / superseded lines; logs `catalog_compact` with before/after bytes.

## Metrics & Stats Additions

| Key | Description |
|-----|-------------|
| flush_events | Number of eviction passes that removed episodes |
| episodes_evicted_total | Total evicted episodes |
| episodes_rehydrated_total | Total rehydrated episodes |
| evicted_catalog_size | Current LRU map size |
| evicted_catalog_bytes | File size after last compaction (if performed) |
| timing.flush_ms / rehydration_ms | Latency distributions |
| ann_backend | `None` or `hnsw` |
| ann_init_error | Initialization failure message (if any) |
| ann_upgrade_threshold | Configured threshold for auto-upgrade |
| ann_index_elements | Current ANN index population |

## ANN Dynamic Upgrade

When using linear `InMemoryIndex` (and no explicit ANN backend) and index length exceeds `ann_upgrade_threshold`:

1. Construct HNSW index with capacity `max(2000, current*2)`.
2. Best-effort migration of existing vectors (if internal storage attribute available).
3. Swap backend & log `ann_upgrade`.
4. Failures logged as `ann_upgrade_failed` without stopping navigation.

## Benchmarking (ann_benchmark.py)

- Generates synthetic normalized vectors.
- Computes brute-force reference neighbor sets for overlap metrics.
- Measures per-query search latency (mean, p95) for linear vs ANN.
- Reports Jaccard & hit-rate vs reference neighbors.

## Failure & Robustness Notes

- All persistence and ANN initialization wrapped in try/except to avoid fatal run termination.
- Import absence (hnswlib) gracefully degrades (logs warning + stays linear).
- Catalog loading skips malformed lines; continues best-effort.

## Future Improvements (Backlog)

- Periodic background compaction (size threshold trigger).
- Adaptive `ann_upgrade_threshold` based on observed wiring_ms latency slope.
- Recall quality tracking in live navigation (sample queries comparing linear fallback vs ANN for drift detection).
- Configurable eviction scoring weights via config file / CLI.

---
Generated: auto doc snapshot.
