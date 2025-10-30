# Changelog

## [Unreleased]

### Added

- Public API entrypoint `insightspike.public` exposing `create_agent` and `quick_demo` for stable external usage.
- Lightweight query transformation visualizer stub `visualization/query_transform_viz.py` with `animate_transformation` and `snapshot`.
- Unit tests covering public API import, L3 current graph exposure, and visualization snapshot.
- geDIG `ab` モード (pure + full の並列計算) と相関モニタリング (`GeDIGABLogger`)。Pearson 相関しきい値監視 (デフォルト 0.85, min_pairs=30) と CSV エクスポート (ヘッダ保証) を追加。
- Config 正規化レイヤ Phase4 スキャフォールド: `legacy/compat_config.py` に `detect_config_type` / `normalize()`; dict 入力を最小安全フィールドで `InsightSpikeConfig` 化。
- ドキュメント: `docs/architecture/config_normalization_overview.md` 作成 (フロー図 / 目的 / FAQ)。

### Changed

- L3 Graph Reasoner now exposes the latest analyzed graph via `L3GraphReasoner.current_graph` to support Query-as-Node workflows.
- ConfigurableAgent aligns `QueryTransformer(use_gnn)` with L3 `graph.use_gnn` setting for consistent GNN behavior.
- CI: Non-selector `compute_gedig` calls are now enforced in STRICT mode in `.github/workflows/test.yml` (set `STRICT_GEDIG_SELECTOR=1`).
- MainAgent の geDIG モード分岐を単一ファサード (`full|pure|ab`) へ整理し分岐密度削減。
- A/B CSV エクスポート `export_gedig_ab_csv` でヘッダのみ状態でも 1 行とカウントしテスト安定化。

### Technical Debt / Planned

- MainAgent 受領時の dict 自動 `normalize()` 適用 (次リリース予定)。
- `GeDIGABLogger` auto-flush と threshold WARN のユニットテスト追加。


### Removed

- GeDIG experimental adapter (maze experiments): deprecated hop1 approximation fallback path removed. `allow_hop1_approx` parameter now inert; synthetic hop1 values are no longer generated.
- Experimental GeDIG 'legacy' scoring mode removed from maze experiments adapter (`GeDIGEvaluator`). Use `mode='core_raw'`. Threshold scaling logic simplified; attempts to pass `mode='legacy'` now raise.

### Changed

- Multihop evaluation now reports only actual collected hops; missing hops surfaced via `multihop_missing` without approximation.
- Updated tests (`test_multihop_no_fallback`) to assert absence of hop1 approximation flags.

### Docs
- README: Added Quick Start with `insightspike.public`, Query Transform usage, and CI selector policy notes.
- Updated plans: UNUSED_LOGIC_REFACTORING_PLAN_2025_09.md and CODE_ENHANCEMENT_PROPOSAL.md to reflect current implementation status (selector guard, Query-as-Node, visualization stub).


## [0.9.0] - 2024-07-24

### Added
- **Layer1 Bypass Mechanism** - Fast-path processing for known concepts with low uncertainty
  - 10x performance improvement for production systems
  - Configurable uncertainty thresholds
  - New `production_optimized` preset with bypass enabled
  
- **Insight Auto-Registration and Search** - Automatic capture and reuse of discovered insights
  - Insights automatically extracted when spikes detected
  - Quality evaluation with multiple criteria
  - Persistent storage in SQLite database
  - Integration with memory search for future queries
  
- **Mode-Aware Prompt Building** - Dynamic prompt sizing based on model capabilities
  - Minimal mode for small models (DistilGPT2, TinyLlama)
  - Standard mode for medium models (GPT-3.5)
  - Detailed mode for large models (GPT-4, Claude)
  - Prevents token limit issues and optimizes response quality
  
- **Graph-Based Memory Search** - Multi-hop graph traversal for associative retrieval
  - 2-hop neighbor exploration with configurable limits
  - Path-based relevance scoring with exponential decay
  - Enables "associative leaps" between concepts
  - New `graph_enhanced` preset with full graph features

### Changed
- Enhanced Layer1 to calculate cacheability and suggest processing paths
- Layer2 memory search now includes relevant insights from registry
- Layer3 auto-registers high-quality insights when spikes detected
- Layer4 selects appropriate prompt mode based on model capabilities
- Updated configuration system with new processing options

### Fixed
- Method name mismatches in MainAgent (merge, prune, split operations)
- Missing `split_episode()` implementation in L2MemoryManager
- Graph features from GNN now properly used in reasoning quality calculation
- DataStoreAgent now returns actual responses, not just reasoning
- Working memory properly loads episode content from DataStore

### Documentation
- Added comprehensive feature documentation in `docs/architecture/recent_features_2024_07.md`
- Created quick start guide in `docs/user-guide/july_2024_features_quickstart.md`
- Updated architecture documentation with new features
- Updated geDIG implementation status report

## [0.8.2] - 2025-07-10

### Changed
- Renamed CLI commands to follow industry standards:
  - `ask` → `query` (standard term for RAG/database systems)
  - `learn` → `embed` (technically accurate for document vectorization)
- Updated command aliases:
  - `q` → `query` (was `ask`)
  - `e` → `embed` (new alias)
  - `l` → `embed` (was `learn`)
  - Legacy commands still work for backward compatibility:
    - `spike ask` redirects to `spike query`
    - `spike learn` redirects to `spike embed`
  - Consistent error handling across providers
  - Better streaming support

### Changed
- Legacy CLI (`insightspike`) commands renamed:
  - `ask` → `legacy-ask` (with deprecation warning)
  - `stats` → `legacy-stats` (with deprecation warning)
  - Removed 13 redundant/experimental commands

- README documentation updated with:
  - New CLI commands and examples
  - Simplified troubleshooting section
  - Clear distinction between new and legacy CLIs

### Removed
- Experimental commands from legacy CLI:
  - `embed`, `query` (replaced by new commands)
  - `experiment`, `benchmark`, `experiment_suite` (use scripts directly)
  - `insight_experiment`, `compare_experiments` (use scripts directly)
  - `insights_validate`, `insights_cleanup` (maintenance commands)
  - `test_safe` (development command)

### Fixed
- Configuration compatibility issues
- CLI command discovery in Poetry environment
- Test suite reliability (39/39 tests passing)

## [0.8.0] - Previous release
- Initial production-ready release with 4-layer architecture
- Full InsightSpike system implementation
