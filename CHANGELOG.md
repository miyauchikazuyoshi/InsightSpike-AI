# Changelog

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

### Updated
- README documentation with new command names
- CLI guide with updated examples
- Legacy CLI help text to show new commands

## [0.8.1] - 2025-07-10

### Added
- New improved CLI (`spike`) with better user experience
  - Interactive chat mode (`spike chat`)
  - Configuration management (`spike config`)
  - Command aliases (q, c, l)
  - Interactive demo (`spike demo`)
  - Insights display and search (`spike insights`, `spike insights-search`)
  - Progress indicators and rich output

- Simplified configuration system
  - `SimpleConfig` class with flat structure
  - 5 presets: development, testing, production, experiment, cloud
  - Environment variable overrides (INSIGHTSPIKE_* prefix)
  - JSON save/load functionality

- Comprehensive error handling and logging
  - Custom exception hierarchy
  - Unified logging to ~/.insightspike/logs/
  - Error decorators for consistent handling
  - User-friendly error messages

- Unified LLM Provider system
  - Single interface for all LLM providers
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