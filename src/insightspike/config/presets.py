"""
Configuration Presets
===================

Pre-defined configuration sets for common use cases.
"""

from typing import Any, Dict

from .models import (
    EmbeddingConfig,
    GraphConfig,
    InsightSpikeConfig,
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    MonitoringConfig,
)


class ConfigPresets:
    """Configuration presets for different environments and use cases"""

    @staticmethod
    def development() -> InsightSpikeConfig:
        """Development preset - safe mode, mock LLM, fast iteration"""
        return InsightSpikeConfig(
            environment="development",
            llm=LLMConfig(
                provider="mock",
                model="mock",
                temperature=0.3,
                max_tokens=256,
                prompt_style="standard",
                max_context_docs=5,
            ),
            memory=MemoryConfig(
                episodic_memory_capacity=60,
                max_retrieved_docs=10,
                similarity_threshold=0.3,
            ),
            embedding=EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
            ),
            graph=GraphConfig(
                spike_ged_threshold=-0.5,
                spike_ig_threshold=0.2,
                similarity_threshold=0.3,
                use_gnn=False,
                ged_algorithm="advanced",
                ig_algorithm="advanced",
            ),
            monitoring=MonitoringConfig(
                enabled=False,
                performance_tracking=False,
            ),
            logging=LoggingConfig(
                level="DEBUG",
                # Repo-relative by default; error_handler further resolves safely
                file_path="results/logs",
            ),
        )

    @staticmethod
    def experiment() -> InsightSpikeConfig:
        """Experiment preset - local LLM, moderate settings"""
        return InsightSpikeConfig(
            environment="experiment",
            llm=LLMConfig(
                provider="local",
                model="distilgpt2",
                temperature=0.7,
                max_tokens=512,
                prompt_style="minimal",
                use_simple_prompt=True,
                max_context_docs=3,
            ),
            memory=MemoryConfig(
                episodic_memory_capacity=100,
                max_retrieved_docs=15,
                similarity_threshold=0.35,
            ),
            embedding=EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
            ),
            graph=GraphConfig(
                spike_ged_threshold=-0.4,
                spike_ig_threshold=0.25,
                similarity_threshold=0.35,
                use_gnn=False,
                ged_algorithm="advanced",
                ig_algorithm="advanced",
            ),
            monitoring=MonitoringConfig(
                enabled=True,
                performance_tracking=False,
            ),
            logging=LoggingConfig(
                level="INFO",
                file_path="results/logs",
            ),
        )

    @staticmethod
    def production() -> InsightSpikeConfig:
        """Production preset - optimized settings, real LLM"""
        return InsightSpikeConfig(
            environment="production",
            llm=LLMConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=1024,
                api_key=None,  # Should be set via environment variable
                prompt_style="standard",
                max_context_docs=7,
            ),
            memory=MemoryConfig(
                episodic_memory_capacity=200,
                max_retrieved_docs=20,
                similarity_threshold=0.4,
            ),
            embedding=EmbeddingConfig(
                model_name="sentence-transformers/all-mpnet-base-v2",
                dimension=768,
            ),
            graph=GraphConfig(
                spike_ged_threshold=-0.3,
                spike_ig_threshold=0.3,
                similarity_threshold=0.4,
                use_gnn=True,
                ged_algorithm="advanced",
                ig_algorithm="advanced",
            ),
            monitoring=MonitoringConfig(
                enabled=True,
                performance_tracking=True,
                metrics_port=9090,
            ),
            logging=LoggingConfig(
                level="WARNING",
                file_path="/var/log/insightspike",
                max_size_mb=100,
                backup_count=5,
            ),
        )

    @staticmethod
    def research() -> InsightSpikeConfig:
        """Research preset - full features, detailed logging"""
        return InsightSpikeConfig(
            environment="research",
            llm=LLMConfig(
                provider="anthropic",
                model="claude-2",
                temperature=0.5,
                max_tokens=2048,
                api_key=None,  # Should be set via environment variable
                prompt_style="detailed",
                max_context_docs=10,
                include_metadata=True,
            ),
            memory=MemoryConfig(
                episodic_memory_capacity=500,
                max_retrieved_docs=30,
                similarity_threshold=0.25,
            ),
            embedding=EmbeddingConfig(
                model_name="sentence-transformers/all-mpnet-base-v2",
                dimension=768,
            ),
            graph=GraphConfig(
                spike_ged_threshold=-0.6,
                spike_ig_threshold=0.15,
                similarity_threshold=0.25,
                use_gnn=True,
                ged_algorithm="advanced",
                ig_algorithm="advanced",
            ),
            monitoring=MonitoringConfig(
                enabled=True,
                performance_tracking=True,
                detailed_tracing=True,
            ),
            logging=LoggingConfig(
                level="DEBUG",
                file_path="results/logs",
                log_to_console=True,
            ),
        )

    @staticmethod
    def cloud() -> InsightSpikeConfig:
        """Cloud/testing preset — ultra‑light, mock LLM, conservative logging.

        Notes:
        - Uses mock LLM to avoid network.
        - Disables GNN and heavy graph options.
        - Keeps small memory sizes and safe thresholds.
        - Environment is set to 'testing' to match allowed values.
        """
        return InsightSpikeConfig(
            pre_warm_models=False,
            environment="testing",
            llm=LLMConfig(
                provider="mock",
                model="mock",
                temperature=0.2,
                max_tokens=128,
                prompt_style="standard",
                max_context_docs=3,
            ),
            memory=MemoryConfig(
                episodic_memory_capacity=40,
                max_retrieved_docs=8,
                similarity_threshold=0.3,
            ),
            embedding=EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
            ),
            graph=GraphConfig(
                spike_ged_threshold=-0.5,
                spike_ig_threshold=0.2,
                similarity_threshold=0.3,
                use_gnn=False,
                ged_algorithm="advanced",
                ig_algorithm="advanced",
            ),
            monitoring=MonitoringConfig(
                enabled=False,
                performance_tracking=False,
            ),
            logging=LoggingConfig(
                level="WARNING",
                file_path="logs",
                log_to_console=True,
            ),
        )

    @staticmethod
    def paper() -> InsightSpikeConfig:
        """Paper-aligned preset (query-centric, linkset IG, candidate-base GED).

        Intended for parity between experiment evaluator and Layer3 via adapter.
        """
        cfg = ConfigPresets.development()
        # Graph/paper controls
        cfg.graph.sp_scope_mode = "union"
        cfg.graph.sp_eval_mode = "fixed_before_pairs"
        cfg.graph.ged_norm_scheme = "candidate_base"
        cfg.graph.ig_source_mode = "linkset"
        cfg.graph.linkset_query_weight = 1.0
        cfg.graph.lambda_weight = 1.0
        cfg.graph.sp_beta = 1.0
        # Metrics controls
        cfg.metrics.ig_denominator = "fixed_kstar"
        cfg.metrics.use_local_normalization = True
        return cfg
    
    @staticmethod
    def production_optimized() -> InsightSpikeConfig:
        """Production preset with Layer1 bypass enabled for high performance"""
        config = ConfigPresets.production()
        # Enable Layer1 bypass for production optimization
        config.processing.enable_layer1_bypass = True
        config.processing.bypass_uncertainty_threshold = 0.2
        config.processing.bypass_known_ratio_threshold = 0.9
        return config
    
    @staticmethod
    def minimal() -> InsightSpikeConfig:
        """Minimal preset with advanced features disabled"""
        config = ConfigPresets.development()
        # Disable advanced features
        config.processing.enable_layer1_bypass = False
        config.processing.enable_insight_registration = False
        config.processing.enable_insight_search = False
        config.graph.use_gnn = False
        config.graph.enable_graph_search = False
        return config
    
    @staticmethod
    def graph_enhanced() -> InsightSpikeConfig:
        """Preset with graph-based features enabled"""
        config = ConfigPresets.experiment()
        # Enable graph features
        config.graph.use_gnn = True
        config.graph.enable_graph_search = True
        config.graph.hop_limit = 2
        config.graph.neighbor_threshold = 0.35
        config.graph.path_decay = 0.7
        # Also enable insights for better reasoning
        config.processing.enable_insight_registration = True
        config.processing.enable_insight_search = True
        return config
    
    @staticmethod
    def adaptive_learning() -> InsightSpikeConfig:
        """Preset with adaptive learning enabled"""
        config = ConfigPresets.graph_enhanced()
        # Enable learning
        config.processing.enable_learning = True
        config.processing.learning_rate = 0.1
        config.processing.exploration_rate = 0.15
        # Enable all advanced features for maximum learning
        config.processing.enable_layer1_bypass = True
        config.processing.enable_insight_registration = True
        config.processing.enable_insight_search = True
        config.graph.enable_graph_search = True
        return config

    @staticmethod
    def get_preset(name: str) -> Dict[str, Any]:
        """Get preset by name (for backward compatibility)"""
        presets = {
            "development": ConfigPresets.development(),
            "experiment": ConfigPresets.experiment(),
            "production": ConfigPresets.production(),
            "research": ConfigPresets.research(),
            "cloud": ConfigPresets.cloud(),
            "paper": ConfigPresets.paper(),
            # Alias for CLI convenience; testing → cloud (ultra-light)
            "testing": ConfigPresets.cloud(),
            "production_optimized": ConfigPresets.production_optimized(),
            "minimal": ConfigPresets.minimal(),
            "graph_enhanced": ConfigPresets.graph_enhanced(),
            "adaptive_learning": ConfigPresets.adaptive_learning(),
        }

        if name not in presets:
            raise ValueError(
                f"Unknown preset: {name}. Available: {list(presets.keys())}"
            )

        # Return as dictionary for backward compatibility
        return presets[name].dict()

    # Legacy dictionary format for backward compatibility
    DEVELOPMENT = {
        "core": {
            "model_name": "paraphrase-MiniLM-L6-v2",
            "llm_provider": "mock",
            "llm_model": "mock",
            "max_tokens": 256,
            "temperature": 0.3,
            "device": "cpu",
            "use_gpu": False,
            "safe_mode": True,
        },
        "memory": {
            "max_retrieved_docs": 10,
            "short_term_capacity": 5,
            "working_memory_capacity": 10,
            "episodic_memory_capacity": 30,
            "pattern_cache_capacity": 10,
        },
        "retrieval": {
            "similarity_threshold": 0.3,
            "top_k": 10,
            "layer1_top_k": 15,
            "layer2_top_k": 10,
            "layer3_top_k": 8,
        },
        "graph": {
            "spike_ged_threshold": 0.4,
            "spike_ig_threshold": 0.15,
            "use_gnn": False,
            "ged_algorithm": "advanced",
            "ig_algorithm": "advanced",
        },
        "processing": {"batch_size": 16, "max_workers": 2, "timeout_seconds": 60},
        "output": {
            "verbose": True,
            "save_results": False,
            "generate_visualizations": False,
        },
        "environment": "local",
    }

    PRODUCTION = {
        "core": {
            "model_name": "paraphrase-MiniLM-L6-v2",
            "llm_provider": "local",
            "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "max_tokens": 512,
            "temperature": 0.3,
            "device": "cpu",
            "use_gpu": False,
            "safe_mode": False,
        },
        "memory": {
            "max_retrieved_docs": 20,
            "short_term_capacity": 10,
            "working_memory_capacity": 30,
            "episodic_memory_capacity": 100,
            "pattern_cache_capacity": 20,
        },
        "retrieval": {
            "similarity_threshold": 0.35,
            "top_k": 15,
            "layer1_top_k": 20,
            "layer2_top_k": 15,
            "layer3_top_k": 12,
        },
        "graph": {
            "spike_ged_threshold": 0.5,
            "spike_ig_threshold": 0.2,
            "use_gnn": False,
            "ged_algorithm": "advanced",
            "ig_algorithm": "advanced",
            "hybrid_weights": {"structure": 0.4, "semantic": 0.4, "quality": 0.2},
        },
        "reasoning": {
            "episode_merge_threshold": 0.85,
            "episode_split_threshold": 0.25,
            "episode_prune_threshold": 0.05,
        },
        "processing": {"batch_size": 32, "max_workers": 4, "timeout_seconds": 300},
        "output": {
            "verbose": False,
            "save_results": True,
            "generate_visualizations": True,
            "results_dir": "production_results",
        },
        "environment": "production",
        "monitoring": {
            "enable_telemetry": True,
            "enable_performance_tracking": True,
            "log_level": "INFO",
        },
    }

    EXPERIMENT = {
        "core": {
            "model_name": "paraphrase-MiniLM-L6-v2",
            "llm_provider": "local",
            "llm_model": "distilgpt2",
            "max_tokens": 512,
            "temperature": 0.7,
            "device": "cpu",
            "use_gpu": False,
            "safe_mode": False,
        },
        "memory": {
            "max_retrieved_docs": 15,
            "short_term_capacity": 8,
            "working_memory_capacity": 20,
            "episodic_memory_capacity": 50,
            "pattern_cache_capacity": 15,
        },
        "retrieval": {
            "similarity_threshold": 0.25,
            "top_k": 12,
            "layer1_top_k": 18,
            "layer2_top_k": 12,
            "layer3_top_k": 10,
        },
        "graph": {
            "spike_ged_threshold": 0.45,
            "spike_ig_threshold": 0.18,
            "use_gnn": False,
            "ged_algorithm": "advanced",
            "ig_algorithm": "advanced",
        },
        "processing": {"batch_size": 24, "max_workers": 3, "timeout_seconds": 180},
        "output": {
            "verbose": True,
            "save_results": True,
            "generate_visualizations": True,
            "results_dir": "experiment_results",
        },
        "environment": "experiment",
        "experimental": {"enable_feature_x": True, "feature_x_threshold": 0.5},
    }

    RESEARCH = {
        "core": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "llm_provider": "local",
            "llm_model": "EleutherAI/gpt-neo-1.3B",
            "max_tokens": 1024,
            "temperature": 0.5,
            "device": "cuda"
            if False
            else "cpu",  # Would check torch.cuda.is_available()
            "use_gpu": False,
            "safe_mode": False,
        },
        "memory": {
            "max_retrieved_docs": 30,
            "short_term_capacity": 15,
            "working_memory_capacity": 40,
            "episodic_memory_capacity": 200,
            "pattern_cache_capacity": 30,
        },
        "retrieval": {
            "similarity_threshold": 0.2,
            "top_k": 20,
            "layer1_top_k": 30,
            "layer2_top_k": 20,
            "layer3_top_k": 15,
        },
        "graph": {
            "spike_ged_threshold": 0.35,
            "spike_ig_threshold": 0.12,
            "use_gnn": True,
            "ged_algorithm": "advanced",
            "ig_algorithm": "advanced",
            "gnn_hidden_dim": 256,
            "gnn_num_layers": 3,
        },
        "reasoning": {
            "episode_merge_threshold": 0.8,
            "episode_split_threshold": 0.3,
            "episode_prune_threshold": 0.1,
            "enable_meta_reasoning": True,
            "meta_reasoning_depth": 3,
        },
        "processing": {
            "batch_size": 64,
            "max_workers": 8,
            "timeout_seconds": 600,
            "enable_gpu_acceleration": False,
        },
        "output": {
            "verbose": True,
            "save_results": True,
            "generate_visualizations": True,
            "results_dir": "research_results",
            "save_intermediate_states": True,
            "export_formats": ["json", "csv", "parquet"],
        },
        "environment": "research",
        "monitoring": {
            "enable_telemetry": True,
            "enable_performance_tracking": True,
            "enable_detailed_tracing": True,
            "log_level": "DEBUG",
            "profile_memory": True,
            "profile_cpu": True,
        },
    }
