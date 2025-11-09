"""
Config Normalizer - Unified configuration handling
"""

import logging
from typing import Union, Dict, Any, Optional
from pydantic import BaseModel

from .models import (
    InsightSpikeConfig, LLMConfig, MemoryConfig, EmbeddingConfig,
    GraphConfig, ProcessingConfig
)

logger = logging.getLogger(__name__)


class ConfigNormalizer:
    """
    Normalizes various configuration formats to standard Pydantic models.
    
    This ensures consistent configuration handling throughout the system.
    """
    
    @staticmethod
    def normalize(config: Union[Dict[str, Any], InsightSpikeConfig, BaseModel]) -> InsightSpikeConfig:
        """
        Normalize any configuration format to InsightSpikeConfig.
        
        Args:
            config: Configuration in dict or Pydantic format
            
        Returns:
            InsightSpikeConfig: Normalized configuration
        """
        if isinstance(config, InsightSpikeConfig):
            return config
        
        if isinstance(config, dict):
            return ConfigNormalizer._dict_to_config(config)
        
        if isinstance(config, BaseModel):
            # Other Pydantic model, convert to dict first
            return ConfigNormalizer._dict_to_config(config.dict())
        
        raise TypeError(f"Unsupported config type: {type(config)}")
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> InsightSpikeConfig:
        """
        Convert dictionary to InsightSpikeConfig.
        
        Handles legacy formats and missing sections gracefully.
        """
        # Extract sections with defaults
        llm_dict = config_dict.get('llm', {})
        memory_dict = config_dict.get('memory', {})
        embedding_dict = config_dict.get('embedding', {})
        graph_dict = config_dict.get('graph', {})
        processing_dict = config_dict.get('processing', {})
        
        # Create sub-configs
        llm_config = ConfigNormalizer._create_llm_config(llm_dict)
        memory_config = ConfigNormalizer._create_memory_config(memory_dict)
        embedding_config = ConfigNormalizer._create_embedding_config(embedding_dict)
        graph_config = ConfigNormalizer._create_graph_config(graph_dict)
        processing_config = ConfigNormalizer._create_processing_config(processing_dict)
        
        # Create main config
        return InsightSpikeConfig(
            llm=llm_config,
            memory=memory_config,
            embedding=embedding_config,
            graph=graph_config,
            processing=processing_config
        )
    
    @staticmethod
    def _create_llm_config(llm_dict: Dict[str, Any]) -> LLMConfig:
        """
        Create LLMConfig with support for extra fields.
        """
        # Standard fields
        config = LLMConfig(
            provider=llm_dict.get('provider', 'mock'),
            api_key=llm_dict.get('api_key', ''),
            model=llm_dict.get('model', ''),
            temperature=llm_dict.get('temperature', 0.7),
            max_tokens=llm_dict.get('max_tokens', 1000),
            timeout=llm_dict.get('timeout', 30)
        )
        
        # Store extra fields as attributes
        extra_fields = ['prompt_style', 'use_simple_prompt', 'system_prompt']
        for field in extra_fields:
            if field in llm_dict:
                setattr(config, field, llm_dict[field])
        
        return config
    
    @staticmethod
    def _create_memory_config(memory_dict: Dict[str, Any]) -> MemoryConfig:
        """Create MemoryConfig with defaults."""
        # Track which defaults we auto-applied for diagnostics
        defaults_applied = []
        faiss_index_type = memory_dict.get('faiss_index_type')
        if not faiss_index_type:
            faiss_index_type = 'FlatL2'
            defaults_applied.append('faiss_index_type')
        metric = memory_dict.get('metric')
        if not metric:
            metric = 'l2'
            defaults_applied.append('metric')
        mc = MemoryConfig(
            episodic_memory_capacity=memory_dict.get('episodic_memory_capacity', 100),
            max_retrieved_docs=memory_dict.get('max_retrieved_docs', 10),
            similarity_threshold=memory_dict.get('similarity_threshold', 0.3),
            enable_graph_search=memory_dict.get('enable_graph_search', False),
            cache_size=memory_dict.get('cache_size', 100)
        )
        # Attach dynamic attributes (non-schema) so downstream still tolerant (layer2 uses dataclass MemoryConfig)
        try:
            setattr(mc, 'faiss_index_type', faiss_index_type)
            setattr(mc, 'metric', metric)
            setattr(mc, 'defaults_applied', defaults_applied)
        except Exception:
            # Best-effort; ignore if underlying model forbids extras
            pass
        return mc
    
    @staticmethod
    def _create_embedding_config(embedding_dict: Dict[str, Any]) -> EmbeddingConfig:
        """Create EmbeddingConfig with defaults."""
        # Treat None/empty as missing to avoid pydantic validation error when tests
        # explicitly set model_name=None for lightweight runs.
        model_name = embedding_dict.get('model_name')
        if not model_name:  # None or ""
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        return EmbeddingConfig(
            model_name=model_name,
            dimension=embedding_dict.get('dimension', 384),
            batch_size=embedding_dict.get('batch_size', 32),
            normalize=embedding_dict.get('normalize', True)
        )
    
    @staticmethod
    def _create_graph_config(graph_dict: Dict[str, Any]) -> GraphConfig:
        """Create GraphConfig with defaults."""
        # Handle multihop config
        multihop_dict = graph_dict.get('multihop_config', {})
        
        config_params = {
            'similarity_threshold': graph_dict.get('similarity_threshold', 0.3),
            'spike_ged_threshold': graph_dict.get('spike_ged_threshold', -0.5),
            'spike_ig_threshold': graph_dict.get('spike_ig_threshold', 0.2),
            'conflict_threshold': graph_dict.get('conflict_threshold', 0.5),
            'use_gnn': graph_dict.get('use_gnn', False),
            'gnn_hidden_dim': graph_dict.get('gnn_hidden_dim', 64),
            'hop_limit': graph_dict.get('hop_limit', 2),
            'neighbor_threshold': graph_dict.get('neighbor_threshold', 0.4),
            'path_decay': graph_dict.get('path_decay', 0.7),
            'enable_graph_search': graph_dict.get('enable_graph_search', False),
        }
        
        # Handle legacy fields
        if 'use_multihop_gedig' in graph_dict:
            config_params['enable_graph_search'] = graph_dict['use_multihop_gedig']
        
        return GraphConfig(**config_params)
    
    @staticmethod
    def _create_processing_config(processing_dict: Dict[str, Any]) -> ProcessingConfig:
        """Create ProcessingConfig with defaults."""
        # Handle adaptive loop config
        adaptive_dict = processing_dict.get('adaptive_loop', {})
        
        return ProcessingConfig(
            enable_layer1_bypass=processing_dict.get('enable_layer1_bypass', True),
            dynamic_doc_adjustment=processing_dict.get('dynamic_doc_adjustment', False),
            enable_insight_registration=processing_dict.get('enable_insight_registration', False),
            min_confidence_threshold=processing_dict.get('min_confidence_threshold', 0.1),
            # AdaptiveLoopConfig if needed
            adaptive_loop=adaptive_dict if adaptive_dict else None
        )
    
    @staticmethod
    def get_llm_config(config: Union[Dict[str, Any], InsightSpikeConfig, BaseModel]) -> LLMConfig:
        """
        Extract LLM configuration from any config format.
        
        Args:
            config: Configuration in any format
            
        Returns:
            LLMConfig: LLM configuration
        """
        if isinstance(config, dict):
            llm_dict = config.get('llm', {})
            return ConfigNormalizer._create_llm_config(llm_dict)
        
        if hasattr(config, 'llm'):
            return config.llm
        
        # Fallback to empty LLM config
        logger.warning("No LLM config found, using defaults")
        return LLMConfig()
    
    @staticmethod
    def merge_configs(base: InsightSpikeConfig, override: Dict[str, Any]) -> InsightSpikeConfig:
        """
        Merge override dictionary into base config.
        
        Args:
            base: Base configuration
            override: Dictionary of overrides
            
        Returns:
            InsightSpikeConfig: Merged configuration
        """
        # Convert base to dict
        base_dict = base.dict()
        
        # Deep merge
        merged = ConfigNormalizer._deep_merge(base_dict, override)
        
        # Convert back to config
        return ConfigNormalizer._dict_to_config(merged)
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigNormalizer._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
