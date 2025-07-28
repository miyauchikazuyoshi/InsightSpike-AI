"""
Apply Fixes - Apply all pipeline fixes to ensure consistency
"""

import logging
import numpy as np
from typing import Union, Dict, Any, List

logger = logging.getLogger(__name__)

# Import fix functions (will be imported when needed to avoid circular imports)


def apply_embedder_fix():
    """Fix embedder to always return 1D arrays."""
    from ..processing.embedder import EmbeddingManager
    
    # Patch EmbeddingManager's get_embedding method
    original_get_embedding = EmbeddingManager.get_embedding
    
    def patched_get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with fixed shape."""
        emb = self.encode(text)  # Call encode directly
        if isinstance(emb, np.ndarray) and emb.ndim > 1:
            emb = emb.squeeze()
        return emb
    
    EmbeddingManager.get_embedding = patched_get_embedding
    
    # Also patch encode method
    original_encode = EmbeddingManager.encode
    
    def patched_encode(self, text: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode with fixed shape."""
        result = original_encode(self, text, **kwargs)
        
        # Fix shape for single text
        if isinstance(text, str) and isinstance(result, np.ndarray):
            if result.ndim > 1 and result.shape[0] == 1:
                result = result.squeeze(0)
        
        return result
    
    EmbeddingManager.encode = patched_encode
    logger.info("Applied embedder shape fix")


def apply_llm_provider_fix():
    """Fix LLMProviderRegistry to handle dict configs."""
    from ..implementations.layers.layer4_llm_interface import LLMProviderRegistry
    
    original_get_instance = LLMProviderRegistry.get_instance
    
    @classmethod
    def patched_get_instance(cls, config: Union[Dict[str, Any], Any]):
        """Get or create provider instance with dict support."""
        # Handle dict config
        if isinstance(config, dict):
            provider = config.get('provider', 'mock')
            # Create a simple object with provider attribute
            class ConfigWrapper:
                def __init__(self, d):
                    self.__dict__.update(d)
            
            config_obj = ConfigWrapper(config)
            config_obj.provider = provider
            return original_get_instance(config_obj)
        
        return original_get_instance(config)
    
    LLMProviderRegistry.get_instance = patched_get_instance
    logger.info("Applied LLM provider registry fix")


def apply_cached_memory_fix():
    """Fix CachedMemoryManager to handle embeddings properly."""
    from ..implementations.layers.cached_memory_manager import CachedMemoryManager
    from ..core.episode import Episode
    
    original_add_episode = CachedMemoryManager.add_episode
    
    def patched_add_episode(self, text: str, c_value: float = 0.5, metadata=None):
        """Add episode with fixed embedding shape."""
        # Create embedding
        embedding = self.embedder.get_embedding(text)
        
        # Fix shape if needed
        if isinstance(embedding, np.ndarray) and embedding.ndim > 1:
            embedding = embedding.squeeze()
            logger.debug(f"Fixed embedding shape in add_episode: {embedding.shape}")
        
        # Create episode with fixed embedding
        episode = Episode(
            text=text,
            vec=embedding,
            c=c_value,  # Use 'c' parameter, not 'confidence'
            metadata=metadata or {}
        )
        
        # Continue with original logic
        import uuid
        episode_id = str(uuid.uuid4())
        self._add_to_cache(episode_id, episode)
        
        # Save to DataStore
        try:
            episodes_to_save = [{
                "id": episode_id,
                "text": text,
                "vec": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "confidence": c_value,
                "c_value": c_value,
                "timestamp": episode.timestamp,
                "metadata": metadata or {}
            }]
            
            existing = self.datastore.load_episodes(namespace="episodes")
            all_episodes = existing + episodes_to_save
            self.datastore.save_episodes(all_episodes, namespace="episodes")
            
            self.cache_stats['misses'] += 1
            self._check_memory()
            
            return len(existing)
            
        except Exception as e:
            logger.error(f"Failed to save episode: {e}")
            return -1
    
    CachedMemoryManager.add_episode = patched_add_episode
    logger.info("Applied cached memory manager fix")


def apply_graph_analyzer_fix():
    """Fix GraphAnalyzer to use GraphTypeAdapter."""
    from ..features.graph_reasoning.graph_analyzer import GraphAnalyzer
    from ..graph.type_adapter import GraphTypeAdapter
    
    original_calculate_metrics = GraphAnalyzer.calculate_metrics
    
    def patched_calculate_metrics(self, current_graph, previous_graph, delta_ged_func, delta_ig_func):
        """Calculate metrics with graph type normalization."""
        # Ensure both graphs are NetworkX
        current_nx, _ = GraphTypeAdapter.ensure_networkx(current_graph)
        previous_nx, _ = GraphTypeAdapter.ensure_networkx(previous_graph) if previous_graph else (None, False)
        
        # Call original with NetworkX graphs
        return original_calculate_metrics(self, current_nx, previous_nx, delta_ged_func, delta_ig_func)
    
    GraphAnalyzer.calculate_metrics = patched_calculate_metrics
    logger.info("Applied graph analyzer fix")


def apply_l4_llm_fix():
    """Fix L4LLMInterface to handle dict configs properly."""
    from ..implementations.layers.layer4_llm_interface import L4LLMInterface
    from ..config.normalizer import ConfigNormalizer
    
    original_init = L4LLMInterface.__init__
    
    def patched_init(self, config: Union[Dict[str, Any], Any]):
        """Initialize with proper config handling."""
        # Normalize config
        if isinstance(config, dict):
            llm_config = ConfigNormalizer.get_llm_config(config)
            # Store extra attributes
            self.prompt_style = config.get('llm', {}).get('prompt_style', 'standard')
            self.use_simple_prompt = config.get('llm', {}).get('use_simple_prompt', False)
            # Call original with normalized config
            original_init(self, llm_config)
        else:
            original_init(self, config)
    
    L4LLMInterface.__init__ = patched_init
    logger.info("Applied L4 LLM interface fix")


def apply_all_fixes():
    """
    Apply all fixes to ensure consistent pipeline behavior.
    
    This should be called early in the application lifecycle.
    """
    logger.info("Applying pipeline fixes...")
    
    # CRITICAL: MainAgent Episode fix is now in source code - skip patch
    # try:
    #     from .main_agent_episode_fix import apply_main_agent_episode_fix
    #     apply_main_agent_episode_fix()
    # except Exception as e:
    #     logger.warning(f"Failed to apply MainAgent Episode fix: {e}")
    logger.info("Skipping MainAgent Episode fix - already in source")
    
    try:
        apply_embedder_fix()
    except Exception as e:
        logger.warning(f"Failed to apply embedder fix: {e}")
    
    try:
        apply_llm_provider_fix()
    except Exception as e:
        logger.warning(f"Failed to apply LLM provider fix: {e}")
    
    try:
        apply_cached_memory_fix()
    except Exception as e:
        logger.warning(f"Failed to apply cached memory fix: {e}")
    
    # Graph analyzer fix is now in source code - skip patch
    # try:
    #     apply_graph_analyzer_fix()
    # except Exception as e:
    #     logger.warning(f"Failed to apply graph analyzer fix: {e}")
    logger.info("Skipping graph analyzer fix - already in source")
    
    try:
        apply_l4_llm_fix()
    except Exception as e:
        logger.warning(f"Failed to apply L4 LLM fix: {e}")
    
    try:
        from .enum_value_fix import fix_llm_interface_enum_access
        fix_llm_interface_enum_access()
    except Exception as e:
        logger.warning(f"Failed to apply enum value fix: {e}")
    
    # Graph analyzer NetworkX fix is now in source code - skip patch
    # try:
    #     from .graph_analyzer_networkx_fix import apply_graph_analyzer_networkx_fix
    #     apply_graph_analyzer_networkx_fix()
    # except Exception as e:
    #     logger.warning(f"Failed to apply graph analyzer NetworkX fix: {e}")
    logger.info("Skipping graph analyzer NetworkX fix - already in source")
    
    logger.info("Pipeline fixes applied successfully")


# Auto-apply fixes on import
if __name__ != "__main__":
    apply_all_fixes()