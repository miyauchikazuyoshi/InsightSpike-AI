"""
Service Providers
================

Factory functions for creating services with dependencies.
"""

from typing import Optional
import logging

from ..config import InsightSpikeConfig
from ..interfaces import (
    IDataStore,
    IEmbedder,
    IGraphBuilder,
    ILLMProvider,
    IMemoryManager
)
from ..implementations.datastore import DataStoreFactory
from ..processing.embedder import EmbeddingManager
from ..implementations.graph.pyg_graph_builder import PyGGraphBuilder
from ..providers import LLMProviderRegistry
from ..implementations.layers.layer2_memory_manager import L2MemoryManager

logger = logging.getLogger(__name__)


class DataStoreProvider:
    """Provider for IDataStore implementations."""
    
    @staticmethod
    def create(container, config: Optional[InsightSpikeConfig] = None) -> IDataStore:
        """
        Create a datastore instance.
        
        Args:
            container: DI container
            config: Optional configuration
            
        Returns:
            IDataStore implementation
        """
        if config is None:
            config = container.resolve(InsightSpikeConfig)
        
        # For experiments, always use filesystem
        if config.environment in ["experiment", "test", "testing"]:
            return DataStoreFactory.create("filesystem", base_path="data/experiment")
        
        # For production, use configured store
        return DataStoreFactory.create("filesystem", base_path=str(config.paths.data_dir))


class EmbedderProvider:
    """Provider for IEmbedder implementations."""
    
    @staticmethod
    def create(container, config: Optional[InsightSpikeConfig] = None) -> IEmbedder:
        """
        Create an embedder instance.
        
        Args:
            container: DI container
            config: Optional configuration
            
        Returns:
            IEmbedder implementation
        """
        if config is None:
            config = container.resolve(InsightSpikeConfig)
        
        return EmbeddingManager(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            dimension=config.embedding.dimension
        )


class LLMProviderFactory:
    """Factory for ILLMProvider implementations."""
    
    @staticmethod
    def create(container, config: Optional[InsightSpikeConfig] = None) -> ILLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            container: DI container
            config: Optional configuration
            
        Returns:
            ILLMProvider implementation
        """
        if config is None:
            config = container.resolve(InsightSpikeConfig)
        
        registry = LLMProviderRegistry()
        provider_name = config.llm.provider
        
        # Get provider class
        provider_class = registry.get_provider(provider_name)
        
        # Create provider with config
        return provider_class(config)


class GraphBuilderProvider:
    """Provider for IGraphBuilder implementations."""
    
    @staticmethod
    def create(container, config: Optional[InsightSpikeConfig] = None) -> IGraphBuilder:
        """
        Create a graph builder instance.
        
        Args:
            container: DI container
            config: Optional configuration
            
        Returns:
            IGraphBuilder implementation
        """
        if config is None:
            config = container.resolve(InsightSpikeConfig)
        
        return PyGGraphBuilder(config)


class MemoryManagerProvider:
    """Provider for IMemoryManager implementations."""
    
    @staticmethod
    def create(container, config: Optional[InsightSpikeConfig] = None) -> IMemoryManager:
        """
        Create a memory manager instance.
        
        Args:
            container: DI container
            config: Optional configuration
            
        Returns:
            IMemoryManager implementation
        """
        if config is None:
            config = container.resolve(InsightSpikeConfig)
        
        # Get dependencies
        datastore = container.resolve(IDataStore)
        embedder = container.resolve(IEmbedder)
        
        return L2MemoryManager(
            config=config,
            datastore=datastore,
            embedder=embedder
        )