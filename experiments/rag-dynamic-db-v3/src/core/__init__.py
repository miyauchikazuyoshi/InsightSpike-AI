"""Core components for geDIG-RAG v3 system.

This module contains the fundamental components for the geDIG-based
self-growing RAG system:

- gedig_evaluator: Core geDIG evaluation functionality
- knowledge_graph: Dynamic knowledge graph management
- config: Experiment configuration system
"""

from .gedig_evaluator import GeDIGEvaluator, GeDIGResult
from .knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from .config import ExperimentConfig

__all__ = [
    "GeDIGEvaluator",
    "GeDIGResult", 
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "ExperimentConfig"
]