"""Baseline RAG systems for geDIG-RAG v3 comparison experiments.

This module contains implementations of different RAG approaches for comparison:

- BaseRAGSystem: Abstract base class defining common interface
- StaticRAG: Baseline with no knowledge updates
- FrequencyBasedRAG: Updates based on query frequency and time
- CosineOnlyRAG: Updates based on cosine similarity thresholds  
- GeDIGRAG: Proposed method using geDIG evaluation
"""

from .base_rag import BaseRAGSystem, RAGResponse, UpdateDecision
from .static_rag import StaticRAG
from .frequency_rag import FrequencyBasedRAG
from .cosine_rag import CosineOnlyRAG
from .gedig_rag import GeDIGRAG

__all__ = [
    "BaseRAGSystem",
    "RAGResponse", 
    "UpdateDecision",
    "StaticRAG",
    "FrequencyBasedRAG",
    "CosineOnlyRAG",
    "GeDIGRAG"
]