"""
Data Processing Module
=====================

Handles data loading, retrieval, and embedding operations.
"""

from .loader import load_corpus
from .retrieval import retrieve
from .embedder import get_model

__all__ = ["load_corpus", "retrieve", "get_model"]
