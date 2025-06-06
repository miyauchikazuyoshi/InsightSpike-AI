"""
Data Processing Module
=====================

Handles data loading, retrieval, and embedding operations.
"""

from .embedder import get_model
from .loader import load_corpus
from .retrieval import retrieve

__all__ = ["load_corpus", "retrieve", "get_model"]
