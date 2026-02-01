"""Baseline implementations for HotpotQA benchmark."""

from .base import BaseRAG
from .bm25_gpt import BM25GPTBaseline
from .closed_book_gpt import ClosedBookGPTBaseline
from .colbert_gpt import ColBERTGPTBaseline
from .contriever_gpt import ContrieverGPTBaseline
from .dpr_gpt import DPRGPTBaseline
from .static_graphrag import StaticGraphRAGBaseline

__all__ = [
    "BaseRAG",
    "BM25GPTBaseline",
    "ClosedBookGPTBaseline",
    "ColBERTGPTBaseline",
    "ContrieverGPTBaseline",
    "DPRGPTBaseline",
    "StaticGraphRAGBaseline",
]
