"""
Processing Pipeline
==================

Manages the flow of data through InsightSpike layers.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Import from appropriate locations
from ..core.base.layers import L1Embedder, L2Indexer, L3GraphReasoner, L4PromptAugmenter
from ..memory import Memory

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with metadata"""

    documents: List[Dict[str, Any]]
    answer: str
    metadata: Dict[str, Any]


class ProcessingPipeline:
    """
    Main processing pipeline for InsightSpike.

    Orchestrates data flow through:
    - L1: Embedding generation
    - L2: Indexing and retrieval
    - L3: Graph reasoning and spike detection
    - L4: Prompt augmentation and response generation
    """

    def __init__(
        self,
        config,
        memory: Memory,
        l1: L1Embedder,
        l2: L2Indexer,
        l3: L3GraphReasoner,
        l4: L4PromptAugmenter,
    ):
        """Initialize pipeline with components"""
        self.config = config
        self.memory = memory
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

        # Initialize layers
        self.l1.initialize()
        self.l2.initialize()
        self.l3.initialize()
        self.l4.initialize()

        logger.info("Processing pipeline initialized")

    def add_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add content to memory through layers"""
        try:
            # L1: Generate embedding
            embedding_result = self.l1.process(content)
            embedding = (
                embedding_result.get("embedding")
                if isinstance(embedding_result, dict)
                else embedding_result
            )

            # Add to memory
            doc_id = self.memory.add_document(content, embedding, metadata)

            # L2: Index the document
            self.l2.add_document(
                {
                    "id": doc_id,
                    "content": content,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )

            return {
                "doc_id": doc_id,
                "embedding_shape": embedding.shape
                if hasattr(embedding, "shape")
                else None,
            }

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

    def search(self, query: str, top_k: int = 10) -> SearchResult:
        """Search through pipeline layers"""
        try:
            # L1: Embed query
            query_embedding_result = self.l1.process(query)
            query_embedding = (
                query_embedding_result.get("embedding")
                if isinstance(query_embedding_result, dict)
                else query_embedding_result
            )

            # L2: Retrieve documents
            retrieved_docs = self.l2.search(query_embedding, top_k=top_k)

            # L3: Graph reasoning
            # Attach NormSpec for consistent evaluation across layers
            try:
                from ..config import get_config as _get_cfg
                from ..config.normalized import NormalizedConfig as _NC
                _nc = _NC.from_any(_get_cfg())
                _norm_spec = _nc.norm_spec
            except Exception:
                _norm_spec = None
            graph_result = self.l3.analyze_documents(
                retrieved_docs, {"query": query, "norm_spec": _norm_spec}
            )

            # L4: Generate response
            l4_input = {
                "query": query,
                "documents": retrieved_docs,
                "graph_analysis": graph_result,
            }

            response = self.l4.process(l4_input)

            # Prepare result
            return SearchResult(
                documents=retrieved_docs,
                answer=response.get("answer", "")
                if isinstance(response, dict)
                else str(response),
                metadata={
                    "graph_metrics": graph_result.get("metrics", {}),
                    "spike_detected": graph_result.get("spike_detected", False),
                    "reasoning_quality": graph_result.get("reasoning_quality", 0.0),
                },
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty result on error
            return SearchResult(documents=[], answer=f"Error: {str(e)}", metadata={})

    def process(self, input_data: Any) -> Any:
        """Process data through pipeline (interface method)"""
        if isinstance(input_data, str):
            # Assume it's a query
            return self.search(input_data)
        elif isinstance(input_data, dict):
            if "query" in input_data:
                return self.search(input_data["query"], input_data.get("top_k", 10))
            elif "content" in input_data:
                return self.add_memory(
                    input_data["content"], input_data.get("metadata")
                )

        raise ValueError(f"Unsupported input type: {type(input_data)}")

    def cleanup(self):
        """Cleanup pipeline resources"""
        self.l1.cleanup()
        self.l2.cleanup()
        self.l3.cleanup()
        self.l4.cleanup()
        logger.info("Pipeline cleaned up")
