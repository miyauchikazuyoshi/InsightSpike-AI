"""
Enhanced Vector Decoder
======================

Efficient vector-to-text decoding with caching, interpolation, and fallback strategies.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorDecoder:
    """
    Enhanced vector decoder with multiple strategies:
    1. Cache lookup (fastest)
    2. Template generation (fast)
    3. Text interpolation (medium)
    4. LLM generation (slowest but highest quality)
    """
    
    def __init__(
        self,
        llm_interface=None,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        similarity_threshold: float = 0.9
    ):
        """
        Initialize vector decoder.
        
        Args:
            llm_interface: L4LLMInterface for text generation
            cache_size: Maximum cache entries
            cache_ttl: Cache time-to-live in seconds
            similarity_threshold: Threshold for similarity matching
        """
        self.llm = llm_interface
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.similarity_threshold = similarity_threshold
        
        # LRU cache with TTL
        self.cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()
        self.usage_stats: Dict[str, int] = {}
        
        # Precomputed base concepts (can be extended)
        self.base_concepts = self._initialize_base_concepts()
        
        logger.info(
            f"VectorDecoder initialized with cache_size={cache_size}, "
            f"similarity_threshold={similarity_threshold}"
        )
    
    def _initialize_base_concepts(self) -> Dict[str, str]:
        """Initialize common base concepts for fast lookup."""
        return {
            # Japanese concepts
            "果物": "fruit",
            "技術": "technology", 
            "赤い": "red",
            "甘い": "sweet",
            "会社": "company",
            # English concepts
            "fruit": "fruit",
            "technology": "technology",
            "red": "red",
            "sweet": "sweet",
            "company": "company",
            # Abstract concepts
            "connection": "connection",
            "relationship": "relationship",
            "insight": "insight",
            "discovery": "discovery"
        }
    
    def decode(
        self,
        vector: np.ndarray,
        generation_type: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Decode vector to text using progressive fallback strategy.
        
        Args:
            vector: Input vector to decode
            generation_type: Type of generation (insight, branch, summary, etc.)
            context: Additional context for generation
            
        Returns:
            Decoded text
        """
        context = context or {}
        
        # 1. Check cache first
        cache_key = self._make_cache_key(vector, generation_type)
        if cached_text := self._get_from_cache(cache_key):
            self._record_usage(cache_key, "cache")
            return cached_text
        
        # 2. Try template-based generation (fastest)
        if self._can_use_template(generation_type, context):
            text = self._template_generation(generation_type, context)
            if text:
                self._add_to_cache(cache_key, text)
                self._record_usage(cache_key, "template")
                return text
        
        # 3. Try interpolation if we have good neighbors
        if self._has_good_neighbors(context):
            text = self._interpolate_generation(context)
            if text:
                self._add_to_cache(cache_key, text)
                self._record_usage(cache_key, "interpolation")
                return text
        
        # 4. Fall back to LLM generation
        if self.llm and hasattr(self.llm, '_generate_text_from_vector_legacy'):
            text = self.llm._generate_text_from_vector_legacy(
                vector=vector,
                generation_type=generation_type,
                context=context
            )
            self._add_to_cache(cache_key, text)
            self._record_usage(cache_key, "llm")
            return text
        
        # 5. Ultimate fallback
        self._record_usage(cache_key, "fallback")
        return self._fallback_generation(generation_type, context)
    
    def _make_cache_key(self, vector: np.ndarray, generation_type: str) -> str:
        """Create cache key from vector and type."""
        # Use first 8 dimensions for hash to save computation
        vec_hash = hashlib.md5(
            vector[:8].tobytes() if len(vector) > 8 else vector.tobytes()
        ).hexdigest()[:16]
        return f"{generation_type}_{vec_hash}"
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get from cache with TTL check."""
        if key in self.cache:
            text, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return text
            else:
                # Expired
                del self.cache[key]
        return None
    
    def _add_to_cache(self, key: str, text: str):
        """Add to cache with LRU eviction."""
        # Evict if needed
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            self.cache.popitem(last=False)
        
        self.cache[key] = (text, time.time())
    
    def _can_use_template(
        self, generation_type: str, context: Dict[str, Any]
    ) -> bool:
        """Check if template generation is applicable."""
        if generation_type == "branch":
            return (
                "parent_text" in context and 
                "context_neighbors" in context
            )
        elif generation_type == "insight":
            return "retrieved_documents" in context
        elif generation_type == "summary":
            return "texts" in context
        return False
    
    def _template_generation(
        self, generation_type: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate text using simple templates."""
        try:
            if generation_type == "branch":
                parent = context.get("parent_text", "")
                neighbors = context.get("context_neighbors", [])
                
                if not neighbors:
                    return None
                
                # Classify context based on neighbors
                context_type = self._classify_context(neighbors)
                return f"{parent}({context_type})"
            
            elif generation_type == "insight":
                docs = context.get("retrieved_documents", [])
                if len(docs) >= 2:
                    # Extract text from documents
                    texts = []
                    for doc in docs[:2]:
                        if isinstance(doc, dict):
                            texts.append(doc.get("text", ""))
                        else:
                            texts.append(str(doc))
                    
                    if all(texts):
                        return f"{texts[0]}と{texts[1]}の間に新しい関係を発見"
            
            elif generation_type == "summary":
                texts = context.get("texts", [])
                if texts:
                    return f"要約: {', '.join(texts[:3])}の統合概念"
            
        except Exception as e:
            logger.debug(f"Template generation failed: {e}")
        
        return None
    
    def _classify_context(self, neighbors: List[str]) -> str:
        """Classify context based on neighbor content."""
        # Convert to lowercase for matching
        neighbors_lower = [n.lower() for n in neighbors if isinstance(n, str)]
        
        # Food/fruit related
        if any(word in str(neighbors_lower) for word in ["red", "sweet", "fruit", "赤", "甘い", "果物"]):
            return "fruit"
        
        # Technology related
        if any(word in str(neighbors_lower) for word in ["tech", "computer", "jobs", "iphone", "技術", "コンピュータ"]):
            return "technology"
        
        # Academic/scientific
        if any(word in str(neighbors_lower) for word in ["research", "study", "theory", "研究", "理論"]):
            return "academic"
        
        # Default: use first neighbor
        if neighbors:
            return f"{neighbors[0]}系"
        
        return "general"
    
    def _has_good_neighbors(self, context: Dict[str, Any]) -> bool:
        """Check if context has good neighbors for interpolation."""
        if "retrieved_documents" in context:
            return len(context["retrieved_documents"]) >= 2
        if "context_neighbors" in context:
            return len(context["context_neighbors"]) >= 2
        return False
    
    def _interpolate_generation(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate text by interpolating between neighbors."""
        try:
            # For insights
            if "retrieved_documents" in context:
                docs = context["retrieved_documents"]
                if len(docs) >= 3:
                    # Three-way interpolation
                    texts = [self._extract_text(doc) for doc in docs[:3]]
                    texts = [t for t in texts if t]  # Filter empty
                    
                    if len(texts) >= 3:
                        return f"{texts[0]}、{texts[1]}、{texts[2]}の共通原理"
                    elif len(texts) == 2:
                        return f"{texts[0]}と{texts[1]}の融合概念"
            
            # For branches
            if "context_neighbors" in context:
                parent = context.get("parent_text", "概念")
                neighbors = context["context_neighbors"]
                
                if len(neighbors) >= 2:
                    return f"{parent}（{neighbors[0]}と{neighbors[1]}の特徴を持つ）"
                elif neighbors:
                    return f"{parent}（{neighbors[0]}的な）"
            
        except Exception as e:
            logger.debug(f"Interpolation failed: {e}")
        
        return None
    
    def _extract_text(self, doc: Any) -> str:
        """Extract text from various document formats."""
        if isinstance(doc, dict):
            return doc.get("text", "")
        elif isinstance(doc, str):
            return doc
        else:
            return str(doc)
    
    def _fallback_generation(
        self, generation_type: str, context: Dict[str, Any]
    ) -> str:
        """Ultimate fallback when all else fails."""
        if generation_type == "insight":
            return "新しい洞察を発見"
        elif generation_type == "branch":
            parent = context.get("parent_text", "概念")
            return f"{parent}（派生）"
        elif generation_type == "summary":
            return "概念の要約"
        else:
            return "生成されたテキスト"
    
    def _record_usage(self, cache_key: str, method: str):
        """Record usage statistics for analysis."""
        self.usage_stats[cache_key] = self.usage_stats.get(cache_key, 0) + 1
        logger.debug(f"Decoded via {method}: {cache_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decoder statistics."""
        method_counts = {
            "cache": 0,
            "template": 0,
            "interpolation": 0,
            "llm": 0,
            "fallback": 0
        }
        
        # Count by parsing debug logs (simplified)
        total_calls = sum(self.usage_stats.values())
        
        return {
            "cache_size": len(self.cache),
            "cache_hit_rate": len(self.cache) / max(total_calls, 1),
            "total_decodings": total_calls,
            "unique_vectors": len(self.usage_stats),
            "method_distribution": method_counts
        }
    
    def learn_from_feedback(
        self, vector: np.ndarray, text: str, quality_score: float
    ):
        """
        Learn from user feedback to improve future decodings.
        
        Args:
            vector: Vector that was decoded
            text: Generated text
            quality_score: Quality score (0-1)
        """
        if quality_score > 0.8:
            # High quality - prioritize in cache
            cache_key = self._make_cache_key(vector, "learned")
            self._add_to_cache(cache_key, text)
            
            # Could also update base concepts or templates
            logger.info(f"Learned high-quality decoding: {text[:50]}...")
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        logger.info("Vector decoder cache cleared")