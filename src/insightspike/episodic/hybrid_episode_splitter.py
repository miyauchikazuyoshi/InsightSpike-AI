"""
Hybrid Episode Splitter
=======================

Implements episode splitting using both message passing and LLM analysis,
following the same dual-evaluation flow as insight detection.

Key Features:
- Message passing for graph-based vector generation
- LLM semantic analysis for meaning validation
- Quality assurance through dual evaluation
- Reuses insight detection evaluation algorithms
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..algorithms.graph_edit_distance import calculate_ged_change
from ..algorithms.information_gain import calculate_information_gain
from ..core.episode import Episode
from ..implementations.llm_providers import LLMProvider
from ..processing.embedder import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class SplitCandidate:
    """Represents a potential episode split"""
    segments: List[str]
    mp_vectors: List[np.ndarray]  # Message passing vectors
    semantic_weights: List[float]  # LLM-derived weights
    quality_score: float
    ged_improvement: float
    ig_improvement: float
    metadata: Dict[str, Any]


@dataclass
class SplitEvaluationResult:
    """Results of split evaluation (similar to GraphOptimizationResult)"""
    should_split: bool
    ged_before: float
    ged_after: float
    ged_improvement: float
    ig_before: float
    ig_after: float
    ig_improvement: float
    quality_score: float
    semantic_coherence: float


class HybridEpisodeSplitter:
    """
    Splits episodes using both graph structure (message passing) and 
    semantic understanding (LLM), following insight detection's dual evaluation.
    """
    
    def __init__(
        self,
        embedder: Optional[EmbeddingManager] = None,
        quality_threshold: float = 0.6,
        ged_improvement_threshold: float = 0.1,
        ig_improvement_threshold: float = 0.05
    ):
        """
        Initialize the hybrid splitter.
        
        Args:
            embedder: Embedding manager for text encoding
            quality_threshold: Minimum quality score for split
            ged_improvement_threshold: Minimum GED improvement
            ig_improvement_threshold: Minimum IG improvement
        """
        self.embedder = embedder or EmbeddingManager()
        self.quality_threshold = quality_threshold
        self.ged_threshold = ged_improvement_threshold
        self.ig_threshold = ig_improvement_threshold
        
    def split_episode(
        self,
        episode: Episode,
        graph: nx.Graph,
        llm_provider: Optional[LLMProvider] = None,
        force_split: bool = False
    ) -> List[Episode]:
        """
        Split episode using hybrid approach.
        
        Args:
            episode: Episode to potentially split
            graph: Current knowledge graph
            llm_provider: LLM for semantic analysis
            force_split: Force split even if quality checks fail
            
        Returns:
            List of episodes (original if no split, or new splits)
        """
        # 1. Analyze split necessity
        split_points = self._detect_split_boundaries(episode.text, graph)
        if not split_points:
            return [episode]
            
        # 2. Generate segments
        segments = self._split_at_boundaries(episode.text, split_points)
        if len(segments) <= 1:
            return [episode]
            
        # 3. Dual evaluation: Message Passing + LLM
        candidate = self._create_split_candidate(
            episode, segments, graph, llm_provider
        )
        
        # 4. Evaluate quality (like insight detection)
        evaluation = self._evaluate_split_quality(
            episode, candidate, graph
        )
        
        # 5. Decision based on dual criteria
        if evaluation.should_split or force_split:
            return self._finalize_split(episode, candidate, evaluation)
        else:
            logger.info(
                f"Split rejected - Quality: {evaluation.quality_score:.2f}, "
                f"GED improvement: {evaluation.ged_improvement:.2f}, "
                f"IG improvement: {evaluation.ig_improvement:.2f}"
            )
            return [episode]
    
    def _detect_split_boundaries(
        self, 
        text: str, 
        graph: nx.Graph
    ) -> List[int]:
        """
        Detect potential split points based on semantic boundaries.
        """
        boundaries = []
        
        # 1. Paragraph boundaries
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            current_pos = 0
            for i, para in enumerate(paragraphs[:-1]):
                current_pos += len(para) + 2  # +2 for \n\n
                boundaries.append(current_pos)
        
        # 2. Major punctuation boundaries (。！？)
        major_punct_pattern = r'[。！？]\s+'
        for match in re.finditer(major_punct_pattern, text):
            boundaries.append(match.end())
            
        # 3. Topic shift indicators
        topic_indicators = ['しかし', 'ただし', '一方で', 'また', 'さらに', 'つまり']
        for indicator in topic_indicators:
            idx = text.find(indicator)
            if idx > 0:
                boundaries.append(idx)
                
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        # Filter based on minimum segment size
        min_segment_size = 50
        filtered_boundaries = []
        last_pos = 0
        
        for boundary in boundaries:
            if boundary - last_pos >= min_segment_size:
                filtered_boundaries.append(boundary)
                last_pos = boundary
                
        return filtered_boundaries
    
    def _split_at_boundaries(
        self, 
        text: str, 
        boundaries: List[int]
    ) -> List[str]:
        """Split text at given boundaries."""
        segments = []
        last_pos = 0
        
        for boundary in boundaries:
            segment = text[last_pos:boundary].strip()
            if segment:
                segments.append(segment)
            last_pos = boundary
            
        # Add final segment
        final_segment = text[last_pos:].strip()
        if final_segment:
            segments.append(final_segment)
            
        return segments
    
    def _create_split_candidate(
        self,
        parent_episode: Episode,
        segments: List[str],
        graph: nx.Graph,
        llm_provider: Optional[LLMProvider]
    ) -> SplitCandidate:
        """
        Create split candidate with both MP vectors and LLM weights.
        """
        # 1. Message Passing vectors
        mp_vectors = self._generate_message_passing_vectors(
            parent_episode, segments, graph
        )
        
        # 2. LLM semantic analysis
        if llm_provider:
            semantic_analysis = self._analyze_with_llm(
                parent_episode, segments, llm_provider
            )
            semantic_weights = semantic_analysis['weights']
        else:
            # Fallback: equal weights
            semantic_weights = [1.0 / len(segments)] * len(segments)
            
        # 3. Calculate initial quality
        quality_score = self._calculate_candidate_quality(
            segments, mp_vectors, semantic_weights
        )
        
        return SplitCandidate(
            segments=segments,
            mp_vectors=mp_vectors,
            semantic_weights=semantic_weights,
            quality_score=quality_score,
            ged_improvement=0.0,  # Will be calculated in evaluation
            ig_improvement=0.0,   # Will be calculated in evaluation
            metadata={
                'parent_id': parent_episode.id if hasattr(parent_episode, 'id') else None,
                'original_c': parent_episode.c
            }
        )
    
    def _generate_message_passing_vectors(
        self,
        parent_episode: Episode,
        segments: List[str],
        graph: nx.Graph
    ) -> List[np.ndarray]:
        """
        Generate vectors using message passing from graph neighbors.
        """
        vectors = []
        parent_vec = parent_episode.vec
        
        # Get parent's neighbors if in graph
        neighbors_data = []
        if hasattr(parent_episode, 'id') and parent_episode.id in graph:
            for neighbor_id in graph.neighbors(parent_episode.id):
                neighbor_node = graph.nodes[neighbor_id]
                edge_data = graph.edges[parent_episode.id, neighbor_id]
                neighbors_data.append({
                    'id': neighbor_id,
                    'vec': neighbor_node.get('vec', np.zeros_like(parent_vec)),
                    'text': neighbor_node.get('text', ''),
                    'weight': edge_data.get('weight', 1.0)
                })
        
        # Sort neighbors by edge weight
        neighbors_data = sorted(
            neighbors_data, 
            key=lambda x: x['weight'], 
            reverse=True
        )[:10]  # Top 10 neighbors
        
        for i, segment in enumerate(segments):
            # Base embedding
            base_vec = self.embedder.encode(segment)
            
            # Initialize messages
            messages = []
            
            # 1. Parent message (highest weight)
            parent_weight = 0.5
            messages.append((parent_vec, parent_weight))
            
            # 2. Neighbor messages
            remaining_weight = 0.3
            if neighbors_data:
                # Calculate relevance of each neighbor to this segment
                neighbor_relevances = []
                for neighbor in neighbors_data:
                    relevance = self._calculate_text_relevance(
                        segment, neighbor['text']
                    )
                    neighbor_relevances.append(relevance * neighbor['weight'])
                
                # Normalize relevances
                total_relevance = sum(neighbor_relevances)
                if total_relevance > 0:
                    for neighbor, relevance in zip(neighbors_data, neighbor_relevances):
                        weight = (relevance / total_relevance) * remaining_weight
                        messages.append((neighbor['vec'], weight))
            
            # 3. Base vector message
            base_weight = 0.2
            messages.append((base_vec, base_weight))
            
            # Aggregate messages
            final_vec = np.zeros_like(parent_vec)
            for vec, weight in messages:
                final_vec += vec * weight
                
            # Normalize
            norm = np.linalg.norm(final_vec)
            if norm > 0:
                final_vec = final_vec / norm
            else:
                final_vec = base_vec  # Fallback
                
            vectors.append(final_vec)
            
        return vectors
    
    def _calculate_text_relevance(self, text1: str, text2: str) -> float:
        """
        Calculate relevance between two text segments.
        Simple implementation using embedding similarity.
        """
        if not text1 or not text2:
            return 0.0
            
        vec1 = self.embedder.encode(text1)
        vec2 = self.embedder.encode(text2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
        )
        
        return max(0.0, similarity)
    
    def _analyze_with_llm(
        self,
        parent_episode: Episode,
        segments: List[str],
        llm_provider: LLMProvider
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze semantic properties of segments.
        """
        # Build prompt
        prompt = self._build_llm_prompt(parent_episode.text, segments)
        
        try:
            response = llm_provider.generate(
                prompt,
                temperature=0.3,  # Lower temperature for analysis
                max_tokens=1000
            )
            
            # Parse response
            return self._parse_llm_response(response, len(segments))
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            # Fallback
            return {
                'weights': [1.0 / len(segments)] * len(segments),
                'keywords': [[] for _ in segments],
                'relationships': {}
            }
    
    def _build_llm_prompt(self, original_text: str, segments: List[str]) -> str:
        """Build prompt for LLM analysis."""
        prompt = f"""テキストが{len(segments)}個のセグメントに分割されました。
各セグメントの意味的特性を分析してください。

元のテキスト（要約）: {original_text[:300]}...

セグメント:
"""
        
        for i, segment in enumerate(segments):
            prompt += f"\n[セグメント{i+1}]\n{segment[:200]}...\n"
            
        prompt += """
各セグメントについて以下をJSON形式で出力してください:
{
  "segments": [
    {
      "index": 0,
      "keywords": ["キーワード1", "キーワード2", "キーワード3"],
      "importance": 0.0-1.0,
      "independence": 0.0-1.0,
      "parent_inheritance": 0.0-1.0
    }
  ]
}
"""
        
        return prompt
    
    def _parse_llm_response(
        self, 
        response: str, 
        num_segments: int
    ) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        import json
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            data = json.loads(json_str)
            segments_data = data.get('segments', [])
            
            # Extract weights based on importance
            weights = []
            keywords = []
            
            for i in range(num_segments):
                if i < len(segments_data):
                    seg_data = segments_data[i]
                    weights.append(seg_data.get('importance', 0.5))
                    keywords.append(seg_data.get('keywords', []))
                else:
                    weights.append(0.5)
                    keywords.append([])
                    
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / num_segments] * num_segments
                
            return {
                'weights': weights,
                'keywords': keywords,
                'segments_data': segments_data
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {
                'weights': [1.0 / num_segments] * num_segments,
                'keywords': [[] for _ in range(num_segments)],
                'segments_data': []
            }
    
    def _calculate_candidate_quality(
        self,
        segments: List[str],
        mp_vectors: List[np.ndarray],
        semantic_weights: List[float]
    ) -> float:
        """
        Calculate quality score for split candidate.
        """
        scores = []
        
        # 1. Vector diversity (should be different but not too different)
        if len(mp_vectors) > 1:
            similarities = []
            for i in range(len(mp_vectors)):
                for j in range(i+1, len(mp_vectors)):
                    sim = np.dot(mp_vectors[i], mp_vectors[j]) / (
                        np.linalg.norm(mp_vectors[i]) * 
                        np.linalg.norm(mp_vectors[j]) + 1e-8
                    )
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            # Optimal similarity around 0.5-0.7
            diversity_score = 1.0 - abs(avg_similarity - 0.6)
            scores.append(diversity_score)
        
        # 2. Segment length balance
        lengths = [len(seg) for seg in segments]
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        balance_score = 1.0 - min(1.0, std_length / avg_length)
        scores.append(balance_score)
        
        # 3. Semantic weight distribution
        weight_entropy = -sum(
            w * np.log(w + 1e-8) for w in semantic_weights
        ) / np.log(len(semantic_weights))
        scores.append(weight_entropy)
        
        return np.mean(scores)
    
    def _evaluate_split_quality(
        self,
        parent_episode: Episode,
        candidate: SplitCandidate,
        graph: nx.Graph
    ) -> SplitEvaluationResult:
        """
        Evaluate split quality using GED/IG metrics (like insight detection).
        """
        # Calculate metrics before split
        ged_before = self._calculate_local_ged(parent_episode, graph)
        ig_before = self._calculate_local_ig(parent_episode, graph)
        
        # Simulate graph after split
        simulated_graph = self._simulate_split_graph(
            parent_episode, candidate, graph
        )
        
        # Calculate metrics after split
        ged_after = self._calculate_split_ged(candidate, simulated_graph)
        ig_after = self._calculate_split_ig(candidate, simulated_graph)
        
        # Calculate improvements
        ged_improvement = (ged_before - ged_after) / max(ged_before, 1.0)
        ig_improvement = (ig_after - ig_before) / max(ig_before, 1.0)
        
        # Semantic coherence from LLM analysis
        semantic_coherence = self._calculate_semantic_coherence(candidate)
        
        # Should split if all criteria are met
        should_split = (
            candidate.quality_score >= self.quality_threshold and
            ged_improvement >= self.ged_threshold and
            ig_improvement >= self.ig_threshold
        )
        
        return SplitEvaluationResult(
            should_split=should_split,
            ged_before=ged_before,
            ged_after=ged_after,
            ged_improvement=ged_improvement,
            ig_before=ig_before,
            ig_after=ig_after,
            ig_improvement=ig_improvement,
            quality_score=candidate.quality_score,
            semantic_coherence=semantic_coherence
        )
    
    def _calculate_local_ged(
        self, 
        episode: Episode, 
        graph: nx.Graph
    ) -> float:
        """Calculate local GED around episode."""
        if not hasattr(episode, 'id') or episode.id not in graph:
            return 1.0
            
        # Get local subgraph
        neighbors = list(graph.neighbors(episode.id))
        nodes = [episode.id] + neighbors[:10]  # Limit for performance
        subgraph = graph.subgraph(nodes)
        
        # Simple GED approximation: inverse of edge density
        if len(subgraph.nodes) > 1:
            density = 2 * len(subgraph.edges) / (
                len(subgraph.nodes) * (len(subgraph.nodes) - 1)
            )
            return 1.0 - density
        else:
            return 1.0
    
    def _calculate_local_ig(
        self, 
        episode: Episode, 
        graph: nx.Graph
    ) -> float:
        """Calculate local information gain around episode."""
        if not hasattr(episode, 'id') or episode.id not in graph:
            return 0.0
            
        # Information content based on connections and weights
        node = graph.nodes[episode.id]
        
        # Base information from episode
        base_info = episode.c
        
        # Additional information from connections
        connection_info = 0.0
        for neighbor in graph.neighbors(episode.id):
            edge_weight = graph.edges[episode.id, neighbor].get('weight', 0.0)
            neighbor_c = graph.nodes[neighbor].get('c', 0.5)
            connection_info += edge_weight * neighbor_c
            
        return base_info + connection_info * 0.5
    
    def _simulate_split_graph(
        self,
        parent_episode: Episode,
        candidate: SplitCandidate,
        graph: nx.Graph
    ) -> nx.Graph:
        """Simulate graph structure after split."""
        # Create a copy
        new_graph = graph.copy()
        
        # Remove parent if exists
        if hasattr(parent_episode, 'id') and parent_episode.id in new_graph:
            # Save connections
            parent_edges = list(new_graph.edges(parent_episode.id, data=True))
            new_graph.remove_node(parent_episode.id)
        else:
            parent_edges = []
            
        # Add split nodes
        for i, (segment, vec) in enumerate(zip(candidate.segments, candidate.mp_vectors)):
            node_id = f"{parent_episode.id if hasattr(parent_episode, 'id') else 'split'}_{i}"
            new_graph.add_node(
                node_id,
                text=segment,
                vec=vec,
                c=parent_episode.c * candidate.semantic_weights[i]
            )
            
        # Reconnect edges based on relevance
        # This is simplified - real implementation would be more sophisticated
        
        return new_graph
    
    def _calculate_split_ged(
        self,
        candidate: SplitCandidate,
        graph: nx.Graph
    ) -> float:
        """Calculate GED after split."""
        # Simplified: better structure means lower GED
        # More segments with balanced weights = better structure
        weight_variance = np.var(candidate.semantic_weights)
        structure_score = len(candidate.segments) * (1.0 - weight_variance)
        
        return 1.0 / (1.0 + structure_score)
    
    def _calculate_split_ig(
        self,
        candidate: SplitCandidate,
        graph: nx.Graph
    ) -> float:
        """Calculate IG after split."""
        # Information gain from having more specific segments
        total_ig = 0.0
        
        for i, (segment, weight) in enumerate(
            zip(candidate.segments, candidate.semantic_weights)
        ):
            # Each segment contributes information
            segment_ig = weight * np.log(len(candidate.segments))
            total_ig += segment_ig
            
        return total_ig
    
    def _calculate_semantic_coherence(
        self,
        candidate: SplitCandidate
    ) -> float:
        """Calculate semantic coherence of split."""
        # Based on quality score and weight distribution
        coherence = candidate.quality_score
        
        # Penalize very uneven weight distribution
        weight_entropy = -sum(
            w * np.log(w + 1e-8) for w in candidate.semantic_weights
        ) / np.log(len(candidate.semantic_weights))
        
        coherence *= weight_entropy
        
        return coherence
    
    def _finalize_split(
        self,
        parent_episode: Episode,
        candidate: SplitCandidate,
        evaluation: SplitEvaluationResult
    ) -> List[Episode]:
        """
        Create final split episodes with integrated vectors.
        """
        split_episodes = []
        
        for i, (segment, mp_vec, weight) in enumerate(zip(
            candidate.segments,
            candidate.mp_vectors,
            candidate.semantic_weights
        )):
            # Final vector: mainly MP with slight LLM adjustment
            # (MP already incorporates graph structure)
            final_vec = mp_vec  # MP vector is already well-integrated
            
            # C-value based on parent and importance
            new_c = parent_episode.c * weight * evaluation.quality_score
            
            # Create metadata
            metadata = {
                'parent_id': candidate.metadata.get('parent_id'),
                'parent_c': parent_episode.c,
                'split_index': i,
                'split_method': 'hybrid_mp_llm',
                'quality_score': evaluation.quality_score,
                'ged_improvement': evaluation.ged_improvement,
                'ig_improvement': evaluation.ig_improvement,
                'semantic_weight': weight
            }
            
            # Create new episode
            new_episode = Episode(
                text=segment,
                vec=final_vec,
                c=new_c,
                metadata=metadata
            )
            
            split_episodes.append(new_episode)
            
        logger.info(
            f"Split episode into {len(split_episodes)} segments. "
            f"Quality: {evaluation.quality_score:.2f}, "
            f"GED improvement: {evaluation.ged_improvement:.2f}, "
            f"IG improvement: {evaluation.ig_improvement:.2f}"
        )
        
        return split_episodes