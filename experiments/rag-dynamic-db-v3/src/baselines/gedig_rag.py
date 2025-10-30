"""geDIG-based RAG system - the proposed method."""

import time
from typing import Dict, Any, List, Optional

import numpy as np

from .base_rag import BaseRAGSystem, UpdateDecision, RetrievalResult
from ..core.gedig_evaluator import GeDIGEvaluator, UpdateType, GraphUpdate, GeDIGResult
from ..core.config import ExperimentConfig


class GeDIGRAG(BaseRAGSystem):
    """RAG system using geDIG evaluation for knowledge updates.
    
    This is the proposed method that uses geDIG (Graph Edit Distance + Information Gain)
    to make principled decisions about knowledge graph updates.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize geDIG-based RAG system.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, method_name="gedig")
        
        # Core geDIG evaluator
        self.gedig_evaluator = GeDIGEvaluator(
            k_coefficient=config.gedig.k_coefficient,
            radius=config.gedig.radius
        )
        
        # geDIG thresholds from config (F = ΔGED - k·ΔIG; accept if F < θ)
        self.threshold_mode = getattr(config.gedig, 'threshold_mode', 'fixed')
        self.gedig_threshold = getattr(config.gedig, 'threshold_value', 0.0)
        self.threshold_percentile = getattr(config.gedig, 'threshold_percentile', 20.0)
        self.add_ig_threshold = config.gedig.add_ig_threshold
        self.add_ged_min = config.gedig.add_ged_min
        self.add_ged_max = config.gedig.add_ged_max
        self.merge_similarity = config.gedig.merge_similarity
        self.prune_usage_min = config.gedig.prune_usage_min
        
        # Statistics tracking
        self.gedig_evaluations = 0
        self.gedig_history: List[GeDIGResult] = []
        self.update_type_counts = {
            'add': 0,
            'prune': 0,
            'merge': 0,
            'rejected': 0
        }
    
    def should_update_knowledge(self, 
                               query: str, 
                               response: str,
                               retrieval_result: RetrievalResult) -> UpdateDecision:
        """Decide based on geDIG evaluation of potential updates.
        
        Args:
            query: User query
            response: Generated response
            retrieval_result: Result from knowledge retrieval
            
        Returns:
            Update decision based on geDIG evaluation
        """
        # Generate multiple update candidates
        update_candidates = self._generate_update_candidates(query, response, retrieval_result)
        
        if not update_candidates:
            return UpdateDecision(
                should_update=False,
                update_type=UpdateType.ADD,
                reason="geDIG: No valid update candidates generated",
                confidence=0.0,
                metadata={'n_candidates': 0}
            )
        
        # Evaluate each candidate with geDIG
        best_update = None
        best_gedig_score = float('inf')  # minimize geDIG
        best_gedig_result = None
        
        evaluation_results = []
        
        for candidate in update_candidates:
            # Evaluate using geDIG
            gedig_result = self.gedig_evaluator.evaluate_update(
                graph_before=self.knowledge_graph.graph,
                update=candidate,
                metadata={
                    'query': query,
                    'response': response,
                    'retrieval_quality': retrieval_result.stats.get('avg_similarity', 0.0)
                }
            )
            
            evaluation_results.append((candidate, gedig_result))
            self.gedig_evaluations += 1
            self.gedig_history.append(gedig_result)
            
            # Track best candidate (lower geDIG is better)
            if gedig_result.delta_gedig < best_gedig_score:
                best_gedig_score = gedig_result.delta_gedig
                best_update = candidate
                best_gedig_result = gedig_result
        
        # Determine threshold depending on mode
        threshold = self.gedig_threshold
        if self.threshold_mode == 'percentile' and self.gedig_history:
            # Compute percentile over recent scores (use last 200 for stability)
            recent = [r.delta_gedig for r in self.gedig_history[-200:]]
            try:
                threshold = float(np.percentile(recent, self.threshold_percentile))
            except Exception:
                threshold = self.gedig_threshold
        
        # Decision based on best geDIG score (accept if F < θ)
        if best_gedig_score < threshold:
            # Additional checks based on specific thresholds
            decision_valid = self._validate_gedig_decision(best_gedig_result, best_update)
            
            if decision_valid:
                self.update_type_counts[best_update.update_type.value] += 1
                
                reason = (f"geDIG: ACCEPT F={best_gedig_score:.3f} < θ={threshold:.3f} "
                         f"(ΔGED={best_gedig_result.delta_ged:.3f}, "
                         f"ΔIG={best_gedig_result.delta_ig:.3f})")
                
                return UpdateDecision(
                    should_update=True,
                    update_type=best_update.update_type,
                    reason=reason,
                    confidence=best_gedig_result.confidence,
                    update=best_update,
                    gedig_result=best_gedig_result,
                    metadata={
                        'gedig_score': best_gedig_score,
                        'delta_ged': best_gedig_result.delta_ged,
                        'delta_ig': best_gedig_result.delta_ig,
                        'n_candidates_evaluated': len(update_candidates),
                        'threshold': threshold,
                        'evaluation_results': [
                            {
                                'update_type': c.update_type.value,
                                'gedig_score': r.delta_gedig,
                                'delta_ged': r.delta_ged,
                                'delta_ig': r.delta_ig
                            }
                            for c, r in evaluation_results
                        ]
                    }
                )
            else:
                self.update_type_counts['rejected'] += 1
                reason = f"geDIG: REJECT validation failed (F={best_gedig_score:.3f}, θ={threshold:.3f})"
        else:
            self.update_type_counts['rejected'] += 1
            reason = f"geDIG: REJECT F not below θ (F={best_gedig_score:.3f} >= θ={threshold:.3f})"
        
        return UpdateDecision(
            should_update=False,
            update_type=best_update.update_type if best_update else UpdateType.ADD,
            reason=reason,
            confidence=0.0,
            gedig_result=best_gedig_result,
            metadata={
                'gedig_score': best_gedig_score,
                'n_candidates_evaluated': len(update_candidates),
                'threshold': self.gedig_threshold
            }
        )
    
    def _generate_update_candidates(self, 
                                  query: str, 
                                  response: str, 
                                  retrieval_result: RetrievalResult) -> List[GraphUpdate]:
        """Generate multiple update candidates for geDIG evaluation.
        
        Args:
            query: User query
            response: Generated response
            retrieval_result: Retrieval result
            
        Returns:
            List of potential graph updates
        """
        candidates = []
        
        # Candidate 1: Add new QA node
        add_candidate = self._create_addition_candidate(query, response, retrieval_result)
        if add_candidate:
            candidates.append(add_candidate)
        
        # Candidate 2: Merge similar nodes (if applicable)
        merge_candidates = self._create_merge_candidates(retrieval_result)
        candidates.extend(merge_candidates)
        
        # Candidate 3: Prune low-value nodes (periodically)
        if self.query_count % 10 == 0:  # Check pruning every 10 queries
            prune_candidates = self._create_prune_candidates()
            candidates.extend(prune_candidates)
        
        return candidates
    
    def _create_addition_candidate(self, 
                                 query: str, 
                                 response: str,
                                 retrieval_result: RetrievalResult) -> Optional[GraphUpdate]:
        """Create knowledge addition candidate.
        
        Args:
            query: User query
            response: Generated response
            retrieval_result: Retrieval result
            
        Returns:
            Graph update for addition, or None
        """
        # Create combined knowledge text
        knowledge_text = f"Q: {query} A: {response}"
        
        # Generate embedding
        embedding = self.embedder.encode([knowledge_text])[0]
        
        # Check if this would be too similar to existing knowledge
        max_similarity = retrieval_result.stats.get('max_similarity', 0.0)
        if max_similarity > 0.95:  # Very high similarity - likely redundant
            return None
        
        # Create new node data
        new_node_id = f"gedig_node_{int(time.time())}"
        new_node_data = {
            'id': new_node_id,
            'text': knowledge_text,
            'embedding': embedding,
            'node_type': 'qa_pair',
            'confidence': 0.8,  # High confidence for geDIG-validated additions
            'timestamp': time.time(),
            'metadata': {
                'source': 'gedig_addition',
                'query': query,
                'response': response,
                'max_similarity_to_existing': max_similarity
            }
        }
        
        # Create edges to relevant existing nodes
        new_edges = []
        for node_id, similarity in retrieval_result.retrieved_nodes[:3]:  # Top 3
            if similarity > 0.2:  # Meaningful connection threshold
                new_edges.append((new_node_id, node_id, {
                    'relation': 'semantic',
                    'weight': similarity,
                    'semantic_similarity': similarity
                }))
        
        return GraphUpdate(
            update_type=UpdateType.ADD,
            target_nodes=[],
            new_node_data=new_node_data,
            new_edges=new_edges,
            metadata={
                'method': 'gedig_addition',
                'query': query
            }
        )
    
    def _create_merge_candidates(self, retrieval_result: RetrievalResult) -> List[GraphUpdate]:
        """Create merge candidates from similar nodes.
        
        Args:
            retrieval_result: Retrieval result with similar nodes
            
        Returns:
            List of merge update candidates
        """
        merge_candidates = []
        
        # Look for highly similar node pairs
        retrieved_nodes = retrieval_result.retrieved_nodes
        
        for i in range(len(retrieved_nodes)):
            for j in range(i + 1, min(len(retrieved_nodes), i + 3)):  # Check next 2 nodes
                node_id_i, sim_i = retrieved_nodes[i]
                node_id_j, sim_j = retrieved_nodes[j]
                
                # Check if nodes are similar enough to merge
                if node_id_i in self.knowledge_graph.nodes and node_id_j in self.knowledge_graph.nodes:
                    node_i = self.knowledge_graph.nodes[node_id_i]
                    node_j = self.knowledge_graph.nodes[node_id_j]
                    
                    # Calculate direct similarity between nodes
                    direct_similarity = self.knowledge_graph._cosine_similarity(
                        node_i.embedding, node_j.embedding
                    )
                    
                    if direct_similarity > self.merge_similarity:
                        # Create merge candidate
                        merged_text = f"Merged: {node_i.text} | {node_j.text}"
                        merged_embedding = (node_i.embedding + node_j.embedding) / 2
                        
                        merge_candidate = GraphUpdate(
                            update_type=UpdateType.MERGE,
                            target_nodes=[node_id_i, node_id_j],
                            new_node_data={
                                'id': f"merged_{int(time.time())}",
                                'text': merged_text,
                                'embedding': merged_embedding,
                                'node_type': 'merged',
                                'confidence': min(node_i.confidence, node_j.confidence),
                                'timestamp': time.time()
                            },
                            remove_nodes=[node_id_i, node_id_j],
                            metadata={
                                'merge_similarity': direct_similarity,
                                'original_nodes': [node_id_i, node_id_j]
                            }
                        )
                        
                        merge_candidates.append(merge_candidate)
                        
                        # Only create one merge candidate per call to avoid complexity
                        break
            
            if merge_candidates:
                break
        
        return merge_candidates
    
    def _create_prune_candidates(self) -> List[GraphUpdate]:
        """Create pruning candidates for low-value nodes.
        
        Returns:
            List of pruning update candidates
        """
        prune_candidates = []
        
        # Find nodes with low usage
        current_time = time.time()
        
        for node_id, node in self.knowledge_graph.nodes.items():
            # Pruning criteria
            is_old = (current_time - node.timestamp) > 3600  # Older than 1 hour
            is_unused = node.access_count <= self.prune_usage_min
            is_isolated = self.knowledge_graph.graph.degree(node_id) <= 1
            
            if is_old and is_unused and is_isolated:
                prune_candidate = GraphUpdate(
                    update_type=UpdateType.PRUNE,
                    target_nodes=[node_id],
                    remove_nodes=[node_id],
                    metadata={
                        'prune_reason': 'low_usage_isolated',
                        'access_count': node.access_count,
                        'age_hours': (current_time - node.timestamp) / 3600,
                        'degree': self.knowledge_graph.graph.degree(node_id)
                    }
                )
                
                prune_candidates.append(prune_candidate)
                
                # Limit pruning to avoid too many changes at once
                if len(prune_candidates) >= 3:
                    break
        
        return prune_candidates
    
    def _validate_gedig_decision(self, gedig_result: GeDIGResult, update: GraphUpdate) -> bool:
        """Validate geDIG decision with additional domain-specific checks.
        
        Args:
            gedig_result: geDIG evaluation result
            update: Proposed update
            
        Returns:
            True if decision passes validation
        """
        # Check type-specific thresholds
        if update.update_type == UpdateType.ADD:
            return (gedig_result.delta_ig > self.add_ig_threshold and
                   self.add_ged_min <= gedig_result.delta_ged <= self.add_ged_max)
        
        elif update.update_type == UpdateType.MERGE:
            # Merges should have positive information gain
            return gedig_result.delta_ig > 0
        
        elif update.update_type == UpdateType.PRUNE:
            # Pruning should have negative information impact (removing low-value content)
            return gedig_result.delta_ig <= 0
        
        return True
    
    def get_gedig_statistics(self) -> Dict[str, Any]:
        """Get geDIG-specific statistics.
        
        Returns:
            Dictionary of geDIG evaluation statistics
        """
        base_stats = self.gedig_evaluator.get_evaluation_statistics()
        
        # Add local statistics
        total_updates = sum(self.update_type_counts.values())
        
        if total_updates > 0:
            update_type_rates = {
                f"{update_type}_rate": count / total_updates 
                for update_type, count in self.update_type_counts.items()
            }
        else:
            update_type_rates = {}
        
        return {
            'gedig_evaluations': self.gedig_evaluations,
            'update_type_counts': self.update_type_counts,
            'total_updates_attempted': total_updates,
            **update_type_rates,
            'gedig_threshold': self.gedig_threshold,
            'threshold_mode': self.threshold_mode,
            'threshold_percentile': self.threshold_percentile,
            'k_coefficient': self.gedig_evaluator.k,
            'evaluator_statistics': base_stats
        }
    
    def analyze_gedig_performance(self) -> Dict[str, Any]:
        """Analyze geDIG evaluation performance over time.
        
        Returns:
            Analysis of geDIG performance patterns
        """
        if not self.gedig_history:
            return {'status': 'no_evaluations'}
        
        # Extract performance metrics
        delta_geds = [r.delta_ged for r in self.gedig_history]
        delta_igs = [r.delta_ig for r in self.gedig_history]
        delta_gedigs = [r.delta_gedig for r in self.gedig_history]
        computation_times = [r.computation_time for r in self.gedig_history]
        
        # Acceptance analysis
        accepted_scores = [r.delta_gedig for r in self.gedig_history if r.delta_gedig > self.gedig_threshold]
        rejected_scores = [r.delta_gedig for r in self.gedig_history if r.delta_gedig <= self.gedig_threshold]
        
        return {
            'total_evaluations': len(self.gedig_history),
            'acceptances': len(accepted_scores),
            'rejections': len(rejected_scores),
            'acceptance_rate': len(accepted_scores) / len(self.gedig_history),
            'average_scores': {
                'delta_ged': np.mean(delta_geds),
                'delta_ig': np.mean(delta_igs),
                'delta_gedig': np.mean(delta_gedigs)
            },
            'performance_metrics': {
                'avg_computation_time': np.mean(computation_times),
                'total_computation_time': np.sum(computation_times)
            },
            'threshold_analysis': {
                'current_threshold': self.gedig_threshold,
                'avg_accepted_score': np.mean(accepted_scores) if accepted_scores else 0,
                'avg_rejected_score': np.mean(rejected_scores) if rejected_scores else 0
            }
        }
    
    def reset(self):
        """Reset geDIG tracking and parent state."""
        super().reset()
        self.gedig_evaluations = 0
        self.gedig_history.clear()
        self.update_type_counts = {
            'add': 0,
            'prune': 0,  
            'merge': 0,
            'rejected': 0
        }
        # Reset geDIG evaluator history
        self.gedig_evaluator.evaluation_history.clear()
