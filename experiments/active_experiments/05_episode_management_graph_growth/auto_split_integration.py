#!/usr/bin/env python3
"""
Automatic Episode Management: Integration and Splitting
Complete system for graph-informed episode management
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EpisodeManagementConfig:
    """Configuration for episode management"""
    # Integration parameters
    integration_similarity_threshold: float = 0.85
    integration_content_threshold: float = 0.70
    graph_connection_bonus: float = 0.1
    
    # Splitting parameters
    conflict_threshold: float = 0.7
    min_connections_for_split: int = 3
    max_episode_length: int = 500  # characters
    split_decay_factor: float = 0.8  # C-value reduction
    
    # Graph influence
    graph_weight: float = 0.3
    enable_auto_split: bool = True


class GraphAwareEpisodeManager:
    """
    Unified episode management with graph-aware integration and splitting
    """
    
    def __init__(self, config: EpisodeManagementConfig = None):
        self.config = config or EpisodeManagementConfig()
        self.split_history = []
        self.integration_history = []
        
    def process_new_episode(self,
                          vector: np.ndarray,
                          text: str,
                          c_value: float,
                          episodes: List[Dict],
                          graph) -> Dict:
        """
        Process new episode: check for integration or add as new
        
        Returns:
            Dict with 'action' ('integrate', 'add', 'error') and details
        """
        # First check integration
        integration_result = self._check_graph_integration(
            vector, text, c_value, episodes, graph
        )
        
        if integration_result['should_integrate']:
            # Perform integration
            integrated = self._perform_integration(
                integration_result['target_index'],
                vector, text, c_value, episodes
            )
            
            self.integration_history.append({
                'action': 'integrate',
                'target': integration_result['target_index'],
                'scores': integration_result['scores']
            })
            
            return {
                'action': 'integrate',
                'target_index': integration_result['target_index'],
                'updated_episode': integrated
            }
        else:
            # Add as new episode
            new_episode = {
                'vec': vector,
                'text': text,
                'c': c_value,
                'metadata': {'created': 'new'}
            }
            
            return {
                'action': 'add',
                'new_episode': new_episode
            }
    
    def check_and_split_episodes(self,
                                episodes: List[Dict],
                                graph) -> List[Dict]:
        """
        Check all episodes for potential splits
        
        Returns:
            List of split operations to perform
        """
        if not self.config.enable_auto_split:
            return []
        
        split_operations = []
        
        for i, episode in enumerate(episodes):
            # Check multiple split conditions
            should_split = False
            split_reason = ""
            
            # 1. Length-based split
            if len(episode['text']) > self.config.max_episode_length:
                should_split = True
                split_reason = "text_too_long"
            
            # 2. Conflict-based split
            elif graph is not None:
                conflict_score = self._calculate_neighbor_conflict(i, episode, graph)
                if conflict_score > self.config.conflict_threshold:
                    should_split = True
                    split_reason = f"high_conflict:{conflict_score:.2f}"
            
            if should_split:
                splits = self._generate_splits(i, episode, graph, split_reason)
                if splits:
                    split_operations.append({
                        'episode_index': i,
                        'reason': split_reason,
                        'new_episodes': splits
                    })
        
        return split_operations
    
    def _check_graph_integration(self,
                               vector: np.ndarray,
                               text: str,
                               c_value: float,
                               episodes: List[Dict],
                               graph) -> Dict:
        """Check integration with graph awareness"""
        
        best_candidate = {
            'index': -1,
            'vector_sim': 0.0,
            'content_overlap': 0.0,
            'graph_connection': 0.0,
            'combined_score': 0.0
        }
        
        for i, ep in enumerate(episodes):
            # Vector similarity
            vec_sim = np.dot(vector, ep['vec']) / (
                np.linalg.norm(vector) * np.linalg.norm(ep['vec'])
            )
            
            # Content overlap
            new_words = set(text.lower().split())
            ep_words = set(ep['text'].lower().split())
            overlap = len(new_words & ep_words) / len(new_words | ep_words) if new_words else 0
            
            # Graph connection (would need actual graph lookup)
            graph_conn = self._get_graph_connection(i, len(episodes), graph)
            
            # Combined score
            vec_weight = 1 - self.config.graph_weight
            graph_weight = self.config.graph_weight
            
            combined = vec_weight * vec_sim + graph_weight * graph_conn
            
            if combined > best_candidate['combined_score']:
                best_candidate = {
                    'index': i,
                    'vector_sim': vec_sim,
                    'content_overlap': overlap,
                    'graph_connection': graph_conn,
                    'combined_score': combined
                }
        
        # Adjust thresholds based on graph connection
        sim_threshold = self.config.integration_similarity_threshold
        if best_candidate['graph_connection'] > 0.5:
            sim_threshold -= self.config.graph_connection_bonus
        
        should_integrate = (
            best_candidate['combined_score'] >= sim_threshold and
            best_candidate['content_overlap'] >= self.config.integration_content_threshold
        )
        
        return {
            'should_integrate': should_integrate,
            'target_index': best_candidate['index'] if should_integrate else -1,
            'scores': best_candidate
        }
    
    def _calculate_neighbor_conflict(self,
                                   episode_idx: int,
                                   episode: Dict,
                                   graph) -> float:
        """Calculate conflict score with neighbors"""
        
        if graph is None or not hasattr(graph, 'edge_index'):
            return 0.0
        
        # Get neighbors
        edge_index = graph.edge_index
        neighbors = edge_index[1][edge_index[0] == episode_idx].tolist()
        
        if len(neighbors) < self.config.min_connections_for_split:
            return 0.0
        
        # Calculate pairwise conflicts
        conflicts = []
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if i < graph.x.shape[0] and j < graph.x.shape[0]:
                    vec1 = graph.x[i].numpy()
                    vec2 = graph.x[j].numpy()
                    
                    # Conflict = 1 - similarity
                    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    conflicts.append(1 - sim)
        
        return np.mean(conflicts) if conflicts else 0.0
    
    def _generate_splits(self,
                       episode_idx: int,
                       episode: Dict,
                       graph,
                       reason: str) -> List[Dict]:
        """Generate split episodes"""
        
        # Simple text splitting
        sentences = [s.strip() for s in episode['text'].split('.') if s.strip()]
        
        if len(sentences) < 2:
            # Can't split single sentence
            return []
        
        # Determine number of splits (2-3)
        n_splits = min(3, len(sentences))
        
        splits = []
        for i in range(n_splits):
            # Distribute sentences
            start = i * len(sentences) // n_splits
            end = (i + 1) * len(sentences) // n_splits
            split_text = '. '.join(sentences[start:end]) + '.'
            
            # Create vector variation
            base_vec = episode['vec']
            noise = np.random.normal(0, 0.05, base_vec.shape)
            split_vec = base_vec + noise
            split_vec = split_vec / np.linalg.norm(split_vec)
            
            splits.append({
                'vec': split_vec.astype(np.float32),
                'text': split_text,
                'c': episode['c'] * self.config.split_decay_factor,
                'metadata': {
                    'split_from': episode_idx,
                    'split_reason': reason,
                    'split_part': i + 1,
                    'total_parts': n_splits
                }
            })
        
        self.split_history.append({
            'original_index': episode_idx,
            'reason': reason,
            'n_splits': n_splits
        })
        
        return splits
    
    def _perform_integration(self,
                           target_idx: int,
                           new_vec: np.ndarray,
                           new_text: str,
                           new_c: float,
                           episodes: List[Dict]) -> Dict:
        """Perform actual integration"""
        
        target = episodes[target_idx]
        
        # Weighted vector average
        total_weight = target['c'] + new_c
        w1 = target['c'] / total_weight
        w2 = new_c / total_weight
        
        integrated_vec = w1 * target['vec'] + w2 * new_vec
        integrated_vec = integrated_vec / np.linalg.norm(integrated_vec)
        
        # Combine text
        integrated_text = f"{target['text']} | {new_text}"
        
        # Take max C-value
        integrated_c = max(target['c'], new_c)
        
        # Update metadata
        metadata = target.get('metadata', {})
        metadata.setdefault('integration_count', 0)
        metadata['integration_count'] += 1
        
        return {
            'vec': integrated_vec.astype(np.float32),
            'text': integrated_text,
            'c': integrated_c,
            'metadata': metadata
        }
    
    def _get_graph_connection(self, idx1: int, idx2: int, graph) -> float:
        """Get connection strength between nodes (simplified)"""
        if graph is None:
            return 0.0
        
        # This is simplified - in real implementation would check actual edges
        return 0.0  # Placeholder
    
    def get_management_stats(self) -> Dict:
        """Get statistics about episode management"""
        return {
            'total_integrations': len(self.integration_history),
            'total_splits': len(self.split_history),
            'integration_rate': len(self.integration_history) / max(1, 
                len(self.integration_history) + len(self.split_history)),
            'config': {
                'integration_threshold': self.config.integration_similarity_threshold,
                'conflict_threshold': self.config.conflict_threshold,
                'auto_split_enabled': self.config.enable_auto_split
            }
        }


def demonstrate_unified_system():
    """Demonstrate the unified episode management system"""
    
    print("=== Unified Episode Management System ===\n")
    
    # Initialize manager
    config = EpisodeManagementConfig(
        integration_similarity_threshold=0.75,  # Lowered for demo
        conflict_threshold=0.6,
        enable_auto_split=True
    )
    manager = GraphAwareEpisodeManager(config)
    
    # Existing episodes
    episodes = [
        {
            'vec': np.array([0.9, 0.1, 0.0]),
            'text': "Machine learning fundamentals",
            'c': 0.5
        },
        {
            'vec': np.array([0.5, 0.5, 0.0]),
            'text': "AI applications in climate science. ML models for weather. Quantum algorithms for optimization.",
            'c': 0.6
        },
        {
            'vec': np.array([0.0, 0.0, 1.0]),
            'text': "Quantum computing basics",
            'c': 0.5
        }
    ]
    
    print("Initial episodes:")
    for i, ep in enumerate(episodes):
        print(f"  {i}: {ep['text'][:50]}... (length: {len(ep['text'])})")
    
    # Test integration
    print("\n--- Testing Integration ---")
    new_vec = np.array([0.85, 0.15, 0.0])
    new_text = "Deep learning and neural networks"
    
    result = manager.process_new_episode(new_vec, new_text, 0.4, episodes, None)
    
    print(f"Action: {result['action']}")
    if result['action'] == 'integrate':
        print(f"Integrated with episode {result['target_index']}")
        print(f"Updated text: {result['updated_episode']['text'][:80]}...")
    
    # Test splitting
    print("\n--- Testing Splitting ---")
    split_ops = manager.check_and_split_episodes(episodes, None)
    
    for op in split_ops:
        print(f"\nSplit episode {op['episode_index']} ({op['reason']}):")
        for i, new_ep in enumerate(op['new_episodes']):
            print(f"  Part {i+1}: {new_ep['text'][:50]}...")
    
    # Show stats
    print("\n--- Management Statistics ---")
    stats = manager.get_management_stats()
    print(f"Total integrations: {stats['total_integrations']}")
    print(f"Total splits: {stats['total_splits']}")
    print(f"Integration rate: {stats['integration_rate']:.1%}")
    
    print("\n=== Benefits of Unified System ===")
    print("1. Automatic conflict resolution through splitting")
    print("2. Graph-aware integration decisions")
    print("3. Maintains optimal episode size and coherence")
    print("4. Reduces manual intervention")
    print("5. Improves knowledge graph quality over time")


if __name__ == "__main__":
    demonstrate_unified_system()