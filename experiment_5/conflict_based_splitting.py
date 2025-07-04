#!/usr/bin/env python3
"""
Conflict-based Episode Splitting
Split episodes when their connections have extreme conflicts
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ConflictMetrics:
    """Metrics for measuring conflict between connected nodes"""
    semantic_conflict: float  # How different the meanings are
    directional_conflict: float  # How opposite the directions are
    cluster_conflict: float  # How different the topic clusters are
    total_conflict: float  # Combined conflict score


class ConflictBasedSplitter:
    """
    Split episodes based on conflicts in their graph connections
    """
    
    def __init__(self,
                 conflict_threshold: float = 0.7,
                 min_connections: int = 3,
                 split_ratio: float = 0.4):
        """
        Initialize conflict-based splitter
        
        Args:
            conflict_threshold: Minimum conflict score to trigger split
            min_connections: Minimum connections needed to evaluate conflicts
            split_ratio: Minimum ratio of conflicting connections to trigger split
        """
        self.conflict_threshold = conflict_threshold
        self.min_connections = min_connections
        self.split_ratio = split_ratio
    
    def calculate_conflict(self, 
                         episode_vec: np.ndarray,
                         neighbor_vecs: List[np.ndarray]) -> ConflictMetrics:
        """
        Calculate conflict between an episode and its neighbors
        
        Args:
            episode_vec: Vector of the episode
            neighbor_vecs: Vectors of connected neighbors
            
        Returns:
            ConflictMetrics object
        """
        if len(neighbor_vecs) < 2:
            return ConflictMetrics(0, 0, 0, 0)
        
        conflicts = []
        
        # Calculate pairwise conflicts among neighbors
        for i in range(len(neighbor_vecs)):
            for j in range(i + 1, len(neighbor_vecs)):
                vec1, vec2 = neighbor_vecs[i], neighbor_vecs[j]
                
                # Semantic conflict (1 - similarity)
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                semantic_conflict = 1 - similarity
                
                # Directional conflict (opposing vectors)
                direction_sim = np.dot(vec1, -vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                directional_conflict = max(0, direction_sim)  # High when vectors point opposite
                
                conflicts.append({
                    'semantic': semantic_conflict,
                    'directional': directional_conflict
                })
        
        if not conflicts:
            return ConflictMetrics(0, 0, 0, 0)
        
        # Average conflicts
        avg_semantic = np.mean([c['semantic'] for c in conflicts])
        avg_directional = np.mean([c['directional'] for c in conflicts])
        
        # Cluster conflict (variance in neighbor directions)
        neighbor_array = np.array(neighbor_vecs)
        cluster_variance = np.mean(np.var(neighbor_array, axis=0))
        cluster_conflict = min(1.0, cluster_variance * 2)  # Normalize to 0-1
        
        # Total conflict score
        total_conflict = (
            0.5 * avg_semantic +
            0.3 * avg_directional +
            0.2 * cluster_conflict
        )
        
        return ConflictMetrics(
            semantic_conflict=avg_semantic,
            directional_conflict=avg_directional,
            cluster_conflict=cluster_conflict,
            total_conflict=total_conflict
        )
    
    def identify_conflicting_groups(self,
                                  episode_vec: np.ndarray,
                                  neighbor_data: List[Dict]) -> List[List[int]]:
        """
        Identify groups of neighbors that conflict with each other
        
        Args:
            episode_vec: Vector of the episode
            neighbor_data: List of dicts with 'vec', 'idx', and 'text'
            
        Returns:
            List of groups, each containing indices of similar neighbors
        """
        if len(neighbor_data) < 2:
            return [list(range(len(neighbor_data)))]
        
        # Simple clustering based on similarity
        groups = []
        assigned = set()
        
        for i, data_i in enumerate(neighbor_data):
            if i in assigned:
                continue
                
            group = [i]
            assigned.add(i)
            
            for j, data_j in enumerate(neighbor_data):
                if j <= i or j in assigned:
                    continue
                
                # Check similarity
                sim = np.dot(data_i['vec'], data_j['vec']) / (
                    np.linalg.norm(data_i['vec']) * np.linalg.norm(data_j['vec'])
                )
                
                if sim > 0.7:  # Similar enough to be in same group
                    group.append(j)
                    assigned.add(j)
            
            groups.append(group)
        
        return groups
    
    def should_split_episode(self,
                           episode_idx: int,
                           episode_data: Dict,
                           graph) -> Tuple[bool, Optional[Dict]]:
        """
        Check if an episode should be split based on conflicts
        
        Args:
            episode_idx: Index of episode in graph
            episode_data: Episode data with 'vec' and 'text'
            graph: PyTorch Geometric graph
            
        Returns:
            (should_split, split_info)
        """
        if graph is None or not hasattr(graph, 'edge_index'):
            return False, None
        
        # Get neighbors
        edge_index = graph.edge_index
        neighbors_mask = edge_index[0] == episode_idx
        neighbor_indices = edge_index[1][neighbors_mask].tolist()
        
        if len(neighbor_indices) < self.min_connections:
            return False, {"reason": "insufficient_connections"}
        
        # Get neighbor vectors
        neighbor_vecs = []
        neighbor_data = []
        for idx in neighbor_indices:
            if idx < graph.x.shape[0]:
                vec = graph.x[idx].numpy()
                neighbor_vecs.append(vec)
                neighbor_data.append({
                    'vec': vec,
                    'idx': idx,
                    'text': f"Neighbor {idx}"  # Would get actual text in real implementation
                })
        
        # Calculate conflict metrics
        conflict = self.calculate_conflict(episode_data['vec'], neighbor_vecs)
        
        if conflict.total_conflict < self.conflict_threshold:
            return False, {"reason": "low_conflict", "conflict_score": conflict.total_conflict}
        
        # Identify conflicting groups
        groups = self.identify_conflicting_groups(episode_data['vec'], neighbor_data)
        
        # Check if we have significant conflicting groups
        if len(groups) < 2:
            return False, {"reason": "no_distinct_groups"}
        
        # Calculate split ratio
        largest_group = max(len(g) for g in groups)
        split_potential = 1 - (largest_group / len(neighbor_indices))
        
        if split_potential < self.split_ratio:
            return False, {"reason": "insufficient_split_ratio", "ratio": split_potential}
        
        # Should split!
        split_info = {
            "reason": "high_conflict",
            "conflict_metrics": conflict,
            "neighbor_groups": groups,
            "split_potential": split_potential,
            "suggested_splits": len(groups)
        }
        
        return True, split_info
    
    def generate_split_episodes(self,
                              episode_data: Dict,
                              split_info: Dict) -> List[Dict]:
        """
        Generate new episodes from a split
        
        Args:
            episode_data: Original episode data
            split_info: Information about the split
            
        Returns:
            List of new episode dictionaries
        """
        groups = split_info['neighbor_groups']
        n_splits = min(len(groups), 3)  # Max 3 splits
        
        # Parse original text
        sentences = [s.strip() for s in episode_data['text'].split('.') if s.strip()]
        
        # If not enough sentences, split by other means
        if len(sentences) < n_splits:
            # Split by words or phrases
            words = episode_data['text'].split()
            chunk_size = len(words) // n_splits
            sentences = []
            for i in range(n_splits):
                start = i * chunk_size
                end = start + chunk_size if i < n_splits - 1 else len(words)
                sentences.append(' '.join(words[start:end]))
        
        new_episodes = []
        base_vec = episode_data['vec']
        
        for i in range(n_splits):
            # Create variation of vector aligned with each group
            if i < len(groups) and groups[i]:
                # Adjust vector towards this group's direction
                group_center = np.mean([split_info['neighbor_data'][idx]['vec'] 
                                      for idx in groups[i]], axis=0)
                # Blend original vector with group direction
                new_vec = 0.7 * base_vec + 0.3 * group_center
                new_vec = new_vec / np.linalg.norm(new_vec)
            else:
                # Add small random variation
                noise = np.random.normal(0, 0.1, base_vec.shape)
                new_vec = base_vec + noise
                new_vec = new_vec / np.linalg.norm(new_vec)
            
            # Create new episode
            new_episode = {
                'vec': new_vec.astype(np.float32),
                'text': sentences[i] if i < len(sentences) else f"Split {i+1} of original",
                'c': episode_data.get('c', 0.5) * 0.8,  # Reduce importance
                'metadata': {
                    'split_from': episode_data.get('idx', -1),
                    'split_reason': 'conflict_resolution',
                    'conflict_score': split_info['conflict_metrics'].total_conflict,
                    'aligned_with_group': i
                }
            }
            new_episodes.append(new_episode)
        
        return new_episodes


def demonstrate_conflict_splitting():
    """Demonstrate conflict-based splitting"""
    
    print("=== Conflict-Based Episode Splitting ===\n")
    
    # Create example episode with conflicting connections
    episode = {
        'idx': 0,
        'vec': np.array([0.5, 0.5, 0.0]),  # Mixed topic
        'text': "AI can help with climate research. Quantum computing challenges traditional ML. Biology uses neural networks.",
        'c': 0.6
    }
    
    # Conflicting neighbors
    neighbors = [
        # Group 1: AI/ML focused
        {'vec': np.array([0.9, 0.1, 0.0]), 'idx': 1, 'text': "Deep learning advances"},
        {'vec': np.array([0.8, 0.2, 0.0]), 'idx': 2, 'text': "Neural network research"},
        # Group 2: Climate/Biology focused  
        {'vec': np.array([0.1, 0.9, 0.0]), 'idx': 3, 'text': "Climate modeling"},
        {'vec': np.array([0.0, 0.8, 0.2]), 'idx': 4, 'text': "Biological systems"},
        # Group 3: Quantum focused
        {'vec': np.array([0.0, 0.0, 1.0]), 'idx': 5, 'text': "Quantum algorithms"},
    ]
    
    splitter = ConflictBasedSplitter()
    
    # Calculate conflicts
    neighbor_vecs = [n['vec'] for n in neighbors]
    conflict = splitter.calculate_conflict(episode['vec'], neighbor_vecs)
    
    print("Episode:", episode['text'][:50] + "...")
    print("\nNeighbors:")
    for n in neighbors:
        print(f"  {n['idx']}: {n['text']}")
    
    print(f"\nConflict Metrics:")
    print(f"  Semantic conflict: {conflict.semantic_conflict:.3f}")
    print(f"  Directional conflict: {conflict.directional_conflict:.3f}")
    print(f"  Cluster conflict: {conflict.cluster_conflict:.3f}")
    print(f"  Total conflict: {conflict.total_conflict:.3f}")
    
    # Identify groups
    groups = splitter.identify_conflicting_groups(episode['vec'], neighbors)
    print(f"\nIdentified {len(groups)} conflicting groups:")
    for i, group in enumerate(groups):
        members = [neighbors[idx]['text'] for idx in group]
        print(f"  Group {i+1}: {members}")
    
    # Generate splits
    split_info = {
        'neighbor_groups': groups,
        'conflict_metrics': conflict,
        'neighbor_data': neighbors
    }
    
    new_episodes = splitter.generate_split_episodes(episode, split_info)
    
    print(f"\nGenerated {len(new_episodes)} split episodes:")
    for i, ep in enumerate(new_episodes):
        print(f"\n  Split {i+1}:")
        print(f"    Text: {ep['text']}")
        print(f"    Vector direction: {ep['vec'][:3]}")
        print(f"    Aligned with group: {ep['metadata']['aligned_with_group']}")
    
    print("\n=== Benefits ===")
    print("1. Reduces conflicts in graph structure")
    print("2. Creates more coherent topic clusters")
    print("3. Improves future integration accuracy")
    print("4. Maintains graph harmony")


if __name__ == "__main__":
    demonstrate_conflict_splitting()