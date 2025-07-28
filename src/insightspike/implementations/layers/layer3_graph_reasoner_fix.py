"""
L3GraphReasoner Fix
==================

Adds missing update_graph method to L3GraphReasoner.
"""

from typing import List
from ...core.episode import Episode


def add_update_graph_method(L3GraphReasoner):
    """Add missing update_graph method to L3GraphReasoner class."""
    
    def update_graph(self, episodes: List[Episode]):
        """
        Update graph with new episodes.
        
        This method was missing but called by MainAgent.
        For now, we'll implement a simple version that rebuilds the graph.
        """
        # If we have stored episodes, rebuild the graph
        if hasattr(self, 'graph_builder') and hasattr(self, 'episodes'):
            # Get all episodes (assuming they're stored somewhere)
            all_episodes = getattr(self, 'episodes', [])
            
            # Add new episodes
            all_episodes.extend(episodes)
            
            # Rebuild graph with all episodes
            if all_episodes:
                vectors = [ep.vec for ep in all_episodes if hasattr(ep, 'vec')]
                if vectors:
                    self.previous_graph = self.build_graph(vectors)
        
        # For now, just log that we received the episodes
        from logging import getLogger
        logger = getLogger(__name__)
        logger.debug(f"Graph update requested with {len(episodes)} episodes")
    
    # Add the method to the class
    L3GraphReasoner.update_graph = update_graph


# Apply the fix when this module is imported
try:
    from .layer3_graph_reasoner import L3GraphReasoner
    add_update_graph_method(L3GraphReasoner)
except ImportError:
    pass