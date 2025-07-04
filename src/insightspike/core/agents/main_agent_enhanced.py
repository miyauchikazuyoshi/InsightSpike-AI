"""
Enhanced Main Agent with Graph-Aware Episode Management
======================================================

Integrates enhanced L2 memory manager with graph-based episode management.
"""

import logging
from typing import Dict, Any, Optional

from .main_agent import MainAgent
from ..layers.layer2_enhanced import EnhancedL2MemoryManager, upgrade_to_enhanced

logger = logging.getLogger(__name__)


class EnhancedMainAgent(MainAgent):
    """
    Enhanced version of MainAgent with graph-aware episode management.
    
    Features:
    - Graph-informed episode integration
    - Automatic conflict-based splitting
    - Self-organizing knowledge structure
    """
    
    def __init__(self):
        super().__init__()
        self.use_enhanced_l2 = True
        
    def initialize(self):
        """Initialize layers with enhanced L2 memory manager"""
        logger.info("Initializing Enhanced InsightSpike Main Agent")
        
        # Initialize base components
        super().initialize()
        
        # Upgrade L2 to enhanced version if needed
        if self.use_enhanced_l2 and not isinstance(self.l2_memory, EnhancedL2MemoryManager):
            logger.info("Upgrading to Enhanced L2 Memory Manager")
            self.l2_memory = upgrade_to_enhanced(self.l2_memory)
            
            # Connect L3 graph reference
            if hasattr(self.l2_memory, 'set_layer3_graph'):
                self.l2_memory.set_layer3_graph(self.l3_graph)
                logger.info("Connected L2 to L3 graph")
        
        self._is_initialized = True
        logger.info("Enhanced initialization complete")
        
        return True
    
    def add_episode_with_graph_update(self, text: str) -> Dict[str, Any]:
        """
        Enhanced episode addition with graph-aware integration and splitting.
        
        This method:
        1. Adds episode with graph-aware integration
        2. Updates the knowledge graph
        3. Checks for conflicts and performs splitting if needed
        4. Returns comprehensive results
        """
        try:
            # Encode text using embedder
            from ...utils.embedder import get_model
            model = get_model()
            vector = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
            
            # Initial C-value
            c_value = 0.5  # Default C-value
            
            # Add episode with enhanced integration
            episode_idx = self.l2_memory.add_episode(vector, text, c_value)
            
            if episode_idx < 0:
                return {
                    'success': False,
                    'error': 'Failed to add episode'
                }
            
            # Update graph with all current episodes
            all_documents = []
            for i, episode in enumerate(self.l2_memory.episodes):
                doc = {
                    "text": episode.text,
                    "embedding": episode.vec,
                    "c_value": episode.c,
                    "episode_idx": i
                }
                all_documents.append(doc)
            
            # Analyze with Layer3
            if self.l3_graph:
                graph_analysis = self.l3_graph.analyze_documents(all_documents)
            else:
                graph_analysis = {}
            
            # Get enhanced statistics
            enhanced_stats = {}
            if hasattr(self.l2_memory, 'get_enhanced_stats'):
                enhanced_stats = self.l2_memory.get_enhanced_stats()
            
            result = {
                'success': True,
                'episode_idx': episode_idx,
                'vector': vector,
                'c_value': c_value,
                'total_episodes': len(self.l2_memory.episodes),
                'graph_analysis': graph_analysis,
                'enhanced_stats': enhanced_stats
            }
            
            # Add integration/splitting info if available
            if enhanced_stats:
                integration_stats = enhanced_stats.get('integration_stats', {})
                splitting_stats = enhanced_stats.get('splitting_stats', {})
                
                result['integration_info'] = {
                    'total_integrations': integration_stats.get('successful_integrations', 0),
                    'graph_assisted': integration_stats.get('graph_assisted', 0),
                    'integration_rate': enhanced_stats.get('integration_rate', 0.0)
                }
                
                result['splitting_info'] = {
                    'conflicts_detected': splitting_stats.get('conflicts_detected', 0),
                    'episodes_split': splitting_stats.get('episodes_split', 0),
                    'new_episodes_created': splitting_stats.get('total_new_episodes', 0)
                }
            
            logger.info(f"Enhanced episode processing complete. Total episodes: {len(self.l2_memory.episodes)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced episode addition: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def configure_episode_management(self, 
                                   integration_config: Optional[Dict] = None,
                                   splitting_config: Optional[Dict] = None):
        """
        Configure the enhanced episode management parameters.
        
        Args:
            integration_config: Dict with integration parameters
                - similarity_threshold: float (0-1)
                - graph_weight: float (0-1)
                - graph_connection_bonus: float (0-1)
                - enable_graph_integration: bool
                
            splitting_config: Dict with splitting parameters
                - conflict_threshold: float (0-1)
                - min_connections_for_split: int
                - max_episode_length: int
                - enable_auto_split: bool
        """
        if not hasattr(self.l2_memory, 'integration_config'):
            logger.warning("L2 memory manager is not enhanced version")
            return
        
        if integration_config:
            for key, value in integration_config.items():
                if hasattr(self.l2_memory.integration_config, key):
                    setattr(self.l2_memory.integration_config, key, value)
            logger.info(f"Updated integration config: {integration_config}")
        
        if splitting_config:
            for key, value in splitting_config.items():
                if hasattr(self.l2_memory.splitting_config, key):
                    setattr(self.l2_memory.splitting_config, key, value)
            logger.info(f"Updated splitting config: {splitting_config}")
    
    def trigger_global_optimization(self):
        """
        Manually trigger global conflict checking and resolution.
        
        This can be called periodically to maintain graph quality.
        """
        if hasattr(self.l2_memory, '_check_global_conflicts'):
            logger.info("Triggering global conflict resolution")
            self.l2_memory._check_global_conflicts()
            
            # Get statistics
            stats = self.l2_memory.get_enhanced_stats()
            splitting_stats = stats.get('splitting_stats', {})
            
            logger.info(f"Global optimization complete. Episodes split: {splitting_stats.get('episodes_split', 0)}")
            
            return {
                'success': True,
                'episodes_split': splitting_stats.get('episodes_split', 0),
                'new_episodes': splitting_stats.get('total_new_episodes', 0)
            }
        else:
            return {
                'success': False,
                'error': 'Enhanced L2 not available'
            }
    
    def get_episode_graph_info(self, episode_idx: int) -> Dict[str, Any]:
        """
        Get detailed graph information for a specific episode.
        
        Returns:
            Dict containing:
            - connections: List of connected episode indices
            - conflict_score: Current conflict score
            - integration_candidates: Potential integration targets
        """
        if not (0 <= episode_idx < len(self.l2_memory.episodes)):
            return {'error': 'Invalid episode index'}
        
        info = {
            'episode_idx': episode_idx,
            'text': self.l2_memory.episodes[episode_idx].text[:100] + '...',
            'c_value': self.l2_memory.episodes[episode_idx].c
        }
        
        # Get connections from graph
        if self.l3_graph and hasattr(self.l3_graph, 'previous_graph'):
            graph = self.l3_graph.previous_graph
            if graph and hasattr(graph, 'edge_index'):
                edge_index = graph.edge_index
                connections = edge_index[1][edge_index[0] == episode_idx].tolist()
                info['connections'] = connections
                info['connection_count'] = len(connections)
        
        # Get conflict score
        if hasattr(self.l2_memory, '_calculate_episode_conflict'):
            conflict_score = self.l2_memory._calculate_episode_conflict(episode_idx)
            info['conflict_score'] = conflict_score
            info['needs_splitting'] = conflict_score > self.l2_memory.splitting_config.conflict_threshold
        
        return info