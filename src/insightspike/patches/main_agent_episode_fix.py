"""
Fix for MainAgent Episode creation issue
"""

import logging
from typing import Optional, Dict, Any
import time
import numpy as np

logger = logging.getLogger(__name__)


def apply_main_agent_episode_fix():
    """Fix MainAgent.add_knowledge to use 'confidence' parameter."""
    from ..implementations.agents.main_agent import MainAgent
    from ..core.episode import Episode
    
    # Save original method
    if hasattr(MainAgent, '_original_add_knowledge'):
        # Already patched
        return
    
    MainAgent._original_add_knowledge = MainAgent.add_knowledge
    
    def patched_add_knowledge(self, text: str, c_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Add new knowledge to the agent's memory.
        
        Args:
            text: The knowledge text to add
            c_value: Confidence value (0-1)
            
        Returns:
            Dict with success status and details
        """
        if not self._initialized:
            return {"success": False, "error": "Agent not initialized"}
        
        try:
            # Get embedding
            embedding = self.l1_embedder.get_embedding(text)
            
            # Ensure 1D shape
            if isinstance(embedding, np.ndarray) and embedding.ndim > 1:
                embedding = embedding.squeeze()
            
            # Create episode with correct parameter name
            episode = Episode(
                text=text,
                vec=embedding,
                confidence=c_value or 0.5,  # Use 'confidence' not 'c'
                timestamp=time.time(),
                metadata={"c_value": c_value or 0.5}
            )
            
            # Add to memory
            episode_idx = self.l2_memory.add_episode(episode)
            
            # Update graph if available
            if hasattr(self, 'l3_graph') and self.l3_graph:
                try:
                    self.l3_graph.update_graph([episode])
                except Exception as e:
                    logger.warning(f"Graph update failed: {e}")
            
            return {
                "success": True,
                "episode_idx": episode_idx,
                "text": text,
                "c_value": c_value or 0.5
            }
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return {"success": False, "error": str(e)}
    
    # Apply patch
    MainAgent.add_knowledge = patched_add_knowledge
    logger.info("Applied MainAgent Episode creation fix")


def verify_episode_fix():
    """Verify that Episode creation works correctly."""
    try:
        from ..core.episode import Episode
        import numpy as np
        
        # Test creating episode with 'confidence' parameter
        ep = Episode(
            text="test",
            vec=np.zeros(384),
            confidence=0.5,
            timestamp=time.time()
        )
        logger.info("Episode creation fix verified")
        return True
        
    except Exception as e:
        logger.error(f"Episode creation still broken: {e}")
        return False