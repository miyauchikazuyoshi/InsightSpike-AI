#!/usr/bin/env python3
"""
Minimal InsightSpike Test
========================

Test InsightSpike's core functionality with minimal dependencies.
"""

import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Any

from sentence_transformers import SentenceTransformer
import numpy as np

from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.layers.layer2_memory_manager import CompatibleL2MemoryManager as Memory
from insightspike.core.episode import Episode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinimalInsightSpikeTest:
    """Minimal test of InsightSpike functionality."""
    
    def __init__(self):
        self.experiment_dir = Path(__file__).parent.parent
        self.data_dir = self.experiment_dir / "data"
        self.results_dir = self.experiment_dir / "results"
        
        # Initialize encoder
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize memory manager directly
        self.memory = Memory(dim=384)
        
        # Questions
        self.questions = [
            "What is the relationship between energy and information?",
            "What is entropy?"
        ]
    
    def load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load knowledge base."""
        kb_path = self.data_dir / "input" / "english_knowledge_base.json"
        with open(kb_path, 'r') as f:
            data = json.load(f)
        return data['episodes']
    
    def populate_memory(self):
        """Populate memory with knowledge base."""
        logger.info("Populating memory with knowledge base...")
        
        knowledge_base = self.load_knowledge_base()
        
        for item in knowledge_base:
            # Encode text
            vector = self.encoder.encode(item['text'])
            
            # Add to memory
            self.memory.add(
                vec=vector,
                text=item['text'],
                c=0.5,
                metadata={
                    'id': item['id'],
                    'phase': item['phase']
                }
            )
        
        logger.info(f"Added {len(knowledge_base)} episodes to memory")
    
    def test_retrieval(self):
        """Test simple retrieval."""
        logger.info("\n=== Testing Retrieval ===")
        
        results = []
        
        for question in self.questions:
            logger.info(f"\nQuestion: {question}")
            
            # Encode question
            q_vector = self.encoder.encode(question)
            
            # Search memory
            distances, indices = self.memory.search(q_vector, k=5)
            
            # Get retrieved episodes
            retrieved = []
            phases = set()
            
            for idx in indices:
                if idx < len(self.memory.episodes):
                    episode = self.memory.episodes[idx]
                    phases.add(episode.metadata.get('phase', 0))
                    retrieved.append({
                        'text': episode.text,
                        'phase': episode.metadata.get('phase', 0),
                        'distance': float(distances[retrieved.__len__()])
                    })
            
            logger.info(f"Retrieved {len(retrieved)} documents from {len(phases)} phases")
            
            # Check for multi-phase integration
            spike_potential = len(phases) >= 3
            logger.info(f"Spike potential: {spike_potential} (phases: {sorted(phases)})")
            
            results.append({
                'question': question,
                'retrieved_count': len(retrieved),
                'phases_integrated': len(phases),
                'spike_potential': spike_potential,
                'top_results': retrieved[:3]
            })
        
        return results
    
    def test_with_agent(self):
        """Test with MainAgent (simplified)."""
        logger.info("\n=== Testing with MainAgent ===")
        
        # Create config
        config = ConfigPresets.experiment()
        config.llm.provider = "clean"  # Use clean provider to avoid LLM issues
        
        # Create a simple datastore-like object
        class SimpleDataStore:
            def __init__(self, memory):
                self.memory = memory
        
        datastore = SimpleDataStore(self.memory)
        
        # Test one question
        question = self.questions[0]
        logger.info(f"\nTesting: {question}")
        
        try:
            # Create agent
            agent = MainAgent(config=config, datastore=datastore)
            
            # Just test initialization
            if agent.initialize():
                logger.info("Agent initialized successfully")
                
                # Try to get a simple response
                result = agent.l4_llm.generate_response(
                    context={'retrieved_documents': []},
                    question=question
                )
                logger.info(f"LLM Response: {result[:100]}...")
            else:
                logger.error("Failed to initialize agent")
                
        except Exception as e:
            logger.error(f"Agent test failed: {e}")
    
    def run(self):
        """Run the minimal test."""
        logger.info("Starting minimal InsightSpike test")
        
        # Populate memory
        self.populate_memory()
        
        # Test retrieval
        retrieval_results = self.test_retrieval()
        
        # Test with agent
        self.test_with_agent()
        
        # Save results
        results = {
            'test': 'minimal_insightspike',
            'retrieval_results': retrieval_results,
            'memory_stats': {
                'total_episodes': len(self.memory.episodes),
                'index_size': self.memory.index.ntotal if hasattr(self.memory.index, 'ntotal') else len(self.memory.episodes)
            }
        }
        
        output_path = self.results_dir / "outputs" / "minimal_test_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {output_path}")
        
        # Summary
        logger.info("\n=== Summary ===")
        for r in retrieval_results:
            logger.info(f"Question: {r['question'][:50]}...")
            logger.info(f"  - Retrieved: {r['retrieved_count']} docs")
            logger.info(f"  - Phases: {r['phases_integrated']}")
            logger.info(f"  - Spike potential: {r['spike_potential']}")


def main():
    """Main entry point."""
    test = MinimalInsightSpikeTest()
    test.run()


if __name__ == "__main__":
    main()