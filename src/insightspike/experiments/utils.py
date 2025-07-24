"""Experiment utilities for InsightSpike experiments."""

import logging
import time
from typing import Dict, Any, List, Optional
import numpy as np

from ..implementations.agents.main_agent import MainAgent

logger = logging.getLogger(__name__)


def create_experiment_config(provider='mock', model='mock-model'):
    """Create a simple configuration for experiments.
    
    Args:
        provider: LLM provider ('mock', 'clean', 'local', 'openai', etc.)
        model: Model name to use
    
    Returns:
        Config object with necessary attributes
    """
    class Config:
        def __init__(self):
            self.graph = type('GraphConfig', (), {
                'similarity_threshold': 0.7,
                'conflict_threshold': 0.5,
                'ged_threshold': 0.3
            })()
            
            self.embedding = type('EmbeddingConfig', (), {
                'dimension': 384,  # all-MiniLM-L6-v2 uses 384 dimensions
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
            })()
            
            self.llm = type('LLMConfig', (), {
                'provider': provider,
                'model_name': model,
                'temperature': 0.7,
                'max_tokens': 512,
                'top_p': 0.9,
                'device': 'cpu',
                'timeout': 60
            })()
            
            self.memory = type('MemoryConfig', (), {
                'max_episodes': 1000,
                'compression_enabled': False,
                'max_retrieved_docs': 10
            })()
            
            self.insight = type('InsightConfig', (), {
                'detection_threshold': 0.5,
                'min_confidence': 0.3
            })()
    
    return Config()


class ExperimentRunner:
    """Helper class for running experiments."""
    
    @staticmethod
    def quick_test(provider='mock', n_questions=5):
        """Run a quick test with the specified provider.
        
        Args:
            provider: LLM provider to use
            n_questions: Number of test questions
        
        Returns:
            List of results
        """
        # Create config
        config = create_experiment_config(provider)
        
        # Initialize agent
        logger.info(f"Initializing agent with {provider} provider")
        agent = MainAgent(config)
        
        # Add some initial knowledge
        knowledge_items = [
            "The sky is blue because of Rayleigh scattering.",
            "Water freezes at 0 degrees Celsius.",
            "The Earth orbits around the Sun.",
            "Photosynthesis converts light energy into chemical energy.",
            "Gravity causes objects to attract each other."
        ]
        
        for item in knowledge_items:
            result = agent.add_knowledge(item)
            logger.info(f"Added knowledge: {item[:50]}... (success: {result.get('success', False)})")
        
        # Generate test questions
        test_questions = [
            "Why is the sky blue?",
            "What happens to water at 0 degrees?",
            "How does the Earth move in space?",
            "What is photosynthesis?",
            "What causes things to fall?"
        ][:n_questions]
        
        # Run tests
        results = []
        for i, question in enumerate(test_questions):
            logger.info(f"Processing question {i+1}/{n_questions}: {question}")
            start_time = time.time()
            
            try:
                result = agent.process_question(question)
                processing_time = time.time() - start_time
                
                # Handle both CycleResult object and dict
                if hasattr(result, 'has_spike'):
                    results.append({
                        'question': question,
                        'response': result.response,
                        'has_spike': result.has_spike,
                        'confidence': getattr(result, 'confidence', 0.0),
                        'processing_time': processing_time,
                        'success': True
                    })
                else:
                    # Fallback for dict format
                    results.append({
                        'question': question,
                        'response': result.get('response', ''),
                        'has_spike': result.get('spike_detected', False),
                        'confidence': result.get('reasoning_quality', 0.0),
                        'processing_time': processing_time,
                        'success': result.get('success', False)
                    })
                    
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                results.append({
                    'question': question,
                    'response': f"Error: {str(e)}",
                    'has_spike': False,
                    'confidence': 0.0,
                    'processing_time': time.time() - start_time,
                    'success': False
                })
        
        # Summary
        logger.info(f"\nQuick test completed:")
        logger.info(f"- Provider: {provider}")
        logger.info(f"- Questions processed: {len(results)}")
        logger.info(f"- Successful: {sum(1 for r in results if r['success'])}")
        logger.info(f"- Spikes detected: {sum(1 for r in results if r['has_spike'])}")
        logger.info(f"- Average processing time: {np.mean([r['processing_time'] for r in results]):.2f}s")
        
        return results
    
    @staticmethod
    def test_providers(providers=['mock', 'clean'], n_questions=3):
        """Test multiple providers and compare results.
        
        Args:
            providers: List of providers to test
            n_questions: Number of questions per provider
        
        Returns:
            Dict with results for each provider
        """
        all_results = {}
        
        for provider in providers:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing provider: {provider}")
            logger.info(f"{'='*50}")
            
            try:
                results = ExperimentRunner.quick_test(provider, n_questions)
                all_results[provider] = {
                    'results': results,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                logger.error(f"Failed to test {provider}: {e}")
                all_results[provider] = {
                    'results': [],
                    'success': False,
                    'error': str(e)
                }
        
        # Compare results
        logger.info(f"\n{'='*50}")
        logger.info("Provider Comparison Summary")
        logger.info(f"{'='*50}")
        
        for provider, data in all_results.items():
            if data['success']:
                results = data['results']
                logger.info(f"\n{provider}:")
                logger.info(f"  - Success rate: {sum(1 for r in results if r['success'])/len(results)*100:.1f}%")
                logger.info(f"  - Spike detection rate: {sum(1 for r in results if r['has_spike'])/len(results)*100:.1f}%")
                logger.info(f"  - Avg processing time: {np.mean([r['processing_time'] for r in results]):.2f}s")
            else:
                logger.info(f"\n{provider}: FAILED - {data['error']}")
        
        return all_results


def run_minimal_test():
    """Run a minimal test to verify basic functionality."""
    logger.info("Running minimal functionality test...")
    
    try:
        # Test with mock provider (fastest)
        config = create_experiment_config('mock')
        agent = MainAgent(config)
        
        # Add one piece of knowledge
        logger.info("Adding test knowledge...")
        result = agent.add_knowledge("Test knowledge")
        logger.info(f"Add knowledge result: {result}")
        
        if not result.get('success', False):
            # Try direct store_episode
            logger.info("Trying direct store_episode...")
            idx = agent.l2_memory.store_episode("Test knowledge", 0.5)
            logger.info(f"Direct store_episode returned: {idx}")
            
            if idx < 0:
                # Check if embedding model is initialized
                logger.info(f"Embedding model: {agent.l2_memory.embedding_model}")
                logger.info(f"L2Memory initialized: {agent.l2_memory.initialized}")
                logger.info(f"Episodes count: {len(agent.l2_memory.episodes)}")
        
        assert result.get('success', False) or idx >= 0, "Failed to add knowledge"
        
        # Ask one question
        response = agent.process_question("Test question")
        
        # Check response format
        if hasattr(response, 'response'):
            logger.info(f"✓ Got response: {response.response[:50]}...")
        else:
            logger.info(f"✓ Got response: {response.get('response', '')[:50]}...")
        
        logger.info("✓ Minimal test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Minimal test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False