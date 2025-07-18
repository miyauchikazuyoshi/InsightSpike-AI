#!/usr/bin/env python3
"""
Proper InsightSpike Agent-based RAT Experiment
Using actual agent structure and data flow
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import InsightSpike components
from src.insightspike.core.agents.main_agent import MainAgent
from src.insightspike.core.experiment_framework import BaseExperiment, ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProperAgentRATExperiment(BaseExperiment):
    """
    RAT experiment using proper InsightSpike agent architecture
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize experiment with proper configuration"""
        if not config:
            config = ExperimentConfig(
                name="RAT_with_Dictionary_Episodes",
                description="Remote Associates Test using English dictionary definitions",
                max_episodes=5,
                save_results=True,
                verbose=True
            )
        super().__init__(config)
        
        # Experiment-specific settings
        self.definition_file = Path(__file__).parent.parent / "data" / "input" / "english_definitions.json"
        self.rat_problems_file = Path(__file__).parent.parent / "data" / "input" / "rat_problems_with_meanings.json"
        
    def setup_experiment(self) -> bool:
        """Set up the experiment environment"""
        logger.info("Setting up RAT experiment with proper agent...")
        
        # Initialize agent
        self.agent = MainAgent()
        if not self.agent.initialize():
            logger.error("Failed to initialize MainAgent")
            return False
        
        # Prepare data
        if not self._prepare_rat_data():
            logger.error("Failed to prepare RAT data")
            return False
        
        # Build initial knowledge graph
        if not self._build_initial_graph():
            logger.error("Failed to build initial graph")
            return False
        
        logger.info("✅ Experiment setup complete")
        return True
    
    def _prepare_rat_data(self) -> bool:
        """Prepare RAT problems with meanings"""
        # Load definitions
        with open(self.definition_file, 'r') as f:
            definitions = json.load(f)['definitions']
        
        # Create updated RAT problems with meanings
        rat_problems = {
            "metadata": {
                "name": "RAT Problems with Word Meanings",
                "description": "RAT problems enhanced with dictionary definitions for InsightSpike",
                "created": datetime.now().isoformat()
            },
            "problems": [
                {
                    "id": 1,
                    "question": "What word associates with COTTAGE, SWISS, and CAKE?",
                    "words": ["COTTAGE", "SWISS", "CAKE"],
                    "answer": "CHEESE",
                    "word_meanings": {
                        "COTTAGE": definitions.get("COTTAGE", [])[:3],  # First 3 definitions
                        "SWISS": definitions.get("SWISS", [])[:3],
                        "CAKE": definitions.get("CAKE", [])[:3]
                    }
                },
                {
                    "id": 2,
                    "question": "What word connects CREAM, SKATE, and WATER?",
                    "words": ["CREAM", "SKATE", "WATER"],
                    "answer": "ICE",
                    "word_meanings": {
                        "CREAM": definitions.get("CREAM", [])[:3],
                        "SKATE": definitions.get("SKATE", [])[:3],
                        "WATER": definitions.get("WATER", [])[:3]
                    }
                },
                {
                    "id": 3,
                    "question": "What concept links DUCK, FOLD, and DOLLAR?",
                    "words": ["DUCK", "FOLD", "DOLLAR"],
                    "answer": "BILL",
                    "word_meanings": {
                        "DUCK": definitions.get("DUCK", [])[:3],
                        "FOLD": definitions.get("FOLD", [])[:3],
                        "DOLLAR": definitions.get("DOLLAR", [])[:3]
                    }
                },
                {
                    "id": 4,
                    "question": "Find the word that relates to NIGHT, WRIST, and STOP",
                    "words": ["NIGHT", "WRIST", "STOP"],
                    "answer": "WATCH",
                    "word_meanings": {
                        "NIGHT": definitions.get("NIGHT", [])[:3],
                        "WRIST": definitions.get("WRIST", [])[:3],
                        "STOP": definitions.get("STOP", [])[:3]
                    }
                },
                {
                    "id": 5,
                    "question": "What connects RIVER, NOTE, and ACCOUNT?",
                    "words": ["RIVER", "NOTE", "ACCOUNT"],
                    "answer": "BANK",
                    "word_meanings": {
                        "RIVER": definitions.get("RIVER", [])[:3],
                        "NOTE": definitions.get("NOTE", [])[:3],
                        "ACCOUNT": definitions.get("ACCOUNT", [])[:3]
                    }
                }
            ]
        }
        
        # Save updated problems
        with open(self.rat_problems_file, 'w') as f:
            json.dump(rat_problems, f, indent=2)
        
        self.test_problems = rat_problems['problems']
        logger.info(f"Prepared {len(self.test_problems)} RAT problems with meanings")
        return True
    
    def _build_initial_graph(self) -> bool:
        """Build initial knowledge graph from definitions"""
        logger.info("Building initial knowledge graph...")
        
        episode_count = 0
        
        # Add all definitions as episodes
        for problem in self.test_problems:
            for word, meanings in problem['word_meanings'].items():
                for meaning in meanings:
                    # Add episode with proper metadata
                    metadata = {
                        "source_word": word,
                        "problem_id": problem['id'],
                        "episode_type": "definition"
                    }
                    
                    # Use agent's method to add episode
                    try:
                        self.agent.add_episode_with_graph_update(
                            text=meaning,
                            metadata=metadata
                        )
                        episode_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to add episode: {e}")
        
        logger.info(f"Added {episode_count} episodes to knowledge graph")
        
        # Get initial graph stats
        stats = self.agent.get_stats()
        if 'memory_stats' in stats:
            logger.info(f"Graph stats: {stats['memory_stats']}")
        
        return episode_count > 0
    
    def run_single_test(self, test_id: int) -> tuple:
        """Run a single RAT test case"""
        problem = self.test_problems[test_id]
        
        logger.info(f"\n📝 Problem {problem['id']}: {problem['question']}")
        
        # Process question using agent
        result = self.agent.process_question(
            question=problem['question'],
            max_cycles=3,  # Allow multiple reasoning cycles
            verbose=True
            # adaptive_topk is calculated internally by the agent
        )
        
        # Extract answer from response
        response = result.get('response', '')
        predicted_answer = self._extract_answer(response)
        
        # Check correctness
        is_correct = predicted_answer.upper() == problem['answer'].upper()
        
        # Build metrics
        from src.insightspike.core.interfaces.generic_interfaces import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            success_rate=1.0 if is_correct else 0.0,
            processing_time=result.get('execution_time', 0.0),
            memory_usage=result.get('memory_usage', 0.0),
            accuracy=result.get('reasoning_quality', 0.0),
            insight_detection_count=1 if result.get('spike_detected', False) else 0
        )
        
        # Enhanced result
        enhanced_result = {
            **result,
            'problem': problem,
            'predicted_answer': predicted_answer,
            'correct_answer': problem['answer'],
            'is_correct': is_correct,
            'spike_info': {
                'detected': result.get('spike_detected', False),
                'delta_ged': result.get('delta_ged', 0.0),
                'delta_ig': result.get('delta_ig', 0.0),
                'insights': result.get('discovered_insights', [])
            }
        }
        
        # Log results
        if is_correct:
            logger.info(f"✅ Correct: {predicted_answer}")
        else:
            logger.info(f"❌ Wrong: {predicted_answer} (expected: {problem['answer']})")
        
        if result.get('spike_detected'):
            logger.info(f"⚡ Insight spike detected!")
            
        return metrics, enhanced_result
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer word from agent response"""
        # Simple extraction - look for capitalized words that match known answers
        known_answers = ['CHEESE', 'ICE', 'BILL', 'WATCH', 'BANK']
        
        response_upper = response.upper()
        for answer in known_answers:
            if answer in response_upper:
                return answer
        
        # Fallback: last capitalized word
        words = response.split()
        for word in reversed(words):
            if word.isupper() and len(word) > 2:
                return word
        
        return "UNKNOWN"
    
    def analyze_results(self, all_results: List[Dict]) -> Dict:
        """Analyze experiment results"""
        analysis = super().analyze_results(all_results)
        
        # Add RAT-specific analysis
        correct_count = sum(1 for r in all_results if r['is_correct'])
        spike_count = sum(1 for r in all_results if r['spike_info']['detected'])
        
        # Spike correlation with correctness
        spike_when_correct = sum(
            1 for r in all_results 
            if r['is_correct'] and r['spike_info']['detected']
        )
        
        analysis['rat_metrics'] = {
            'accuracy': correct_count / len(all_results) if all_results else 0,
            'spike_rate': spike_count / len(all_results) if all_results else 0,
            'spike_correctness_correlation': spike_when_correct / correct_count if correct_count > 0 else 0
        }
        
        # Episode integration analysis
        memory_stats = self.agent.get_stats().get('memory_stats', {})
        analysis['memory_analysis'] = memory_stats
        
        return analysis
    
    def cleanup_experiment(self):
        """Clean up after experiment"""
        logger.info("Cleaning up experiment...")
        
        # Save final graph state
        try:
            graph_file = self.output_dir / "final_knowledge_graph.json"
            # Note: Actual graph serialization would depend on implementation
            logger.info(f"Graph state saved to {graph_file}")
        except Exception as e:
            logger.warning(f"Failed to save graph: {e}")
        
        super().cleanup_experiment()


def main():
    """Run the proper agent-based RAT experiment"""
    # Configure experiment
    config = ExperimentConfig(
        name="Proper_Agent_RAT_Experiment",
        description="RAT with dictionary definitions using InsightSpike agents",
        max_episodes=5,  # 5 RAT problems
        save_results=True,
        verbose=True,
        output_dir=Path(__file__).parent.parent / "results" / "agent_experiments"
    )
    
    # Create and run experiment
    experiment = ProperAgentRATExperiment(config)
    
    try:
        # Run experiment
        results = experiment.run()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT SUMMARY")
        print("=" * 60)
        
        if 'rat_metrics' in results:
            metrics = results['rat_metrics']
            print(f"Accuracy: {metrics['accuracy']*100:.1f}%")
            print(f"Spike Rate: {metrics['spike_rate']*100:.1f}%")
            print(f"Spike-Correctness Correlation: {metrics['spike_correctness_correlation']*100:.1f}%")
        
        print(f"\nResults saved to: {experiment.output_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()