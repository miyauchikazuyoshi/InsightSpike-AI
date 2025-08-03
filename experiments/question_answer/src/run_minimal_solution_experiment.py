#!/usr/bin/env python3
"""
Minimal Solution Experiment with InsightSpike Integration
========================================================

Compare geDIG's minimal knowledge selection with direct LLM answers.
This implementation integrates with the actual InsightSpike system.
"""

import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# InsightSpike imports
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.features.graph_reasoning.graph_analyzer import GraphAnalyzer
from insightspike.algorithms.metrics_selector import UnifiedMetricsSelector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Knowledge:
    """Knowledge entry"""
    id: str
    content: str
    category: str
    embedding: Optional[np.ndarray] = None


@dataclass
class Question:
    """Question with metadata"""
    id: str
    question: str
    difficulty: str
    expected_knowledge_ids: List[str]
    embedding: Optional[np.ndarray] = None


class MinimalSolutionExperimentInsightSpike:
    """Run minimal solution selection experiment with InsightSpike"""
    
    def __init__(self, knowledge_path: str, questions_path: str, config_path: Optional[str] = None):
        self.knowledge_base = self._load_knowledge(knowledge_path)
        self.questions = self._load_questions(questions_path)
        self.results = []
        
        # Initialize InsightSpike components
        self._init_insightspike(config_path)
        
    def _init_insightspike(self, config_path: Optional[str] = None):
        """Initialize InsightSpike components"""
        # Create a simple config for MainAgent
        config = {
            'llm': {
                'provider': 'mock',
                'mock': {
                    'response_template': 'Based on the provided knowledge: [response]',
                    'spike_probability': 0.0
                }
            },
            'datastore': {
                'type': 'filesystem',
                'base_path': 'data/experiment_temp'
            },
            'algorithms': {
                'gedig': {
                    'w_ged': 2.0,
                    'k_temperature': 0.1
                }
            }
        }
        
        # Initialize MainAgent
        self.agent = MainAgent(config)
        
        # Initialize geDIG core
        self.gedig = GeDIGCore(
            w_ged=2.0,
            k_temperature=0.1
        )
        
        # Initialize graph analyzer
        self.graph_analyzer = GraphAnalyzer()
        
        # Initialize metrics selector
        self.metrics_selector = UnifiedMetricsSelector(metric_type="scipy")
        
        logger.info("InsightSpike components initialized")
        
    def _load_knowledge(self, path: str) -> List[Knowledge]:
        """Load knowledge base"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        knowledge_list = []
        for item in data:
            knowledge_list.append(Knowledge(
                id=item['id'],
                content=item['content'],
                category=item.get('category', 'general')
            ))
        
        logger.info(f"Loaded {len(knowledge_list)} knowledge entries")
        return knowledge_list
    
    def _load_questions(self, path: str) -> List[Question]:
        """Load questions"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            questions.append(Question(
                id=item['id'],
                question=item['question'],
                difficulty=item['difficulty'],
                expected_knowledge_ids=item.get('expected_knowledge_ids', [])
            ))
        
        logger.info(f"Loaded {len(questions)} questions")
        return questions
    
    def _add_knowledge_to_agent(self):
        """Add all knowledge to the agent"""
        logger.info("Adding knowledge to agent...")
        for knowledge in self.knowledge_base:
            self.agent.add_knowledge(knowledge.content)
        logger.info("Knowledge added to agent")
    
    def find_minimal_solution_gedig(self, question: Question) -> Tuple[List[Knowledge], Dict]:
        """
        Find minimal knowledge set using geDIG optimization.
        
        Returns:
            Tuple of (selected_knowledge, selection_metrics)
        """
        # Process question through agent to get current state
        _ = self.agent.process_question(question.question)
        
        # Get graph structure
        graph = self.agent.layers['graph_reasoning']['layer']._build_networkx_graph()
        
        # Analyze graph for relevant nodes
        analysis = self.graph_analyzer.analyze_relevance(graph, question.question)
        
        # Sort knowledge by relevance
        relevance_scores = []
        for knowledge in self.knowledge_base:
            # Find corresponding node in graph
            node_id = None
            for nid, data in graph.nodes(data=True):
                if data.get('label', '') == knowledge.content[:50]:  # Match by content prefix
                    node_id = nid
                    break
            
            if node_id and node_id in analysis['node_relevance']:
                score = analysis['node_relevance'][node_id]
            else:
                score = 0.0
            
            relevance_scores.append((knowledge, score))
        
        # Sort by relevance
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select minimal set using geDIG optimization
        selected = []
        cumulative_score = 0.0
        threshold = 0.8  # Coverage threshold
        
        for knowledge, score in relevance_scores:
            if cumulative_score >= threshold:
                break
            if score > 0.3:  # Minimum relevance
                selected.append(knowledge)
                cumulative_score += score * 0.5  # Diminishing returns
        
        # Limit to reasonable number
        selected = selected[:5]
        
        # Calculate selection metrics
        metrics = {
            'total_candidates': len(self.knowledge_base),
            'selected_count': len(selected),
            'avg_relevance': np.mean([s[1] for k, s in relevance_scores[:len(selected)]]) if selected else 0.0,
            'coverage_score': cumulative_score
        }
        
        return selected, metrics
    
    def get_llm_answer_with_knowledge(self, question: str, knowledge_list: List[Knowledge]) -> str:
        """Get LLM answer using selected knowledge"""
        if not knowledge_list:
            return "No relevant knowledge found."
        
        # Create context from selected knowledge
        context = "\n".join([f"- {k.content}" for k in knowledge_list])
        
        # Use agent's LLM to answer with context
        prompt = f"""Based on the following knowledge:
{context}

Please answer this question: {question}"""
        
        # Process through agent
        result = self.agent.process_question(prompt)
        if hasattr(result, 'response'):
            return result.response
        else:
            return result.get('response', 'No response generated')
    
    def get_llm_direct_answer(self, question: str) -> str:
        """Get direct LLM answer without knowledge base"""
        # Create a fresh agent without knowledge
        config = {
            'llm': {
                'provider': 'mock',
                'mock': {
                    'response_template': 'Direct answer: [response]',
                    'spike_probability': 0.0
                }
            },
            'datastore': {
                'type': 'filesystem',
                'base_path': 'data/experiment_temp_fresh'
            }
        }
        fresh_agent = MainAgent(config)
        
        result = fresh_agent.process_question(question)
        if hasattr(result, 'response'):
            return result.response
        else:
            return result.get('response', 'No response generated')
    
    def evaluate_selection(self, question: Question, selected: List[Knowledge]) -> Dict:
        """Evaluate knowledge selection quality"""
        selected_ids = {k.id for k in selected}
        expected_ids = set(question.expected_knowledge_ids)
        
        # Calculate metrics
        true_positive = len(selected_ids & expected_ids)
        false_positive = len(selected_ids - expected_ids)
        false_negative = len(expected_ids - selected_ids)
        
        precision = true_positive / len(selected_ids) if selected_ids else 0.0
        recall = true_positive / len(expected_ids) if expected_ids else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'selected_count': len(selected),
            'expected_count': len(expected_ids),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
    
    def run_experiment(self):
        """Run the full experiment"""
        logger.info("Starting minimal solution experiment with InsightSpike...")
        
        # Add knowledge to agent
        self._add_knowledge_to_agent()
        
        for i, question in enumerate(self.questions):
            logger.info(f"\nProcessing question {i+1}/{len(self.questions)}: {question.id}")
            logger.info(f"Question: {question.question[:100]}...")
            
            try:
                # Time the selection
                start_time = time.time()
                
                # Find minimal solution with geDIG
                selected_knowledge, selection_metrics = self.find_minimal_solution_gedig(question)
                
                selection_time = time.time() - start_time
                
                # Get answers
                gedig_answer = self.get_llm_answer_with_knowledge(
                    question.question, 
                    selected_knowledge
                )
                direct_answer = self.get_llm_direct_answer(question.question)
                
                # Evaluate selection
                evaluation = self.evaluate_selection(question, selected_knowledge)
                
                # Store results
                result = {
                    'question_id': question.id,
                    'question': question.question,
                    'difficulty': question.difficulty,
                    'selected_knowledge_count': len(selected_knowledge),
                    'selected_knowledge_ids': [k.id for k in selected_knowledge],
                    'selected_knowledge_content': [k.content[:100] + '...' for k in selected_knowledge],
                    'selection_time': selection_time,
                    'selection_metrics': selection_metrics,
                    'evaluation': evaluation,
                    'gedig_answer': gedig_answer[:500],  # Truncate for storage
                    'direct_answer': direct_answer[:500]  # Truncate for storage
                }
                
                self.results.append(result)
                
                # Log summary
                logger.info(f"  Selected {len(selected_knowledge)} knowledge items")
                logger.info(f"  Precision: {evaluation['precision']:.2f}, "
                           f"Recall: {evaluation['recall']:.2f}, "
                           f"F1: {evaluation['f1']:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing question {question.id}: {e}")
                self.results.append({
                    'question_id': question.id,
                    'error': str(e)
                })
    
    def save_results(self, output_dir: str):
        """Save experiment results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        valid_results = [r for r in self.results if 'error' not in r]
        
        summary = {
            'experiment_date': datetime.now().isoformat(),
            'total_questions': len(self.results),
            'successful_questions': len(valid_results),
            'failed_questions': len(self.results) - len(valid_results),
            'avg_selected_knowledge': np.mean([r['selected_knowledge_count'] for r in valid_results]) if valid_results else 0,
            'avg_precision': np.mean([r['evaluation']['precision'] for r in valid_results]) if valid_results else 0,
            'avg_recall': np.mean([r['evaluation']['recall'] for r in valid_results]) if valid_results else 0,
            'avg_f1': np.mean([r['evaluation']['f1'] for r in valid_results]) if valid_results else 0,
            'avg_selection_time': np.mean([r['selection_time'] for r in valid_results]) if valid_results else 0,
            'by_difficulty': {}
        }
        
        # Group by difficulty
        for difficulty in ['easy', 'medium', 'hard', 'very_hard']:
            difficulty_results = [r for r in valid_results if r['difficulty'] == difficulty]
            if difficulty_results:
                summary['by_difficulty'][difficulty] = {
                    'count': len(difficulty_results),
                    'avg_selected': np.mean([r['selected_knowledge_count'] for r in difficulty_results]),
                    'avg_f1': np.mean([r['evaluation']['f1'] for r in difficulty_results]),
                    'avg_precision': np.mean([r['evaluation']['precision'] for r in difficulty_results]),
                    'avg_recall': np.mean([r['evaluation']['recall'] for r in difficulty_results])
                }
        
        # Save full results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"minimal_solution_results_{timestamp}.json"
        
        output_data = {
            'summary': summary,
            'results': self.results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save summary separately
        summary_file = output_path / f"minimal_solution_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Minimal Solution Experiment Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {summary['experiment_date']}\n")
            f.write(f"Total questions: {summary['total_questions']}\n")
            f.write(f"Successful: {summary['successful_questions']}\n")
            f.write(f"Failed: {summary['failed_questions']}\n\n")
            
            f.write("Overall Performance:\n")
            f.write(f"  Average selected knowledge: {summary['avg_selected_knowledge']:.1f}\n")
            f.write(f"  Average F1 score: {summary['avg_f1']:.3f}\n")
            f.write(f"  Average precision: {summary['avg_precision']:.3f}\n")
            f.write(f"  Average recall: {summary['avg_recall']:.3f}\n")
            f.write(f"  Average selection time: {summary['avg_selection_time']:.3f}s\n\n")
            
            f.write("Performance by Difficulty:\n")
            for difficulty, stats in summary['by_difficulty'].items():
                f.write(f"\n  {difficulty.upper()}:\n")
                f.write(f"    Questions: {stats['count']}\n")
                f.write(f"    Avg selected: {stats['avg_selected']:.1f}\n")
                f.write(f"    Avg F1: {stats['avg_f1']:.3f}\n")
                f.write(f"    Avg Precision: {stats['avg_precision']:.3f}\n")
                f.write(f"    Avg Recall: {stats['avg_recall']:.3f}\n")
        
        logger.info(f"\nResults saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")
        
        # Print summary
        print(f"\n{'='*50}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*50}")
        print(f"Average selected knowledge: {summary['avg_selected_knowledge']:.1f}")
        print(f"Average F1 score: {summary['avg_f1']:.3f}")
        print(f"Success rate: {summary['successful_questions']}/{summary['total_questions']}")


def main():
    """Run the experiment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run minimal solution experiment")
    parser.add_argument("--knowledge", default="data/input/knowledge_base/knowledge_500.json",
                        help="Path to knowledge base file")
    parser.add_argument("--questions", default="data/input/questions/questions_100.json",
                        help="Path to questions file")
    parser.add_argument("--config", default=None,
                        help="Path to experiment config file")
    parser.add_argument("--output", default="results/metrics",
                        help="Output directory for results")
    parser.add_argument("--test", action="store_true",
                        help="Run with test data")
    
    args = parser.parse_args()
    
    # Use test data if requested
    if args.test:
        args.knowledge = "data/input/knowledge_base/test_knowledge.json"
        args.questions = "data/input/questions/test_questions.json"
    
    # Create experiment
    experiment = MinimalSolutionExperimentInsightSpike(
        args.knowledge,
        args.questions,
        args.config
    )
    
    # Run experiment
    experiment.run_experiment()
    
    # Save results
    experiment.save_results(args.output)
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()