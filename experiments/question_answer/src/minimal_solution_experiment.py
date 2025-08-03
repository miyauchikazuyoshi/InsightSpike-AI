#!/usr/bin/env python3
"""
Minimal Solution Experiment
===========================

Compare geDIG's minimal knowledge selection with direct LLM answers.
"""

import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
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
    expected_knowledge_ids: List[str]  # For evaluation
    embedding: Optional[np.ndarray] = None


class MinimalSolutionExperiment:
    """Run minimal solution selection experiment"""
    
    def __init__(self, knowledge_path: str, questions_path: str):
        self.knowledge_base = self._load_knowledge(knowledge_path)
        self.questions = self._load_questions(questions_path)
        self.results = []
        
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
    
    def find_minimal_solution_gedig(self, question: Question) -> List[Knowledge]:
        """
        Find minimal knowledge set using geDIG optimization.
        
        This is a simplified version - in real implementation,
        we would use actual geDIG with graph operations.
        """
        # Step 1: Compute relevance scores
        relevance_scores = []
        for knowledge in self.knowledge_base:
            score = self._compute_relevance(question.question, knowledge.content)
            relevance_scores.append((knowledge, score))
        
        # Step 2: Sort by relevance
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Select minimal set using threshold
        # In real geDIG, this would involve graph optimization
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
        return selected[:5]
    
    def _compute_relevance(self, question: str, knowledge: str) -> float:
        """
        Compute relevance score between question and knowledge.
        
        In real implementation, this would use embeddings and
        more sophisticated similarity measures.
        """
        # Simple keyword overlap for demo
        q_words = set(question.lower().split())
        k_words = set(knowledge.lower().split())
        
        if not q_words or not k_words:
            return 0.0
        
        overlap = len(q_words & k_words)
        return overlap / max(len(q_words), len(k_words))
    
    def get_llm_answer_with_knowledge(self, question: str, knowledge_list: List[Knowledge]) -> str:
        """Get LLM answer using selected knowledge"""
        # In real implementation, this would call actual LLM
        # For now, return a structured response
        if not knowledge_list:
            return "No relevant knowledge found."
        
        knowledge_text = "\n".join([f"- {k.content}" for k in knowledge_list])
        return f"Based on the following knowledge:\n{knowledge_text}\n\nAnswer: [LLM response here]"
    
    def get_llm_direct_answer(self, question: str) -> str:
        """Get direct LLM answer without knowledge base"""
        # In real implementation, this would call actual LLM
        return f"Direct answer to '{question}': [LLM direct response here]"
    
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
            'expected_count': len(expected_ids)
        }
    
    def run_experiment(self):
        """Run the full experiment"""
        logger.info("Starting minimal solution experiment...")
        
        for i, question in enumerate(self.questions):
            logger.info(f"\nProcessing question {i+1}/{len(self.questions)}: {question.id}")
            
            # Time the selection
            start_time = time.time()
            
            # Find minimal solution with geDIG
            selected_knowledge = self.find_minimal_solution_gedig(question)
            
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
                'selection_time': selection_time,
                'evaluation': evaluation,
                'gedig_answer': gedig_answer,
                'direct_answer': direct_answer
            }
            
            self.results.append(result)
            
            # Log summary
            logger.info(f"  Selected {len(selected_knowledge)} knowledge items")
            logger.info(f"  Precision: {evaluation['precision']:.2f}, "
                       f"Recall: {evaluation['recall']:.2f}, "
                       f"F1: {evaluation['f1']:.2f}")
    
    def save_results(self, output_path: str):
        """Save experiment results"""
        # Calculate summary statistics
        summary = {
            'total_questions': len(self.results),
            'avg_selected_knowledge': np.mean([r['selected_knowledge_count'] for r in self.results]),
            'avg_precision': np.mean([r['evaluation']['precision'] for r in self.results]),
            'avg_recall': np.mean([r['evaluation']['recall'] for r in self.results]),
            'avg_f1': np.mean([r['evaluation']['f1'] for r in self.results]),
            'by_difficulty': {}
        }
        
        # Group by difficulty
        for difficulty in ['easy', 'medium', 'hard', 'very_hard']:
            difficulty_results = [r for r in self.results if r['difficulty'] == difficulty]
            if difficulty_results:
                summary['by_difficulty'][difficulty] = {
                    'count': len(difficulty_results),
                    'avg_selected': np.mean([r['selected_knowledge_count'] for r in difficulty_results]),
                    'avg_f1': np.mean([r['evaluation']['f1'] for r in difficulty_results])
                }
        
        # Save full results
        output_data = {
            'summary': summary,
            'results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to {output_path}")
        logger.info(f"Summary:")
        logger.info(f"  Average selected knowledge: {summary['avg_selected_knowledge']:.1f}")
        logger.info(f"  Average F1 score: {summary['avg_f1']:.2f}")


def main():
    """Run the experiment"""
    # Setup paths
    knowledge_path = "data/input/knowledge_base/knowledge_500.json"
    questions_path = "data/input/questions/questions_100.json"
    output_path = "results/minimal_solution_experiment.json"
    
    # Create experiment
    experiment = MinimalSolutionExperiment(knowledge_path, questions_path)
    
    # Run experiment
    experiment.run_experiment()
    
    # Save results
    experiment.save_results(output_path)
    
    print("\nExperiment completed!")
    print("This is a demonstration implementation.")
    print("For real experiment, integrate with actual geDIG and LLM.")


if __name__ == "__main__":
    main()