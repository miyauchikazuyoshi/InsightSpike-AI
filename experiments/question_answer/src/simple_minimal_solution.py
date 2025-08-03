#!/usr/bin/env python3
"""
Simple Minimal Solution Experiment
==================================

A simplified version for testing the concept.
"""

import json
import time
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Knowledge:
    """Knowledge entry"""
    id: str
    content: str
    category: str


@dataclass
class Question:
    """Question with metadata"""
    id: str
    question: str
    difficulty: str
    expected_knowledge_ids: List[str]


class SimpleMinimalSolutionExperiment:
    """Simple implementation of minimal solution experiment"""
    
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
    
    def find_minimal_solution_simple(self, question: Question) -> List[Knowledge]:
        """
        Simple knowledge selection based on keyword matching.
        This is a placeholder for actual geDIG implementation.
        """
        # Extract keywords from question
        q_words = set(question.question.lower().split())
        stopwords = {'what', 'is', 'the', 'who', 'when', 'where', 'how', 'a', 'an', 'and', 'or', 'of', 'to', 'in'}
        q_keywords = q_words - stopwords
        
        # Score each knowledge item
        scored_knowledge = []
        for knowledge in self.knowledge_base:
            k_words = set(knowledge.content.lower().split())
            k_keywords = k_words - stopwords
            
            # Calculate overlap score
            overlap = len(q_keywords & k_keywords)
            if overlap > 0:
                score = overlap / max(len(q_keywords), 1)
                scored_knowledge.append((knowledge, score))
        
        # Sort by score
        scored_knowledge.sort(key=lambda x: x[1], reverse=True)
        
        # Select top items with diminishing returns
        selected = []
        cumulative_score = 0.0
        
        for knowledge, score in scored_knowledge:
            if cumulative_score >= 0.8 or len(selected) >= 5:
                break
            selected.append(knowledge)
            cumulative_score += score * (0.7 ** len(selected))  # Diminishing returns
        
        return selected
    
    def generate_mock_answer(self, question: str, knowledge_list: List[Knowledge]) -> str:
        """Generate a mock answer based on selected knowledge"""
        if not knowledge_list:
            return "No relevant knowledge found to answer this question."
        
        # Create a simple answer combining knowledge
        knowledge_points = [k.content for k in knowledge_list[:3]]  # Use top 3
        
        return f"Based on the available knowledge: {' '.join(knowledge_points[:2])}"
    
    def generate_direct_answer(self, question: str) -> str:
        """Generate a direct answer without knowledge base"""
        return f"Direct answer attempt: This question requires specific knowledge that I don't have access to."
    
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
        """Run the experiment"""
        logger.info("Starting simple minimal solution experiment...")
        
        for i, question in enumerate(self.questions):
            logger.info(f"\nProcessing question {i+1}/{len(self.questions)}: {question.id}")
            
            # Time the selection
            start_time = time.time()
            
            # Find minimal solution
            selected_knowledge = self.find_minimal_solution_simple(question)
            
            selection_time = time.time() - start_time
            
            # Generate answers
            knowledge_answer = self.generate_mock_answer(question.question, selected_knowledge)
            direct_answer = self.generate_direct_answer(question.question)
            
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
                'knowledge_answer': knowledge_answer,
                'direct_answer': direct_answer
            }
            
            self.results.append(result)
            
            # Log summary
            logger.info(f"  Selected {len(selected_knowledge)} knowledge items")
            logger.info(f"  F1 score: {evaluation['f1']:.2f}")
    
    def save_results(self, output_dir: str):
        """Save experiment results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary
        summary = {
            'experiment_date': datetime.now().isoformat(),
            'total_questions': len(self.results),
            'avg_selected_knowledge': np.mean([r['selected_knowledge_count'] for r in self.results]),
            'avg_f1': np.mean([r['evaluation']['f1'] for r in self.results]),
            'avg_selection_time': np.mean([r['selection_time'] for r in self.results])
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"simple_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'results': self.results
            }, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        logger.info(f"Average selected knowledge: {summary['avg_selected_knowledge']:.1f}")
        logger.info(f"Average F1 score: {summary['avg_f1']:.2f}")


def main():
    """Run the experiment"""
    # Setup paths
    knowledge_path = "data/input/knowledge_base/minimal_test_knowledge.json"
    questions_path = "data/input/questions/minimal_test_questions.json"
    output_dir = "results/test"
    
    # Create and run experiment
    experiment = SimpleMinimalSolutionExperiment(knowledge_path, questions_path)
    experiment.run_experiment()
    experiment.save_results(output_dir)
    
    print("\nSimple experiment completed!")


if __name__ == "__main__":
    main()