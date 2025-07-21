"""
Experiment and Demo Runners
===========================

Separate module for running experiments and demos using MainAgent.
This follows the Single Responsibility Principle by keeping
MainAgent focused on its core reasoning functionality.
"""

import logging
from typing import Any, Dict, List, Optional

from ..implementations.agents.main_agent import MainAgent

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs various experiments using a MainAgent instance."""
    
    def __init__(self, agent: MainAgent):
        """
        Initialize with an existing MainAgent instance.
        
        Args:
            agent: Initialized MainAgent to use for experiments
        """
        self.agent = agent
        
    def run_simple_experiment(self, episodes: int = 5) -> Dict[str, Any]:
        """Run simple Q&A experiment."""
        test_questions = [
            "What is machine learning?",
            "How does deep learning work?",
            "What is the difference between AI and ML?",
            "Explain neural networks",
            "What is reinforcement learning?",
            "How does natural language processing work?",
            "What are transformers in AI?",
            "Explain gradient descent",
            "What is overfitting?",
            "How do convolutional neural networks work?"
        ]
        
        results = {"type": "simple", "episodes": []}
        
        for i, question in enumerate(test_questions[:episodes]):
            logger.info(f"Simple experiment episode {i+1}/{episodes}")
            result = self.agent.process_question(question, max_cycles=3)
            
            results["episodes"].append({
                "question": question,
                "response": result.response,
                "success": len(result.response) > 50
            })
            
        return results
        
    def run_insight_experiment(self, episodes: int = 5) -> Dict[str, Any]:
        """Run insight detection experiment."""
        # First, add some knowledge
        knowledge_pieces = [
            "Machine learning uses algorithms to learn from data.",
            "Neural networks are inspired by biological neurons.",
            "Deep learning is a subset of machine learning using deep neural networks.",
            "Backpropagation is used to train neural networks.",
            "Gradient descent optimizes model parameters."
        ]
        
        for knowledge in knowledge_pieces:
            self.agent.add_knowledge(text=knowledge)
            
        # Then ask insight-provoking questions
        insight_questions = [
            "How are neural networks and deep learning related?",
            "What connects backpropagation and gradient descent?",
            "Why is deep learning considered part of machine learning?",
            "How do algorithms learn from biological inspiration?",
            "What optimization methods are used in neural networks?"
        ]
        
        results = {"type": "insight", "episodes": []}
        
        for i, question in enumerate(insight_questions[:episodes]):
            logger.info(f"Insight experiment episode {i+1}/{episodes}")
            result = self.agent.process_question(question, max_cycles=5)
            
            results["episodes"].append({
                "question": question,
                "response": result.response,
                "spike_detected": result.spike_detected,
                "reasoning_quality": result.reasoning_quality
            })
            
        return results
        
    def run_math_experiment(self, episodes: int = 5) -> Dict[str, Any]:
        """Run mathematical reasoning experiment."""
        math_problems = [
            "What is 15 + 27?",
            "Calculate 156 / 12",
            "What is the square root of 144?",
            "Solve: 2x + 5 = 13",
            "What is 25% of 80?"
        ]
        
        results = {"type": "math", "episodes": []}
        
        for i, problem in enumerate(math_problems[:episodes]):
            logger.info(f"Math experiment episode {i+1}/{episodes}")
            result = self.agent.process_question(problem, max_cycles=2)
            
            results["episodes"].append({
                "question": problem,
                "answer": result.response
            })
            
        return results
        
    def run(self, experiment_type: str, episodes: int = 5) -> Dict[str, Any]:
        """
        Run specified experiment type.
        
        Args:
            experiment_type: Type of experiment (simple, insight, math)
            episodes: Number of episodes to run
            
        Returns:
            Dict containing experiment results
        """
        try:
            if experiment_type == "simple":
                return self.run_simple_experiment(episodes)
            elif experiment_type == "insight":
                return self.run_insight_experiment(episodes)
            elif experiment_type == "math":
                return self.run_math_experiment(episodes)
            else:
                return {
                    "type": experiment_type,
                    "error": f"Unknown experiment type: {experiment_type}"
                }
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return {
                "type": experiment_type,
                "error": str(e)
            }


class DemoRunner:
    """Runs interactive demos using a MainAgent instance."""
    
    def __init__(self, agent: MainAgent):
        """
        Initialize with an existing MainAgent instance.
        
        Args:
            agent: Initialized MainAgent to use for demos
        """
        self.agent = agent
        
    def run(self) -> List[Dict[str, Any]]:
        """Run interactive demo showcasing capabilities."""
        demo_data = [
            {
                "title": "Basic Knowledge Storage",
                "action": "store",
                "content": "Python is a high-level programming language known for its simplicity.",
            },
            {
                "title": "Knowledge Retrieval",
                "action": "query",
                "question": "What do you know about Python?",
            },
            {
                "title": "Adding Related Knowledge",
                "action": "store",
                "content": "Python was created by Guido van Rossum and emphasizes code readability.",
            },
            {
                "title": "Insight Detection",
                "action": "query",
                "question": "Why is Python popular for beginners?",
            },
            {
                "title": "Complex Knowledge Integration",
                "action": "store",
                "content": "Machine learning frameworks like TensorFlow and PyTorch are written in Python.",
            },
            {
                "title": "Insight Spike Detection",
                "action": "query",
                "question": "How does Python relate to modern AI development?",
            }
        ]
        
        results = []
        
        for step in demo_data:
            result = {"step": step["title"]}
            
            if step["action"] == "store":
                # Store knowledge
                self.agent.add_knowledge(text=step["content"])
                result["action"] = "stored"
                logger.info(f"Demo: Stored knowledge - {step['title']}")
                
            elif step["action"] == "query":
                # Query and check for insights
                response = self.agent.process_question(step["question"], max_cycles=3)
                result.update({
                    "question": step["question"],
                    "answer": response.response,
                    "spike_detected": response.spike_detected,
                    "quality": response.reasoning_quality
                })
                logger.info(f"Demo: Query - {step['title']}")
                
            results.append(result)
            
        return results