#!/usr/bin/env python3
"""
Large-Scale Insight Task Benchmarks for Google Colab
===================================================

Based on docs/experiment_design/01_insight_task_benchmarks.md
Implements comprehensive benchmarks for insight discovery tasks.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import wandb

# Add project root to path
import sys
sys.path.append('/content/InsightSpike-AI/src')

from insightspike.config import InsightSpikeConfig
from insightspike.core.agents.main_agent import MainAgent
from insightspike.legacy.agent_loop import cycle as insight_cycle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightBenchmarkSuite:
    """Comprehensive benchmark suite for insight discovery tasks"""
    
    def __init__(self, config_path: Optional[str] = None, use_wandb: bool = True):
        """Initialize benchmark suite"""
        self.config_path = config_path or "experiments/colab_experiments/colab_config.yaml"
        self.use_wandb = use_wandb
        self.results_dir = Path("experiments/colab_experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Weights & Biases if requested
        if self.use_wandb:
            wandb.init(
                project="insightspike-benchmarks",
                config={
                    "environment": "colab",
                    "experiment_type": "insight_benchmarks"
                }
            )
    
    def run_rat_benchmark(self, n_problems: int = 100) -> Dict[str, Any]:
        """
        Run Remote Associates Test benchmark
        
        Args:
            n_problems: Number of RAT problems to test
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"🧪 Running RAT Benchmark with {n_problems} problems...")
        
        # Load or generate RAT problems
        rat_problems = self._load_rat_problems(n_problems)
        
        results = {
            'task': 'Remote Associates Test',
            'n_problems': n_problems,
            'problems': [],
            'metrics': {}
        }
        
        # Initialize agent
        config = InsightSpikeConfig()
        if self.config_path and Path(self.config_path).exists():
            # Load custom config
            import yaml
            with open(self.config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                # Apply custom settings
                config.core.device = custom_config.get('core', {}).get('device', 'cuda')
                config.core.use_gpu = custom_config.get('core', {}).get('use_gpu', True)
        
        agent = MainAgent(config)
        
        # Process each problem
        correct = 0
        spike_count = 0
        total_time = 0
        
        for i, problem in enumerate(tqdm(rat_problems, desc="Processing RAT problems")):
            start_time = datetime.now()
            
            try:
                # Format question
                question = f"What word associates with {problem['word1']}, {problem['word2']}, and {problem['word3']}?"
                
                # Process with InsightSpike
                result = agent.process_question(question)
                
                # Extract answer
                predicted = self._extract_answer(result.get('response', ''))
                is_correct = predicted.lower() == problem['answer'].lower()
                
                if is_correct:
                    correct += 1
                
                # Check for spike
                spike_detected = result.get('spike_detected', False)
                if spike_detected:
                    spike_count += 1
                
                # Record result
                problem_result = {
                    'problem_id': i + 1,
                    'words': [problem['word1'], problem['word2'], problem['word3']],
                    'expected': problem['answer'],
                    'predicted': predicted,
                    'correct': is_correct,
                    'spike_detected': spike_detected,
                    'delta_ged': result.get('delta_ged', 0.0),
                    'delta_ig': result.get('delta_ig', 0.0),
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
                
                results['problems'].append(problem_result)
                total_time += problem_result['processing_time']
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'rat/accuracy': correct / (i + 1),
                        'rat/spike_rate': spike_count / (i + 1),
                        'rat/processing_time': problem_result['processing_time']
                    })
                    
            except Exception as e:
                logger.error(f"Error processing problem {i+1}: {e}")
                results['problems'].append({
                    'problem_id': i + 1,
                    'error': str(e)
                })
        
        # Calculate metrics
        results['metrics'] = {
            'accuracy': correct / n_problems * 100,
            'spike_rate': spike_count / n_problems * 100,
            'avg_processing_time': total_time / n_problems,
            'total_time': total_time,
            'spike_accuracy_correlation': self._calculate_spike_correlation(results['problems'])
        }
        
        # Save results
        self._save_results(results, 'rat_benchmark')
        
        logger.info(f"✅ RAT Benchmark complete: {results['metrics']['accuracy']:.1f}% accuracy")
        
        return results
    
    def run_scientific_discovery_benchmark(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run scientific discovery benchmark
        
        Tests ability to discover relationships in scientific data
        """
        logger.info("🔬 Running Scientific Discovery Benchmark...")
        
        # Load scientific discovery problems
        problems = self._load_scientific_problems(dataset_path)
        
        results = {
            'task': 'Scientific Discovery',
            'n_problems': len(problems),
            'problems': [],
            'metrics': {}
        }
        
        # Initialize agent
        config = InsightSpikeConfig()
        agent = MainAgent(config)
        
        discovered = 0
        novel_insights = 0
        
        for i, problem in enumerate(tqdm(problems, desc="Processing scientific problems")):
            try:
                # Process with InsightSpike
                result = agent.process_question(problem['question'])
                
                # Evaluate discovery
                is_discovered = self._evaluate_discovery(
                    result.get('response', ''),
                    problem['expected_relationships']
                )
                
                if is_discovered:
                    discovered += 1
                
                # Check for novel insights
                if result.get('spike_detected', False):
                    novel_insights += 1
                
                problem_result = {
                    'problem_id': i + 1,
                    'domain': problem['domain'],
                    'discovered': is_discovered,
                    'spike_detected': result.get('spike_detected', False),
                    'reasoning_quality': result.get('reasoning_quality', 0.0)
                }
                
                results['problems'].append(problem_result)
                
            except Exception as e:
                logger.error(f"Error in scientific problem {i+1}: {e}")
        
        # Calculate metrics
        results['metrics'] = {
            'discovery_rate': discovered / len(problems) * 100,
            'novel_insight_rate': novel_insights / len(problems) * 100,
            'avg_reasoning_quality': np.mean([p.get('reasoning_quality', 0) for p in results['problems']])
        }
        
        self._save_results(results, 'scientific_discovery_benchmark')
        
        return results
    
    def run_analogy_completion_benchmark(self, n_problems: int = 50) -> Dict[str, Any]:
        """
        Run analogy completion benchmark
        
        Tests: A is to B as C is to ?
        """
        logger.info("🔗 Running Analogy Completion Benchmark...")
        
        analogies = self._load_analogy_problems(n_problems)
        
        results = {
            'task': 'Analogy Completion',
            'n_problems': n_problems,
            'problems': [],
            'metrics': {}
        }
        
        config = InsightSpikeConfig()
        agent = MainAgent(config)
        
        correct = 0
        semantic_valid = 0
        
        for i, analogy in enumerate(tqdm(analogies, desc="Processing analogies")):
            try:
                question = f"{analogy['a']} is to {analogy['b']} as {analogy['c']} is to what?"
                
                result = agent.process_question(question)
                predicted = self._extract_answer(result.get('response', ''))
                
                is_correct = predicted.lower() == analogy['d'].lower()
                if is_correct:
                    correct += 1
                
                # Check semantic validity
                is_valid = self._check_semantic_validity(
                    analogy['a'], analogy['b'],
                    analogy['c'], predicted
                )
                if is_valid:
                    semantic_valid += 1
                
                problem_result = {
                    'problem_id': i + 1,
                    'analogy': f"{analogy['a']}:{analogy['b']}::{analogy['c']}:?",
                    'expected': analogy['d'],
                    'predicted': predicted,
                    'correct': is_correct,
                    'semantically_valid': is_valid
                }
                
                results['problems'].append(problem_result)
                
            except Exception as e:
                logger.error(f"Error in analogy {i+1}: {e}")
        
        results['metrics'] = {
            'accuracy': correct / n_problems * 100,
            'semantic_validity_rate': semantic_valid / n_problems * 100
        }
        
        self._save_results(results, 'analogy_benchmark')
        
        return results
    
    def run_creative_problem_solving_benchmark(self, n_problems: int = 30) -> Dict[str, Any]:
        """
        Run creative problem solving benchmark
        
        Tests ability to find non-obvious solutions
        """
        logger.info("💡 Running Creative Problem Solving Benchmark...")
        
        problems = self._load_creative_problems(n_problems)
        
        results = {
            'task': 'Creative Problem Solving',
            'n_problems': n_problems,
            'problems': [],
            'metrics': {}
        }
        
        config = InsightSpikeConfig()
        agent = MainAgent(config)
        
        solved = 0
        creative_solutions = 0
        
        for i, problem in enumerate(tqdm(problems, desc="Processing creative problems")):
            try:
                result = agent.process_question(problem['question'])
                
                # Evaluate solution
                solution_quality = self._evaluate_creative_solution(
                    result.get('response', ''),
                    problem['constraints'],
                    problem['criteria']
                )
                
                if solution_quality['is_valid']:
                    solved += 1
                
                if solution_quality['creativity_score'] > 0.7:
                    creative_solutions += 1
                
                problem_result = {
                    'problem_id': i + 1,
                    'problem_type': problem['type'],
                    'solved': solution_quality['is_valid'],
                    'creativity_score': solution_quality['creativity_score'],
                    'spike_detected': result.get('spike_detected', False)
                }
                
                results['problems'].append(problem_result)
                
            except Exception as e:
                logger.error(f"Error in creative problem {i+1}: {e}")
        
        results['metrics'] = {
            'solution_rate': solved / n_problems * 100,
            'creative_solution_rate': creative_solutions / n_problems * 100,
            'avg_creativity_score': np.mean([p.get('creativity_score', 0) for p in results['problems']])
        }
        
        self._save_results(results, 'creative_problem_solving_benchmark')
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and create comprehensive report"""
        logger.info("🏃 Running Comprehensive Benchmark Suite...")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'environment': 'Google Colab',
            'benchmarks': {}
        }
        
        # Run all benchmarks
        benchmarks = [
            ('rat', lambda: self.run_rat_benchmark(100)),
            ('scientific_discovery', self.run_scientific_discovery_benchmark),
            ('analogy', lambda: self.run_analogy_completion_benchmark(50)),
            ('creative_problem_solving', lambda: self.run_creative_problem_solving_benchmark(30))
        ]
        
        for name, benchmark_fn in benchmarks:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running {name} benchmark...")
                logger.info(f"{'='*60}")
                
                results = benchmark_fn()
                all_results['benchmarks'][name] = results
                
            except Exception as e:
                logger.error(f"Failed to run {name} benchmark: {e}")
                all_results['benchmarks'][name] = {'error': str(e)}
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        # Save comprehensive results
        output_path = self.results_dir / f"comprehensive_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\n✅ Comprehensive benchmark complete!")
        logger.info(f"📊 Results saved to: {output_path}")
        
        return all_results
    
    # Helper methods
    def _load_rat_problems(self, n_problems: int) -> List[Dict]:
        """Load RAT problems from dataset"""
        # For demo, generate sample problems
        # In production, load from actual dataset
        problems = []
        
        # Classic RAT problems
        classic_problems = [
            {"word1": "COTTAGE", "word2": "SWISS", "word3": "CAKE", "answer": "CHEESE"},
            {"word1": "CREAM", "word2": "SKATE", "word3": "WATER", "answer": "ICE"},
            {"word1": "DUCK", "word2": "FOLD", "word3": "DOLLAR", "answer": "BILL"},
            {"word1": "NIGHT", "word2": "WRIST", "word3": "STOP", "answer": "WATCH"},
            {"word1": "RIVER", "word2": "NOTE", "word3": "ACCOUNT", "answer": "BANK"},
        ]
        
        # Extend with more problems
        for i in range(n_problems):
            if i < len(classic_problems):
                problems.append(classic_problems[i])
            else:
                # Generate variations
                problems.append({
                    "word1": f"WORD{i}_1",
                    "word2": f"WORD{i}_2", 
                    "word3": f"WORD{i}_3",
                    "answer": f"ANSWER{i}"
                })
        
        return problems[:n_problems]
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from response"""
        # Simple extraction logic
        response = response.upper()
        
        # Look for common answer patterns
        patterns = [
            "THE ANSWER IS ",
            "ANSWER: ",
            "IS ",
        ]
        
        for pattern in patterns:
            if pattern in response:
                after_pattern = response.split(pattern)[-1]
                word = after_pattern.split()[0] if after_pattern.split() else ""
                return word.strip('.,!?')
        
        # Return first uppercase word
        words = response.split()
        for word in words:
            if word.isupper() and len(word) > 2:
                return word.strip('.,!?')
        
        return "UNKNOWN"
    
    def _calculate_spike_correlation(self, problems: List[Dict]) -> float:
        """Calculate correlation between spikes and correct answers"""
        correct_with_spike = sum(
            1 for p in problems 
            if p.get('correct', False) and p.get('spike_detected', False)
        )
        total_correct = sum(1 for p in problems if p.get('correct', False))
        
        if total_correct == 0:
            return 0.0
        
        return correct_with_spike / total_correct
    
    def _save_results(self, results: Dict, benchmark_name: str):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_name}_{timestamp}.json"
        
        output_path = self.results_dir / "outputs" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_path}")
        
        # Log summary to wandb
        if self.use_wandb:
            wandb.log({f"{benchmark_name}/metrics": results['metrics']})
    
    def _generate_summary_report(self, all_results: Dict):
        """Generate markdown summary report"""
        report = f"""# InsightSpike Comprehensive Benchmark Report

## Experiment Information
- **Date**: {all_results['timestamp']}
- **Environment**: {all_results['environment']}

## Benchmark Results Summary

"""
        
        for name, results in all_results['benchmarks'].items():
            if 'error' in results:
                report += f"### {name}\n**Error**: {results['error']}\n\n"
            else:
                report += f"### {name}\n"
                if 'metrics' in results:
                    for metric, value in results['metrics'].items():
                        if isinstance(value, float):
                            report += f"- **{metric}**: {value:.2f}\n"
                        else:
                            report += f"- **{metric}**: {value}\n"
                report += "\n"
        
        # Save report
        report_path = self.results_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {report_path}")
    
    def _load_scientific_problems(self, dataset_path: Optional[str]) -> List[Dict]:
        """Load scientific discovery problems"""
        # Placeholder implementation
        return [
            {
                "question": "What is the relationship between temperature and enzyme activity?",
                "domain": "biochemistry",
                "expected_relationships": ["optimal temperature", "denaturation", "activation energy"]
            },
            # Add more problems
        ]
    
    def _evaluate_discovery(self, response: str, expected_relationships: List[str]) -> bool:
        """Evaluate if key relationships were discovered"""
        response_lower = response.lower()
        discovered = sum(1 for rel in expected_relationships if rel.lower() in response_lower)
        return discovered >= len(expected_relationships) * 0.5
    
    def _load_analogy_problems(self, n_problems: int) -> List[Dict]:
        """Load analogy problems"""
        analogies = [
            {"a": "dog", "b": "puppy", "c": "cat", "d": "kitten"},
            {"a": "hot", "b": "cold", "c": "day", "d": "night"},
            # Add more
        ]
        return analogies[:n_problems]
    
    def _check_semantic_validity(self, a: str, b: str, c: str, d: str) -> bool:
        """Check if analogy is semantically valid"""
        # Placeholder - would use embeddings to check relationships
        return True
    
    def _load_creative_problems(self, n_problems: int) -> List[Dict]:
        """Load creative problem solving tasks"""
        return [
            {
                "question": "How can you connect 9 dots arranged in a 3x3 grid with 4 straight lines without lifting your pen?",
                "type": "lateral_thinking",
                "constraints": ["4 lines", "no lifting pen"],
                "criteria": ["all dots connected", "constraint satisfaction"]
            },
            # Add more
        ]
    
    def _evaluate_creative_solution(self, response: str, constraints: List[str], criteria: List[str]) -> Dict:
        """Evaluate creative solution quality"""
        # Placeholder evaluation
        return {
            "is_valid": True,
            "creativity_score": 0.8
        }


# Convenience functions for Colab
def run_rat_benchmark(n_problems: int = 100, use_wandb: bool = False) -> Dict:
    """Quick function to run RAT benchmark"""
    suite = InsightBenchmarkSuite(use_wandb=use_wandb)
    return suite.run_rat_benchmark(n_problems)


def run_all_benchmarks(use_wandb: bool = True) -> Dict:
    """Run comprehensive benchmark suite"""
    suite = InsightBenchmarkSuite(use_wandb=use_wandb)
    return suite.run_comprehensive_benchmark()


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting InsightSpike Benchmarks...")
    
    # Run comprehensive benchmarks
    results = run_all_benchmarks(use_wandb=False)
    
    print("\n✅ Benchmarks complete!")
    print(f"📊 Results saved to: experiments/colab_experiments/results/")