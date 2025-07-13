#!/usr/bin/env python3
"""
Run Comparative Study: LLM vs RAG vs InsightSpike
================================================

Execute experiments comparing three approaches.
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import openai
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.insightspike.core.config import Config
from src.insightspike.core.agents.main_agent_optimized import MainAgentOptimized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result from a single experiment run"""
    
    test_id: str
    system: str  # "baseline_llm", "rag", "insightspike"
    run_number: int
    response: str
    response_time: float
    tokens_used: int
    
    # Evaluation scores
    correctness_score: float = 0.0
    completeness_score: float = 0.0
    reasoning_depth_score: float = 0.0
    insight_quality_score: float = 0.0
    
    # Additional metrics for InsightSpike
    transformation_cycles: int = 0
    insights_discovered: List[str] = None
    confidence_trajectory: List[float] = None
    spike_detected: bool = False
    
    def __post_init__(self):
        if self.insights_discovered is None:
            self.insights_discovered = []
        if self.confidence_trajectory is None:
            self.confidence_trajectory = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaselineLLM:
    """Baseline LLM without any augmentation"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.provider = provider
        self.model = model
        
        # Initialize based on provider
        if provider == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            # Mock for demonstration
            self.client = None
    
    async def answer_question(self, question: str, context: List[str]) -> Tuple[str, int]:
        """Answer question using raw LLM"""
        
        # Format prompt
        prompt = f"Question: {question}\n\nPlease provide a comprehensive answer."
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                tokens = response.usage.total_tokens
                
                return answer, tokens
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._mock_response(question), 100
        else:
            return self._mock_response(question), 100
    
    def _mock_response(self, question: str) -> str:
        """Mock response for testing"""
        return f"This is a baseline response to: {question}. Without additional context or reasoning capabilities, I can only provide a basic answer based on my training data."


class RAGSystem:
    """Traditional RAG system with vector search"""
    
    def __init__(self, knowledge_base: Dict[str, List[str]], 
                 provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.knowledge_base = knowledge_base
        self.provider = provider
        self.model = model
        self.all_facts = []
        
        # Flatten knowledge base
        for domain, facts in knowledge_base.items():
            self.all_facts.extend(facts)
        
        # Initialize LLM
        if provider == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = None
    
    async def answer_question(self, question: str, context: List[str]) -> Tuple[str, int]:
        """Answer question using RAG"""
        
        # Simple retrieval: find relevant facts (mock embedding search)
        relevant_facts = self._retrieve_relevant_facts(question, context)
        
        # Format prompt with retrieved context
        context_str = "\n".join(relevant_facts[:5])  # Top 5 facts
        prompt = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {question}

Provide a comprehensive answer using the given context."""
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                tokens = response.usage.total_tokens
                
                return answer, tokens
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._mock_response(question, relevant_facts), 150
        else:
            return self._mock_response(question, relevant_facts), 150
    
    def _retrieve_relevant_facts(self, question: str, context: List[str]) -> List[str]:
        """Mock retrieval based on keyword matching"""
        question_words = set(question.lower().split())
        relevant = []
        
        # Add provided context first
        relevant.extend(context)
        
        # Find relevant facts from knowledge base
        for fact in self.all_facts:
            fact_words = set(fact.lower().split())
            if len(question_words & fact_words) > 2:  # At least 2 common words
                relevant.append(fact)
        
        return relevant[:10]  # Return top 10
    
    def _mock_response(self, question: str, facts: List[str]) -> str:
        """Mock response for testing"""
        facts_summary = " ".join(facts[:3])
        return f"Based on the retrieved information about '{facts_summary[:100]}...', here's my answer to '{question}'. The RAG system found relevant context to provide a more informed response."


class InsightSpikeSystem:
    """InsightSpike with Query Transformation"""
    
    def __init__(self, knowledge_base: Dict[str, List[str]]):
        self.knowledge_base = knowledge_base
        
        # Initialize InsightSpike
        config = Config()
        config.llm.provider = "openai"  # or "mock" for testing
        
        self.agent = MainAgentOptimized(
            config=config,
            enable_cache=True,
            enable_learning=True,
            enable_async=True
        )
        
        # Pre-load knowledge into InsightSpike
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load knowledge base into InsightSpike"""
        for domain, facts in self.knowledge_base.items():
            for fact in facts:
                # Add as episode
                self.agent.add_episode_with_graph_update(fact)
        
        logger.info(f"Loaded {sum(len(f) for f in self.knowledge_base.values())} facts into InsightSpike")
    
    async def answer_question(self, question: str, context: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Answer question using InsightSpike with Query Transformation"""
        
        # Add specific context as temporary episodes
        for ctx in context:
            self.agent.add_episode_with_graph_update(ctx)
        
        # Process with InsightSpike
        result = await self.agent.process_question_async(question)
        
        # Extract response and metrics
        response = result.get("response", "No response generated")
        
        # Extract InsightSpike-specific metrics
        metrics = {
            "transformation_cycles": len(result.get("transformation_history", {}).get("states", [])),
            "insights_discovered": result.get("synthesis", {}).get("total_insights", []),
            "confidence_trajectory": result.get("transformation_history", {}).get("confidence_trajectory", []),
            "spike_detected": result.get("spike_detected", False),
            "tokens_used": 200  # Estimate
        }
        
        return response, metrics


class Evaluator:
    """Evaluate experiment results"""
    
    def __init__(self, test_cases: List[Dict[str, Any]]):
        self.test_cases = {tc["id"]: tc for tc in test_cases}
    
    def evaluate_response(self, result: ExperimentResult, test_case: Dict[str, Any]) -> ExperimentResult:
        """Evaluate a single response"""
        
        # Simple keyword-based evaluation (in practice, use LLM-as-judge)
        response_lower = result.response.lower()
        expected_insights = test_case["expected_insights"]
        
        # Correctness: Does it mention key concepts?
        key_concepts = self._extract_key_concepts(test_case["question"], test_case["context"])
        mentioned_concepts = sum(1 for concept in key_concepts if concept.lower() in response_lower)
        result.correctness_score = mentioned_concepts / len(key_concepts) if key_concepts else 0
        
        # Completeness: Does it cover expected insights?
        covered_insights = sum(1 for insight in expected_insights 
                             if any(word in response_lower for word in insight.lower().split()[:3]))
        result.completeness_score = covered_insights / len(expected_insights) if expected_insights else 0
        
        # Reasoning depth: Length and structure indicators
        result.reasoning_depth_score = min(1.0, len(result.response) / 1000)  # Simple proxy
        
        # Insight quality: Specific to InsightSpike
        if result.system == "insightspike" and result.insights_discovered:
            result.insight_quality_score = min(1.0, len(result.insights_discovered) / 3)
        else:
            result.insight_quality_score = result.completeness_score * 0.5
        
        return result
    
    def _extract_key_concepts(self, question: str, context: List[str]) -> List[str]:
        """Extract key concepts from question and context"""
        # Simple extraction based on nouns and important words
        important_words = []
        
        # From question
        question_words = question.lower().split()
        important_words.extend([w for w in question_words if len(w) > 4])
        
        # From context (first few words of each)
        for ctx in context[:3]:
            ctx_words = ctx.lower().split()[:5]
            important_words.extend([w for w in ctx_words if len(w) > 4])
        
        return list(set(important_words))[:5]


async def run_single_experiment(test_case: Dict[str, Any], 
                              systems: Dict[str, Any],
                              run_number: int) -> List[ExperimentResult]:
    """Run experiment for all systems on a single test case"""
    
    results = []
    
    # 1. Baseline LLM
    logger.info(f"Running Baseline LLM on {test_case['id']} (run {run_number})")
    start_time = time.time()
    
    response, tokens = await systems["baseline"].answer_question(
        test_case["question"], test_case["context"]
    )
    
    results.append(ExperimentResult(
        test_id=test_case["id"],
        system="baseline_llm",
        run_number=run_number,
        response=response,
        response_time=time.time() - start_time,
        tokens_used=tokens
    ))
    
    # 2. RAG System
    logger.info(f"Running RAG on {test_case['id']} (run {run_number})")
    start_time = time.time()
    
    response, tokens = await systems["rag"].answer_question(
        test_case["question"], test_case["context"]
    )
    
    results.append(ExperimentResult(
        test_id=test_case["id"],
        system="rag",
        run_number=run_number,
        response=response,
        response_time=time.time() - start_time,
        tokens_used=tokens
    ))
    
    # 3. InsightSpike
    logger.info(f"Running InsightSpike on {test_case['id']} (run {run_number})")
    start_time = time.time()
    
    response, metrics = await systems["insightspike"].answer_question(
        test_case["question"], test_case["context"]
    )
    
    results.append(ExperimentResult(
        test_id=test_case["id"],
        system="insightspike",
        run_number=run_number,
        response=response,
        response_time=time.time() - start_time,
        tokens_used=metrics["tokens_used"],
        transformation_cycles=metrics["transformation_cycles"],
        insights_discovered=metrics["insights_discovered"],
        confidence_trajectory=metrics["confidence_trajectory"],
        spike_detected=metrics["spike_detected"]
    ))
    
    return results


async def run_all_experiments(test_cases: List[Dict[str, Any]], 
                            knowledge_base: Dict[str, List[str]],
                            num_runs: int = 3) -> pd.DataFrame:
    """Run all experiments and return results DataFrame"""
    
    # Initialize systems
    systems = {
        "baseline": BaselineLLM(provider="mock"),  # Use mock for demo
        "rag": RAGSystem(knowledge_base, provider="mock"),
        "insightspike": InsightSpikeSystem(knowledge_base)
    }
    
    # Initialize evaluator
    evaluator = Evaluator(test_cases)
    
    # Run experiments
    all_results = []
    
    for test_case in tqdm(test_cases, desc="Test cases"):
        for run in range(num_runs):
            # Run experiment
            results = await run_single_experiment(test_case, systems, run)
            
            # Evaluate results
            for result in results:
                evaluated = evaluator.evaluate_response(result, test_case)
                all_results.append(evaluated.to_dict())
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Clean up InsightSpike
    systems["insightspike"].agent.cleanup()
    
    return df


def analyze_results(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize results"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Overall Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ["correctness_score", "completeness_score", 
               "reasoning_depth_score", "insight_quality_score"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Box plot
        df.boxplot(column=metric, by='system', ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('System')
        ax.set_ylabel('Score')
    
    plt.suptitle('Performance Comparison: LLM vs RAG vs InsightSpike')
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Response Time Analysis
    plt.figure(figsize=(10, 6))
    df.boxplot(column='response_time', by='system')
    plt.title('Response Time Comparison')
    plt.xlabel('System')
    plt.ylabel('Time (seconds)')
    plt.savefig(output_dir / 'response_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. InsightSpike-specific Analysis
    insightspike_df = df[df['system'] == 'insightspike']
    
    if not insightspike_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Spike detection rate
        spike_rate = insightspike_df.groupby('test_id')['spike_detected'].mean()
        axes[0].bar(range(len(spike_rate)), spike_rate.values)
        axes[0].set_title('Spike Detection Rate by Test Case')
        axes[0].set_xlabel('Test Case')
        axes[0].set_ylabel('Detection Rate')
        
        # Transformation cycles
        axes[1].hist(insightspike_df['transformation_cycles'], bins=10)
        axes[1].set_title('Distribution of Transformation Cycles')
        axes[1].set_xlabel('Number of Cycles')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'insightspike_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Statistical Summary
    summary = df.groupby('system')[metrics].agg(['mean', 'std'])
    summary.to_csv(output_dir / 'performance_summary.csv')
    
    # Print summary
    print("\n📊 Performance Summary:")
    print("="*60)
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for system in df['system'].unique():
            system_data = df[df['system'] == system][metric]
            print(f"  {system}: {system_data.mean():.3f} ± {system_data.std():.3f}")
    
    # 5. Best performing system by test case
    best_by_test = df.groupby(['test_id', 'system'])['completeness_score'].mean().unstack()
    best_system = best_by_test.idxmax(axis=1)
    
    print("\n🏆 Best System by Test Case:")
    print("="*60)
    for test_id, system in best_system.items():
        score = best_by_test.loc[test_id, system]
        print(f"{test_id}: {system} (score: {score:.3f})")
    
    return summary


async def main():
    """Main experiment execution"""
    
    base_path = Path(__file__).parent
    
    # Load experiment data
    with open(base_path / "data/test_cases/all_cases.json") as f:
        test_cases = json.load(f)
    
    with open(base_path / "data/knowledge_base/knowledge.json") as f:
        knowledge_base = json.load(f)
    
    # Run small subset for demo (use all for full experiment)
    test_subset = test_cases[:5]  # First 5 test cases
    num_runs = 2  # Reduced for demo
    
    print(f"\n🔬 Running Comparative Study")
    print(f"📊 Test cases: {len(test_subset)}")
    print(f"🔄 Runs per test: {num_runs}")
    print(f"💡 Systems: Baseline LLM, RAG, InsightSpike\n")
    
    # Run experiments
    results_df = await run_all_experiments(test_subset, knowledge_base, num_runs)
    
    # Save raw results
    results_path = base_path / "results" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Analyze results
    analysis_dir = base_path / "analysis" / "figures"
    analyze_results(results_df, analysis_dir)
    
    print(f"\n📈 Analysis complete! Check {analysis_dir} for visualizations.")


if __name__ == "__main__":
    # First setup the experiment
    from setup_experiment import create_experiment_structure
    create_experiment_structure()
    
    # Then run the comparison
    asyncio.run(main())