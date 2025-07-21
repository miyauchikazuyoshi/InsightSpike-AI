#!/usr/bin/env python3
"""
English Insight Experiment V2
============================

Re-implementation of the English insight experiment using the new InsightSpike architecture.
Tests multi-phase knowledge integration and spike detection with improved GED/IG metrics.

Author: InsightSpike Team
Date: 2025-07-20
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Import InsightSpike components
from insightspike.config import load_config
from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
from insightspike.core.episode import Episode
from insightspike.metrics import compute_fusion_reward

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable some verbose logging
logging.getLogger('insightspike.implementations.layers.layer4_llm_interface').setLevel(logging.WARNING)


class EnglishInsightExperiment:
    """Main experiment class for English insight generation."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.data_dir = self.experiment_dir / "data"
        self.results_dir = self.experiment_dir / "results"
        
        # Load configuration
        self.config = ConfigPresets.experiment()
        
        # Update config for this experiment
        self.config.llm.provider = "local"
        self.config.llm.model = "distilgpt2"
        self.config.llm.temperature = 0.7
        self.config.llm.max_tokens = 100
        
        # Spike detection thresholds
        self.config.graph.spike_ged_threshold = -0.5  # Negative for simplification
        self.config.graph.spike_ig_threshold = 0.2    # Positive for information gain
        
        # Initialize components
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.datastore = FileSystemDataStore(base_path=str(self.data_dir))
        
        # Questions to test
        self.questions = [
            "What is the relationship between energy and information?",
            "Why does consciousness emerge?",
            "What is the mechanism of creativity at the edge of chaos?",
            "What is entropy?",
            "Can you explain quantum entanglement?",
            "Is there a principle that unifies all phenomena?"
        ]
        
    def load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load the knowledge base from JSON file."""
        kb_path = self.data_dir / "input" / "english_knowledge_base.json"
        with open(kb_path, 'r') as f:
            data = json.load(f)
            # Extract episodes array from the structure
            return data['episodes']
    
    def prepare_datastore(self):
        """Prepare the datastore with knowledge base episodes."""
        logger.info("Preparing datastore with knowledge base...")
        
        # Load knowledge base
        knowledge_base = self.load_knowledge_base()
        
        # Clear existing data - FileSystemDataStore doesn't have clear()
        # Just proceed with storing episodes
        
        # Prepare episodes for batch storage
        episodes = []
        phase_names = {
            1: "Basic Concepts",
            2: "Relationships",
            3: "Deep Integration",
            4: "Emergent Insights",
            5: "Integration and Circulation"
        }
        
        for item in knowledge_base:
            # Encode the text
            vector = self.encoder.encode(item['text'])
            
            # Create episode dict
            episode_dict = {
                'text': item['text'],
                'vec': vector.tolist(),  # Convert to list for JSON serialization
                'c': 0.5,  # Default C-value
                'metadata': {
                    'id': item['id'],
                    'phase': item['phase'],
                    'phase_name': phase_names.get(item['phase'], f"Phase {item['phase']}")
                }
            }
            episodes.append(episode_dict)
        
        # Save all episodes at once
        self.datastore.save_episodes(episodes)
        
        logger.info(f"Stored {len(knowledge_base)} episodes in datastore")
    
    def run_baseline_llm(self, question: str) -> Dict[str, Any]:
        """Run direct LLM without any knowledge base."""
        logger.info(f"Running baseline LLM for: {question}")
        
        # Create a minimal agent with clean LLM
        config = ConfigPresets.experiment()
        config.llm.provider = "local"
        config.llm.model = "distilgpt2"
        
        agent = MainAgent(config=config, datastore=self.datastore)
        agent.initialize()
        
        # Run with empty context
        start_time = time.time()
        
        # For baseline, we'll use the LLM directly
        llm_response = agent.l4_llm.generate_response(
            context={},
            question=question
        )
        
        response_time = time.time() - start_time
        
        return {
            'method': 'baseline_llm',
            'question': question,
            'response': llm_response,
            'response_time': response_time,
            'retrieved_docs': 0,
            'phases_integrated': 0,
            'spike_detected': False
        }
    
    def run_standard_rag(self, question: str) -> Dict[str, Any]:
        """Run standard RAG approach."""
        logger.info(f"Running standard RAG for: {question}")
        
        # Encode question
        question_vector = self.encoder.encode(question)
        
        # Simple similarity search (top-3)
        start_time = time.time()
        
        # Load episodes directly from saved JSON file
        episodes_file = self.data_dir / "episodes" / "default.json"
        if not episodes_file.exists():
            # Try alternative path
            episodes_file = self.data_dir / "core" / "default.json"
            
        if not episodes_file.exists():
            logger.error(f"No episodes found at {episodes_file}")
            return {
                'method': 'standard_rag',
                'question': question,
                'response': 'No knowledge base found',
                'response_time': 0,
                'retrieved_docs': 0,
                'phases_integrated': 0,
                'spike_detected': False
            }
        
        with open(episodes_file, 'r') as f:
            episodes_data = json.load(f)
        
        # Convert back to Episode objects
        all_episodes = []
        for ep_data in episodes_data:
            episode = Episode(
                text=ep_data['text'],
                vec=np.array(ep_data['vec']),
                c=ep_data.get('c', 0.5),
                metadata=ep_data.get('metadata', {})
            )
            all_episodes.append(episode)
        similarities = []
        
        for episode in all_episodes:
            sim = np.dot(question_vector, episode.vec) / (
                np.linalg.norm(question_vector) * np.linalg.norm(episode.vec)
            )
            similarities.append((sim, episode))
        
        # Get top-3
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_episodes = [ep for _, ep in similarities[:3]]
        
        # Build context
        context = "\n".join([ep.text for ep in top_episodes])
        
        # Generate response
        config = ConfigPresets.experiment()
        config.llm.provider = "local"
        config.llm.model = "distilgpt2"
        
        agent = MainAgent(config=config, datastore=self.datastore)
        agent.initialize()
        
        llm_context = {
            'retrieved_documents': [{'text': ep.text} for ep in top_episodes]
        }
        
        llm_response = agent.l4_llm.generate_response(
            context=llm_context,
            question=question
        )
        
        response_time = time.time() - start_time
        
        return {
            'method': 'standard_rag',
            'question': question,
            'response': llm_response,
            'response_time': response_time,
            'retrieved_docs': len(top_episodes),
            'phases_integrated': len(set(ep.metadata.get('phase', 0) for ep in top_episodes)),
            'spike_detected': False,
            'context': context
        }
    
    def run_insightspike(self, question: str) -> Dict[str, Any]:
        """Run InsightSpike approach."""
        logger.info(f"Running InsightSpike for: {question}")
        
        # Create a memory manager and populate it with knowledge base
        from insightspike.implementations.layers.layer2_memory_manager import L2MemoryManager, MemoryConfig, MemoryMode
        memory_config = MemoryConfig(
            mode=MemoryMode.SCALABLE,
            embedding_dim=384,
            use_graph_integration=True,
            use_importance_scoring=True
        )
        memory = L2MemoryManager(config=memory_config, legacy_config=self.config)
        
        # Load episodes and add to memory
        episodes_file = self.data_dir / "core" / "default.json"
        if episodes_file.exists():
            with open(episodes_file, 'r') as f:
                episodes_data = json.load(f)
            
            for ep_data in episodes_data:
                # Store episode with text and metadata
                memory.store_episode(
                    text=ep_data['text'],
                    c_value=ep_data.get('c', 0.5),
                    metadata=ep_data.get('metadata', {})
                )
            logger.info(f"Loaded {len(episodes_data)} episodes into memory")
        
        # Create agent with populated memory
        agent = MainAgent(config=self.config, datastore=self.datastore)
        agent.l2_memory = memory  # Replace the memory with our populated one
        agent.initialize()
        
        # Process question
        start_time = time.time()
        result = agent.process_question(question, max_cycles=1)
        response_time = time.time() - start_time
        
        # Extract results
        result_dict = result.to_dict()
        
        # Get phase integration
        retrieved_docs = result_dict.get('documents', [])
        phases = set()
        for doc in retrieved_docs:
            if 'metadata' in doc and 'phase' in doc['metadata']:
                phases.add(doc['metadata']['phase'])
        
        # Get graph metrics if available
        graph_metrics = result_dict.get('metrics', {})
        
        return {
            'method': 'insightspike',
            'question': question,
            'response': result_dict.get('response', ''),
            'response_time': response_time,
            'retrieved_docs': len(retrieved_docs),
            'phases_integrated': len(phases),
            'spike_detected': result_dict.get('spike_detected', False),
            'graph_metrics': graph_metrics,
            'reasoning_quality': result_dict.get('reasoning_quality', 0.0),
            'context': "\n".join([doc.get('text', '') for doc in retrieved_docs])
        }
    
    def calculate_quality_score(self, response: str) -> float:
        """Calculate quality score for a response."""
        if not response:
            return 0.0
        
        # Simple quality metrics
        score = 0.0
        
        # Length bonus (normalized)
        length_score = min(len(response.split()) / 50, 1.0) * 0.2
        score += length_score
        
        # Keyword presence
        keywords = {
            'depth': ['because', 'therefore', 'specifically', 'for example', 'this means'],
            'specificity': ['energy', 'information', 'entropy', 'quantum', 'consciousness'],
            'integration': ['relationship', 'connection', 'integration', 'unified', 'perspective'],
            'insight': ['insight', 'discovery', 'emerges', 'fundamental', 'principle']
        }
        
        for category, words in keywords.items():
            category_score = sum(1 for word in words if word.lower() in response.lower())
            category_score = min(category_score / len(words), 1.0) * 0.2
            score += category_score
        
        return score
    
    def run_experiment(self):
        """Run the complete experiment."""
        logger.info("Starting English Insight Experiment V2")
        
        # Prepare datastore
        self.prepare_datastore()
        
        # Results storage
        all_results = []
        
        # Run experiments for each question
        for i, question in enumerate(self.questions):
            logger.info(f"\n{'='*60}")
            logger.info(f"Question {i+1}/{len(self.questions)}: {question}")
            logger.info(f"{'='*60}")
            
            results = {}
            
            # 1. Baseline LLM
            try:
                baseline_result = self.run_baseline_llm(question)
                baseline_result['quality_score'] = self.calculate_quality_score(
                    baseline_result['response']
                )
                results['baseline'] = baseline_result
            except Exception as e:
                logger.error(f"Baseline LLM failed: {e}")
                results['baseline'] = {'error': str(e)}
            
            # 2. Standard RAG
            try:
                rag_result = self.run_standard_rag(question)
                rag_result['quality_score'] = self.calculate_quality_score(
                    rag_result['response']
                )
                results['rag'] = rag_result
            except Exception as e:
                logger.error(f"Standard RAG failed: {e}")
                results['rag'] = {'error': str(e)}
            
            # 3. InsightSpike
            try:
                spike_result = self.run_insightspike(question)
                spike_result['quality_score'] = self.calculate_quality_score(
                    spike_result['response']
                )
                results['insightspike'] = spike_result
            except Exception as e:
                logger.error(f"InsightSpike failed: {e}")
                results['insightspike'] = {'error': str(e)}
            
            all_results.append(results)
            
            # Log summary
            logger.info("\nResults Summary:")
            for method in ['baseline', 'rag', 'insightspike']:
                if method in results and 'error' not in results[method]:
                    r = results[method]
                    logger.info(f"{method}: Quality={r.get('quality_score', 0):.3f}, "
                              f"Spike={r.get('spike_detected', False)}, "
                              f"Phases={r.get('phases_integrated', 0)}")
        
        # Save results
        self.save_results(all_results)
        
        # Generate summary report
        self.generate_report(all_results)
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save experiment results."""
        # Save raw results as JSON
        output_path = self.results_dir / "outputs" / "experiment_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        
        # Save summary as CSV
        summary_data = []
        for i, result in enumerate(results):
            for method in ['baseline', 'rag', 'insightspike']:
                if method in result and 'error' not in result[method]:
                    r = result[method]
                    summary_data.append({
                        'question_id': i + 1,
                        'question': self.questions[i],
                        'method': method,
                        'quality_score': r.get('quality_score', 0),
                        'spike_detected': r.get('spike_detected', False),
                        'phases_integrated': r.get('phases_integrated', 0),
                        'response_time': r.get('response_time', 0)
                    })
        
        df = pd.DataFrame(summary_data)
        csv_path = self.results_dir / "outputs" / "experiment_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """Generate experiment report."""
        report_lines = [
            "# English Insight Experiment V2 Results",
            f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "\n## Summary Statistics\n"
        ]
        
        # Calculate averages
        methods = ['baseline', 'rag', 'insightspike']
        stats = {method: {
            'quality_scores': [],
            'spike_detections': 0,
            'phases_integrated': [],
            'response_times': []
        } for method in methods}
        
        for result in results:
            for method in methods:
                if method in result and 'error' not in result[method]:
                    r = result[method]
                    stats[method]['quality_scores'].append(r.get('quality_score', 0))
                    if r.get('spike_detected', False):
                        stats[method]['spike_detections'] += 1
                    stats[method]['phases_integrated'].append(r.get('phases_integrated', 0))
                    stats[method]['response_times'].append(r.get('response_time', 0))
        
        # Create summary table
        report_lines.append("| Method | Avg Quality | Spike Rate | Avg Phases | Avg Time |")
        report_lines.append("|--------|------------|------------|------------|----------|")
        
        for method in methods:
            s = stats[method]
            avg_quality = np.mean(s['quality_scores']) if s['quality_scores'] else 0
            spike_rate = s['spike_detections'] / len(self.questions) * 100
            avg_phases = np.mean(s['phases_integrated']) if s['phases_integrated'] else 0
            avg_time = np.mean(s['response_times']) if s['response_times'] else 0
            
            report_lines.append(
                f"| {method.capitalize()} | {avg_quality:.3f} | {spike_rate:.1f}% | "
                f"{avg_phases:.1f} | {avg_time:.2f}s |"
            )
        
        # Add individual results
        report_lines.append("\n## Individual Question Results\n")
        
        for i, (question, result) in enumerate(zip(self.questions, results)):
            report_lines.append(f"### Question {i+1}: {question}\n")
            
            if 'insightspike' in result and 'error' not in result['insightspike']:
                r = result['insightspike']
                report_lines.append(f"- **Spike Detected**: {r.get('spike_detected', False)}")
                report_lines.append(f"- **Phases Integrated**: {r.get('phases_integrated', 0)}")
                report_lines.append(f"- **Quality Score**: {r.get('quality_score', 0):.3f}")
                report_lines.append(f"- **Response**: {r.get('response', '')[:200]}...")
            
            report_lines.append("")
        
        # Save report
        report_path = self.results_dir / "experiment_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Generated report at {report_path}")


def main():
    """Main entry point."""
    experiment_dir = Path(__file__).parent.parent
    experiment = EnglishInsightExperiment(experiment_dir)
    experiment.run_experiment()


if __name__ == "__main__":
    main()