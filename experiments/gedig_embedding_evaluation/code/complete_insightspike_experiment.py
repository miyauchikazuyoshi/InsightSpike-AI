#!/usr/bin/env python3
"""
Complete InsightSpike-AI RAG Experiment with Graph Updates
==========================================================

This version properly saves both episodes and graph data.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.algorithms.information_gain import InformationGain, EntropyMethod
from insightspike.algorithms.graph_edit_distance import GraphEditDistance, OptimizationLevel
from insightspike.core.agents.main_agent import MainAgent

import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteInsightSpikeRAGExperiment:
    """Complete experiment using InsightSpike-AI with proper state saving"""
    
    def __init__(self, backup_dir="experiment_backup"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize InsightSpike components
        self.agent = MainAgent()
        self.ig_calculator = InformationGain(method=EntropyMethod.SHANNON)
        self.ged_calculator = GraphEditDistance(optimization_level=OptimizationLevel.STANDARD)
        
        # Initialize agent
        if not self.agent.initialize():
            logger.warning("Failed to fully initialize MainAgent, continuing anyway")
        
        self.insights = []
        self.processing_times = []
        self.all_responses = []
        
    def backup_data(self):
        """Backup current data state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"data_backup_{timestamp}"
        
        # Backup data directory if it exists
        data_dir = Path("data")
        if data_dir.exists():
            shutil.copytree(data_dir, backup_path / "data")
            logger.info(f"Backed up data to {backup_path}")
        
        # Save agent state before backup
        if hasattr(self.agent, 'save_state'):
            try:
                self.agent.save_state()
                logger.info("Agent state saved before backup")
            except Exception as e:
                logger.warning(f"Could not save agent state: {e}")
                
        return backup_path
    
    def create_test_documents(self, n_docs=50):  # Reduced for testing
        """Create test documents for the experiment"""
        topics = [
            "machine learning", "deep learning", "neural networks",
            "natural language processing", "computer vision",
            "reinforcement learning", "data science", "algorithms"
        ]
        
        documents = []
        for i in range(n_docs):
            topic = topics[i % len(topics)]
            doc_variations = [
                f"{topic} is a field of artificial intelligence that enables systems to learn from data.",
                f"The principles of {topic} involve mathematical models and computational algorithms.",
                f"Applications of {topic} include pattern recognition, prediction, and automation.",
                f"{topic} has revolutionized how we approach complex computational problems.",
                f"Recent advances in {topic} have led to breakthrough discoveries in various domains."
            ]
            doc = doc_variations[i % len(doc_variations)]
            documents.append(doc)
        
        return documents
    
    def process_documents(self, documents):
        """Process documents through InsightSpike-AI"""
        logger.info(f"Processing {len(documents)} documents...")
        
        results = []
        save_frequency = 10  # Save state every 10 documents
        
        for i, doc in enumerate(documents):
            start_time = time.time()
            
            # Process through agent
            result = self.agent.process_question(
                f"Learn and remember this information: {doc}",
                max_cycles=2
            )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Store all responses
            self.all_responses.append({
                'document': doc,
                'result': result,
                'index': i
            })
            
            # Check for insights using correct field name
            if 'reasoning_quality' in result:
                quality = result['reasoning_quality']
                is_spike = quality > 0.6  # Adjusted threshold
                
                if is_spike:
                    self.insights.append({
                        'document': doc,
                        'quality': quality,
                        'processing_time': processing_time,
                        'index': i
                    })
                    logger.info(f"Insight detected at document {i}: quality={quality:.3f}")
            
            results.append(result)
            
            # Save state periodically
            if (i + 1) % save_frequency == 0:
                logger.info(f"Progress: {i+1}/{len(documents)} - Saving state...")
                if self.agent.save_state():
                    logger.info("State saved successfully")
                else:
                    logger.warning("Failed to save state")
        
        # Final save after all documents
        logger.info("Processing complete - Final state save...")
        if self.agent.save_state():
            logger.info("Final state saved successfully")
        else:
            logger.warning("Failed to save final state")
        
        return results
    
    def check_data_files(self):
        """Check the status of data files"""
        data_dir = Path("data")
        files_to_check = ["episodes.json", "index.faiss", "graph_pyg.pt"]
        
        logger.info("\nData file status:")
        for filename in files_to_check:
            filepath = data_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                logger.info(f"  {filename}: {size} bytes, modified {mtime}")
                
                # Special handling for episodes.json
                if filename == "episodes.json":
                    try:
                        with open(filepath, 'r') as f:
                            episodes = json.load(f)
                        logger.info(f"    Episodes count: {len(episodes)}")
                    except:
                        pass
            else:
                logger.info(f"  {filename}: NOT FOUND")
    
    def test_retrieval(self, queries):
        """Test retrieval performance"""
        logger.info(f"\nTesting retrieval with {len(queries)} queries...")
        
        retrieval_results = []
        
        for i, query in enumerate(queries):
            start_time = time.time()
            
            # Ask agent
            result = self.agent.process_question(query, max_cycles=3)
            
            retrieval_time = time.time() - start_time
            
            # More flexible evaluation
            response_text = result.get('response', '').lower()
            
            # Extract key topic from query
            topic_keywords = {
                "machine learning": ["machine", "learning", "ml"],
                "deep learning": ["deep", "learning", "neural"],
                "neural networks": ["neural", "network", "neuron"],
                "natural language processing": ["natural", "language", "nlp", "text"],
                "computer vision": ["computer", "vision", "image", "visual"],
                "reinforcement learning": ["reinforcement", "learning", "reward", "agent"],
                "data science": ["data", "science", "analysis", "statistics"],
                "algorithms": ["algorithm", "computational", "complexity"]
            }
            
            # Check if any relevant keywords are mentioned
            relevant_found = False
            for topic, keywords in topic_keywords.items():
                if any(keyword in query.lower() for keyword in keywords):
                    if any(keyword in response_text for keyword in keywords):
                        relevant_found = True
                        break
            
            quality = result.get('reasoning_quality', 0)
            
            retrieval_results.append({
                'query': query,
                'response': result.get('response', ''),
                'quality': quality,
                'relevant': relevant_found,
                'time': retrieval_time
            })
            
            if (i + 1) % 5 == 0:
                logger.info(f"Retrieval progress: {i+1}/{len(queries)}")
        
        return retrieval_results
    
    def analyze_results(self, processing_results, retrieval_results):
        """Analyze experiment results"""
        all_qualities = []
        for resp in self.all_responses:
            if 'reasoning_quality' in resp['result']:
                all_qualities.append(resp['result']['reasoning_quality'])
        
        analysis = {
            'processing': {
                'total_documents': len(processing_results),
                'total_insights': len(self.insights),
                'insight_ratio': len(self.insights) / max(1, len(processing_results)),
                'avg_processing_time': np.mean(self.processing_times),
                'total_processing_time': sum(self.processing_times)
            },
            'retrieval': {
                'total_queries': len(retrieval_results),
                'successful_retrievals': sum(1 for r in retrieval_results if r['relevant']),
                'retrieval_accuracy': sum(1 for r in retrieval_results if r['relevant']) / max(1, len(retrieval_results)),
                'avg_retrieval_time': np.mean([r['time'] for r in retrieval_results]),
                'avg_quality_score': np.mean([r['quality'] for r in retrieval_results])
            },
            'insights': {
                'count': len(self.insights),
                'avg_quality': np.mean([i['quality'] for i in self.insights]) if self.insights else 0,
                'indices': [i['index'] for i in self.insights]
            },
            'quality_distribution': {
                'mean': np.mean(all_qualities) if all_qualities else 0,
                'std': np.std(all_qualities) if all_qualities else 0,
                'min': min(all_qualities) if all_qualities else 0,
                'max': max(all_qualities) if all_qualities else 0
            }
        }
        
        return analysis
    
    def visualize_results(self, analysis):
        """Create simple visualization"""
        output_dir = Path("results_complete_insightspike")
        output_dir.mkdir(exist_ok=True)
        
        # Save analysis results
        with open(output_dir / 'complete_experiment_analysis.json', 'w') as f:
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                return obj
            
            json.dump(convert_to_serializable(analysis), f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
        
        return output_dir

def main():
    """Run the complete experiment"""
    print("="*60)
    print("Complete InsightSpike-AI RAG Experiment")
    print("="*60)
    
    # Create experiment instance
    experiment = CompleteInsightSpikeRAGExperiment()
    
    # Check initial state
    print("\nInitial data state:")
    experiment.check_data_files()
    
    # Backup current data
    print("\nBacking up current data...")
    backup_path = experiment.backup_data()
    
    try:
        # Create test data
        print("\nCreating test documents...")
        documents = experiment.create_test_documents(50)  # Reduced for testing
        queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks",
            "What are the applications of computer vision?",
            "How does reinforcement learning differ from supervised learning?",
            "What is natural language processing used for?",
            "Describe data science methodologies",
            "What algorithms are used in machine learning?",
            "Tell me about artificial intelligence",
            "What is pattern recognition?"
        ]
        
        # Process documents
        print("\nProcessing documents through InsightSpike-AI...")
        processing_results = experiment.process_documents(documents)
        
        # Check state after processing
        print("\nData state after processing:")
        experiment.check_data_files()
        
        # Test retrieval
        print("\nTesting retrieval performance...")
        retrieval_results = experiment.test_retrieval(queries)
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = experiment.analyze_results(processing_results, retrieval_results)
        
        # Visualize results
        print("\nSaving results...")
        output_dir = experiment.visualize_results(analysis)
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Documents processed: {analysis['processing']['total_documents']}")
        print(f"Insights detected: {analysis['processing']['total_insights']} ({analysis['processing']['insight_ratio']:.1%})")
        print(f"Average processing time: {analysis['processing']['avg_processing_time']:.3f}s")
        print(f"Retrieval accuracy: {analysis['retrieval']['retrieval_accuracy']:.1%}")
        print(f"Average quality score: {analysis['retrieval']['avg_quality_score']:.3f}")
        print(f"\nResults saved to: {output_dir}")
        
        # Final state check
        print("\nFinal data state:")
        experiment.check_data_files()
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise
    
    finally:
        # Restore clean data
        print("\n" + "="*60)
        print("Restoring clean data state...")
        print("="*60)
        
        # Run the restore script
        restore_script = Path("scripts/utilities/restore_clean_data.py")
        if restore_script.exists():
            os.system(f"python {restore_script}")
            print("Data restored to clean state!")
        else:
            print("Warning: Could not find restore script")
            print(f"Backup is available at: {backup_path}")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()