#!/usr/bin/env python3
"""
RAG Systems Benchmark - Local Execution Version
Converted from Colab notebook for robust local execution

This script implements the same RAG benchmark experiments as the Colab notebook
but runs reliably in local environments without Colab-specific dependencies.

Usage:
    python rag_benchmark_local.py --profile demo
    python rag_benchmark_local.py --profile research
    python rag_benchmark_local.py --profile presentation
    python rag_benchmark_local.py --profile insightspike_only
"""

import sys
import os
from pathlib import Path
import argparse
import datetime
import json
import logging
from typing import Dict, List, Any, Optional
import gc
import psutil
import time
import pickle

# Add src directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# InsightSpike imports
try:
    from insightspike.core.layers.mock_llm_provider import MockLLMProvider
    from insightspike.core.config_manager import ConfigManager
    INSIGHTSPIKE_AVAILABLE = True
    print("✅ InsightSpike modules imported successfully")
except ImportError as e:
    INSIGHTSPIKE_AVAILABLE = False
    print(f"❌ InsightSpike import failed: {e}")

# ML/NLP imports  
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import faiss
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"❌ ML library import failed: {e}")

# External RAG frameworks (optional)
try:
    import langchain
    from langchain.llms import OpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS as LangChainFAISS
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import llama_index
    from llama_index import VectorStoreIndex, SimpleDirectoryReader
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

try:
    import haystack
    from haystack.document_stores import FAISSDocumentStore
    from haystack.nodes import EmbeddingRetriever, FARMReader
    from haystack.pipelines import ExtractiveQAPipeline
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False


class EvalConfig:
    """Experiment configuration management"""
    
    def __init__(self, profile="demo"):
        self.profile = profile
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{profile}_{self.timestamp}"
        
        # Profile definitions
        self.profiles = {
            "demo": {
                "description": "Lightweight demo execution - basic functionality check",
                "sample_sizes": [1000],
                "max_queries": 50,
                "datasets": ["squad_fallback", "test_fallback"],
                "systems": ["llm_only", "bm25_llm", "insightspike"],
                "enable_visualization": True,
                "save_results": True,
                "memory_cleanup": True,
                "strict_error_handling": False
            },
            "research": {
                "description": "Full research execution - all features, large scale",
                "sample_sizes": [1000, 5000, 10000, 50000],
                "max_queries": 1000,
                "datasets": ["squad", "ms_marco", "synthetic"],
                "systems": ["llm_only", "bm25_llm", "insightspike", "langchain", "llama_index", "haystack"],
                "enable_visualization": True,
                "save_results": True,
                "memory_cleanup": True,
                "strict_error_handling": False
            },
            "presentation": {
                "description": "Presentation execution - balanced focus",
                "sample_sizes": [1000, 5000, 10000],
                "max_queries": 200,
                "datasets": ["squad", "squad_fallback", "synthetic"],
                "systems": ["llm_only", "bm25_llm", "insightspike", "langchain"],
                "enable_visualization": True,
                "save_results": True,
                "memory_cleanup": True,
                "strict_error_handling": False
            },
            "insightspike_only": {
                "description": "InsightSpike specialized experiment - detailed analysis",
                "sample_sizes": [1000, 5000, 10000, 20000],
                "max_queries": 500,
                "datasets": ["squad", "ms_marco", "synthetic"],
                "systems": ["insightspike"],
                "enable_visualization": True,
                "save_results": True,
                "memory_cleanup": True,
                "strict_error_handling": False
            }
        }
        
        # Apply profile configuration
        if profile not in self.profiles:
            print(f"❌ Invalid profile: {profile}")
            print(f"Available profiles: {list(self.profiles.keys())}")
            profile = "demo"
            
        profile_config = self.profiles[profile]
        
        # Set configuration as instance variables
        self.description = profile_config["description"]
        self.sample_sizes = profile_config["sample_sizes"]
        self.max_queries = profile_config["max_queries"]
        self.datasets = profile_config["datasets"]
        self.systems = profile_config["systems"]
        self.enable_visualization = profile_config["enable_visualization"]
        self.save_results = profile_config["save_results"]
        self.memory_cleanup = profile_config["memory_cleanup"]
        self.strict_error_handling = profile_config["strict_error_handling"]
        
        # Section execution control
        self.sections_to_run = {
            "setup": True,
            "rag_systems": True,
            "datasets": True,
            "benchmark": True,
            "insightspike_specialized": "insightspike" in self.systems,
            "visualization": self.enable_visualization
        }
        
        # InsightSpike availability flag
        self.insightspike_available = INSIGHTSPIKE_AVAILABLE
        
        # Results directory
        self.results_dir = project_root / "experiments" / "results" / self.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def print_config(self):
        """Display configuration"""
        print(f"📋 Selected Profile: {self.profile}")
        print(f"📝 Description: {self.description}")
        print(f"📊 Data sizes: {self.sample_sizes}")
        print(f"🔍 Max queries per dataset: {self.max_queries}")
        print(f"📚 Datasets: {self.datasets}")
        print(f"🤖 RAG systems: {self.systems}")
        print(f"\n🎯 Active Sections:")
        for section, enabled in self.sections_to_run.items():
            status = "✅" if enabled else "⏭️ "
            print(f"  {status} {section}")
        print(f"\n🆔 Experiment ID: {self.experiment_id}")
        print(f"💾 Results directory: {self.results_dir}")


class RAGSystemManager:
    """Manages different RAG system implementations"""
    
    def __init__(self, eval_config: EvalConfig):
        self.eval_config = eval_config
        self.systems = {}
        self.logger = logging.getLogger(__name__)
        
    def initialize_systems(self):
        """Initialize all available RAG systems"""
        print("\n🤖 Initializing RAG Systems...")
        print("=" * 50)
        
        # Initialize InsightSpike
        if "insightspike" in self.eval_config.systems and INSIGHTSPIKE_AVAILABLE:
            try:
                self.systems["insightspike"] = self._initialize_insightspike()
                print("✅ InsightSpike initialized")
            except Exception as e:
                print(f"❌ InsightSpike initialization failed: {e}")
                
        # Initialize LangChain
        if "langchain" in self.eval_config.systems and LANGCHAIN_AVAILABLE:
            try:
                self.systems["langchain"] = self._initialize_langchain()
                print("✅ LangChain initialized")
            except Exception as e:
                print(f"❌ LangChain initialization failed: {e}")
                
        # Initialize LlamaIndex
        if "llama_index" in self.eval_config.systems and LLAMA_INDEX_AVAILABLE:
            try:
                self.systems["llama_index"] = self._initialize_llama_index()
                print("✅ LlamaIndex initialized")
            except Exception as e:
                print(f"❌ LlamaIndex initialization failed: {e}")
                
        # Initialize Haystack
        if "haystack" in self.eval_config.systems and HAYSTACK_AVAILABLE:
            try:
                self.systems["haystack"] = self._initialize_haystack()
                print("✅ Haystack initialized")
            except Exception as e:
                print(f"❌ Haystack initialization failed: {e}")
                
        # Initialize baseline systems
        if "llm_only" in self.eval_config.systems:
            self.systems["llm_only"] = self._initialize_llm_only()
            print("✅ LLM-only baseline initialized")
            
        if "bm25_llm" in self.eval_config.systems:
            self.systems["bm25_llm"] = self._initialize_bm25_llm()
            print("✅ BM25+LLM baseline initialized")
            
        print(f"\n📊 Initialized {len(self.systems)} RAG systems")
        return self.systems
    
    def _initialize_insightspike(self):
        """Initialize InsightSpike system"""
        try:
            # Initialize core components
            llm_provider = MockLLMProvider()
            llm_provider.initialize()
            
            config_manager = ConfigManager()
            
            return {
                "name": "InsightSpike",
                "llm_provider": llm_provider,
                "config_manager": config_manager,
                "type": "insightspike"
            }
        except Exception as e:
            self.logger.error(f"InsightSpike initialization failed: {e}")
            raise
    
    def _initialize_langchain(self):
        """Initialize LangChain system"""
        try:
            # Mock implementation for demonstration
            return {
                "name": "LangChain",
                "pipeline": "langchain_mock",
                "type": "langchain"
            }
        except Exception as e:
            self.logger.error(f"LangChain initialization failed: {e}")
            raise
    
    def _initialize_llama_index(self):
        """Initialize LlamaIndex system"""
        try:
            # Mock implementation for demonstration
            return {
                "name": "LlamaIndex",
                "pipeline": "llama_index_mock",
                "type": "llama_index"
            }
        except Exception as e:
            self.logger.error(f"LlamaIndex initialization failed: {e}")
            raise
    
    def _initialize_haystack(self):
        """Initialize Haystack system"""
        try:
            # Mock implementation for demonstration
            return {
                "name": "Haystack",
                "pipeline": "haystack_mock",
                "type": "haystack"
            }
        except Exception as e:
            self.logger.error(f"Haystack initialization failed: {e}")
            raise
    
    def _initialize_llm_only(self):
        """Initialize LLM-only baseline"""
        return {
            "name": "LLM Only",
            "pipeline": "llm_only_mock",
            "type": "baseline"
        }
    
    def _initialize_bm25_llm(self):
        """Initialize BM25+LLM baseline"""
        return {
            "name": "BM25 + LLM",
            "pipeline": "bm25_llm_mock",
            "type": "baseline"
        }


class DatasetManager:
    """Manages benchmark datasets"""
    
    def __init__(self, eval_config: EvalConfig):
        self.eval_config = eval_config
        self.datasets = {}
        self.logger = logging.getLogger(__name__)
        
    def load_datasets(self):
        """Load all configured datasets"""
        print("\n📚 Loading Datasets...")
        print("=" * 40)
        
        for dataset_name in self.eval_config.datasets:
            try:
                if dataset_name == "squad":
                    self.datasets[dataset_name] = self._load_squad()
                elif dataset_name == "squad_fallback":
                    self.datasets[dataset_name] = self._load_squad_fallback()
                elif dataset_name == "ms_marco":
                    self.datasets[dataset_name] = self._load_ms_marco()
                elif dataset_name == "synthetic":
                    self.datasets[dataset_name] = self._load_synthetic()
                elif dataset_name == "test_fallback":
                    self.datasets[dataset_name] = self._load_test_fallback()
                    
                print(f"✅ Loaded {dataset_name}")
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")
                
        print(f"\n📊 Loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def _load_squad(self):
        """Load SQuAD dataset"""
        # Mock implementation - replace with actual HuggingFace datasets loading
        return {
            "name": "SQuAD",
            "questions": [f"What is question {i}?" for i in range(100)],
            "contexts": [f"Context for question {i}" for i in range(100)],
            "answers": [f"Answer {i}" for i in range(100)]
        }
    
    def _load_squad_fallback(self):
        """Load SQuAD fallback dataset"""
        return {
            "name": "SQuAD Fallback",
            "questions": [f"Fallback question {i}?" for i in range(50)],
            "contexts": [f"Fallback context for question {i}" for i in range(50)],
            "answers": [f"Fallback answer {i}" for i in range(50)]
        }
    
    def _load_ms_marco(self):
        """Load MS MARCO dataset"""
        return {
            "name": "MS MARCO",
            "questions": [f"MARCO question {i}?" for i in range(100)],
            "contexts": [f"MARCO context for question {i}" for i in range(100)],
            "answers": [f"MARCO answer {i}" for i in range(100)]
        }
    
    def _load_synthetic(self):
        """Load synthetic dataset"""
        return {
            "name": "Synthetic",
            "questions": [f"Synthetic question {i}?" for i in range(100)],
            "contexts": [f"Synthetic context for question {i}" for i in range(100)],
            "answers": [f"Synthetic answer {i}" for i in range(100)]
        }
    
    def _load_test_fallback(self):
        """Load test fallback dataset"""
        return {
            "name": "Test Fallback",
            "questions": [f"Test question {i}?" for i in range(20)],
            "contexts": [f"Test context for question {i}" for i in range(20)],
            "answers": [f"Test answer {i}" for i in range(20)]
        }


class BenchmarkRunner:
    """Runs RAG system benchmarks"""
    
    def __init__(self, eval_config: EvalConfig, systems: Dict, datasets: Dict):
        self.eval_config = eval_config
        self.systems = systems
        self.datasets = datasets
        self.results = {}
        self.logger = logging.getLogger(__name__)
        
    def run_benchmark(self):
        """Run complete benchmark"""
        print("\n🚀 Running RAG Systems Benchmark...")
        print("=" * 50)
        
        start_time = time.time()
        
        for system_name, system in self.systems.items():
            print(f"\n🤖 Testing {system['name']}...")
            self.results[system_name] = {}
            
            for dataset_name, dataset in self.datasets.items():
                print(f"  📚 Dataset: {dataset['name']}")
                
                for sample_size in self.eval_config.sample_sizes:
                    print(f"    📊 Sample size: {sample_size}")
                    
                    try:
                        result = self._run_single_test(
                            system, dataset, sample_size
                        )
                        
                        key = f"{dataset_name}_{sample_size}"
                        self.results[system_name][key] = result
                        
                        print(f"    ✅ Completed - Accuracy: {result['accuracy']:.3f}")
                        
                    except Exception as e:
                        print(f"    ❌ Failed: {e}")
                        self.results[system_name][f"{dataset_name}_{sample_size}"] = {
                            "error": str(e),
                            "accuracy": 0.0,
                            "response_time": 0.0
                        }
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total benchmark time: {total_time:.2f} seconds")
        
        return self.results
    
    def _run_single_test(self, system, dataset, sample_size):
        """Run single system test"""
        # Limit queries to min of available data and max_queries
        available_queries = min(len(dataset["questions"]), sample_size)
        num_queries = min(available_queries, self.eval_config.max_queries)
        
        correct_answers = 0
        total_time = 0
        
        for i in range(num_queries):
            start_time = time.time()
            
            # Mock RAG system query
            question = dataset["questions"][i]
            context = dataset["contexts"][i]
            expected_answer = dataset["answers"][i]
            
            # Simulate system response based on type
            if system["type"] == "insightspike":
                # Use actual InsightSpike if available
                if INSIGHTSPIKE_AVAILABLE and "llm_provider" in system:
                    try:
                        # Actual InsightSpike processing
                        response = self._query_insightspike(system["llm_provider"], question, context)
                    except Exception as e:
                        response = f"Mock InsightSpike answer for: {question}"
                else:
                    response = f"Mock InsightSpike answer for: {question}"
            else:
                response = f"Mock {system['name']} answer for: {question}"
            
            # Simple accuracy check (mock evaluation)
            is_correct = self._evaluate_answer(response, expected_answer)
            if is_correct:
                correct_answers += 1
            
            query_time = time.time() - start_time
            total_time += query_time
        
        accuracy = correct_answers / num_queries if num_queries > 0 else 0.0
        avg_response_time = total_time / num_queries if num_queries > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "response_time": avg_response_time,
            "total_queries": num_queries,
            "correct_answers": correct_answers
        }
    
    def _query_insightspike(self, llm_provider, question, context):
        """Query InsightSpike LLM provider"""
        try:
            # Use InsightSpike LLM provider for question answering
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            context_dict = {"question": question, "context": context}
            response = llm_provider.generate_response(context_dict, prompt)
            return response.get("response", "No answer generated") if isinstance(response, dict) else str(response)
        except Exception as e:
            self.logger.error(f"InsightSpike query failed: {e}")
            return f"InsightSpike error: {e}"
    
    def _evaluate_answer(self, response, expected_answer):
        """Evaluate answer quality (mock implementation)"""
        # Simple keyword matching for demonstration
        response_words = set(response.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        # Calculate simple overlap
        overlap = len(response_words & expected_words)
        total_expected = len(expected_words)
        
        # Consider correct if >50% overlap
        return overlap / total_expected > 0.5 if total_expected > 0 else False


class ResultsAnalyzer:
    """Analyzes and visualizes benchmark results"""
    
    def __init__(self, eval_config: EvalConfig, results: Dict):
        self.eval_config = eval_config
        self.results = results
        self.logger = logging.getLogger(__name__)
        
    def analyze_results(self):
        """Analyze benchmark results"""
        print("\n📊 Analyzing Results...")
        print("=" * 40)
        
        if not self.results:
            print("❌ No results to analyze")
            return
        
        # Create summary statistics
        summary = self._create_summary()
        
        # Print summary
        self._print_summary(summary)
        
        # Create visualizations if enabled
        if self.eval_config.enable_visualization:
            self._create_visualizations()
        
        # Save results if enabled
        if self.eval_config.save_results:
            self._save_results(summary)
        
        return summary
    
    def _create_summary(self):
        """Create summary statistics"""
        summary = {
            "experiment_id": self.eval_config.experiment_id,
            "timestamp": self.eval_config.timestamp,
            "profile": self.eval_config.profile,
            "systems_tested": list(self.results.keys()),
            "datasets_used": self.eval_config.datasets,
            "sample_sizes": self.eval_config.sample_sizes,
            "results": self.results
        }
        
        # Calculate aggregate statistics
        system_averages = {}
        for system_name, system_results in self.results.items():
            accuracies = []
            response_times = []
            
            for test_name, test_result in system_results.items():
                if "error" not in test_result:
                    accuracies.append(test_result["accuracy"])
                    response_times.append(test_result["response_time"])
            
            if accuracies:
                system_averages[system_name] = {
                    "avg_accuracy": np.mean(accuracies),
                    "avg_response_time": np.mean(response_times),
                    "std_accuracy": np.std(accuracies),
                    "std_response_time": np.std(response_times)
                }
        
        summary["system_averages"] = system_averages
        return summary
    
    def _print_summary(self, summary):
        """Print summary statistics"""
        print("\n📋 BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"🆔 Experiment ID: {summary['experiment_id']}")
        print(f"📅 Timestamp: {summary['timestamp']}")
        print(f"📋 Profile: {summary['profile']}")
        print(f"🤖 Systems: {', '.join(summary['systems_tested'])}")
        print(f"📚 Datasets: {', '.join(summary['datasets_used'])}")
        print(f"📊 Sample sizes: {summary['sample_sizes']}")
        
        print("\n🏆 SYSTEM PERFORMANCE RANKINGS:")
        print("-" * 50)
        
        if "system_averages" in summary:
            # Sort by accuracy
            sorted_systems = sorted(
                summary["system_averages"].items(),
                key=lambda x: x[1]["avg_accuracy"],
                reverse=True
            )
            
            for i, (system_name, stats) in enumerate(sorted_systems, 1):
                print(f"{i}. {system_name}:")
                print(f"   📈 Accuracy: {stats['avg_accuracy']:.3f} (±{stats['std_accuracy']:.3f})")
                print(f"   ⏱️  Response Time: {stats['avg_response_time']:.3f}s (±{stats['std_response_time']:.3f})")
                print()
    
    def _create_visualizations(self):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("📊 Creating visualizations...")
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figures directory
            viz_dir = self.eval_config.results_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. Accuracy comparison plot
            self._plot_accuracy_comparison(viz_dir)
            
            # 2. Response time comparison plot  
            self._plot_response_time_comparison(viz_dir)
            
            # 3. Combined performance plot
            self._plot_combined_performance(viz_dir)
            
            print(f"✅ Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            print(f"❌ Visualization creation failed: {e}")
    
    def _plot_accuracy_comparison(self, viz_dir):
        """Plot accuracy comparison"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            systems = []
            accuracies = []
            
            for system_name, system_results in self.results.items():
                for test_name, test_result in system_results.items():
                    if "error" not in test_result:
                        systems.append(system_name)
                        accuracies.append(test_result["accuracy"])
            
            if systems and accuracies:
                df = pd.DataFrame({"System": systems, "Accuracy": accuracies})
                sns.boxplot(data=df, x="System", y="Accuracy", ax=ax)
                ax.set_title("RAG Systems Accuracy Comparison")
                ax.set_ylabel("Accuracy")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plt.savefig(viz_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
                plt.close()
            
        except Exception as e:
            self.logger.error(f"Accuracy plot creation failed: {e}")
    
    def _plot_response_time_comparison(self, viz_dir):
        """Plot response time comparison"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            systems = []
            response_times = []
            
            for system_name, system_results in self.results.items():
                for test_name, test_result in system_results.items():
                    if "error" not in test_result:
                        systems.append(system_name)
                        response_times.append(test_result["response_time"])
            
            if systems and response_times:
                df = pd.DataFrame({"System": systems, "Response Time": response_times})
                sns.boxplot(data=df, x="System", y="Response Time", ax=ax)
                ax.set_title("RAG Systems Response Time Comparison")
                ax.set_ylabel("Response Time (seconds)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plt.savefig(viz_dir / "response_time_comparison.png", dpi=300, bbox_inches="tight")
                plt.close()
            
        except Exception as e:
            self.logger.error(f"Response time plot creation failed: {e}")
    
    def _plot_combined_performance(self, viz_dir):
        """Plot combined performance (accuracy vs response time)"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            for system_name, system_results in self.results.items():
                accuracies = []
                response_times = []
                
                for test_name, test_result in system_results.items():
                    if "error" not in test_result:
                        accuracies.append(test_result["accuracy"])
                        response_times.append(test_result["response_time"])
                
                if accuracies and response_times:
                    ax.scatter(response_times, accuracies, label=system_name, s=50, alpha=0.7)
            
            ax.set_xlabel("Response Time (seconds)")
            ax.set_ylabel("Accuracy")
            ax.set_title("RAG Systems Performance: Accuracy vs Response Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(viz_dir / "combined_performance.png", dpi=300, bbox_inches="tight")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Combined performance plot creation failed: {e}")
    
    def _save_results(self, summary):
        """Save results to files"""
        try:
            # Save JSON results
            results_file = self.eval_config.results_dir / "benchmark_results.json"
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Save pickle for detailed data
            pickle_file = self.eval_config.results_dir / "benchmark_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(summary, f)
            
            print(f"✅ Results saved to: {self.eval_config.results_dir}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def setup_logging(eval_config: EvalConfig):
    """Setup logging configuration"""
    log_file = eval_config.results_dir / "benchmark.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="RAG Systems Benchmark - Local Execution")
    parser.add_argument(
        "--profile", 
        choices=["demo", "research", "presentation", "insightspike_only"],
        default="demo",
        help="Execution profile (default: demo)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    print("🔍 RAG SYSTEMS BENCHMARK - LOCAL EXECUTION")
    print("=" * 60)
    
    # Initialize configuration
    eval_config = EvalConfig(args.profile)
    eval_config.print_config()
    
    # Setup logging
    setup_logging(eval_config)
    logger = logging.getLogger(__name__)
    
    try:
        # Check dependencies
        print("\n🔧 Checking Dependencies...")
        if not ML_AVAILABLE:
            print("❌ ML libraries not available. Please install: pip install sentence-transformers scikit-learn faiss-cpu")
            return
        
        # Initialize RAG systems
        system_manager = RAGSystemManager(eval_config)
        systems = system_manager.initialize_systems()
        
        if not systems:
            print("❌ No RAG systems available. Check your configuration and dependencies.")
            return
        
        # Load datasets
        dataset_manager = DatasetManager(eval_config)
        datasets = dataset_manager.load_datasets()
        
        if not datasets:
            print("❌ No datasets loaded. Check your configuration.")
            return
        
        # Run benchmark
        benchmark_runner = BenchmarkRunner(eval_config, systems, datasets)
        results = benchmark_runner.run_benchmark()
        
        # Analyze results
        analyzer = ResultsAnalyzer(eval_config, results)
        summary = analyzer.analyze_results()
        
        print("\n✅ Benchmark completed successfully!")
        print(f"📁 Results available in: {eval_config.results_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\n❌ Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
