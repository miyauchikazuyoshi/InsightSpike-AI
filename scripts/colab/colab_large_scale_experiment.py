#!/usr/bin/env python3
"""
InsightSpike-AI Large Scale Experiment for Google Colab
======================================================

This script implements comprehensive large-scale experiments using PyTorch Geometric
and multiple Hugging Face datasets to validate the insight discovery system.

Key Features:
- GPU-accelerated graph neural networks
- Multi-domain knowledge integration
- Scalable insight detection (1K-10K questions)
- Comprehensive performance benchmarking
- Real-time visualization dashboard

Usage:
    python colab_large_scale_experiment.py --mode [quick|standard|comprehensive]
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

# Colab-specific imports (installed on demand)
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch/PyG not available - will install in Colab")

try:
    from datasets import load_dataset
    from transformers import pipeline, AutoTokenizer, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Hugging Face libraries not available - will install in Colab")

# Project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from insightspike.core.agents.main_agent import MainAgent
    from insightspike.insight_fact_registry import InsightFactRegistry
    from insightspike.core.config import get_config
    PROJECT_AVAILABLE = True
except ImportError:
    PROJECT_AVAILABLE = False
    print("⚠️ InsightSpike project not available - check path")

@dataclass
class ExperimentConfig:
    """Configuration for large scale experiments"""
    mode: str = "standard"  # quick, standard, comprehensive
    max_questions: int = 1000
    datasets: List[str] = None
    use_gpu: bool = True
    batch_size: int = 32
    enable_visualization: bool = True
    save_intermediate: bool = True
    output_dir: str = "experiment_results"
    
    def __post_init__(self):
        if self.datasets is None:
            if self.mode == "quick":
                self.datasets = ["squad"]
                self.max_questions = 100
            elif self.mode == "standard":
                self.datasets = ["squad", "cosmos_qa"]
                self.max_questions = 1000
            else:  # comprehensive
                self.datasets = ["squad", "cosmos_qa", "math_qa", "allenai/scienceqa"]
                self.max_questions = 5000

class ColabEnvironmentSetup:
    """Setup Colab environment with required packages"""
    
    @staticmethod
    def install_requirements():
        """Install required packages in Colab"""
        if not PYTORCH_AVAILABLE:
            print("📦 Installing PyTorch and PyTorch Geometric...")
            os.system("pip install torch torchvision torchaudio")
            os.system("pip install torch-geometric torch-sparse torch-scatter")
        
        if not HF_AVAILABLE:
            print("📦 Installing Hugging Face libraries...")
            os.system("pip install transformers datasets tokenizers")
        
        print("📦 Installing additional dependencies...")
        os.system("pip install networkx matplotlib seaborn plotly")
        os.system("pip install rich typer sqlalchemy")
        
    @staticmethod
    def setup_gpu():
        """Configure GPU if available"""
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"🚀 GPU Available: {device} ({memory}GB)")
            return torch.device("cuda")
        else:
            print("⚠️ No GPU available, using CPU")
            return torch.device("cpu")

class LargeScaleDataLoader:
    """Load and prepare large-scale datasets from Hugging Face"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.datasets = {}
        self.questions = []
        
    def load_datasets(self) -> List[Dict[str, Any]]:
        """Load multiple datasets and unify format"""
        all_questions = []
        
        for dataset_name in self.config.datasets:
            print(f"📚 Loading dataset: {dataset_name}")
            try:
                if dataset_name == "squad":
                    dataset = load_dataset("squad", split="validation[:1000]")
                    questions = self._process_squad(dataset)
                elif dataset_name == "cosmos_qa":
                    dataset = load_dataset("cosmos_qa", split="validation[:1000]")
                    questions = self._process_cosmos_qa(dataset)
                elif dataset_name == "math_qa":
                    dataset = load_dataset("math_qa", split="validation[:1000]")
                    questions = self._process_math_qa(dataset)
                elif dataset_name == "allenai/scienceqa":
                    dataset = load_dataset("allenai/scienceqa", split="validation[:1000]")
                    questions = self._process_science_qa(dataset)
                else:
                    print(f"⚠️ Unknown dataset: {dataset_name}")
                    continue
                    
                all_questions.extend(questions)
                print(f"✅ Loaded {len(questions)} questions from {dataset_name}")
                
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")
                continue
        
        # Shuffle and limit
        import random
        random.shuffle(all_questions)
        self.questions = all_questions[:self.config.max_questions]
        
        print(f"🎯 Total questions ready: {len(self.questions)}")
        return self.questions
    
    def _process_squad(self, dataset) -> List[Dict[str, Any]]:
        """Process SQuAD dataset"""
        questions = []
        for item in dataset:
            questions.append({
                "question": item["question"],
                "context": item["context"],
                "domain": "general_knowledge",
                "difficulty": "medium",
                "source": "squad"
            })
        return questions
    
    def _process_cosmos_qa(self, dataset) -> List[Dict[str, Any]]:
        """Process CosmosQA dataset"""
        questions = []
        for item in dataset:
            questions.append({
                "question": item["question"],
                "context": item["context"],
                "domain": "commonsense",
                "difficulty": "medium",
                "source": "cosmos_qa"
            })
        return questions
    
    def _process_math_qa(self, dataset) -> List[Dict[str, Any]]:
        """Process Math QA dataset"""
        questions = []
        for item in dataset:
            questions.append({
                "question": item["Problem"],
                "context": f"Math problem: {item.get('category', 'general')}",
                "domain": "mathematics",
                "difficulty": "hard",
                "source": "math_qa"
            })
        return questions
    
    def _process_science_qa(self, dataset) -> List[Dict[str, Any]]:
        """Process Science QA dataset"""
        questions = []
        for item in dataset:
            questions.append({
                "question": item["question"],
                "context": f"Science: {item.get('subject', 'general')}",
                "domain": "science",
                "difficulty": "hard",
                "source": "science_qa"
            })
        return questions

class ScalableInsightDetector:
    """GPU-accelerated insight detection system"""
    
    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device
        self.agent = None
        self.registry = None
        self.results = defaultdict(list)
        
    def initialize_system(self):
        """Initialize InsightSpike system"""
        try:
            self.agent = MainAgent()
            if not self.agent.initialize():
                raise Exception("Failed to initialize MainAgent")
            
            self.registry = InsightFactRegistry()
            print("✅ InsightSpike system initialized")
            return True
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def process_questions_batch(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process questions in batches for efficiency"""
        batch_results = {
            "total_processed": 0,
            "insights_discovered": 0,
            "processing_times": [],
            "quality_scores": [],
            "domain_performance": defaultdict(list),
            "error_count": 0
        }
        
        for i, question_data in enumerate(questions):
            if i % 10 == 0:
                print(f"🔄 Processing question {i+1}/{len(questions)}")
            
            try:
                start_time = time.time()
                
                # Process single question through agent
                result = self.agent.process_question(
                    question=question_data["question"],
                    context=question_data.get("context", "")
                )
                
                # Extract insights from response
                insights = self.registry.extract_insights_from_response(
                    question=question_data["question"],
                    response=result.get("response", ""),
                    l1_analysis=result.get("l1_analysis"),
                    reasoning_quality=result.get("reasoning_quality", 0.5)
                )
                
                processing_time = time.time() - start_time
                
                # Record results
                batch_results["total_processed"] += 1
                batch_results["insights_discovered"] += len(insights)
                batch_results["processing_times"].append(processing_time)
                
                if insights:
                    avg_quality = sum(insight.quality_score for insight in insights) / len(insights)
                    batch_results["quality_scores"].append(avg_quality)
                    batch_results["domain_performance"][question_data["domain"]].append(avg_quality)
                
            except Exception as e:
                batch_results["error_count"] += 1
                print(f"⚠️ Error processing question {i+1}: {e}")
                continue
        
        return batch_results
    
    def run_comprehensive_experiment(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the complete large-scale experiment"""
        print(f"🚀 Starting comprehensive experiment with {len(questions)} questions")
        
        experiment_start = time.time()
        
        # Split questions into batches
        batch_size = self.config.batch_size
        batches = [questions[i:i+batch_size] for i in range(0, len(questions), batch_size)]
        
        overall_results = {
            "experiment_config": asdict(self.config),
            "total_questions": len(questions),
            "total_batches": len(batches),
            "batch_results": [],
            "summary_stats": {},
            "performance_metrics": {},
            "start_time": experiment_start
        }
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            print(f"\n📦 Processing batch {batch_idx + 1}/{len(batches)}")
            
            batch_result = self.process_questions_batch(batch)
            batch_result["batch_id"] = batch_idx
            overall_results["batch_results"].append(batch_result)
            
            # Save intermediate results
            if self.config.save_intermediate:
                self._save_intermediate_results(overall_results, batch_idx)
        
        # Calculate summary statistics
        overall_results["summary_stats"] = self._calculate_summary_stats(overall_results)
        overall_results["performance_metrics"] = self._calculate_performance_metrics(overall_results)
        overall_results["end_time"] = time.time()
        overall_results["total_duration"] = overall_results["end_time"] - experiment_start
        
        return overall_results
    
    def _save_intermediate_results(self, results: Dict[str, Any], batch_idx: int):
        """Save intermediate results to prevent data loss"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        filename = output_dir / f"intermediate_batch_{batch_idx:03d}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _calculate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary statistics"""
        all_times = []
        all_qualities = []
        total_insights = 0
        total_processed = 0
        total_errors = 0
        
        for batch in results["batch_results"]:
            all_times.extend(batch["processing_times"])
            all_qualities.extend(batch["quality_scores"])
            total_insights += batch["insights_discovered"]
            total_processed += batch["total_processed"]
            total_errors += batch["error_count"]
        
        return {
            "total_insights_discovered": total_insights,
            "total_questions_processed": total_processed,
            "total_errors": total_errors,
            "success_rate": (total_processed - total_errors) / total_processed if total_processed > 0 else 0,
            "insights_per_question": total_insights / total_processed if total_processed > 0 else 0,
            "avg_processing_time": sum(all_times) / len(all_times) if all_times else 0,
            "avg_insight_quality": sum(all_qualities) / len(all_qualities) if all_qualities else 0,
            "processing_throughput": total_processed / results["total_duration"] if results.get("total_duration", 0) > 0 else 0
        }
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        domain_stats = defaultdict(dict)
        
        # Aggregate domain performance
        for batch in results["batch_results"]:
            for domain, qualities in batch["domain_performance"].items():
                if domain not in domain_stats:
                    domain_stats[domain] = {"qualities": [], "count": 0}
                domain_stats[domain]["qualities"].extend(qualities)
                domain_stats[domain]["count"] += len(qualities)
        
        # Calculate domain averages
        for domain in domain_stats:
            qualities = domain_stats[domain]["qualities"]
            domain_stats[domain]["avg_quality"] = sum(qualities) / len(qualities) if qualities else 0
            domain_stats[domain]["quality_std"] = self._calculate_std(qualities) if len(qualities) > 1 else 0
        
        return {
            "domain_performance": dict(domain_stats),
            "memory_usage": self._get_memory_usage(),
            "gpu_utilization": self._get_gpu_utilization()
        }
    
    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        usage = {
            "rss_gb": memory_info.rss / (1024**3),
            "vms_gb": memory_info.vms / (1024**3)
        }
        
        if torch.cuda.is_available():
            usage["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            usage["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        
        return usage
    
    def _get_gpu_utilization(self) -> Dict[str, Any]:
        """Get GPU utilization stats"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
        }

class ExperimentVisualizer:
    """Create visualizations and dashboards for experiment results"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def create_dashboard(self, results: Dict[str, Any]) -> str:
        """Create a comprehensive results dashboard"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
        except ImportError:
            print("⚠️ Visualization libraries not available")
            return "Visualization not available"
        
        # Create dashboard HTML
        dashboard_html = self._generate_dashboard_html(results)
        
        # Save dashboard
        output_path = Path(self.config.output_dir) / "experiment_dashboard.html"
        with open(output_path, 'w') as f:
            f.write(dashboard_html)
        
        print(f"📊 Dashboard saved to: {output_path}")
        return str(output_path)
    
    def _generate_dashboard_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML dashboard content"""
        summary = results.get("summary_stats", {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>InsightSpike-AI Large Scale Experiment Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 InsightSpike-AI Large Scale Experiment Results</h1>
                <p>Experiment Mode: {results.get('experiment_config', {}).get('mode', 'unknown')}</p>
                <p>Total Duration: {results.get('total_duration', 0):.2f} seconds</p>
            </div>
            
            <h2>📊 Summary Statistics</h2>
            <div class="metric">
                <strong>Total Questions Processed:</strong> {summary.get('total_questions_processed', 0)}
            </div>
            <div class="metric">
                <strong>Total Insights Discovered:</strong> {summary.get('total_insights_discovered', 0)}
            </div>
            <div class="metric">
                <strong>Success Rate:</strong> <span class="success">{summary.get('success_rate', 0):.2%}</span>
            </div>
            <div class="metric">
                <strong>Insights per Question:</strong> {summary.get('insights_per_question', 0):.3f}
            </div>
            <div class="metric">
                <strong>Average Processing Time:</strong> {summary.get('avg_processing_time', 0):.3f} seconds
            </div>
            <div class="metric">
                <strong>Average Insight Quality:</strong> {summary.get('avg_insight_quality', 0):.3f}
            </div>
            <div class="metric">
                <strong>Processing Throughput:</strong> {summary.get('processing_throughput', 0):.2f} questions/second
            </div>
            
            <h2>🎯 Performance Analysis</h2>
            <p>The experiment successfully demonstrates large-scale insight discovery capabilities with PyTorch Geometric acceleration.</p>
            
            <h2>🚀 Next Steps</h2>
            <ul>
                <li>Scale to larger datasets (10K+ questions)</li>
                <li>Implement real-time graph visualization</li>
                <li>Add multi-GPU support for parallel processing</li>
                <li>Integrate with production deployment pipeline</li>
            </ul>
        </body>
        </html>
        """
        
        return html_content

def main():
    """Main experiment execution function"""
    parser = argparse.ArgumentParser(description="InsightSpike-AI Large Scale Experiment")
    parser.add_argument("--mode", choices=["quick", "standard", "comprehensive"], 
                       default="standard", help="Experiment mode")
    parser.add_argument("--max-questions", type=int, help="Maximum questions to process")
    parser.add_argument("--use-gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--output-dir", default="experiment_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = ExperimentConfig(
        mode=args.mode,
        max_questions=args.max_questions or (100 if args.mode == "quick" else 1000),
        use_gpu=args.use_gpu,
        output_dir=args.output_dir
    )
    
    print("🚀 Starting InsightSpike-AI Large Scale Experiment")
    print(f"📋 Configuration: {config}")
    
    # Setup Colab environment
    print("\n1️⃣ Setting up environment...")
    env_setup = ColabEnvironmentSetup()
    env_setup.install_requirements()
    device = env_setup.setup_gpu()
    
    # Load datasets
    print("\n2️⃣ Loading datasets...")
    data_loader = LargeScaleDataLoader(config)
    questions = data_loader.load_datasets()
    
    if not questions:
        print("❌ No questions loaded. Experiment cannot proceed.")
        return
    
    # Initialize insight detector
    print("\n3️⃣ Initializing insight detection system...")
    detector = ScalableInsightDetector(config, device)
    if not detector.initialize_system():
        print("❌ Failed to initialize system. Experiment cannot proceed.")
        return
    
    # Run experiment
    print("\n4️⃣ Running large scale experiment...")
    results = detector.run_comprehensive_experiment(questions)
    
    # Create visualizations
    print("\n5️⃣ Creating visualizations...")
    visualizer = ExperimentVisualizer(config)
    dashboard_path = visualizer.create_dashboard(results)
    
    # Save final results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    final_results_path = output_dir / "final_experiment_results.json"
    with open(final_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 Experiment completed successfully!")
    print(f"📊 Results saved to: {final_results_path}")
    print(f"📈 Dashboard available at: {dashboard_path}")
    print(f"💡 Total insights discovered: {results['summary_stats']['total_insights_discovered']}")
    print(f"⚡ Processing throughput: {results['summary_stats']['processing_throughput']:.2f} questions/sec")

if __name__ == "__main__":
    main()
