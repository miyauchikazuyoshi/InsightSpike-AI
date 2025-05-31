#!/usr/bin/env python3
"""
Hugging Face Dataset Integration Test for InsightSpike-AI
========================================================

Comprehensive testing of dataset loading, GPU acceleration, and processing
throughput for production-level insight discovery experiments.

Usage:
    python test_hf_dataset_integration.py [--gpu] [--datasets squad,cosmos_qa]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import torch
    import torch_geometric
    from transformers import pipeline, AutoTokenizer, AutoModel
    from datasets import load_dataset
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    logger.error(f"Required dependencies not available: {e}")

class HuggingFaceDatasetTester:
    """Test Hugging Face dataset integration with InsightSpike-AI"""
    
    SUPPORTED_DATASETS = {
        "squad": {
            "config": None,
            "split": "validation",
            "text_field": "question",
            "description": "SQuAD 2.0 reading comprehension dataset"
        },
        "cosmos_qa": {
            "config": None,
            "split": "validation",
            "text_field": "question",
            "description": "Commonsense reading comprehension dataset"
        },
        "math_qa": {
            "config": None,
            "split": "validation",
            "text_field": "Problem",
            "description": "Mathematical word problems dataset"
        },
        "allenai/scienceqa": {
            "config": None,
            "split": "validation",
            "text_field": "question",
            "description": "Science question answering dataset"
        }
    }
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.results = {}
        
        logger.info(f"Initialized tester with device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
    
    def test_dataset_loading(self, dataset_name: str, max_samples: int = 100) -> Dict[str, Any]:
        """Test loading and processing a specific dataset"""
        logger.info(f"Testing dataset: {dataset_name}")
        
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        dataset_info = self.SUPPORTED_DATASETS[dataset_name]
        start_time = time.time()
        
        try:
            # Load dataset
            logger.info(f"Loading {dataset_name}...")
            dataset = load_dataset(
                dataset_name,
                config=dataset_info["config"],
                split=dataset_info["split"]
            )
            
            load_time = time.time() - start_time
            logger.info(f"Dataset loaded in {load_time:.2f}s")
            
            # Extract samples
            samples = []
            text_field = dataset_info["text_field"]
            
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                if text_field in item:
                    samples.append({
                        "id": i,
                        "text": item[text_field],
                        "metadata": {k: v for k, v in item.items() if k != text_field}
                    })
            
            # Performance metrics
            total_time = time.time() - start_time
            throughput = len(samples) / total_time if total_time > 0 else 0
            
            results = {
                "dataset_name": dataset_name,
                "description": dataset_info["description"],
                "total_samples": len(dataset),
                "processed_samples": len(samples),
                "load_time": load_time,
                "total_time": total_time,
                "throughput": throughput,
                "success": True,
                "error": None
            }
            
            logger.info(f"✅ {dataset_name}: {len(samples)} samples, {throughput:.2f} samples/sec")
            return results
            
        except Exception as e:
            error_msg = f"Failed to load {dataset_name}: {str(e)}"
            logger.error(error_msg)
            
            return {
                "dataset_name": dataset_name,
                "description": dataset_info["description"],
                "total_samples": 0,
                "processed_samples": 0,
                "load_time": 0,
                "total_time": time.time() - start_time,
                "throughput": 0,
                "success": False,
                "error": error_msg
            }
    
    def test_gpu_acceleration(self, samples: List[str], batch_size: int = 16) -> Dict[str, Any]:
        """Test GPU acceleration for text processing"""
        logger.info("Testing GPU acceleration...")
        
        if not samples:
            return {"gpu_available": False, "error": "No samples provided"}
        
        start_time = time.time()
        
        try:
            # Initialize model on GPU
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(self.device)
            
            # Process in batches
            embeddings = []
            total_batches = (len(samples) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(samples), batch_size), 
                         desc="Processing batches", total=total_batches):
                batch = samples[i:i + batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(batch_embeddings.cpu().numpy())
            
            # Combine results
            all_embeddings = np.vstack(embeddings)
            processing_time = time.time() - start_time
            throughput = len(samples) / processing_time
            
            # GPU utilization (if available)
            gpu_stats = {}
            if self.device.type == "cuda":
                gpu_stats = {
                    "memory_allocated": torch.cuda.memory_allocated() // (1024**2),  # MB
                    "memory_cached": torch.cuda.memory_reserved() // (1024**2),     # MB
                    "max_memory": torch.cuda.max_memory_allocated() // (1024**2)    # MB
                }
            
            results = {
                "gpu_available": self.device.type == "cuda",
                "device": str(self.device),
                "samples_processed": len(samples),
                "embedding_dimension": all_embeddings.shape[1],
                "processing_time": processing_time,
                "throughput": throughput,
                "batch_size": batch_size,
                "gpu_stats": gpu_stats,
                "success": True,
                "error": None
            }
            
            logger.info(f"✅ GPU processing: {throughput:.2f} samples/sec")
            return results
            
        except Exception as e:
            error_msg = f"GPU acceleration test failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "gpu_available": self.device.type == "cuda",
                "device": str(self.device),
                "success": False,
                "error": error_msg
            }
    
    def benchmark_throughput(self, dataset_names: List[str], 
                           max_samples_per_dataset: int = 500) -> Dict[str, Any]:
        """Benchmark processing throughput across multiple datasets"""
        logger.info("Starting throughput benchmark...")
        
        benchmark_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(self.device),
            "max_samples_per_dataset": max_samples_per_dataset,
            "datasets": {},
            "summary": {}
        }
        
        all_samples = []
        total_load_time = 0
        successful_datasets = 0
        
        # Test each dataset
        for dataset_name in dataset_names:
            logger.info(f"Benchmarking {dataset_name}...")
            
            result = self.test_dataset_loading(dataset_name, max_samples_per_dataset)
            benchmark_results["datasets"][dataset_name] = result
            
            if result["success"]:
                successful_datasets += 1
                total_load_time += result["load_time"]
                
                # Collect samples for GPU test
                dataset = load_dataset(
                    dataset_name,
                    config=self.SUPPORTED_DATASETS[dataset_name]["config"],
                    split=self.SUPPORTED_DATASETS[dataset_name]["split"]
                )
                
                text_field = self.SUPPORTED_DATASETS[dataset_name]["text_field"]
                for i, item in enumerate(dataset):
                    if i >= max_samples_per_dataset:
                        break
                    if text_field in item:
                        all_samples.append(item[text_field])
        
        # GPU acceleration test
        if all_samples:
            gpu_result = self.test_gpu_acceleration(all_samples[:1000])  # Limit for performance
            benchmark_results["gpu_acceleration"] = gpu_result
        
        # Summary statistics
        total_samples = sum(r["processed_samples"] for r in benchmark_results["datasets"].values() if r["success"])
        avg_throughput = np.mean([r["throughput"] for r in benchmark_results["datasets"].values() if r["success"]])
        
        benchmark_results["summary"] = {
            "total_datasets_tested": len(dataset_names),
            "successful_datasets": successful_datasets,
            "total_samples_processed": total_samples,
            "total_load_time": total_load_time,
            "average_throughput": avg_throughput,
            "gpu_acceleration_available": self.device.type == "cuda"
        }
        
        logger.info(f"✅ Benchmark complete: {total_samples} samples from {successful_datasets} datasets")
        return benchmark_results
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "hf_dataset_integration_report.json"):
        """Generate comprehensive test report"""
        import json
        
        # Save detailed results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("🔬 HUGGING FACE DATASET INTEGRATION TEST REPORT")
        print("="*60)
        
        summary = results.get("summary", {})
        print(f"📊 Total datasets tested: {summary.get('total_datasets_tested', 0)}")
        print(f"✅ Successful datasets: {summary.get('successful_datasets', 0)}")
        print(f"📝 Total samples processed: {summary.get('total_samples_processed', 0)}")
        print(f"⚡ Average throughput: {summary.get('average_throughput', 0):.2f} samples/sec")
        print(f"🚀 GPU acceleration: {'Available' if summary.get('gpu_acceleration_available') else 'Not available'}")
        
        print("\n📋 Dataset Results:")
        for dataset_name, result in results.get("datasets", {}).items():
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {dataset_name}: {result['processed_samples']} samples "
                  f"({result['throughput']:.2f} samples/sec)")
        
        if "gpu_acceleration" in results:
            gpu_result = results["gpu_acceleration"]
            if gpu_result["success"]:
                print(f"\n🚀 GPU Acceleration Results:")
                print(f"  Device: {gpu_result['device']}")
                print(f"  Throughput: {gpu_result['throughput']:.2f} samples/sec")
                print(f"  Embedding dimension: {gpu_result['embedding_dimension']}")
            else:
                print(f"\n❌ GPU Acceleration Failed: {gpu_result.get('error', 'Unknown error')}")
        
        print(f"\n📄 Detailed report saved to: {output_file}")
        print("="*60)


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test Hugging Face dataset integration")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--datasets", default="squad,cosmos_qa", 
                       help="Comma-separated list of datasets to test")
    parser.add_argument("--max-samples", type=int, default=500,
                       help="Maximum samples per dataset")
    parser.add_argument("--output", default="hf_dataset_integration_report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    if not DEPS_AVAILABLE:
        logger.error("Required dependencies not available. Please run setup_colab.sh first.")
        return 1
    
    # Parse dataset list
    dataset_names = [name.strip() for name in args.datasets.split(",")]
    
    logger.info("🔬 Starting Hugging Face Dataset Integration Test")
    logger.info(f"📊 Datasets: {dataset_names}")
    logger.info(f"📝 Max samples per dataset: {args.max_samples}")
    
    # Initialize tester
    tester = HuggingFaceDatasetTester(use_gpu=args.gpu)
    
    # Run benchmark
    results = tester.benchmark_throughput(dataset_names, args.max_samples)
    
    # Generate report
    tester.generate_report(results, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
