#!/usr/bin/env python3
"""
Google Colab Setup Script for InsightSpike-AI Experiments
=========================================================

This script sets up the Google Colab environment for running
large-scale InsightSpike experiments.
"""

import subprocess
import sys
import os
from pathlib import Path


def setup_colab_environment():
    """Setup Google Colab environment for InsightSpike experiments"""
    print("🚀 Setting up Google Colab for InsightSpike experiments...")
    
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
        print("⚠️ Warning: Not running in Google Colab environment")
    
    # Clone repository if not exists
    if not Path("/content/InsightSpike-AI").exists():
        print("\n📦 Cloning InsightSpike-AI repository...")
        subprocess.run([
            "git", "clone", "https://github.com/Sunwood-ai-labs/InsightSpike-AI.git",
            "/content/InsightSpike-AI"
        ], check=True)
    
    # Change to repository directory
    os.chdir("/content/InsightSpike-AI")
    
    # Install dependencies
    print("\n📚 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "poetry"], check=True)
    subprocess.run(["poetry", "install", "--no-interaction"], check=True)
    
    # Install additional Colab-specific dependencies
    print("\n🔧 Installing Colab-specific dependencies...")
    colab_deps = [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",  # For 8-bit quantization
        "datasets>=2.14.0",
        "wandb>=0.15.0",  # For experiment tracking
        "plotly>=5.17.0",  # For interactive visualizations
        "kaleido>=0.2.1",  # For plotly image export
    ]
    
    for dep in colab_deps:
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
    
    # Setup GPU if available
    print("\n🎮 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("❌ No GPU available, using CPU")
    except:
        print("⚠️ PyTorch not properly installed")
    
    # Create experiment directories
    print("\n📁 Creating experiment directories...")
    exp_dirs = [
        "experiments/colab_experiments",
        "experiments/colab_experiments/data/input",
        "experiments/colab_experiments/data/processed",
        "experiments/colab_experiments/results/metrics",
        "experiments/colab_experiments/results/outputs",
        "experiments/colab_experiments/results/visualizations",
        "experiments/colab_experiments/data_snapshots",
    ]
    
    for dir_path in exp_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Mount Google Drive if in Colab
    if IN_COLAB:
        print("\n☁️ Mounting Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted at /content/drive")
    
    # Setup environment variables
    print("\n🔧 Setting up environment variables...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["INSIGHTSPIKE_ENV"] = "colab"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Create Colab-specific config
    create_colab_config()
    
    print("\n✅ Colab setup complete!")
    print("\n📝 Next steps:")
    print("1. Run experiments with: python experiments/colab_experiments/run_experiment.py")
    print("2. Monitor with Weights & Biases: wandb.login()")
    print("3. Save results to Drive: /content/drive/MyDrive/InsightSpike_Results/")
    
    return True


def create_colab_config():
    """Create Colab-specific configuration"""
    config_content = """# Google Colab Configuration for InsightSpike-AI
# =============================================

# Core settings optimized for Colab
core:
  model_name: "paraphrase-MiniLM-L6-v2"
  llm_provider: "local"
  llm_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  max_tokens: 512
  temperature: 0.3
  device: "cuda"  # Use GPU in Colab
  use_gpu: true
  safe_mode: false

# Memory settings for Colab (adjust based on instance)
memory:
  max_retrieved_docs: 50
  short_term_capacity: 20
  working_memory_capacity: 40
  episodic_memory_capacity: 100
  pattern_cache_capacity: 30

# Increased batch sizes for GPU
retrieval:
  similarity_threshold: 0.35
  top_k: 30
  layer1_top_k: 40
  layer2_top_k: 30
  layer3_top_k: 25

# Graph processing
graph:
  spike_ged_threshold: 0.5
  spike_ig_threshold: 0.2
  use_gnn: true  # Enable GNN on GPU
  gnn_hidden_dim: 128
  ged_algorithm: "hybrid"
  ig_algorithm: "hybrid"

# Processing settings for Colab
processing:
  batch_size: 128  # Larger batch for GPU
  max_workers: 4
  timeout_seconds: 600

# Output settings
output:
  default_format: "json"
  save_results: true
  generate_visualizations: true
  verbose: true

# Paths (Colab-specific)
paths:
  data_dir: "/content/InsightSpike-AI/data/raw"
  log_dir: "/content/InsightSpike-AI/data/logs"
  index_file: "/content/InsightSpike-AI/data/index.faiss"
  graph_file: "/content/InsightSpike-AI/data/graph_pyg.pt"
  
# Environment
environment: "colab"
"""
    
    config_path = Path("experiments/colab_experiments/colab_config.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"📝 Created Colab config at: {config_path}")


def test_installation():
    """Test if InsightSpike is properly installed"""
    print("\n🧪 Testing InsightSpike installation...")
    
    try:
        # Test core imports
        from insightspike.config import InsightSpikeConfig
        from insightspike.core.agents.main_agent import MainAgent
        from insightspike.processing.embedder import EmbeddingManager
        print("✅ Core modules imported successfully")
        
        # Test configuration
        config = InsightSpikeConfig()
        print(f"✅ Configuration loaded: {config.environment}")
        
        # Test embedder
        embedder = EmbeddingManager()
        test_text = "This is a test"
        embedding = embedder.embed_text(test_text)
        print(f"✅ Embedder working: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


if __name__ == "__main__":
    # Run setup
    if setup_colab_environment():
        # Test installation
        test_installation()
        
        print("\n🎉 Setup complete! You can now run experiments.")
        print("\nExample usage:")
        print("```python")
        print("from experiments.colab_experiments.insight_benchmarks import run_rat_benchmark")
        print("results = run_rat_benchmark(n_problems=100)")
        print("```")