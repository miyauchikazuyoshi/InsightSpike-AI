#!/bin/bash
# Setup script for geDIG-RAG v3 experiment environment

set -e  # Exit on error

echo "🚀 Setting up geDIG-RAG v3 Experiment Environment"
echo "=================================================="

# Check Python version
PYTHON_VERSION_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_VERSION_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PYTHON_VERSION_NUM=$((PYTHON_VERSION_MAJOR * 10 + PYTHON_VERSION_MINOR))

if [ "$PYTHON_VERSION_NUM" -lt "39" ]; then
    echo "❌ Python 3.9+ is required. Current version: $(python3 --version)"
    exit 1
fi
echo "✅ Python version check passed: $(python3 --version)"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ Poetry is already installed: $(poetry --version)"
fi

# Create project directories
echo "📁 Creating project directory structure..."
mkdir -p data/{input,processed}
mkdir -p results/{metrics,outputs,visualizations,paper_figures,paper_draft}
mkdir -p data_snapshots
mkdir -p logs
mkdir -p models
mkdir -p tmp

echo "✅ Directory structure created"

# Install dependencies
echo "📦 Installing Python dependencies..."
poetry install --all-extras

echo "🔧 Installing additional NLP resources..."
poetry run python -c "
import nltk
import spacy
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print('✅ NLTK resources downloaded')

# Check spaCy model
try:
    import spacy
    spacy.load('en_core_web_sm')
    print('✅ spaCy model already available')
except OSError:
    print('⚠️  spaCy model not found. Please install with: python -m spacy download en_core_web_sm')
"

# Setup pre-commit hooks (if .pre-commit-config.yaml exists)
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Setting up pre-commit hooks..."
    poetry run pre-commit install
    echo "✅ Pre-commit hooks installed"
fi

# Create default configuration
echo "⚙️  Creating default configuration..."
cat > configs/default_config.yaml << EOF
# geDIG-RAG v3 Default Configuration

# geDIG Parameters
gedig:
  k_coefficient: 0.5
  radius: 2
  thresholds:
    add_ig_threshold: 0.3
    add_ged_min: 0.1
    add_ged_max: 0.8
    merge_similarity: 0.8
    prune_usage_min: 2

# Experiment Parameters
experiment:
  n_sessions: 5
  queries_per_session: 20
  seeds: [42, 43, 44]
  enable_baselines: ["static", "frequency", "cosine", "gedig"]

# Model Parameters
models:
  embedding_model: "all-MiniLM-L6-v2"
  generation_model: "microsoft/DialoGPT-medium"

# Dataset Parameters
datasets:
  hotpot_qa_size: 1000
  domain_qa_path: "data/input/domain_qa.json"

# Output Parameters
output:
  save_detailed_logs: true
  generate_figures: true
  output_formats: ["json", "csv", "png", "pdf"]
  
# Computational Parameters
compute:
  n_processes: 4
  gpu_enabled: false
  memory_limit_gb: 8
EOF

echo "✅ Default configuration created"

# Create example data preparation script
echo "📊 Creating data preparation script..."
cat > src/data/prepare_example_data.py << 'EOF'
#!/usr/bin/env python3
"""Prepare example datasets for geDIG-RAG v3 experiments."""

import json
import random
from pathlib import Path
from datasets import load_dataset

def prepare_hotpot_qa_sample(output_path: Path, size: int = 100):
    """Prepare HotpotQA sample for testing."""
    print(f"📥 Downloading HotpotQA sample ({size} examples)...")
    
    try:
        dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
        
        # Sample random examples
        indices = random.sample(range(len(dataset)), min(size, len(dataset)))
        sampled_data = []
        
        for idx in indices:
            item = dataset[idx]
            sampled_data.append({
                "id": f"hotpot_{idx}",
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "supporting_facts": item["supporting_facts"],
                "context": item["context"]
            })
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sampled_data, f, indent=2)
        
        print(f"✅ HotpotQA sample saved to {output_path}")
        
    except Exception as e:
        print(f"⚠️  Could not download HotpotQA: {e}")
        print("Creating minimal example data instead...")
        
        # Create minimal example data
        example_data = [
            {
                "id": "example_1",
                "question": "What is machine learning?",
                "answer": "Machine learning is a method of data analysis that automates analytical model building.",
                "type": "bridge",
                "level": "easy"
            },
            {
                "id": "example_2", 
                "question": "How do neural networks work?",
                "answer": "Neural networks work by processing data through layers of interconnected nodes that learn patterns.",
                "type": "comparison",
                "level": "medium"
            }
        ]
        
        with open(output_path, 'w') as f:
            json.dump(example_data, f, indent=2)
        
        print(f"✅ Example data created at {output_path}")

def prepare_domain_qa_example(output_path: Path):
    """Prepare example domain QA data."""
    
    domain_data = [
        {
            "id": "domain_1",
            "question": "What are the benefits of distributed systems?",
            "answer": "Distributed systems provide fault tolerance, scalability, and improved performance through parallel processing.",
            "domain": "computer_science",
            "difficulty": "medium"
        },
        {
            "id": "domain_2",
            "question": "Explain the CAP theorem.",
            "answer": "The CAP theorem states that distributed systems can only guarantee two of three properties: Consistency, Availability, and Partition tolerance.",
            "domain": "computer_science", 
            "difficulty": "hard"
        }
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(domain_data, f, indent=2)
    
    print(f"✅ Domain QA example created at {output_path}")

if __name__ == "__main__":
    # Prepare example datasets
    data_dir = Path("data/input")
    
    prepare_hotpot_qa_sample(data_dir / "hotpot_qa_sample.json", size=100)
    prepare_domain_qa_example(data_dir / "domain_qa.json")
    
    print("🎉 Data preparation completed!")
EOF

chmod +x src/data/prepare_example_data.py

# Create basic CLI
echo "🔧 Creating CLI interface..."
mkdir -p src/cli
cat > src/cli/main.py << 'EOF'
#!/usr/bin/env python3
"""Main CLI interface for geDIG-RAG v3."""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="geDIG-RAG v3 Experiment Suite")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup experiment environment")
    
    # Data preparation command
    data_parser = subparsers.add_parser("prepare-data", help="Prepare datasets")
    data_parser.add_argument("--size", type=int, default=100, help="Sample size for HotpotQA")
    
    # Run experiments command
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    run_parser.add_argument("--phase", choices=["baseline", "longterm", "ablation", "all"], default="all")
    
    # Generate results command
    gen_parser = subparsers.add_parser("generate", help="Generate paper results")
    gen_parser.add_argument("--figures", action="store_true", help="Generate figures")
    gen_parser.add_argument("--tables", action="store_true", help="Generate tables")
    gen_parser.add_argument("--paper", action="store_true", help="Generate paper draft")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        print("🔧 Environment already set up!")
        print("Run 'gedig-rag prepare-data' to prepare datasets")
        
    elif args.command == "prepare-data":
        from src.data.prepare_example_data import prepare_hotpot_qa_sample, prepare_domain_qa_example
        data_dir = Path("data/input")
        prepare_hotpot_qa_sample(data_dir / "hotpot_qa_sample.json", size=args.size)
        prepare_domain_qa_example(data_dir / "domain_qa.json")
        
    elif args.command == "run":
        print(f"🚀 Running experiments with config: {args.config}")
        print(f"Phase: {args.phase}")
        print("⚠️  Full experiment runner not yet implemented")
        print("See implementation roadmap for development plan")
        
    elif args.command == "generate":
        print("📊 Generating results...")
        if args.figures:
            print("📈 Generating figures...")
        if args.tables:
            print("📋 Generating tables...")
        if args.paper:
            print("📄 Generating paper draft...")
        print("⚠️  Result generators not yet implemented")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
EOF

# Create quick test script
echo "🧪 Creating quick test script..."
cat > scripts/quick_test.sh << 'EOF'
#!/bin/bash
# Quick functionality test for geDIG-RAG v3

echo "🧪 Running Quick Functionality Test"
echo "=================================="

echo "1. Testing Python environment..."
poetry run python -c "
import sys
print(f'✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
except ImportError:
    print('❌ PyTorch not available')

try:
    import transformers
    print(f'✅ Transformers {transformers.__version__}')
except ImportError:
    print('❌ Transformers not available')

try:
    import networkx
    print(f'✅ NetworkX {networkx.__version__}')
except ImportError:
    print('❌ NetworkX not available')
"

echo -e "\n2. Testing CLI interface..."
poetry run gedig-rag --help > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ CLI interface working"
else
    echo "❌ CLI interface not working"
fi

echo -e "\n3. Testing data preparation..."
poetry run gedig-rag prepare-data --size 5
if [ $? -eq 0 ]; then
    echo "✅ Data preparation working"
else
    echo "❌ Data preparation failed"
fi

echo -e "\n4. Checking directory structure..."
for dir in "data/input" "results" "logs" "configs"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir directory exists"
    else
        echo "❌ $dir directory missing"
    fi
done

echo -e "\n🎉 Quick test completed!"
echo "Ready to begin implementation following the roadmap!"
EOF

chmod +x scripts/quick_test.sh

# Create directory placeholders
touch src/core/__init__.py
touch src/baselines/__init__.py
touch src/experiments/__init__.py
touch src/evaluation/__init__.py
touch src/visualization/__init__.py
touch src/paper/__init__.py
touch src/quality/__init__.py
touch src/data/__init__.py
touch src/cli/__init__.py

# Run quick test
echo "🧪 Running quick functionality test..."
chmod +x scripts/quick_test.sh
./scripts/quick_test.sh

echo ""
echo "🎉 geDIG-RAG v3 Environment Setup Complete!"
echo "==========================================="
echo ""
echo "📋 Next Steps:"
echo "1. Review the implementation roadmap: docs/IMPLEMENTATION_ROADMAP.md"
echo "2. Start with Week 1 implementation: src/core/ components"
echo "3. Run experiments: poetry run gedig-rag run --phase baseline"
echo "4. Generate results: poetry run gedig-rag generate --figures --tables --paper"
echo ""
echo "🔧 Useful Commands:"
echo "- poetry run gedig-rag --help                 # Show CLI help"
echo "- poetry run gedig-rag prepare-data           # Prepare datasets"  
echo "- ./scripts/quick_test.sh                     # Run functionality test"
echo "- poetry run pytest                          # Run test suite"
echo "- poetry run black src/                      # Format code"
echo ""
echo "📚 Documentation:"
echo "- README.md                                   # Project overview"
echo "- docs/SPECIFICATION.md                      # Technical specifications"
echo "- docs/IMPLEMENTATION_ROADMAP.md             # Implementation plan"
echo ""
echo "Ready to implement the geDIG-RAG v3 system! 🚀"