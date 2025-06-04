# Quick Start Guide

## Overview

InsightSpike-AI is a cognitive architecture that detects insight moments in AI reasoning through Graph Edit Distance (GED) and Information Gain (IG) metrics. This guide helps you get started quickly.

## Prerequisites

- Python 3.8 or higher
- Git
- 4GB RAM minimum
- 1GB free disk space

## Installation

### Option 1: Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/InsightSpike-AI.git
cd InsightSpike-AI

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Option 2: pip

```bash
# Clone the repository
git clone https://github.com/your-username/InsightSpike-AI.git
cd InsightSpike-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Option 3: Docker

```bash
# Clone the repository
git clone https://github.com/your-username/InsightSpike-AI.git
cd InsightSpike-AI

# Build Docker image
docker build -t insightspike-ai .

# Run container
docker run -it --rm -v $(pwd):/workspace insightspike-ai
```

## Quick Test

Verify your installation works:

```bash
# Run benchmark tests
python -m pytest tests/test_benchmarks.py -v

# Test basic functionality
python scripts/run_poc_simple.py
```

## First Example: Insight Detection

```python
# examples/basic_insight_detection.py
import sys
sys.path.append('src')

from insightspike.core.layers.mock_llm_provider import MockLLMProvider

# Initialize the system
provider = MockLLMProvider()

# Test insight detection on a paradox
query = "In the Monty Hall problem, should you switch doors?"
response = provider.generate_intelligent_response(query)

print(f"Response: {response['response']}")
print(f"Confidence: {response['confidence']:.3f}")
print(f"Reasoning Quality: {response['reasoning_quality']:.3f}")

# Check for cross-domain insights
if response.get('cross_domain_bonus', 0) > 0.1:
    print("Cross-domain insight detected!")
```

## Common Use Cases

### 1. Educational Assessment

```python
# Test concept understanding
educational_query = "Explain the relationship between entropy and uncertainty"
result = provider.generate_intelligent_response(educational_query)

# Check if student shows conceptual understanding
if result['confidence'] > 0.7:
    print("Strong conceptual understanding detected")
```

### 2. Reinforcement Learning Analysis

```python
# Run maze experiment
from experiments.enhanced_rl_comparison import main as run_rl_experiment

# This will run a small-scale RL experiment with insight detection
run_rl_experiment()
```

### 3. Paradox Resolution

```python
# Test on philosophical paradoxes
queries = [
    "What is the Ship of Theseus paradox?",
    "How is Zeno's paradox resolved?",
    "Explain the trolley problem dilemma"
]

for query in queries:
    result = provider.generate_intelligent_response(query)
    print(f"Query: {query}")
    print(f"Insight Quality: {result['reasoning_quality']:.3f}")
    print("---")
```

## Understanding the Output

### Key Metrics

- **Confidence**: How certain the system is about its response (0.0-1.0)
- **Reasoning Quality**: Quality of the reasoning process (0.0-1.0)
- **Cross-domain Bonus**: Additional score for connecting multiple domains
- **Processing Time**: Time taken to generate the response

### Insight Detection Criteria

- **ΔGED < -0.5**: Significant improvement in graph efficiency
- **ΔIG > 1.5**: Substantial information gain
- **Combined Threshold**: Both criteria must be met for insight detection

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory and environment
   cd InsightSpike-AI
   poetry shell  # or source venv/bin/activate
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Missing Data Files**
   ```bash
   # Create required directories
   mkdir -p data/samples
   mkdir -p data/processed
   mkdir -p experiments/results
   ```

3. **Memory Issues**
   ```bash
   # Run smaller experiments
   python scripts/run_poc_simple.py --small-scale
   ```

### Performance Tips

- Use the MockLLMProvider for fast prototyping
- Run benchmark tests to verify performance
- Monitor memory usage for large-scale experiments
- Check CI status for latest compatibility

## Next Steps

1. **Read the Documentation**: Check `documentation/` for detailed guides
2. **Run Experiments**: Try the examples in `experiments/`
3. **Contribute**: See `CONTRIBUTING.md` for development guidelines
4. **Report Issues**: Use GitHub Issues for bugs or questions

## Support

- **Documentation**: `documentation/guides/`
- **Examples**: `experiments/` directory
- **Tests**: `tests/` directory for reference implementations
- **Community**: GitHub Discussions for questions

---

*This guide gets you started quickly. For comprehensive documentation, see the main documentation directory.*
