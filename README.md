# InsightSpike-AI

**Brain-Inspired AI Architecture for Insight Detection and Knowledge Restructuring**

[![License: InsightSpike Responsible AI](https://img.shields.io/badge/License-InsightSpike--Responsible--AI--1.0-blue)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/LICENSE)  
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-Poetry-blue)](https://python-poetry.org/)

## 🚀 Quick Start

### Google Colab (Recommended)
```python
# Open in Google Colab and run:
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!pip install -e .
```

### Local Installation
```bash
# Clone repository
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Install with Poetry
poetry install

# Run demo
poetry run insightspike demo
```

## 🎯 What is InsightSpike-AI?

InsightSpike-AI is a research project that implements a neurobiologically-inspired AI architecture for detecting and modeling "insight moments" - those "Aha!" moments when knowledge suddenly restructures. The system uses a novel **geDIG** (Graph Edit Distance + Information Gain) methodology to identify when AI systems experience significant conceptual breakthroughs.

### Key Innovation: geDIG Technology

- **ΔGED**: Measures structural simplification in knowledge graphs
- **ΔIG**: Quantifies information entropy changes during learning
- **EurekaSpike**: Triggers when both metrics indicate significant knowledge restructuring

## 📊 Experimental Results

Based on comprehensive experiments across 4 phases:

### ✅ Proven Strengths
- **Memory Efficiency**: 50% reduction in memory usage while maintaining accuracy
- **Insight Detection**: 37.6% improvement in detecting meaningful knowledge patterns
- **Novel Algorithm**: Successfully demonstrated slime mold-inspired optimization

### 🔬 Current Limitations
- **Processing Speed**: Currently optimized for research, not production speed
- **Scalability**: Best suited for small to medium-scale experiments
- **RAG Performance**: Basic retrieval functions need optimization for production use

## 🏗️ Architecture

The system implements a 4-layer architecture inspired by brain structures:

1. **Error Monitor** (Cerebellum analog) - Query analysis
2. **Memory Manager** (Hippocampus analog) - Vector quantized episodic memory
3. **Graph Reasoner** (Prefrontal cortex analog) - GNN-based reasoning with geDIG
4. **Language Interface** (Language area analog) - Natural language synthesis

## 📁 Project Structure

```
InsightSpike-AI/
├── src/insightspike/      # Core implementation
├── experiments/           # Experimental validation
│   ├── phase1_dynamic_memory/
│   ├── phase2_rag_benchmark/
│   ├── phase3_gedig_maze/
│   └── phase4_integrated_evaluation/
├── docs/                  # Documentation
├── tests/                 # Test suite
└── scripts/              # Utility scripts
```

## 🔬 Research Applications

- **Cognitive Science**: Understanding insight and learning mechanisms
- **Educational Technology**: Detecting when students truly understand concepts
- **AI Research**: Novel approaches to knowledge representation and reasoning
- **Optimization**: Bio-inspired algorithms for complex problem solving

## 📚 Documentation

- [Technical Specifications](docs/technical_specifications.md)
- [API Documentation](docs/api/README.md)
- [Experiment Results](experiments/README.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the InsightSpike AI Responsible Use License v1.0. See [LICENSE](LICENSE) for details.

### Patent Notice
Core technologies are patent-pending:
- JP Application No. 特願2025-082988 — "ΔGED/ΔIG 内発報酬生成方法"
- JP Application No. 特願2025-082989 — "階層ベクトル量子化による動的メモリ方法"

## 📧 Contact

For questions, collaborations, or commercial licensing:
- Email: contact@insightspike-ai.org
- GitHub Issues: [Create an issue](https://github.com/miyauchikazuyoshi/InsightSpike-AI/issues)

## 🙏 Acknowledgments

This research builds on insights from neuroscience, graph theory, and bio-inspired computing. Special thanks to the open-source community for foundational tools and libraries.

---

*InsightSpike-AI: Exploring the frontiers of machine insight and analogical reasoning*
