# InsightSpike-AI
**Brain-Inspired Multi-Agent Architecture for “Spike of Insight” (ΔGED × ΔIG)**  

> Quantized RAG ＋ GNN ＋ Internal Reward (ΔGED/ΔIG)  
> Implementing a cerebellum–LC–hippocampus–VTA loop to study *insight*.

[![License: InsightSpike Community License](https://img.shields.io/badge/License-InsightSpike--Community--1.0-blue)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/LICENSE)  
<a href="https://arxiv.org/abs/YYMM.NNNNN"><img src="https://img.shields.io/badge/arXiv-YYMM.NNNNN-b31b1b.svg" alt="arXiv"></a>  
<a href="https://github.com/miyauchikazuyoshi/InsightSpike-AI/releases"><img src="https://img.shields.io/github/v/release/miyauchikazuyoshi/InsightSpike-AI"></a>

## Patent Notice
The core ΔGED/ΔIG intrinsic-reward mechanism and the hierarchical VQ memory module
are **patent-pending** in Japan.

- JP Application No. **特願2025-082988** — “ΔGED/ΔIG 内発報酬生成方法”
- JP Application No. **特願2025-082989** — “階層ベクトル量子化による動的メモリ方法”

<br> Further filings (US/PCT) will follow within the priority year.

---

### ✨ Features
* **ΔGED** – Graph-edit distance between successive RAG search graphs  
* **ΔIG** – Entropy gain from*

## ✨ Why
Human “aha!” moments often arise from abrupt structural re-arrangements of episodic memory.  
InsightSpike-AI models this process and exposes the *spike* as an internal reward signal.

## 🧠 Architecture (Enhanced v0.7-Eureka)

**Layer1: Enhanced Known/Unknown Information Separation**  
- 📋 Intelligent query analysis and concept extraction
- 🎯 Automatic synthesis requirement detection  
- 🔄 Adaptive topK optimization for chain reaction insights
- 🧠 Human-like learning system with weak relationship formation

**Layer2: Quantum-RAG + C-value Memory (Faiss)**  
- 📚 Vector quantized episodic memory with IVF-PQ
- 🔍 Adaptive retrieval with Layer1-optimized topK values

**Layer3: GNN + ΔGED/ΔIG + Conflict Score**  
- 🕸️ Graph neural network reasoning with PyTorch Geometric
- 📊 ΔGED/ΔIG metrics for insight spike detection
- ⚡ Enhanced graph density for chain reaction insights

**Layer4: LLM Interface**  
- 🗣️ Natural language generation with TinyLlama
- 🎨 Context-aware response synthesis

<!-- <p align="center"><img src="docs/diagram/overview_v0.png" width="70%"></p> -->

# InsightSpike AI (v0.7-Eureka)

Proof‑of‑concept brain‑inspired architecture with a 4‑layer subcortical loop.

| Layer | Brain analog  | Main file(s)                  |
|-------|---------------|-------------------------------|
| L1    | Cerebellum    | layer1_error_monitor.py       |
| L2    | LC + Hippocampus | layer2_memory_manager.py  |
| L3    | PFC           | layer3_graph_pyg.py,<br>layer3_reasoner_gnn.py |
| L4    | Language area | layer4_llm.py                 |

EurekaSpike fires when **ΔGED drops ≥ 0.5** *and* **ΔIG rises ≥ 0.2**.

---

## Quick‑start (local)
```bash
git clone ...
cd InsightSpike-AI
chmod +x [setup.sh]
[setup.sh]
```

## 🚀 Quick Demo

Try the insight detection capabilities immediately:

```bash
# Clone and setup
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Run interactive demo (no setup required)
poetry run insightspike demo
```

This demo showcases InsightSpike's ability to synthesize insights across domains like probability theory, mathematics, and philosophy - even when the knowledge base contains no direct answers to the questions!

## Quick‑start on GoogleColab(GPU)
```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Development & PoC/Experiment Environment Setup
---
## 🚀 Quick Start

### 🔧 Three-Environment Installation Strategy

InsightSpike-AI supports three distinct environments, each optimized for specific use cases:

#### 🏠 Local Development Environment (faiss-cpu)
**Best for**: Development, testing, CPU-only machines
```bash
# Clone repository
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Install dependencies for local development
poetry install --with dev

# Run CLI commands
poetry run insightspike loop "What is quantum entanglement?"
```

#### ☁️ Google Colab Environment (faiss-gpu-cu12)
**Best for**: GPU acceleration, large-scale experiments, research
```bash
# Method 1: Enhanced setup script (recommended)
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!chmod +x scripts/colab/setup_colab.sh
!bash scripts/colab/setup_colab.sh

# Method 2: Use pre-configured notebook
# Open: InsightSpike_Colab_Demo.ipynb
```

#### 🔧 CI/Testing Environment (minimal dependencies)
**Best for**: Continuous integration, automated testing
```bash
# Minimal installation for CI
pip install pytest numpy pyyaml networkx scikit-learn psutil faiss-cpu typer rich click
pip install -e .
export INSIGHTSPIKE_LITE_MODE=1

# Run tests
python -m pytest development/tests/unit/ -v
```

### 📦 Dependency Strategy
- **`dev`**: Local development with faiss-cpu, full testing suite
- **`colab`**: Google Colab optimized with faiss-gpu priority installation
- **`ci`**: Minimal dependencies for fast CI/CD pipelines

**Key Innovation**: Our setup prioritizes faiss-gpu installation in Colab before Poetry operations, ensuring GPU acceleration while maintaining compatibility across all environments.

For development, PoC, or experiments, please make sure to install all dependencies including dev packages:
```poetry install --with dev```

This ensures packages like matplotlib (for visualization) and pytest (for testing) are available.
When running `run_poc.py` offline, set the environment variable `EMBED_MODEL_PATH` to a locally downloaded SentenceTransformer model directory.

### Docker
---
The included `Dockerfile` is based on `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`. It installs dependencies from `pyproject.toml` using Poetry, and then installs additional packages such as `torch-geometric` via `pip`. Note that local scripts use `torch==2.2.2`, so be aware of version differences. The base image uses **Python 3.10**, which differs from the Python 3.11 series required in `pyproject.toml`. Also, after installing `faiss-cpu` with Poetry, `faiss-gpu-cu11` is added; if you do not need the CPU version, please uninstall it.

## Minimal Working Example

```bash
# 1. Clone and set up environment
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
chmod +x scripts/colab_setup.sh
./scripts/setup.sh
!chmod +x scripts/databake.py
!chmod +x scripts/run_poc.py

# 2. Prepare data (download & vectorize Wikipedia sentences)
poetry run python scripts/databake.py

# 3. Embed your own corpus (Specify any text file)
# Example: Convert data/raw/your_corpus.txt into episodic memory
# *Note: Each line in the text file is treated as a separate document.*
poetry run insightspike embed --path data/raw/your_corpus.txt

# 4. Build similarity graph
poetry run insightspike graph

# 5. Run a reasoning loop with a sample question
poetry run insightspike loop "What is quantum entanglement?"

# 6. Run the PoC pipeline (with visualization)
poetry run python scripts/run_poc.py
```

---

## CLI Commands

| Command                                      | Description                                                                                 |
|-----------------------------------------------|--------------------------------------------------------------------------------------------|
| `poetry run insightspike ask "question"`      | Ask a question using the new MainAgent architecture                                        |
| `poetry run insightspike demo`                | **Run interactive demo of insight detection capabilities**                             |
| `poetry run insightspike load_documents <path>` | Load documents into the agent's memory from file or directory                            |
| `poetry run insightspike stats`               | Show agent and memory statistics                                                           |
| `poetry run insightspike config_info`         | Display current configuration settings                                                      |
| `poetry run insightspike true_insight`        | **Run rigorous insight detection experiment (no direct answers)**                     |
| `poetry run insightspike compare_experiments` | **Compare different experimental designs (direct vs insight)**                        |
| `poetry run insightspike experiment_suite`    | **Run complete experimental validation suite with multiple experiment types**              |
| `poetry run insightspike experiment`          | **Run complete experimental validation suite** *(Legacy - use experiment_suite)*           |
| `poetry run insightspike benchmark [--dataset]` | **Run performance benchmarks (simple/enhanced/custom)**                                  |
| `poetry run insightspike embed --path <file>` | *(Legacy)* Embed a text corpus and save episodic memory                                   |
| `poetry run insightspike query "question"`    | *(Legacy)* Run one L1-L4 reasoning cycle                                                   |
| `poetry run databake`                         | Download 10,000 Wikipedia sentences, vectorize, and index with faiss                      |
| `poetry run run-poc`                          | Run the full PoC pipeline with visualization and logging                                   |

---

## Common Errors & Troubleshooting

| Error Message                                      | Cause / Solution                                                                                 |
|----------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: No module named 'matplotlib'`| Run `poetry install` to include dev dependencies, or add `matplotlib` to your environment.       |
| `FileNotFoundError: ... episodic memory ...`       | Run `poetry run insightspike embed` or `poetry run databake` to generate the required data files.|
| `torch version mismatch`                           | Ensure Docker and local environments use the same torch version (see Dockerfile notes).          |
| `CUDA not available`                               | If running on CPU, make sure to use CPU versions of torch/faiss; for GPU, check CUDA drivers.    |
| `RuntimeError: Failed to load embedding model`     | Pre-download the SentenceTransformer model and set `EMBED_MODEL_PATH` to its directory.           |

---

## Data Preparation & Preprocessing

To obtain 10,000 sentences from Wikipedia, save them in `data/raw/`, vectorize them using sentence-transformers, and index them with faiss:

```bash
poetry run databake
```

## PoC (Proof of Concept) Usage

To run the PoC pipeline after data preparation, follow these steps:

1. **Start the main process**  
    Run the main script to launch the multi-agent architecture:
    ```bash
    poetry run run-poc
    ```

2. **Monitor outputs**  
    Results, logs, and EurekaSpike events will be saved in the `outputs/` directory.

3. **Experiment with parameters**  
    You can adjust parameters such as thresholds for ΔGED and ΔIG in the configuration files (e.g., `config.yaml`) to observe different behaviors.

For detailed experiments or custom runs, refer to the scripts in the `experiments/` directory.

---

## 🧪 Experimental Validation

InsightSpike-AI has been rigorously tested through controlled experiments demonstrating its effectiveness in detecting cognitive insights and improving response quality through two distinct experimental frameworks.

### Insight Detection Experiment Results (Rigorous Validation)

**🎯 Breakthrough: 108.3% improvement in synthesis tasks requiring genuine cross-domain reasoning**

- **✅ 83.3% response quality** vs 40.0% baseline (108.3% improvement)
- **✅ 66.7% synthesis rate** vs 0% baseline (successful cross-domain connections)  
- **✅ 4/6 successful insight syntheses** on questions with NO direct answers in knowledge base
- **✅ Insight detection** validates genuine reasoning rather than mere information retrieval

### Traditional Experiment Results (Legacy Framework)

- **✅ 133.3% improvement** in response quality over baseline systems
- **✅ 100% insight detection rate** on cognitive paradoxes (Monty Hall, Zeno's, Ship of Theseus)
- **✅ 0% false positive rate** on control questions
- **✅ 287x faster processing** than baseline approaches

### Running New Experimental Framework

```bash
# Run rigorous insight experiment (no direct answers in knowledge base)
poetry run insightspike true_insight

# Compare experimental designs (traditional vs insight)
poetry run insightspike compare_experiments

# Run complete validation suite with multiple experiment types
poetry run insightspike experiment_suite

# Legacy experiments
poetry run python scripts/databake_simple.py
poetry run python scripts/run_poc_simple.py
```

### Experimental Framework Design

InsightSpike-AI uses two complementary experimental approaches to validate its insight detection capabilities:

#### 🎯 Insight Experiments (Rigorous Validation)
**Revolutionary experimental design with NO direct answers in knowledge base**

- **Indirect Knowledge Base**: 57 facts containing only related concepts, NOT direct answers
- **Synthesis-Required Questions**: 6 questions requiring genuine cross-domain reasoning
- **Examples**: Monty Hall (probability + information theory), Zeno's paradox (calculus + motion), Ship of Theseus (philosophy + practical criteria)
- **Validation**: Tests true synthesis capability rather than information retrieval

#### 📊 Traditional Experiments (Legacy Framework)
**Standard evaluation on cognitive paradoxes with complete knowledge base**

- **Direct Knowledge Base**: Contains answers alongside related information
- **Cognitive Paradoxes**: Monty Hall problem variations, mathematical paradoxes, philosophical questions
- **Validation**: Tests insight detection on known challenging problems

#### 🔬 Comparative Analysis
The experimental suite includes comparative analysis showing:
- **Insight experiments** eliminate confounding factors and validate genuine synthesis
- **Traditional experiments** demonstrate performance on standard cognitive challenges
- **Cross-validation** ensures robust insight detection across multiple domains

### Experimental Framework

The validation framework tests InsightSpike-AI across multiple cognitive domains:

- **Probability Paradoxes**: Monty Hall problem variations
- **Mathematical Paradoxes**: Zeno's paradox resolution  
- **Philosophical Paradoxes**: Ship of Theseus identity questions
- **Concept Hierarchies**: Mathematical abstraction levels
- **Conceptual Revolutions**: Physics paradigm shifts
- **Control Conditions**: Standard academic content

Results demonstrate that the ΔGED/ΔIG mechanism effectively identifies breakthrough moments in cognitive processing, validating the core hypothesis of spike-based insight detection.

📄 **Full Reports**: 
- [`EXPERIMENTAL_VALIDATION_REPORT.md`](EXPERIMENTAL_VALIDATION_REPORT.md) - Traditional framework results
- [`COMPARATIVE_EXPERIMENTAL_ANALYSIS.md`](COMPARATIVE_EXPERIMENTAL_ANALYSIS.md) - Insight vs traditional comparison

---

## 🎯 Layer1 Enhanced Features (v0.7-Eureka)

### Known/Unknown Information Separation
- **Concept Extraction**: Automatic identification of key concepts using regex and NLP
- **Certainty Analysis**: Context-based confidence scoring for concept familiarity  
- **Synthesis Detection**: Intelligent detection of queries requiring multi-concept synthesis

### Adaptive TopK Optimization  
- **Dynamic Scaling**: topK values scale 1.5x-6x based on query complexity
- **Chain Reaction Enablement**: Higher graph density for "連鎖反応的洞察向上"
- **Layer-Specific Tuning**: L1(20→50), L2(15→30), L3(12→25) adaptive scaling

### Human-Like Learning System
- **Weak Relationships**: Automatic registration of concept co-occurrences (confidence: 0.1)
- **Sleep-Mode Cleanup**: Background pruning of relationships below 0.15 confidence
- **Gradual Reinforcement**: +0.05 confidence boost per concept reappearance
- **Graph Explosion Prevention**: Maximum 1000 weak edges with natural pruning

### Integration Results
- ✅ **75% synthesis prediction accuracy** for Layer1 analysis
- ✅ **2.5x average topK scaling** for complex queries  
- ✅ **329 relationships learned** from 5 test questions
- ✅ **61.6% → 84.1% chain reaction potential** scaling

---

## Makefile コマンド例
- `make test` ... テスト実行
- `make embed` ... テスト用Memory生成
- `make clean` ... テスト成果物の削除

## .envファイルについて
- `PYTHONPATH=src` など、環境変数を一元管理できます。
- 必要に応じて `DATA_DIR` や `API_KEY` などを追加してください。
