# InsightSpike-AI
**Brain## ✨ Why
Human "aha!" moments often arise from abrupt structural re-arrangements of episodic memory.  
InsightSpike-AI models this process and exposes the *spike* as an internal reward signal.

> **⚠️ 免責事項**: このプロジェクトは **概念実証 (Proof-of-Concept)** 段階です。実験結果にはモック実装とシミュレーションデータが含まれています。実装における制限と改善点については [documentation/reports/](documentation/reports/) をご確認ください。spired Multi-Agent Architecture for “Spike of Insight” (ΔGED × ΔIG)**  

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

## ⚡ Quick Start on Google Colab (GPU)

### 🐳 Docker版 - 超高速セットアップ（推奨）
最新のDocker版により、1分で完全環境構築が可能になりました！

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyauchikazuyoshi/InsightSpike-AI/blob/main/InsightSpike_Docker_Colab_Setup.ipynb)

**特徴:**
- ⚡ **1分セットアップ**: 複雑な依存関係管理不要
- 🔒 **完全再現性**: Docker環境による一貫した動作
- 🚀 **即座利用**: Pre-built Imageで最速起動
- 🛠️ **開発対応**: ソースからのカスタマイズも可能

```python
# Colab内で1行実行
!wget https://raw.githubusercontent.com/miyauchikazuyoshi/InsightSpike-AI/main/scripts/colab/setup_docker.py && python setup_docker.py
```

### 🎯 従来版 - One-Click Setup
The easiest way to get started is with our simplified Colab notebook:

1. **Open the notebook**: [`InsightSpike_Colab_Demo.ipynb`](InsightSpike_Colab_Demo.ipynb) in Google Colab
2. **Choose GPU runtime**: Runtime > Change runtime type > GPU  
3. **Run cells in order**: The notebook guides you through setup and demo

### 🚀 Validated Setup Scripts

#### ⚡ Standard Setup (8-12 minutes, recommended)
```bash
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!chmod +x scripts/colab/setup_colab.sh
!./scripts/colab/setup_colab.sh
```

#### 🔥 Minimal Setup (<60 seconds, for testing)
```bash
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!chmod +x scripts/colab/setup_colab.sh
!./scripts/colab/setup_colab.sh minimal
```

#### 📋 Production Setup (10-15 minutes, complete)
```bash
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!chmod +x scripts/colab/setup_colab.sh
!./scripts/colab/setup_colab.sh
```

#### 🔍 Debug Setup (15-20 minutes, troubleshooting)
```bash
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!chmod +x scripts/colab/setup_colab_debug.sh
!./scripts/colab/setup_colab_debug.sh
```

> **✅ All setup scripts are fully tested and validated**  
> See [`scripts/colab/VALIDATION_SUMMARY.md`](scripts/colab/VALIDATION_SUMMARY.md) for detailed validation results.

#### 🚀 Ultra-Fast Setup (<60 seconds)
```bash
# Essential dependencies only, good for testing
scripts/colab/setup_colab.sh minimal
```

#### 📋 Standard Setup (8-12 minutes)
```bash
# Complete installation with all features
scripts/colab/setup_colab.sh standard
```

#### 🔍 Debug Setup (15-20 minutes)
```bash
# Detailed logging for troubleshooting
scripts/colab/setup_colab.sh debug
```

### 📔 Interactive Notebook
Our simplified [`InsightSpike_Colab_Demo.ipynb`](InsightSpike_Colab_Demo.ipynb) provides:
- **One-click setup** with 4 options (fast/minimal/standard/debug)
- **Guided demo** with preset questions
- **Built-in troubleshooting** guide
- **Clean interface** - no complex menus or redundant options

💡 **Simplified**: We've streamlined the Colab experience! The new notebook uses our comprehensive setup script, eliminating redundant setup options and complex menus for a much cleaner user experience.

### 🔧 Troubleshooting
If you encounter issues, see our [Colab Troubleshooting Guide](documentation/guides/COLAB_TROUBLESHOOTING_GUIDE.md).

### Development & PoC/Experiment Environment Setup
---
## 🚀 Quick Start

### 🔧 Three-Environment Installation Strategy

InsightSpike-AI supports three distinct environments, each optimized for specific use cases:

#### 🏠 Local Development Environment (faiss-cpu)
**Best for**: Development, testing, CPU-only machines

**✅ DEPENDENCY CONFLICTS RESOLVED**: NumPy 1.x compatibility across all packages
```bash
# Clone repository
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Automated setup with dependency resolution
./scripts/setup/setup.sh

# Alternative: Manual Poetry setup
poetry install --with dev

# Verify installation
poetry run insightspike --help
poetry run config-info
```

**Key Benefits**:
- ✅ NumPy 1.26.4 + FAISS 1.11.0 + spaCy 3.7.5 compatibility 
- ✅ Poetry-managed dependencies with resolved lock file
- ✅ Full development environment with testing tools

#### ☁️ Google Colab Environment (2025 T4 GPU Optimized)
**Best for**: GPU acceleration, large-scale experiments, research

**🔧 MODERN ENVIRONMENT COMPATIBILITY**: NumPy 2.x with intelligent FAISS fallback
**2025 Reality**: Google Colab now ships with NumPy 2.2.6+ pre-installed. Our setup intelligently handles this modern environment:

```bash
# Method 1: Enhanced automated setup (recommended)
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!chmod +x scripts/colab/setup_colab.sh
!bash scripts/colab/setup_colab.sh
```

**Setup Options** (modern NumPy 2.x compatible):
- `setup_colab.sh` - Smart FAISS installation with GPU/CPU fallback (8-12 min)
- `setup_colab.sh minimal` - Ultra-fast essential only (<60 sec)
- `setup_colab.sh debug` - Detailed logging for troubleshooting (15-20 min)
- `setup_colab_debug.sh` - Alternative debug script with comprehensive diagnostics

**Modern 2025 Strategy**:
1. **NumPy 2.2.6+** (pre-installed in 2025 Colab, modern standard)
2. **Intelligent FAISS handling**: Try GPU → fallback to CPU if NumPy 2.x incompatible
3. **PyTorch 2.6.0+cu124** for optimal T4 GPU performance
4. **Realistic expectations**: CPU-mode FAISS for compatibility when needed
5. **Modern setup**: No forced downgrades, work with Colab's modern environment

```bash
# Method 2: Use pre-configured notebook (simplified)
# Open: InsightSpike_Colab_Demo.ipynb
```

**Verification Commands**:
```bash
# Test modern environment compatibility
!python -c "import numpy, faiss; print(f'NumPy: {numpy.__version__} (2025 standard), FAISS: {faiss.__version__} ({\"GPU\" if hasattr(faiss, \"StandardGpuResources\") else \"CPU\"} mode)')"
!poetry run insightspike --help
```

#### 🔧 CI/Testing Environment (minimal dependencies)
**Best for**: Continuous integration, automated testing

**✅ DEPENDENCY CONFLICTS RESOLVED**: LITE_MODE with NumPy 1.x compatibility
```bash
# Poetry-based CI setup (automated via .github/workflows/ci.yml)
./scripts/setup/setup.sh
export INSIGHTSPIKE_LITE_MODE=1

# Alternative: Minimal installation for CI
pip install pytest numpy==1.26.4 pyyaml networkx scikit-learn psutil faiss-cpu typer rich click
pip install -e .

# Run tests with environment detection
python -m pytest development/tests/unit/ -v
```

**Key Benefits**:
- ✅ NumPy 1.x compatibility maintained in CI
- ✅ Mock models for fast testing (no model downloads)
- ✅ Unified setup script across all environments

### 📦 Strategic Dependency Management (2025 Updated)

**🔧 MODERN ENVIRONMENT ADAPTATION**: Cross-platform compatibility with NumPy 2.x reality

**Current Status**: 
- ✅ Local Development: **NumPy 1.x ecosystem** (stable, all features)
- ✅ Google Colab 2025: **NumPy 2.x adaptation** (intelligent fallbacks)
- ✅ CI/Testing: **NumPy 1.x controlled** (consistent testing)
- ✅ Production Deployment: **Environment-aware** (adapts to host NumPy version)

Our multi-environment approach ensures optimal performance across different deployment contexts:

- **`dev`**: Local development with Poetry + NumPy 1.26.4 + FAISS 1.11.0 (stable ecosystem)
- **`colab-2025`**: Google Colab with NumPy 2.2.6+ + intelligent FAISS fallback (modern compatibility)  
- **`ci`**: CI testing with Poetry + NumPy 1.26.4 + LITE_MODE (consistent testing)

**Key Innovation**: 
- **Environment Detection**: Automatic NumPy version detection and adaptation
- **FAISS Intelligence**: GPU attempt → CPU fallback for NumPy 2.x compatibility
- **Realistic Messaging**: No false promises about GPU when not available
- **Modern Standards**: Work with 2025 environments rather than forcing downgrades
- **Graceful Degradation**: Full functionality maintained even with CPU-only FAISS

**Resolution Status**: ✅ MODERNIZED FOR 2025
```bash
# Current Environment Compatibility
✅ Local Development - NumPy 1.x stable ecosystem with all features
✅ Google Colab 2025 - NumPy 2.x adaptation with intelligent FAISS handling
✅ Dependency Intelligence - Automatic version detection and fallback
✅ Production Ready - Environment-aware deployment
✅ Realistic User Experience - Clear expectations, no false promises
```

**Modern Environment Handling**:
```bash
# 2025 REALITY: Adaptive approach
✅ NumPy 2.2.6+ (2025 Colab standard, work with it)
✅ FAISS GPU attempt → CPU fallback (graceful degradation) 
✅ PyTorch 2.6.0+cu124 (optimal T4 performance)
✅ Intelligent messaging (realistic expectations)
✅ Full functionality maintained regardless of FAISS mode
```

**Requirements File Structure**:
```
deployment/configs/
├── requirements-colab.txt              # Poetry-managed (NumPy 1.x coordinated)
├── requirements-colab-comprehensive.txt # Complete reference documentation
├── requirements-torch.txt              # PyTorch with CUDA support
└── requirements-PyG.txt                # PyTorch Geometric components
```

**Development Setup**:
```bash
# Enhanced development environment with resolved dependencies
poetry install --with dev

# Verify installation 
poetry run insightspike --help
poetry run config-info

# Check resolved dependencies
poetry run python -c "import numpy, faiss, spacy; print(f'NumPy: {numpy.__version__}, FAISS: {faiss.__version__}, spaCy: {spacy.__version__}')"
```

For development, PoC, or experiments, the full development environment includes:
- ✅ matplotlib (for visualization) 
- ✅ pytest (for testing)
- ✅ All resolved dependencies with NumPy 1.x ecosystem

When running `run_poc.py` offline, set the environment variable `EMBED_MODEL_PATH` to a locally downloaded SentenceTransformer model directory.

### Docker
---
The included `Dockerfile` is based on `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`. It installs dependencies from `pyproject.toml` using Poetry, and then installs additional packages such as `torch-geometric` via `pip`. Note that local scripts use `torch==2.2.2`, so be aware of version differences. The base image uses **Python 3.10**, which differs from the Python 3.11 series required in `pyproject.toml`. Also, after installing `faiss-cpu` with Poetry, `faiss-gpu-cu11` is added; if you do not need the CPU version, please uninstall it.

**Note**: Docker configuration needs updating for NumPy 1.x compatibility. Use local or Colab environments for immediate usage.

## ✅ Minimal Working Example (Dependency Conflicts Resolved)

```bash
# 1. Clone and set up environment with resolved dependencies
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# 2. Automated setup with dependency resolution
./scripts/setup/setup.sh

# 3. Verify resolved installation
poetry run insightspike --help
poetry run config-info

# 4. Check dependency compatibility
poetry run python -c "import numpy, faiss, spacy; print(f'✅ NumPy: {numpy.__version__}, FAISS: {faiss.__version__}, spaCy: {spacy.__version__}')"

# 5. Prepare data (download & vectorize Wikipedia sentences)
poetry run python scripts/databake.py

# 6. Embed your own corpus (Specify any text file)
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

## 🧪 概念実証段階の実験結果 (Experimental Validation - Proof-of-Concept Stage)

> **重要**: 以下の結果は概念実証段階のもので、モック実装とシミュレーションを含みます。詳細は [BIAS_CONFIRMATION_AND_IMPROVEMENT_REPORT.md](BIAS_CONFIRMATION_AND_IMPROVEMENT_REPORT.md) をご確認ください。

InsightSpike-AIは2つの異なる実験フレームワークを通じて制御された実験によってテストされています。

### 洞察検出実験結果 (Rigorous Validation)

**🎯 合成タスクでの改善を確認: 108.3% improvement**

- **✅ 83.3% response quality** vs 40.0% baseline (108.3% improvement)
- **✅ 66.7% synthesis rate** vs 0% baseline (cross-domain connections)  
- **✅ 4/6 insight syntheses** on synthesis-requiring questions
- **⚠️ 注意**: 実験設計により、知識ベースには直接的な回答は含まれていません

### 従来実験フレームワーク結果 (Legacy Framework - Mock Implementation)

> **⚠️ モック実装による結果**: 以下の結果はハードコードされた応答を含む概念実証です

- **133.3% improvement** in response quality (模擬実験)
- **100% insight detection rate** on cognitive paradoxes (予め定義された応答)
- **0% false positive rate** on control questions (制御された条件)
- **287x faster processing** (シミュレーション環境での測定)

### 新しい実験フレームワークの実行

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
**Novel experimental design with NO direct answers in knowledge base**

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

Results demonstrate that the ΔGED/ΔIG mechanism effectively identifies significant moments in cognitive processing, validating the core hypothesis of spike-based insight detection.

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
