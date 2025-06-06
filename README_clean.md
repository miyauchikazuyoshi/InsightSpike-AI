# InsightSpike-AI
**Brain-Inspired Multi-Agent Architecture for "Spike of Insight" (ΔGED × ΔIG)**

> Quantized RAG ＋ GNN ＋ Internal Reward (ΔGED/ΔIG)  
> Implementing a cerebellum–LC–hippocampus–VTA loop to study *insight*.

[![License: InsightSpike Community License](https://img.shields.io/badge/License-InsightSpike--Community--1.0-blue)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/LICENSE)  
<a href="https://arxiv.org/abs/YYMM.NNNNN"><img src="https://img.shields.io/badge/arXiv-YYMM.NNNNN-b31b1b.svg" alt="arXiv"></a>  
<a href="https://github.com/miyauchikazuyoshi/InsightSpike-AI/releases"><img src="https://img.shields.io/github/v/release/miyauchikazuyoshi/InsightSpike-AI"></a>

> **⚠️ 免責事項**: このプロジェクトは **概念実証 (Proof-of-Concept)** 段階です。実験結果にはモック実装とシミュレーションデータが含まれています。実装における制限と改善点については [documentation/reports/](documentation/reports/) をご確認ください。

## Patent Notice
The core ΔGED/ΔIG intrinsic-reward mechanism and the hierarchical VQ memory module are **patent-pending** in Japan.

- JP Application No. **特願2025-082988** — "ΔGED/ΔIG 内発報酬生成方法"
- JP Application No. **特願2025-082989** — "階層ベクトル量子化による動的メモリ方法"

Further filings (US/PCT) will follow within the priority year.

---

## ✨ Why
Human "aha!" moments often arise from abrupt structural re-arrangements of episodic memory.  
InsightSpike-AI models this process and exposes the *spike* as an internal reward signal.

### Key Features
* **ΔGED** – Graph-edit distance between successive RAG search graphs  
* **ΔIG** – Entropy gain from concept restructuring
* **VQ Memory** – Vector quantized episodic memory with FAISS
* **GNN Reasoning** – Graph neural network with PyTorch Geometric
* **Insight Detection** – EurekaSpike fires when ΔGED drops ≥ 0.5 and ΔIG rises ≥ 0.2

## 🧠 Architecture (v0.7-Eureka)

Proof‑of‑concept brain‑inspired architecture with a 4‑layer subcortical loop.

| Layer | Brain Analog      | Main File                       | Function                          |
|-------|-------------------|---------------------------------|-----------------------------------|
| L1    | Cerebellum        | `layer1_error_monitor.py`       | Query analysis & topK optimization|
| L2    | LC + Hippocampus  | `layer2_memory_manager.py`      | Vector quantized episodic memory  |
| L3    | PFC               | `layer3_graph_reasoner.py`      | GNN reasoning with ΔGED/ΔIG      |
| L4    | Language Area     | `layer4_llm.py`                 | Natural language synthesis       |

**Enhanced Features (v0.7-Eureka)**:
- 📋 Intelligent known/unknown information separation
- 🎯 Automatic synthesis requirement detection  
- 🔄 Adaptive topK optimization for chain reaction insights
- 🧠 Human-like learning system with weak relationship formation
- 📚 Vector quantized episodic memory with IVF-PQ
- 🕸️ Graph neural network reasoning with enhanced graph density
- ⚡ Real-time insight spike detection

---

## 🚀 Quick Start

### 🎯 Interactive Demo
Try the insight detection capabilities immediately:

```bash
# Clone and setup
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Run interactive demo (no setup required)
poetry run insightspike demo
```

This demo showcases InsightSpike's ability to synthesize insights across domains like probability theory, mathematics, and philosophy - even when the knowledge base contains no direct answers to the questions!

### 🏠 Local Development Setup

**Automated platform detection with Poetry**:
```bash
# Clone repository
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Automated setup with platform detection
poetry install --with dev

# Verify installation
poetry run insightspike --help
poetry run python -c "import torch, faiss; print(f'PyTorch: {torch.__version__}, FAISS: {faiss.__version__}')"
```

**Platform-Specific Support**:
- 🍎 **macOS**: torch==2.2.2 + faiss-cpu (Intel/AMD compatibility)
- 🐧 **Linux**: torch>=2.4.0 + faiss-gpu (CI/GPU environments)
- 🪟 **Windows**: torch>=2.4.0 + faiss-cpu (fallback)

### ⚡ Google Colab Setup (GPU Optimized)

**One-Click Setup with Colab Notebook**:
1. Open [`InsightSpike_Colab_Demo.ipynb`](InsightSpike_Colab_Demo.ipynb) in Google Colab
2. Choose GPU runtime: Runtime > Change runtime type > GPU  
3. Run cells in order: The notebook guides you through setup and demo

**Manual Setup Options**:
```bash
# Standard setup (8-12 minutes, recommended)
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI
!chmod +x scripts/colab/setup_colab.sh
!./scripts/colab/setup_colab.sh

# Minimal setup (<60 seconds, for testing)
!./scripts/colab/setup_colab.sh minimal

# Debug setup (15-20 minutes, troubleshooting)
!chmod +x scripts/colab/setup_colab_debug.sh
!./scripts/colab/setup_colab_debug.sh
```

**2025 Modern Environment Features**:
- ✅ NumPy 2.x compatibility with intelligent FAISS fallback
- ✅ PyTorch 2.6.0+cu124 for optimal T4 GPU performance
- ✅ Automatic GPU → CPU fallback for compatibility
- ✅ Realistic messaging about available capabilities

---

## 🎮 CLI Commands

| Command                                      | Description                                                                                 |
|-----------------------------------------------|--------------------------------------------------------------------------------------------|
| `poetry run insightspike demo`                | **Run interactive demo of insight detection capabilities**                             |
| `poetry run insightspike ask "question"`      | Ask a question using the MainAgent architecture                                        |
| `poetry run insightspike true_insight`        | **Run rigorous insight detection experiment (no direct answers)**                     |
| `poetry run insightspike compare_experiments` | **Compare different experimental designs (direct vs insight)**                        |
| `poetry run insightspike experiment_suite`    | **Run complete experimental validation suite**                                        |
| `poetry run insightspike benchmark [--dataset]` | **Run performance benchmarks (simple/enhanced/custom)**                                  |
| `poetry run insightspike load_documents <path>` | Load documents into the agent's memory from file or directory                            |
| `poetry run insightspike stats`               | Show agent and memory statistics                                                           |
| `poetry run insightspike config_info`         | Display current configuration settings                                                      |
| `poetry run databake`                         | Download 10,000 Wikipedia sentences, vectorize, and index with FAISS                      |
| `poetry run run-poc`                          | Run the full PoC pipeline with visualization and logging                                   |

---

## ✅ Complete Workflow Example

```bash
# 1. Clone and set up environment
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
poetry install --with dev

# 2. Verify installation
poetry run insightspike --help
poetry run insightspike config_info

# 3. Prepare data (download & vectorize Wikipedia sentences)
poetry run databake

# 4. Test with interactive demo
poetry run insightspike demo

# 5. Run insight detection experiments
poetry run insightspike true_insight

# 6. Run the full PoC pipeline
poetry run run-poc
```

---

## 🧪 Experimental Validation

> **重要**: 以下の結果は概念実証段階のもので、モック実装とシミュレーションを含みます。詳細は [documentation/reports/](documentation/reports/) をご確認ください。

### Rigorous Insight Detection Results

**🎯 合成タスクでの改善確認: 108.3% improvement**

- **✅ 83.3% response quality** vs 40.0% baseline (108.3% improvement)
- **✅ 66.7% synthesis rate** vs 0% baseline (cross-domain connections)  
- **✅ 4/6 insight syntheses** on synthesis-requiring questions
- **⚠️ 注意**: 実験設計により、知識ベースには直接的な回答は含まれていません

### Experimental Framework Design

InsightSpike-AI uses two complementary experimental approaches:

#### 🎯 Insight Experiments (Rigorous Validation)
**Novel experimental design with NO direct answers in knowledge base**

- **Indirect Knowledge Base**: 57 facts containing only related concepts, NOT direct answers
- **Synthesis-Required Questions**: 6 questions requiring genuine cross-domain reasoning
- **Examples**: Monty Hall (probability + information theory), Zeno's paradox (calculus + motion), Ship of Theseus (philosophy + practical criteria)
- **Validation**: Tests true synthesis capability rather than information retrieval

#### 📊 Traditional Experiments (Legacy Framework)
**Standard evaluation on cognitive paradoxes with complete knowledge base**

- **Direct Knowledge Base**: Contains answers alongside related information  
- **Cognitive Paradoxes**: Monty Hall variations, mathematical paradoxes, philosophical questions
- **Validation**: Tests insight detection on known challenging problems

### Test Commands
```bash
# Run rigorous insight experiment
poetry run insightspike true_insight

# Compare experimental designs
poetry run insightspike compare_experiments

# Run complete validation suite
poetry run insightspike experiment_suite
```

📄 **Full Reports**: 
- [`EXPERIMENTAL_VALIDATION_REPORT.md`](EXPERIMENTAL_VALIDATION_REPORT.md) - Traditional framework results
- [`COMPARATIVE_EXPERIMENTAL_ANALYSIS.md`](COMPARATIVE_EXPERIMENTAL_ANALYSIS.md) - Cross-validation analysis

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

## 🔧 Troubleshooting

### Common Issues

| Error Message                                      | Cause / Solution                                                                                 |
|----------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: No module named 'matplotlib'`| Run `poetry install --with dev` to include dev dependencies                                     |
| `FileNotFoundError: ... episodic memory ...`       | Run `poetry run databake` to generate the required data files                                   |
| `CUDA not available`                               | Expected on CPU-only machines; CPU-mode FAISS will be used automatically                        |
| `RuntimeError: Failed to load embedding model`     | Pre-download SentenceTransformer model and set `EMBED_MODEL_PATH` environment variable          |

### Environment-Specific Troubleshooting
- **Google Colab**: See [Colab Troubleshooting Guide](documentation/guides/COLAB_TROUBLESHOOTING_GUIDE.md)
- **Local Development**: Check Poetry environment with `poetry env info`
- **CI/CD**: Set `INSIGHTSPIKE_LITE_MODE=1` for minimal testing dependencies

---

## 📦 Dependency Management Strategy

**Multi-Environment Compatibility (2025)**:
- ✅ **Local Development**: NumPy 1.x ecosystem (stable, all features)
- ✅ **Google Colab 2025**: NumPy 2.x adaptation (intelligent fallbacks)
- ✅ **CI/Testing**: NumPy 1.x controlled (consistent testing)
- ✅ **Production Deployment**: Environment-aware (adapts to host NumPy version)

**Key Innovations**:
- **Environment Detection**: Automatic NumPy version detection and adaptation
- **FAISS Intelligence**: GPU attempt → CPU fallback for NumPy 2.x compatibility
- **Realistic Messaging**: Clear expectations about available capabilities
- **Graceful Degradation**: Full functionality maintained regardless of FAISS mode

---

## Development Commands

**Makefile shortcuts**:
- `make test` — Run test suite
- `make embed` — Generate test memory data
- `make clean` — Clean test artifacts

**Environment configuration**:
- Create `.env` file for environment variables like `PYTHONPATH=src`
- Add API keys or custom data directories as needed
