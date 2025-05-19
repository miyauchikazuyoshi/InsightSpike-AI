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

## 🧠 Architecture (MVP)
L1 Error Monitor (τ_err)<br>L2 Quantum-RAG + C-value (Faiss)<br>L3 GNN + ΔGED/ΔIG + Conflict Score<br>L4 LLM interface<br>

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

## Quick‑start on GoogleColab(GPU)
```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Development & PoC/Experiment Environment Setup
---
For development, PoC, or experiments, please make sure to install all dependencies including dev packages:
```poetry install```<br>
Or, if you want to explicitly include only dev dependencies:```poetry install --with dev```

This will ensure that packages like matplotlib (for visualization) and pytest (for testing) are available in your environment.

### Docker
---
The included `Dockerfile` is based on `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`. It installs dependencies from `pyproject.toml` using Poetry, and then installs additional packages such as `torch-geometric` via `pip`. Note that local scripts use `torch==2.2.2`, so be aware of version differences. The base image uses **Python 3.10**, which differs from the Python 3.11 series required in `pyproject.toml`. Also, after installing `faiss-cpu` with Poetry, `faiss-gpu-cu11` is added; if you do not need the CPU version, please uninstall it.

## Minimal Working Example

```bash
# 1. Clone and set up environment
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
chmod +x scripts/setup.sh
./scripts/setup.sh

# 2. Prepare data (download & vectorize Wikipedia sentences)
poetry run databake

# 3. Embed your own corpus (任意のテキストファイルを指定)
# 例: data/raw/your_corpus.txt をMemory化
poetry run insightspike embed --path data/raw/your_corpus.txt

# 4. Build similarity graph
poetry run insightspike graph

# 5. Run a reasoning loop with a sample question
poetry run insightspike loop "What is quantum entanglement?"

# 6. Run the PoC pipeline (with visualization)
poetry run run-poc
```

---

## CLI Commands

| Command                                      | Description                                                                                 |
|-----------------------------------------------|--------------------------------------------------------------------------------------------|
| `poetry run insightspike embed --path <file>` | Embed a text corpus and save episodic memory (vectorization)                               |
| `poetry run insightspike graph`               | Build a similarity graph from episodic memory                                              |
| `poetry run insightspike loop "question"`     | Run one L1-L4 reasoning cycle with the given question                                      |
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

## Makefile コマンド例
- `make test` ... テスト実行
- `make embed` ... テスト用Memory生成
- `make clean` ... テスト成果物の削除

## .envファイルについて
- `PYTHONPATH=src` など、環境変数を一元管理できます。
- 必要に応じて `DATA_DIR` や `API_KEY` などを追加してください。
