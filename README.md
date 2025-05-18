# InsightSpike-AI
**Brain-Inspired Multi-Agent Architecture for “Spike of Insight” (ΔGED × ΔIG)**  

> Quantized RAG ＋ GNN ＋ Internal Reward (ΔGED/ΔIG)  
> Implementing a cerebellum–LC–hippocampus–VTA loop to study *insight*.

[![License: InsightSpike-OpenRAIL-M](https://img.shields.io/badge/License-InsightSpike--OpenRAIL--M-blue)](./LICENSE)  <a href="https://arxiv.org/abs/YYMM.NNNNN"><img src="https://img.shields.io/badge/arXiv-YYMM.NNNNN-b31b1b.svg" alt="arXiv"></a>  <a href="https://github.com/miyauchikazuyoshi/InsightSpike-AI/releases"><img src="https://img.shields.io/github/v/release/miyauchikazuyoshi/InsightSpike-AI"></a>

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

## Quick‑start (local CPU/MPS)
Poetry が未インストールの場合は `pip install poetry` を実行してください。
```bash
poetry install --no-root        # 依存を入れる
poetry run insightspike embed   # L1+L2 初期化
poetry run insightspike graph   # L3 グラフ初期化
poetry run insightspike loop "光速不変が崩れたら？"
```

🔬 Research Roadmap

## Quick‑start on GoogleColab(GPU)
```bash
# 1. リポジトリを Drive 経由で展開（または PAT で clone）
!unzip -q /content/drive/MyDrive/insightspike-ai.zip -d .
%cd insightspike-ai

# 2. GPU 版バイナリをインストール
!pip install -q torch==2.2.2+cu118 torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu118
!pip install -q torch-geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
!pip install -q faiss-gpu-cu11

# 3. 残りの依存
!pip install -q sentence-transformers transformers rich typer scikit-learn networkx

# 4. コーパスを置き、パイプラインを実行
!python -m insightspike.cli embed
!python -m insightspike.cli graph
!python -m insightspike.cli loop "ブラックホールは情報を失うのか？"
```

Phase	Goal	Status
0	License / README / Contrib guide	✅
1	Docker + CI	🟡
2	L1-L4 MVP	⏳
3	QA Benchmark & ΔGED spike demo	⏳
📄 License
InsightSpike Open RAIL-M (research-only) – commercial use requires written permission.
See LICENSE for details.

## PoC の動かし方

## 環境構築

### ローカル環境 (CPU)
以下のスクリプトで Python 3.11 の仮想環境と依存ライブラリを整えます。

```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
chmod +x scripts/setup.sh
./scripts/setup.sh
```

`setup.sh` の処理順序は次の通りです。
1. `.venv` がなければ `python3.11 -m venv .venv` を作成し `source .venv/bin/activate`
2. `pip`, `setuptools`, `wheel` を最新版へ更新
3. `torch==2.2.2` (CPU版) をインストール
4. `torch-scatter` など PyG 関連パッケージを個別に導入
5. `poetry lock --no-cache --regenerate` と `poetry install --no-root` を実行

### Google Colab (GPU)
Colab 上では次を実行します。

```bash
bash scripts/setup_colab.sh
```

`requirements-colab.txt` で CUDA 11.8 対応の PyTorch 2.2.2 と PyG を導入後、
`python -m insightspike.cli embed` が走ります。

### Docker
同梱の `Dockerfile` は `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` をベースに、
Poetry で `pyproject.toml` の依存を入れた後、追加で `torch-geometric` などを
`pip` でインストールします。ローカルのスクリプトでは `torch==2.2.2` を用いる
ため、バージョン差に注意してください。ベースイメージは **Python 3.10** のた
め、`pyproject.toml` で要求する 3.11 系とは異なります。また Poetry で `faiss-cpu`
を入れた後に `faiss-gpu-cu11` を追加しているため、不要であれば CPU 版をアンイ
ンストールしてください。

```bash
# サンプルデータを用いたデモ実行
python examples/demo.py

# あるいは Notebook で確認
jupyter notebook examples/demo.ipynb
[![CI](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci.yml)
```


