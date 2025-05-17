# InsightSpike-AI
**Brain-Inspired Multi-Agent Architecture for “Spike of Insight” (ΔGED × ΔIG)**  

> Quantized RAG ＋ GNN ＋ Internal Reward (ΔGED/ΔIG)  
> Implementing a cerebellum–LC–hippocampus–VTA loop to study *insight*.

[![License: InsightSpike-OpenRAIL-M](https://img.shields.io/badge/License-InsightSpike--OpenRAIL--M-blue)](./LICENSE)

## ✨ Why
Human “aha!” moments often arise from abrupt structural re-arrangements of episodic memory.  
InsightSpike-AI models this process and exposes the *spike* as an internal reward signal.

## 🧠 Architecture (MVP)
<br>L1 Error Monitor (τ_err)<br>L2 Quantum-RAG + C-value (Faiss)<br>L3 GNN + ΔGED/ΔIG + Conflict Score<br>L4 LLM interface<br>

<!-- <p align="center"><img src="docs/diagram/overview_v0.png" width="70%"></p> -->

# InsightSpike AI (v0.7‑Eureka)

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
!pip install -q faiss-gpu==1.7.4.post118

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

```bash
## 環境構築

以下の順番で実行してください。

```bash
## 環境構築

まずはリポジトリをクローンし、スクリプトを実行してください：

```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
chmod +x scripts/setup.sh
./scripts/setup.sh

# サンプルデータを用いたデモ実行
python examples/demo.py

# あるいは Notebook で確認
jupyter notebook examples/demo.ipynb
```

[![CI](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci.yml)
