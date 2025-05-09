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

## 📦 Quick start (Docker)
```bash
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI
docker compose up --build
python demo_cli.py

🔬 Research Roadmap

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
git clone https://github.com/your-username/InsightSpike-AI.git
cd InsightSpike-AI
chmod +x scripts/setup.sh
./scripts/setup.sh


# 2) Poetry 管理下の残りの依存をインストール
poetry install

# サンプルデータを用いたデモ実行
python examples/demo.py

# あるいは Notebook で確認
jupyter notebook examples/demo.ipynb
yaml
コピーする
編集する

[![CI](https://github.com/miyauchikazuyoshi/InsightSpike-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/InsightSpike-AI/actions/workflows/ci.yml)
