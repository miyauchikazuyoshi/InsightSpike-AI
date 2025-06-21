# InsightSpike-AI Colab Experiments 🚀

Google Colab 上で GPU を使って Phase 1〜4 の実験をワンクリック実行できるノートブックを配置しています。

| Phase | Notebook | 主な内容 |
|-------|----------|----------|
| Phase 1 – Dynamic Memory | `phase1_dynamic_memory.ipynb` | 動的記憶構築実験（GPU/FAISS GPU 有効化） |
| Phase 2 – RAG Benchmark  | `phase2_rag_benchmark.ipynb` | LangChain / LlamaIndex / Haystack / InsightSpike 比較実験（GPU 推論） |
| Phase 3 – GEDIG Maze     | `phase3_gedig_maze.ipynb` | GED×IG 迷路実験 + GIF 可視化（GPU 版 PyG 使用） |
| Phase 4 – Integrated Evaluation | `phase4_integrated_evaluation.ipynb` | 4 フェーズ統合評価（統合 CLI 実行） |

各ノートブックは共通セットアップセルで以下を実行します。
1. リポジトリのクローン
2. `scripts/colab/setup_unified.sh` を実行して依存をインストール
3. 対応する `experiments/phase*/XXXX_experiment.py` を `--quick` + GPU 設定で実行

> **注**: Colab T4/Tesla GPU でも動作しますが、FAISS + PyG の CUDA バイナリは毎回ビルドに数分かかります。無料枠では時間制限にご注意ください。 