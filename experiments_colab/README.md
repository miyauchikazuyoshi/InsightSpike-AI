# Experiments Colab - GPU大規模実験環境 🚀

GPU Colab環境での大規模実験実行用ディレクトリです。HuggingFaceライブラリを活用した高性能実験を提供します。

## 🎯 大規模実験の特徴

| Phase | Notebook | GPU最適化 | HuggingFace活用 |
|-------|----------|-----------|----------------|
| Phase 1 – Dynamic Memory | `dynamic_memory_colab.ipynb` | CUDA対応・大規模データセット | sentence-transformers, datasets |
| Phase 2 – RAG Benchmark  | `rag_benchmark_colab.ipynb` | GPU推論・並列処理 | transformers, accelerate |
| Phase 3 – GEDIG Maze     | `gedig_maze_colab.ipynb` | PyTorch Geometric GPU | torch-geometric, plotly |

## 🚀 GPU最適化機能
- **CUDA対応**: 高速計算・大規模データ処理
- **バッチ処理**: メモリ効率化・並列実行
- **HuggingFace統合**: 事前訓練済みモデル・datasets活用

## 📦 依存関係
`pyproject_colab.toml`でColab最適化された環境：
- torch ^2.3.0 (NumPy 2.x対応)
- transformers ^4.40.0
- sentence-transformers ^2.5.0
- torch-geometric ^2.4.0

> **注**: GPU環境（T4/V100）推奨。CUDA最適化により従来比3-10倍の高速化を実現。 