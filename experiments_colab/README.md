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

## 🔧 インポート問題の解決

### モジュールパスの修正

正しいInsightSpike-AIモジュールの構造に基づいてインポート文を修正：

```python
# ❌ 間違った形式
from insightspike.agents import MainAgent
from insightspike.memory import KnowledgeGraphMemory

# ✅ 正しい形式
from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory
from insightspike.config import Config
from insightspike.utils import logger
```

### CLIコマンドの改善

`setup_unified.sh`でCLIコマンドを直接パスから実行可能に：

```bash
# 直接実行可能（改善後）
insightspike --version

# モジュール経由（フォールバック）
python -m insightspike.cli.main --version
```

### 依存関係の最適化

`pyproject_colab.toml`で重複依存を削除し、NumPy 2.x互換性を改善。

## 🧪 実験実行手順

1. **統一セットアップ実行**: `scripts/colab/setup_unified.sh`
2. **インポート確認**: 各ノートブックの確認セル実行
3. **FAISS修正**: 自動修正・フォールバック実装
4. **実験実行**: GPU最適化による大規模実験
5. **結果保存**: Google Drive連携・可視化

## 🎉 改善済み機能

- ✅ モジュールインポートの自動修正
- ✅ CLIコマンドの直接実行対応
- ✅ FAISS問題の自動フォールバック
- ✅ NumPy 2.x完全対応
- ✅ Phase3アルゴリズム実装完了
- ✅ 依存関係重複の解決