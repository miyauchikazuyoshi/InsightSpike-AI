# 🧠 InsightSpike-AI Colab実行ガイド

Google Colab環境でInsightSpike-AIを実行するための完全ガイドです。

## 📋 前提条件

- Google Colabアカウント
- **GPU Runtime必須** (Runtime > Change runtime type > GPU)

## 🚀 クイックスタート（Colab）

### 方法1: Jupyter Notebook使用

1. **専用ノートブックを開く**
   - [`InsightSpike_Colab_Demo.ipynb`](InsightSpike_Colab_Demo.ipynb) をColabで開く
   - セルを順番に実行

### 方法2: ターミナルコマンド使用

```bash
# 1. リポジトリクローン
!git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
%cd InsightSpike-AI

# 2. 環境セットアップ
!chmod +x scripts/setup_colab.sh
!./scripts/setup_colab.sh

# 3. 問題がある場合の診断
!python scripts/colab_diagnostic.py

# 4. データ準備
!PYTHONPATH=src python scripts/databake.py

# 5. メモリ構築
!PYTHONPATH=src python -m insightspike.cli embed --path data/raw/test_sentences.txt

# 6. グラフ構築
!PYTHONPATH=src python -m insightspike.cli graph

# 7. PoC実行
!PYTHONPATH=src python scripts/run_poc.py "What is quantum entanglement?"
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

| エラー | 原因 | 解決方法 |
|--------|------|----------|
| `CUDA not available` | GPU Runtimeが無効 | Runtime > Change runtime type > GPU |
| `ModuleNotFoundError` | 依存関係未インストール | `!python scripts/colab_diagnostic.py` 実行 |
| `FileNotFoundError: episodic memory` | データ未生成 | `!PYTHONPATH=src python scripts/databake.py` 実行 |
| `Poetry lock failed` | バージョン競合 | `!pip install` で直接インストール |
| `Out of memory` | GPU/RAMリソース不足 | Runtime restart または小さなデータセット使用 |

### 診断コマンド

```bash
# 環境診断（推奨）
!python scripts/colab_diagnostic.py

# 手動確認
!nvidia-smi  # GPU確認
!python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
!python -c "import faiss; print(f'Faiss: {faiss.__version__}')"
```

## 📊 期待される出力

正常に動作している場合の出力例：

```
=== Loop 1 ===
ΔGED: -0.15, ΔIG: 0.08, Eureka: False
更新エピソード数: 3

=== Loop 2 ===
ΔGED: -0.62, ΔIG: 0.25, Eureka: True  ← エウレカスパイク！
更新エピソード数: 7
```

## 🎯 パフォーマンスチューニング

### GPU最適化

```python
# バッチサイズ調整
os.environ['BATCH_SIZE'] = '16'  # デフォルト：32

# メモリ効率化
torch.cuda.empty_cache()  # GPU メモリクリア
```

### データサイズ調整

```python
# 小規模テスト用
!PYTHONPATH=src python scripts/databake.py --max_sentences 1000

# 大規模実験用（Colab Pro推奨）
!PYTHONPATH=src python scripts/databake.py --max_sentences 50000
```

## 📈 実験パラメータ

主要なハイパーパラメータ：

```python
# config.py で調整可能
SPIKE_GED = 0.5      # ΔGED閾値
SPIKE_IG = 0.2       # ΔIG閾値
LOOP_NUM = 10        # 探索ループ数
EMBED_DIM = 384      # 埋め込み次元数
```

## 🔬 研究用機能

### データセット変更

```python
# カスタムコーパス使用
!echo "Your custom text here" > data/raw/custom.txt
!PYTHONPATH=src python -m insightspike.cli embed --path data/raw/custom.txt
```

### 可視化オプション

```python
# matplotlib有効化
import matplotlib.pyplot as plt
plt.style.use('seaborn')  # 見やすいスタイル

# GNNグラフ可視化
!PYTHONPATH=src python -c "
from insightspike.layer3_graph_pyg import visualize_graph
visualize_graph('data/graph_pyg.pt')
"
```

## 💡 Tips

1. **Runtime定期再起動**: 長時間実行時はメモリリーク防止のため
2. **段階的実行**: 大きなデータセットは段階的にテスト
3. **ログ保存**: 実験結果を`/content/drive/MyDrive/`に保存
4. **GPU使用量モニタリング**: `!nvidia-smi` で定期確認

---

## 🔗 リンク

- [メインREADME](README.md)
- [開発者ガイド](docs/CONTRIBUTING.md)
- [ロードマップ](docs/ROADMAP.md)
- [Issues](https://github.com/miyauchikazuyoshi/InsightSpike-AI/issues)
