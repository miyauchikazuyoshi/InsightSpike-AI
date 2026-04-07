# SP計算GPU高速化検討書

## 概要

geDIG実験においてSP（Shortest Path）計算がボトルネックとなっている。GPU（cuGraph）を活用した高速化の可能性を検討する。

## 現状分析

### ボトルネック箇所

`src/insightspike/algorithms/sp_distcache.py`:

```python
# All-Pairs Shortest Path - O(V*(V+E))
nx.all_pairs_shortest_path_length(g)

# Single-Source Shortest Path - O(V+E)
nx.single_source_shortest_path_length(g, src)
```

### 現在の最適化

- `--sp-pair-samples 128`: ペアサンプリングで計算量削減
- `--sp-cand-topk 12`: 候補数制限
- キャッシュ機構（`DistanceCache`クラス）

### 実測値（51x51迷路）

| 設定 | avg_time_ms | p95_time_ms |
|------|-------------|-------------|
| Baseline (512/24, hops=15) | 264.8ms | 1714ms |
| Optimized (128/12, hops=10) | 42.5ms | 291ms |

最適化で84%高速化（6.2倍）済みだが、さらなる高速化の余地あり。

## GPU高速化の可能性

### アプローチ比較

| アプローチ | ライブラリ | 実装難易度 | 期待効果 | 備考 |
|-----------|-----------|-----------|---------|------|
| **cuGraph SSSP** | NVIDIA RAPIDS | 低 | 5-10倍 | ドロップイン置換可能 |
| 行列べき乗法 | PyTorch/CuPy | 中 | 3-5倍 | 密グラフ向き |
| Batched BFS | CuPy | 中 | 5-10倍 | 複数ソース並列 |
| PyTorch Geometric | PyG | 中 | 3-5倍 | GNN向けだがSP計算も可 |

### 推奨: cuGraph

**理由**:
1. NetworkXとの互換APIで移行コストが低い
2. NVIDIA公式でメンテナンスが安定
3. Colab Pro+のA100で動作確認済み

## 実装計画

### Phase 1: 実験ディレクトリでプロトタイプ

```
experiments/maze/
  sp_gpu.py          ← 新規作成（GPU版SP計算）
  run_paper_51x51.py ← 環境変数でGPU切り替え
```

**実装案**:

```python
# experiments/maze/sp_gpu.py
"""GPU-accelerated SP computation using cuGraph."""
import os
from typing import Dict, Any

_USE_GPU = os.getenv("INSIGHTSPIKE_SP_GPU", "0") == "1"

if _USE_GPU:
    try:
        import cugraph
        import cudf
        _HAS_CUGRAPH = True
    except ImportError:
        _HAS_CUGRAPH = False
        print("Warning: cuGraph not available, falling back to CPU")
else:
    _HAS_CUGRAPH = False

import networkx as nx


def sssp(g: nx.Graph, src: Any) -> Dict[Any, int]:
    """Single-Source Shortest Path with optional GPU acceleration."""
    if _HAS_CUGRAPH and _USE_GPU:
        return _sssp_gpu(g, src)
    return dict(nx.single_source_shortest_path_length(g, src))


def _sssp_gpu(g: nx.Graph, src: Any) -> Dict[Any, int]:
    """GPU implementation using cuGraph."""
    # Convert to cuGraph format
    G_cu = cugraph.from_networkx(g)

    # Run SSSP on GPU
    df = cugraph.sssp(G_cu, src)

    # Convert back to dict
    return dict(zip(df['vertex'].to_pandas(), df['distance'].to_pandas().astype(int)))


def apsp(g: nx.Graph) -> Dict[Any, Dict[Any, int]]:
    """All-Pairs Shortest Path with optional GPU acceleration."""
    if _HAS_CUGRAPH and _USE_GPU:
        return _apsp_gpu(g)
    return dict(nx.all_pairs_shortest_path_length(g))


def _apsp_gpu(g: nx.Graph) -> Dict[Any, Dict[Any, int]]:
    """GPU implementation using cuGraph."""
    result = {}
    G_cu = cugraph.from_networkx(g)

    for node in g.nodes():
        df = cugraph.sssp(G_cu, node)
        result[node] = dict(zip(df['vertex'].to_pandas(), df['distance'].to_pandas().astype(int)))

    return result
```

**使用方法**:

```bash
# CPU（従来通り）
python experiments/maze/run_paper_51x51.py

# GPU
INSIGHTSPIKE_SP_GPU=1 python experiments/maze/run_paper_51x51.py
```

### Phase 2: 効果検証

| 検証項目 | 方法 |
|---------|------|
| 速度比較 | 25x25, 51x51で計測 |
| 結果一致 | CPU/GPU結果の差分確認 |
| メモリ使用量 | nvidia-smiで監視 |

### Phase 3: メインコードへの統合（オプショナル）

効果が確認できた場合、`src/insightspike/algorithms/sp_distcache.py`に統合:

```python
# オプショナル依存として追加
try:
    import cugraph
    _HAS_CUGRAPH = True
except ImportError:
    _HAS_CUGRAPH = False

class DistanceCache:
    def __init__(self, *, mode: str = "core", pair_samples: int = 400, use_gpu: bool = False):
        self.use_gpu = use_gpu and _HAS_CUGRAPH
        # ...
```

## 依存関係

### cuGraph インストール

```bash
# Colab
!pip install cugraph-cu12 cudf-cu12

# ローカル（CUDA 12.x）
conda install -c rapidsai -c conda-forge -c nvidia cugraph cuda-version=12.0
```

### 要件

- NVIDIA GPU（Compute Capability 7.0+）
- CUDA 11.x または 12.x
- 十分なGPUメモリ（グラフサイズ依存、51x51なら1GB程度）

## リスクと対策

| リスク | 対策 |
|-------|------|
| cuGraph未インストール環境 | フォールバックでCPU使用 |
| GPU/CPU結果不一致 | 浮動小数点は整数に丸める |
| GPUメモリ不足 | バッチサイズ調整、警告表示 |
| Colab無料版はGPUなし | CPU版で動作保証 |

## タイムライン

| フェーズ | 内容 | 期間 |
|---------|------|------|
| Phase 1 | プロトタイプ実装 | 1日 |
| Phase 2 | 効果検証 | 1日 |
| Phase 3 | メイン統合（任意） | 1日 |

## 結論

1. **短期**: 現状の最適化（pair_samples=128, hops=10）で十分実用的
2. **中期**: cuGraph導入で5-10倍の追加高速化が見込める
3. **実装方針**: 実験ディレクトリでプロトタイプ → 効果確認 → メイン統合

## 参考

- [NVIDIA cuGraph Documentation](https://docs.rapids.ai/api/cugraph/stable/)
- [cuGraph GitHub](https://github.com/rapidsai/cugraph)
- [NetworkX to cuGraph Migration Guide](https://docs.rapids.ai/api/cugraph/stable/nx_transition/)
