# gedig_core.py 依存関係分析

**作成日**: 2026-02-01
**Phase**: 1 (準備)
**Status**: 完了 ✓ - 重複コード削除済み

---

## 1. ファイル構造

### 1.1 クリーンアップ後 (1,685行)

```
gedig_core.py (1,685行) ← 2,159行から削減
│
├── Lines 1-18: Imports
├── Lines 20-23: ProcessingMode (Enum)
├── Lines 25-28: SpikeDetectionMode (Enum)
├── Lines 31-59: HopResult (@dataclass)
├── Lines 61-97: GeDIGResult (@dataclass)
├── Lines 99-120: LinksetMetrics (@dataclass) ← 正しい位置に移動
├── Lines 122-1402: GeDIGCore (メインクラス)
├── Lines 1404-1541: GeDIGMonitor
├── Lines 1543-1546: GeDIGPresets
├── Lines 1549-1629: GeDIGLogger
└── Lines 1632-1684: Convenience Functions (calculate_gedig, delta_ged, delta_ig)
```

### 1.2 クリーンアップ前 (参考)

```
gedig_core.py (2,159行) ← 削除済み
├── Lines 1663-2000: ⚠️ 重複/異常コード ← 削除済み
├── Lines 2060-2138: Convenience Functions (重複定義) ← 削除済み
└── Lines 2140-2158: LinksetMetrics (ファイル末尾) ← 正しい位置に移動
```

## 2. 発見された問題（解決済み）

### 2.1 コード重複 ✅ Resolved

~~以下の関数/クラスが**2回定義**されていた：~~

| 項目 | 状態 |
|------|------|
| `GeDIGLogger` | ✅ 重複削除 |
| `calculate_gedig` | ✅ 重複削除 |
| `detect_insight_spike` | ✅ 重複削除 |
| `delta_ged` | ✅ 重複削除 |
| `delta_ig` | ✅ 重複削除 |

### 2.2 構文異常 ✅ Resolved

~~Line 1663-1685にメソッド定義の途中でクラス定義がネストされていた~~

→ 重複コード（Lines 1686-2181）を削除して解決

### 2.3 LinksetMetrics位置 ✅ Resolved

~~ファイル末尾（Line 2140-2158）に定義されていた~~

→ GeDIGResultの直後（Line 99-120）に移動

### 2.3 環境変数の散在 🟡 Medium

GeDIGCore.__init__内で12個の環境変数を参照：

```python
MAZE_GEDIG_LAMBDA
MAZE_GEDIG_NODE_COST
MAZE_GEDIG_EDGE_COST
MAZE_GEDIG_EFF_WEIGHT
MAZE_GEDIG_SPECTRAL
MAZE_GEDIG_SPECTRAL_WEIGHT
MAZE_GEDIG_IG_MODE
MAZE_GEDIG_IG_NORM
MAZE_GEDIG_ENTROPY_TAU / INSIGHTSPIKE_ENTROPY_TAU
MAZE_GEDIG_IG_NONNEG
INSIGHTSPIKE_GED_MIN_DIAG
MAZE_GEDIG_SP_BOUNDARY
```

---

## 3. 依存関係グラフ

### 3.1 外部依存

```
gedig_core.py
    │
    ├── networkx (nx)
    ├── numpy (np)
    ├── collections.deque
    ├── dataclasses
    ├── enum.Enum
    ├── logging
    ├── math
    ├── os
    ├── time
    └── typing
```

### 3.2 内部依存

```
gedig_core.py
    │
    ├── .core.metrics
    │   ├── normalized_ged (→ _calculate_normalized_ged)
    │   └── entropy_ig (→ _calculate_entropy_variance_ig)
    │
    ├── .structural_similarity
    │   ├── StructuralSimilarityEvaluator
    │   └── StructuralSimilarityConfig
    │
    ├── .sp_distcache
    │   └── DistanceCache
    │
    ├── .linkset_adapter
    │   └── build_linkset_info
    │
    └── ..config.models
        └── StructuralSimilarityConfig
```

### 3.3 クラス間依存

```
                    ┌─────────────────┐
                    │  GeDIGConfig    │ (未実装・目標)
                    └────────┬────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌──────────┐          ┌──────────────┐         ┌──────────────┐
│HopResult │◀─────────│  GeDIGCore   │────────▶│ GeDIGResult  │
└──────────┘          └──────┬───────┘         └──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌───────────┐  ┌───────────┐  ┌─────────────────┐
       │ GeDIGMonitor│  │GeDIGLogger│  │LinksetMetrics │
       └───────────┘  └───────────┘  └─────────────────┘
```

### 3.4 GeDIGCoreメソッド依存関係

```
calculate()
    │
    ├── _ensure_networkx()
    │       └── _pyg_to_networkx()
    │
    ├── _extract_features()
    │
    ├── _compute_ged_min_proxy()
    │       └── _avg_shortest_path_length_safe()
    │
    ├── _compute_linkset_metrics()
    │
    ├── _calculate_multihop()                    [if enable_multihop]
    │       ├── _extract_k_hop_subgraph()
    │       ├── _calculate_normalized_ged()
    │       ├── _filter_features()
    │       ├── _calculate_entropy_variance_ig()
    │       ├── _compute_sp_gain_norm()
    │       │       ├── _avg_shortest_path_length_safe()
    │       │       └── sp_distcache.DistanceCache
    │       └── _trim_terminal_edges()
    │
    ├── _calculate_normalized_ged()              [single-hop path]
    │       └── core.metrics.normalized_ged
    │
    ├── _calculate_entropy_variance_ig()
    │       └── core.metrics.entropy_ig
    │
    ├── _update_ig_stats()
    ├── _compute_ig_z()
    ├── _compute_rewards()
    │
    └── _detect_spike()
            └── _ig_variance()
```

---

## 4. 分割境界の確認

### 4.1 types.py への移動対象

| 項目 | 行範囲 | 依存 |
|------|--------|------|
| ProcessingMode | 20-23 | なし |
| SpikeDetectionMode | 25-28 | なし |
| HopResult | 31-59 | なし |
| GeDIGResult | 61-97 | HopResult, LinksetMetrics |
| LinksetMetrics | 2140-2158 | なし |

**循環依存なし** ✓

### 4.2 config.py への移動対象

- 環境変数読み込みロジック（GeDIGCore.__init__内の12箇所）
- GeDIGPresets（1520-1523）

### 4.3 ged.py への移動対象

- `_calculate_normalized_ged()` (1156-1175)
- 関連: `_graph_efficiency()` (959-970)
- 関連: `_calculate_spectral_score()` (1147-1153)

### 4.4 ig.py への移動対象

- `_calculate_entropy_variance_ig()` (1177-1239)
- `_calculate_local_entropies()` (1241-1265)
- `_update_ig_stats()` (1268-1273)
- `_ig_variance()` (1275-1278)
- `_compute_ig_z()` (1318-1324)

### 4.5 spike.py への移動対象

- `_detect_spike()` (1341-1376)

### 4.6 multihop.py への移動対象

- `_calculate_multihop()` (734-956)
- `_extract_k_hop_subgraph()` (1084-1094)
- `_compute_sp_gain_norm()` (1018-1046)
- `_avg_shortest_path_length_safe()` (972-1016)
- `_trim_terminal_edges()` (1048-1082)
- `_compute_ged_min_proxy()` (1280-1316)

### 4.7 monitor.py への移動対象

- GeDIGMonitor (1381-1518)

### 4.8 logger.py への移動対象

- GeDIGLogger (1526-1606)

---

## 5. 緊急対応事項

### 5.1 重複コード削除（Phase 2前に必須）

```bash
# 削除対象行
Lines 1663-2000  # 異常なネスト・重複コード
Lines 2060-2138  # 3回目の関数定義（重複）
```

### 5.2 テスト影響確認

重複削除前に確認が必要なテスト：
- `test_gedig_core_*.py`
- `test_spike_detection_*.py`
- 迷路実験の回帰テスト

---

## 6. ベースラインテスト結果

**実行日時**: 2026-02-01
**コマンド**: `pytest tests/ -v --tb=short`

| 結果 | 件数 |
|------|------|
| Passed | 600 |
| Failed | 46 |
| Skipped | 82 |
| Errors | 8 |
| **合計** | **736** |
| **実行時間** | 88.67s |

### 主要な失敗テスト

```
test_spike_detection_core
test_gedig_small_maze_stability
test_layer_integration
```

→ これらはリファクタリング前から失敗しているため、回帰テストの基準から除外

---

## 7. 次のステップ

### Phase 1 完了 ✓

- [x] ベースラインテスト結果を記録
- [x] 依存関係グラフを作成
- [x] 重複コード削除（474行削減: 2,159行 → 1,685行）
- [x] テストが通ることを確認（600 passed, 変化なし）

### Phase 2 準備完了

1. ~~重複コードを削除してクリーンな状態にする~~ ✅ 完了
2. types.py を作成（循環依存なしで移動可能）
3. LinksetMetrics, HopResult, GeDIGResult, ProcessingMode, SpikeDetectionMode を移動

---

## 参照

- [refactoring_plan.md](./refactoring_plan.md) - 全体計画
- [repository_review_and_plan.md](./repository_review_and_plan.md) - プロジェクト全体のレビュー
