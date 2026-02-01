# 迷路実験における構造類似度の導入

## 背景

現在の迷路実験では、geDIGのIG項は以下で構成されている:

```
ΔIG = ΔH_norm + γ・ΔSP_rel
```

しかし、**構造類似度（Structural Similarity）** は含まれていない。

知識グラフでの成功（F1 +60%）を迷路でも再現できれば、
**geDIGの構造評価がスケール普遍的に有効**であることを示せる。

---

## 現在の8元ベクトル

```python
[0] x_norm:         正規化x座標 [0, 1]
[1] y_norm:         正規化y座標 [0, 1]
[2] dx:             x方向成分 {-1, 0, 1}
[3] dy:             y方向成分 {-1, 0, 1}
[4] wall_flag:      壁/通路 {-1.0, 1.0}
[5] log_visits:     訪問回数(対数) [0, ∞)
[6] success_outcome: 成功/失敗指標 [-1, 1]
[7] goal_flag:      ゴールフラグ {0, 1}
```

**問題点**: 局所的な構造パターン情報が欠如している

---

## 迷路の構造パターン

迷路には基本的に5種類のパターンしかない:

```
1. 直線 (Corridor)     2. L字 (L-turn)        3. T字路 (T-junction)
   ─ ─ ─                  ─ ┐                    ─ ┬ ─
                            │                      │

4. 十字路 (Cross)      5. 行き止まり (Dead-end)
     │                       │
   ─ ┼ ─                     ╵
     │
```

これらのパターンは**グラフ構造として表現可能**:
- 直線: 次数2、直線的接続
- L字: 次数2、90度曲がり
- T字路: 次数3
- 十字路: 次数4
- 行き止まり: 次数1

---

## 提案: 拡張ベクトル設計

### Option A: 10元ベクトル（パターンID追加）

```python
[0-7]  : 既存の8元ベクトル
[8]    : local_pattern_id (0-4: dead-end, corridor, L, T, cross)
[9]    : neighbor_pattern_similarity (近傍との構造類似度)
```

### Option B: 12元ベクトル（パターン埋め込み追加）

```python
[0-7]  : 既存の8元ベクトル
[8]    : degree (次数: 1-4)
[9]    : linearity (直線性: 0-1)
[10]   : branching (分岐度: 0-1)
[11]   : connectivity (連結性スコア)
```

### Option C: 16元ベクトル（4方向構造情報追加）

```python
[0-7]  : 既存の8元ベクトル
[8-11] : N,E,S,W方向の「先の構造」情報
         (その方向に進んだ場合のパターン予測)
[12-15]: 4方向の「構造的価値」スコア
         (ゴールへの構造的近さ)
```

---

## 構造類似度の計算方法

### 1. 局所パターン間の類似度

```python
def pattern_similarity(pattern_a: int, pattern_b: int) -> float:
    """
    パターン間の構造類似度

    パターンID:
    0: dead-end (次数1)
    1: corridor (次数2, 直線)
    2: L-turn   (次数2, 曲がり)
    3: T-junction (次数3)
    4: cross    (次数4)
    """
    # 次数ベースの類似度行列
    similarity_matrix = [
        #  DE   CO   L    T    CR
        [1.0, 0.3, 0.3, 0.2, 0.1],  # dead-end
        [0.3, 1.0, 0.7, 0.5, 0.3],  # corridor
        [0.3, 0.7, 1.0, 0.6, 0.4],  # L-turn
        [0.2, 0.5, 0.6, 1.0, 0.8],  # T-junction
        [0.1, 0.3, 0.4, 0.8, 1.0],  # cross
    ]
    return similarity_matrix[pattern_a][pattern_b]
```

### 2. 探索グラフとゴール構造の類似度

```python
def structural_similarity_to_goal(
    explored_graph: nx.Graph,
    goal_structure: nx.Graph,
    current_pos: Tuple[int, int]
) -> float:
    """
    探索済みグラフの「ゴール方向の構造」との類似度
    """
    # 現在位置からの部分グラフを抽出
    subgraph = extract_neighborhood(explored_graph, current_pos, radius=3)

    # グラフ編集距離ベースの類似度
    ged = compute_ged(subgraph, goal_structure)
    max_ged = max(len(subgraph), len(goal_structure))

    return 1.0 - (ged / max_ged)
```

---

## 拡張IG項の設計

### 現在

```
F = ΔEPC - λ・ΔIG
ΔIG = ΔH_norm + γ・ΔSP_rel
```

### 提案

```
F = ΔEPC - λ・ΔIG
ΔIG = ΔH_norm + γ・ΔSP_rel + β・ΔSS_local
                              ↑
                    構造類似度項を追加
```

ここで:
- `ΔSS_local`: 移動前後での「ゴール構造への類似度」の変化
- `β`: 構造類似度の重み（ハイパーパラメータ）

---

## 実験設計

### Phase 1: パターン認識の追加（最小変更）

1. 現在位置の局所パターンを認識
2. 8元ベクトルに2次元追加 → 10元ベクトル
3. AG/DGの計算に構造類似度を含める

**期待される改善**:
- デッドエンドの早期回避
- T字路での効率的な探索

### Phase 2: 大域構造の評価

1. 探索済みグラフの構造埋め込み
2. ゴール方向の構造との比較
3. IG項への統合

**期待される改善**:
- 「ゴールに向かう構造」の早期認識
- 全体的なステップ数削減

### Phase 3: 比較実験

| 条件 | IG項の構成 |
|------|-----------|
| Baseline | ΔH + γ・ΔSP |
| +Pattern | ΔH + γ・ΔSP + β・ΔSS_pattern |
| +Global | ΔH + γ・ΔSP + β・ΔSS_global |
| Full | ΔH + γ・ΔSP + β₁・ΔSS_pattern + β₂・ΔSS_global |

**評価指標**:
- 成功率
- 平均ステップ数
- 探索効率（訪問ノード数/最短路長）
- AG/DG発火パターン

---

## 実装計画

### ファイル構成

```
experiments/structural_similarity_maze/
├── DESIGN.md                    # 本ドキュメント
├── src/
│   ├── pattern_detector.py      # 局所パターン検出
│   ├── extended_vector.py       # 拡張ベクトル生成
│   ├── structural_similarity.py # 構造類似度計算
│   ├── extended_gedig.py        # 拡張geDIG (IG項に構造類似度)
│   └── navigator.py             # 拡張ナビゲータ
├── configs/
│   ├── baseline.yaml
│   ├── with_pattern.yaml
│   └── full.yaml
├── scripts/
│   ├── run_experiment.py
│   └── analyze_results.py
└── results/
    └── (実験結果)
```

### 実装順序

1. `pattern_detector.py`: 5種類のパターン検出
2. `extended_vector.py`: 10元/12元ベクトル生成
3. `structural_similarity.py`: パターン間類似度計算
4. `extended_gedig.py`: IG項への統合
5. `navigator.py`: 拡張ナビゲータ
6. 実験スクリプト

---

## 期待される成果

### 定量的

| 指標 | 現在 | 期待（+SS） |
|------|------|------------|
| 15×15 成功率 | 98% | 99%+ |
| 15×15 平均ステップ | ~150 | ~120 (20%削減) |
| 51×51 成功率 | ~50% | ~70% |
| デッドエンド検出 | 後手 | 早期 |

### 定性的

1. **スケール普遍性の証拠**
   - 迷路でも知識グラフでも構造類似度が有効

2. **論文ストーリーの一貫性**
   - 迷路（理解しやすい）→ 知識グラフ（応用）→ 閃き（発見）

3. **geDIG設計原理の強化**
   - 「構造を見る」が全レベルで機能

---

## 次のステップ

1. [ ] `pattern_detector.py` の実装
2. [ ] 10元ベクトルへの拡張
3. [ ] 小規模テスト（5×5迷路）
4. [ ] 15×15での比較実験
5. [ ] 結果分析と論文への統合
