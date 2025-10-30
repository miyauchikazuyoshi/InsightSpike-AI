---
status: active
category: gedig
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# エントロピー分散・マルチホップgeDIG実装計画

## 作成日: 2025-01-27

## 1. 背景と動機

### 1.1 現状の課題
- **IG計算の理論的曖昧さ**: クラスタリングベースで情報理論的根拠が弱い
- **局所性の限界**: 1ホップの直接接続のみ考慮
- **深い洞察の見逃し**: 波及効果や間接的な影響を捉えられない

### 1.2 提案手法の利点
- **理論的明確性**: エントロピーの分散減少 = 情報の均質化
- **マルチスケール分析**: k-hop近傍で異なる粒度の洞察を検出
- **熟考モード**: 深い思考による洞察形成を定量化

## 2. 理論的基盤

### 2.1 エントロピー分散による情報統合
```
IG = Var(H_local_before) - Var(H_local_after)

where:
- H_local(v) = 局所エントロピー（ノードvの近傍）
- Var() = 分散（情報の偏在度）
```

**解釈**: 
- 分散大 → 情報が偏在（不均質）
- 分散小 → 情報が均等分布（統合された）

### 2.2 マルチホップ拡張
```
geDIG(k) = Σ(h=0 to k) w(h) * [GED(h) + IG(h)]

where:
- h = ホップ数
- w(h) = 重み関数（例: decay^h）
- GED(h) = h-hop近傍での構造変化
- IG(h) = h-hop近傍での情報統合
```

## 3. 実装設計

### 3.1 Phase 1: エントロピー分散IG（2日）

#### SimplifiedEntropyVarianceIG
```python
class SimplifiedEntropyVarianceIG:
    """エントロピー分散ベースのIG計算"""
    
    def calculate(self, graph, features_before, features_after):
        # 1. 各ノードの局所エントロピーを計算
        H_local_before = [self._local_entropy(graph, v, features_before) 
                         for v in graph.nodes()]
        H_local_after = [self._local_entropy(graph, v, features_after) 
                        for v in graph.nodes()]
        
        # 2. 分散を計算
        var_before = np.var(H_local_before)
        var_after = np.var(H_local_after)
        
        # 3. 分散の減少 = 情報統合
        return var_before - var_after
    
    def _local_entropy(self, graph, node, features):
        """ノードの局所的なエントロピー"""
        neighbors = list(graph.neighbors(node)) + [node]
        local_features = features[neighbors]
        
        # ヒストグラムベース（シンプル版）
        hist, _ = np.histogram(local_features.flatten(), bins=20)
        probs = hist / hist.sum()
        
        # シャノンエントロピー
        return -np.sum(probs * np.log2(probs + 1e-10))
```

### 3.2 Phase 2: マルチホップ拡張（2日）

#### MultiHopGeDIG
```python
class MultiHopGeDIG:
    """マルチホップgeDIG計算"""
    
    def __init__(self, max_hops=3, decay=0.7):
        self.max_hops = max_hops
        self.decay = decay  # 距離による減衰率
        self.ged_calc = NormalizedGED()
        self.ig_calc = SimplifiedEntropyVarianceIG()
    
    def calculate(self, graph_before, graph_after, features_before, features_after):
        results = {}
        
        for hop in range(self.max_hops + 1):
            # k-hop部分グラフを抽出
            subgraph_before = self._extract_k_hop(graph_before, hop)
            subgraph_after = self._extract_k_hop(graph_after, hop)
            
            # 各hopでのGEDとIG
            ged = self.ged_calc.calculate(subgraph_before, subgraph_after)
            ig = self.ig_calc.calculate(subgraph_after, features_before, features_after)
            
            # 重み付き統合
            weight = self.decay ** hop
            results[f'hop_{hop}'] = {
                'ged': ged,
                'ig': ig,
                'weight': weight,
                'weighted_gedig': weight * (ged + ig)
            }
        
        # 総合スコア
        total_gedig = sum(r['weighted_gedig'] for r in results.values())
        
        return {
            'total': total_gedig,
            'details': results
        }
```

### 3.3 Phase 3: 統合とテスト（1日）

#### 統合ポイント
1. MetricsSelectorへの組み込み
2. フィーチャーフラグの追加
   - `use_entropy_variance_ig`
   - `enable_multihop_gedig`
   - `max_hops`
   - `hop_decay`

## 4. 実験計画

### 4.1 基礎実験
```python
# 1. シンプルなグラフでの検証
# Before: A-B  C-D  （2つの独立成分）
# After:  A-B-E-C-D（Eがハブとして接続）

hop=0: Eの追加のみ検出
hop=1: B,Cへの影響を検出
hop=2: A,Dまでの波及効果を検出
```

### 4.2 数学実験での比較
```python
# 問題: "三角形の内角の和は？"

hop=0: 単純な幾何学的事実
hop=1: 角度の性質、平行線の理解
hop=2: ユークリッド幾何学、公理系への展開
hop=3: 非ユークリッド幾何学への一般化
```

### 4.3 期待される効果
- **浅い洞察 vs 深い洞察**の区別
- **概念の波及効果**の定量化
- **思考の深さ**の可視化

## 5. 評価指標

### 5.1 定量評価
- 各hopでのgeDIG値の変化
- 最適hop数の自動決定
- 計算時間とのトレードオフ

### 5.2 定性評価
- 検出された洞察の「深さ」
- 人間の直感との一致度
- 説明可能性の向上

## 6. リスクと対策

### 6.1 計算コスト
- **リスク**: hop数増加でO(n^k)の計算量
- **対策**: 
  - 適応的hop数（変化が小さければ早期終了）
  - 並列計算の活用
  - キャッシュの利用

### 6.2 過剰な一般化
- **リスク**: 遠すぎる関連まで考慮
- **対策**:
  - 適切な減衰率の設定
  - 最大hop数の制限（通常3-4）

## 7. 実装スケジュール

| フェーズ | 期間 | タスク |
|---------|------|--------|
| Phase 1 | 2日 | エントロピー分散IG実装 |
| Phase 2 | 2日 | マルチホップ拡張 |
| Phase 3 | 1日 | 統合・テスト |
| Phase 4 | 1日 | 実験・評価 |

合計: 6日間

## 8. 成功基準

1. **理論的妥当性**: 情報理論の専門家が納得する実装
2. **実用的価値**: 深い洞察の検出精度向上
3. **計算効率**: hop=3でも実用的な速度
4. **後方互換性**: 既存実験への影響を最小化

## 9. 将来の拡張

### 9.1 適応的ホップ数
```python
# 情報利得が閾値以下になったら停止
if ig_gain(hop) / ig_gain(hop-1) < 0.1:
    break
```

### 9.2 方向性を考慮
```python
# 入力方向と出力方向で異なる重み
in_hop_weight = 0.8 ** hop
out_hop_weight = 0.6 ** hop
```

### 9.3 時間的マルチホップ
```python
# 過去のエピソードも考慮
temporal_hops = get_temporal_neighbors(episode, time_window)
```

## 10. 参考文献

- Shannon, C.E. (1948). A Mathematical Theory of Communication
- Estrada, E. & Hatano, N. (2008). Communicability in complex networks
- Burt, R.S. (2005). Brokerage and Closure: An Introduction to Social Capital
- Newman, M.E.J. (2018). Networks (2nd ed.)

---

これで**シンプルかつ理論的に堅固**なgeDIG実装が実現できます！🚀